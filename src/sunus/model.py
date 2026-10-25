from __future__ import annotations

import http.client
import html
import json
import re
import socket
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

from sunus.config import RuntimeConfig, make_ssl_context, redact_text
from sunus.model_events import ModelEvent


class ModelError(RuntimeError):
    pass


class ModelCancelled(RuntimeError):
    pass


@dataclass
class ModelReply:
    message: dict[str, Any]
    usage: dict[str, Any]
    finish_reason: str


_DSML_DELIMITER = r"(?:｜{1,2}|\|{1,2})"
_DSML_MARKER_RE = re.compile(
    rf"<\s*(?P<close>/\s*)?(?P<delimiter>{_DSML_DELIMITER})"
    r"\s*DSML\s*(?P=delimiter)\s*"
    r"(?P<tag>[A-Za-z_][\w-]*)(?P<attrs>[^>]*)>",
    re.IGNORECASE,
)
_DSML_HINT_RE = re.compile(
    rf"<\s*/?\s*{_DSML_DELIMITER}\s*DSML\b",
    re.IGNORECASE,
)
_DSML_ATTRIBUTE_RE = re.compile(
    r"([A-Za-z_][\w:-]*)\s*=\s*(['\"])(.*?)\2",
    re.DOTALL,
)
_DSML_CALL_TAGS = {
    "calls",
    "function_calls",
    "functioncalls",
    "tool_calls",
    "toolcalls",
}
_DSML_INVOKE_TAGS = {"invoke", "function_call", "tool_call"}
_DSML_PARAMETER_TAGS = {"parameter", "argument", "param"}


def _dsml_attributes(raw: str) -> dict[str, str]:
    return {
        name.lower(): value
        for name, _, value in _DSML_ATTRIBUTE_RE.findall(raw)
    }


def _dsml_parameter_value(raw: str, raw_attributes: str) -> Any:
    attributes = _dsml_attributes(raw_attributes)
    value = html.unescape(raw)
    if attributes.get("string", "").lower() == "true":
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value.strip()


def _extract_dsml_calls(
    content: str,
) -> tuple[list[dict[str, Any]], str | None, bool]:
    matches = list(_DSML_MARKER_RE.finditer(content))
    if not matches:
        return [], None, bool(_DSML_HINT_RE.search(content))

    calls: list[dict[str, Any]] = []
    visible: list[str] = []
    cursor = 0
    current: dict[str, Any] | None = None
    parameter_name: str = ""
    malformed = False

    for match in matches:
        tag = match.group("tag").lower().replace("-", "_")
        closing = bool(match.group("close"))

        if parameter_name:
            if tag in _DSML_PARAMETER_TAGS and closing and current is not None:
                current["arguments"][parameter_name] = _dsml_parameter_value(
                    content[cursor : match.start()],
                    str(current.pop("_parameter_attrs", "")),
                )
                parameter_name = ""
            else:
                malformed = True
            cursor = match.end()
            continue

        if current is not None:
            if (
                tag in _DSML_PARAMETER_TAGS
                and not closing
                and current.get("name")
            ):
                attributes = _dsml_attributes(match.group("attrs"))
                parameter_name = attributes.get("name", "")
                if not parameter_name:
                    malformed = True
                else:
                    current["_parameter_attrs"] = match.group("attrs")
            elif tag in _DSML_INVOKE_TAGS and closing:
                current.pop("_parameter_attrs", None)
                calls.append(current)
                current = None
            else:
                malformed = True
            cursor = match.end()
            continue

        visible.append(content[cursor : match.start()])
        if tag in _DSML_CALL_TAGS:
            pass
        elif tag in _DSML_INVOKE_TAGS and not closing:
            attributes = _dsml_attributes(match.group("attrs"))
            current = {
                "name": attributes.get("name", ""),
                "arguments": {},
            }
            if not current["name"]:
                malformed = True
        elif tag in _DSML_INVOKE_TAGS and closing:
            pass
        else:
            malformed = True
        cursor = match.end()

    if parameter_name or current is not None:
        malformed = True
    else:
        visible.append(content[cursor:])

    clean_content = "\n".join(
        part.strip() for part in visible if part.strip()
    ) or None
    if malformed:
        return [], None, True
    return calls, clean_content, True


def normalize_dsml_message(
    message: dict[str, Any],
    *,
    available_tool_names: set[str],
) -> tuple[dict[str, Any], str]:
    """Convert textual DSML calls into tool_calls and keep markup out of content."""
    normalized = dict(message)
    content = normalized.get("content")
    if not isinstance(content, str):
        return normalized, "none"

    calls, clean_content, detected = _extract_dsml_calls(content)
    if not detected:
        return normalized, "none"

    normalized["content"] = clean_content
    existing_calls = list(normalized.get("tool_calls") or [])
    if existing_calls:
        normalized["tool_calls"] = existing_calls
        return normalized, "stripped"
    if not calls:
        return normalized, "unresolved"

    names = {str(call.get("name") or "") for call in calls}
    if names and names <= available_tool_names:
        normalized["tool_calls"] = [
            {
                "id": f"call_dsml_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": str(call["name"]),
                    "arguments": json.dumps(
                        call["arguments"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for call in calls
        ]
        return normalized, "converted"

    if len(calls) == 1 and calls[0].get("name") == "submit_result":
        summary = calls[0].get("arguments", {}).get("summary")
        if isinstance(summary, str) and summary.strip():
            normalized["content"] = summary.strip()
            return normalized, "salvaged"

    return normalized, "unresolved"


class ModelClient(Protocol):
    def cancel(self) -> None:
        ...

    def list_models(self) -> list[str]:
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> ModelReply:
        ...

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> str:
        ...

    def classify_turn(
        self,
        user_input: str,
        *,
        previous_assistant: str = "",
    ) -> tuple[str, dict[str, Any]]:
        ...


class BaseModelClient:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.provider = config.model_settings.provider
        if not config.api_key:
            raise ModelError(
                f"未找到 {self.provider} API key。"
                "请配置对应的 Provider 凭据，或设置 SUNUS_API_KEY。"
            )
        self._context = make_ssl_context()
        self._cancel_event = threading.Event()
        self._connection_lock = threading.Lock()
        self._connection: http.client.HTTPConnection | None = None
        self._active_socket: socket.socket | None = None

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._connection_lock:
            connection = self._connection
        self._release_connection(connection, shutdown_socket=True)

    def _release_connection(
        self,
        connection: http.client.HTTPConnection | None,
        *,
        shutdown_socket: bool = False,
    ) -> None:
        if connection is None:
            return
        with self._connection_lock:
            if getattr(connection, "_sunus_released", False):
                return
            try:
                setattr(connection, "_sunus_released", True)
            except (AttributeError, TypeError):
                pass
            active_socket = (
                self._active_socket if self._connection is connection else None
            )
            if self._connection is connection:
                self._connection = None
                self._active_socket = None
        if shutdown_socket and active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                active_socket.close()
            except OSError:
                pass
        try:
            connection.close()
        except Exception:
            pass

    def _open_connection(
        self,
        parsed: urllib.parse.SplitResult,
    ) -> http.client.HTTPConnection:
        host = parsed.hostname or ""
        port = parsed.port
        if parsed.scheme == "https":
            return http.client.HTTPSConnection(
                host,
                port,
                timeout=self.config.request_timeout_sec,
                context=self._context,
            )
        if parsed.scheme == "http":
            return http.client.HTTPConnection(
                host,
                port,
                timeout=self.config.request_timeout_sec,
            )
        raise ModelError(f"不支持的模型接口协议: {parsed.scheme}")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Accept": "application/json",
            "User-Agent": "Sunus/0.1",
        }
        if self.provider == "openai":
            organization = self.config.model_settings.options_as_dict().get(
                "organization"
            )
            project = self.config.model_settings.options_as_dict().get("project")
            if organization:
                headers["OpenAI-Organization"] = str(organization)
            if project:
                headers["OpenAI-Project"] = str(project)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        parsed = urllib.parse.urlsplit(url)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        data: bytes | None = None
        headers = self._headers()
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        self._cancel_event.clear()
        last_error = ""
        for attempt in range(3):
            connection: http.client.HTTPConnection | None = None
            try:
                if self._cancel_event.is_set():
                    raise ModelCancelled("思考已中断")
                connection = self._open_connection(parsed)
                with self._connection_lock:
                    self._connection = connection
                connection.request(method, target, body=data, headers=headers)
                with self._connection_lock:
                    if self._connection is connection:
                        self._active_socket = connection.sock
                if self._cancel_event.is_set():
                    raise ModelCancelled("思考已中断")
                response = connection.getresponse()
                body = response.read()
                if self._cancel_event.is_set():
                    raise ModelCancelled("思考已中断")
                raw = body.decode("utf-8", errors="replace")
                if response.status >= 400:
                    safe_body = redact_text(raw, self.config.api_key)[:1000]
                    last_error = (
                        f"HTTP {response.status} {response.reason}: {safe_body}"
                    )
                    if response.status in {408, 409, 429} or response.status >= 500:
                        if attempt < 2:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                    raise ModelError(last_error)
                if not raw:
                    return {}
                return json.loads(raw)
            except ModelCancelled:
                raise
            except (
                http.client.HTTPException,
                ConnectionError,
                OSError,
                TimeoutError,
                socket.timeout,
                json.JSONDecodeError,
            ) as error:
                if self._cancel_event.is_set():
                    raise ModelCancelled("思考已中断") from error
                detail = redact_text(error, self.config.api_key)
                last_error = f"{type(error).__name__}: {detail}"
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise ModelError(f"模型请求失败: {last_error}") from error
            finally:
                self._release_connection(connection)
        raise ModelError(last_error or "模型请求失败")

    def _request_stream(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> Iterator[str]:
        url = self.config.base_url.rstrip("/") + path
        parsed = urllib.parse.urlsplit(url)
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        self._cancel_event.clear()
        connection: http.client.HTTPConnection | None = None
        response: http.client.HTTPResponse | None = None
        last_error = ""
        for attempt in range(3):
            connected = False
            try:
                connection = self._open_connection(parsed)
                with self._connection_lock:
                    self._connection = connection
                connection.request("POST", target, body=body, headers=headers)
                with self._connection_lock:
                    if self._connection is connection:
                        self._active_socket = connection.sock
                response = connection.getresponse()
                if response.status >= 400:
                    raw = response.read().decode("utf-8", errors="replace")
                    raise ModelError(
                        f"HTTP {response.status} {response.reason}: "
                        f"{redact_text(raw, self.config.api_key)[:1000]}"
                    )
                connected = True
                break
            except ModelCancelled:
                raise
            except (
                http.client.HTTPException,
                ConnectionError,
                OSError,
                TimeoutError,
                socket.timeout,
            ) as error:
                self._release_connection(connection)
                last_error = redact_text(error, self.config.api_key)
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise ModelError(f"模型流式请求失败: {last_error}") from error
            finally:
                if not connected and connection is not None:
                    self._release_connection(connection)
        if connection is None or response is None:
            raise ModelError(f"模型流式请求失败: {last_error}")

        data_lines: list[str] = []
        try:
            while True:
                if self._cancel_event.is_set():
                    raise ModelCancelled("思考已中断")
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line == "" and data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
            if data_lines:
                yield "\n".join(data_lines)
        finally:
            self._release_connection(connection)

    def list_models(self) -> list[str]:
        payload = self._request("GET", "/models")
        return sorted(
            str(item.get("id"))
            for item in payload.get("data", [])
            if item.get("id")
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> str:
        reply = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = reply.message.get("content")
        if not content:
            raise ModelError("模型没有返回文本内容")
        return str(content).strip()

    def classify_turn(
        self,
        user_input: str,
        *,
        previous_assistant: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """Ask the active model whether a short, ambiguous turn is chat or work."""
        previous = previous_assistant.strip()
        if len(previous) > 4_000:
            previous = (
                previous[:2_500]
                + "\n...[中间内容省略]...\n"
                + previous[-1_500:]
            )
        system = (
            "你是 Sunus 的回合类型判别器。只根据当前用户消息和上一条助手"
            "回复，判断用户此刻想要哪种交互。\n"
            "CASUAL：只是社交、情绪、闲聊，不需要生成或修改工作成果。\n"
            "TASK：要求回答、解释、分析、生成、改写、继续上文，或执行有明确"
            "交付物的请求。简短但承接上一轮成果的追加要求必须判为 TASK。\n"
            "拿不准时判 TASK。只输出 CASUAL 或 TASK，不要解释。"
        )
        user = (
            f"<current_user_message>\n{user_input.strip()}\n"
            "</current_user_message>\n"
            f"<previous_assistant_reply>\n{previous or '（无）'}\n"
            "</previous_assistant_reply>"
        )
        usage: dict[str, Any] = {}
        for max_tokens in (256, 512):
            reply = self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=None,
                max_tokens=max_tokens,
                temperature=0,
            )
            for key, value in reply.usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            content = str(reply.message.get("content") or "").strip().upper()
            tokens = re.findall(r"\b(CASUAL|TASK)\b", content)
            if tokens:
                return ("task" if "TASK" in tokens else "casual"), usage
            if reply.finish_reason != "length":
                break
        raise ModelError("回合类型判别没有返回 CASUAL 或 TASK")


class ChatCompletionsClient(BaseModelClient):
    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.provider == "minimax":
            return [dict(message) for message in messages]
        result: list[dict[str, Any]] = []
        for message in messages:
            clean = dict(message)
            clean.pop("reasoning_content", None)
            clean.pop("reasoning_details", None)
            result.append(clean)
        return result

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> ModelReply:
        settings = self.config.model_settings
        payload = dict(settings.options_as_dict())
        payload.pop("organization", None)
        payload.pop("project", None)
        payload.update(
            {
                "model": self.config.model,
                "messages": self._messages(messages),
                settings.token_parameter: max_tokens or self.config.max_tokens,
                "stream": False,
            }
        )
        if self.provider not in {"minimax", "openai"} and "temperature" not in payload:
            payload["temperature"] = temperature
        if settings.reasoning_effort:
            payload["reasoning_effort"] = settings.reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = self._request("POST", "/chat/completions", payload)
        choices = response.get("choices") or []
        if not choices:
            raise ModelError("模型返回中没有 choices")
        choice = choices[0]
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise ModelError("模型返回了无效 message")
        return ModelReply(
            message=message,
            usage=response.get("usage") or {},
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        on_event: Callable[[ModelEvent], None] | None = None,
    ) -> ModelReply:
        settings = self.config.model_settings
        payload = dict(settings.options_as_dict())
        payload.pop("organization", None)
        payload.pop("project", None)
        payload.update(
            {
                "model": self.config.model,
                "messages": self._messages(messages),
                settings.token_parameter: max_tokens or self.config.max_tokens,
                "stream": True,
            }
        )
        if self.provider != "minimax":
            payload["stream_options"] = {"include_usage": True}
        if self.provider not in {"minimax", "openai"} and "temperature" not in payload:
            payload["temperature"] = temperature
        if settings.reasoning_effort:
            payload["reasoning_effort"] = settings.reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish_reason = ""
        emit = on_event or (lambda _event: None)
        emit(ModelEvent("message_start", data={"model": self.config.model}))
        for data in self._request_stream("/chat/completions", payload):
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as error:
                raise ModelError(f"模型流式响应不是有效 JSON: {error}") from error
            if isinstance(chunk.get("usage"), dict):
                usage.update(chunk["usage"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text = str(content)
                    content_parts.append(text)
                    emit(ModelEvent("text_delta", text=text))
                reasoning = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning_details")
                )
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                    emit(ModelEvent("reasoning_delta", text=str(reasoning)))
                for item in delta.get("tool_calls") or []:
                    if not isinstance(item, dict):
                        continue
                    index = int(item.get("index", len(calls)))
                    current = calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if item.get("id"):
                        current["id"] += str(item["id"])
                    function = item.get("function") or {}
                    name_delta = str(function.get("name") or "")
                    arguments_delta = str(function.get("arguments") or "")
                    if name_delta:
                        current["function"]["name"] += name_delta
                    if arguments_delta:
                        current["function"]["arguments"] += arguments_delta
                    emit(
                        ModelEvent(
                            "tool_call_delta",
                            data={
                                "index": index,
                                "id": current["id"],
                                "name": current["function"]["name"],
                                "arguments_delta": arguments_delta,
                            },
                        )
                    )
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
        tool_calls = [calls[index] for index in sorted(calls)]
        for index, call in enumerate(tool_calls):
            if not call["id"]:
                call["id"] = f"call_{uuid.uuid4().hex[:12]}"
            call["index"] = index
            emit(
                ModelEvent(
                    "tool_call_end",
                    data={
                        "index": index,
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    },
                )
            )
        content = "".join(content_parts)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        if not finish_reason:
            finish_reason = "stop"
        emit(
            ModelEvent(
                "completed",
                data={
                    "finish_reason": finish_reason,
                    "usage": usage,
                },
            )
        )
        return ModelReply(
            message=message,
            usage=usage,
            finish_reason=finish_reason,
        )


class ResponsesClient(BaseModelClient):
    @staticmethod
    def _text_content(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts)
        return "" if value is None else str(value)

    def _input_items(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = self._text_content(message.get("content"))
            if role in {"system", "developer"}:
                if content:
                    instructions.append(content)
                continue
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id") or ""),
                        "output": content,
                    }
                )
                continue
            if content:
                items.append({"role": role, "content": content})
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments") or "{}"
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": str(call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "arguments": arguments,
                        }
                    )
        return "\n\n".join(instructions), items

    @staticmethod
    def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function") or {}
            result.append(
                {
                    "type": "function",
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters") or {},
                    "strict": False,
                }
            )
        return result

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> ModelReply:
        instructions, input_items = self._input_items(messages)
        settings = self.config.model_settings
        payload = dict(settings.options_as_dict())
        payload.pop("reasoning_split", None)
        payload.pop("organization", None)
        payload.pop("project", None)
        payload.update(
            {
                "model": self.config.model,
                "input": input_items,
                "max_output_tokens": max_tokens or self.config.max_tokens,
                "stream": False,
            }
        )
        if self.provider != "openai" and "temperature" not in payload:
            payload["temperature"] = temperature
        if instructions:
            payload["instructions"] = instructions
        if settings.reasoning_effort:
            payload["reasoning"] = {"effort": settings.reasoning_effort}
        if tools:
            payload["tools"] = self._responses_tools(tools)
            payload["tool_choice"] = "auto"
        response = self._request("POST", "/responses", payload)
        if response.get("status") == "failed":
            error = response.get("error") or {}
            raise ModelError(
                str(error.get("message") or "Responses API 返回 failed")
            )
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning: list[str] = []
        for item in response.get("output") or []:
            item_type = str(item.get("type") or "")
            if item_type == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"}:
                        texts.append(str(part.get("text") or ""))
            elif item_type == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "")
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or ""),
                            "arguments": str(item.get("arguments") or "{}"),
                        },
                    }
                )
            elif item_type == "reasoning":
                for summary in item.get("summary") or []:
                    if summary.get("text"):
                        reasoning.append(str(summary["text"]))
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(filter(None, texts)) or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning:
            message["reasoning_content"] = "\n".join(reasoning)
        return ModelReply(
            message=message,
            usage=response.get("usage") or {},
            finish_reason=(
                "tool_calls"
                if tool_calls
                else "length"
                if response.get("status") == "incomplete"
                else "stop"
            ),
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        on_event: Callable[[ModelEvent], None] | None = None,
    ) -> ModelReply:
        emit = on_event or (lambda _event: None)
        emit(ModelEvent("message_start", data={"model": self.config.model}))
        reply = self.chat(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = reply.message.get("content")
        if content:
            emit(ModelEvent("text_delta", text=str(content)))
        emit(
            ModelEvent(
                "completed",
                data={
                    "finish_reason": reply.finish_reason,
                    "usage": reply.usage,
                    "fallback": "non_stream",
                },
            )
        )
        return reply


DeepSeekClient = ChatCompletionsClient


def create_model_client(config: RuntimeConfig) -> ModelClient:
    if config.model_settings.api_mode == "responses":
        return ResponsesClient(config)
    return ChatCompletionsClient(config)
