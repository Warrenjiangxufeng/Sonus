from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Protocol

from sonus.config import RuntimeConfig, make_ssl_context, redact_text


class ModelError(RuntimeError):
    pass


class ModelCancelled(RuntimeError):
    pass


@dataclass
class ModelReply:
    message: dict[str, Any]
    usage: dict[str, Any]
    finish_reason: str


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


class BaseModelClient:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.provider = config.model_settings.provider
        if not config.api_key:
            raise ModelError(
                f"未找到 {self.provider} API key。"
                "请配置对应的 Provider 凭据，或设置 SONUS_API_KEY。"
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
            active_socket = self._active_socket
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                active_socket.close()
            except OSError:
                pass
        if connection is not None:
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
            "User-Agent": "Sonus/0.1",
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
                with self._connection_lock:
                    if self._connection is connection:
                        self._connection = None
                        self._active_socket = None
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
        raise ModelError(last_error or "模型请求失败")

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


DeepSeekClient = ChatCompletionsClient


def create_model_client(config: RuntimeConfig) -> ModelClient:
    if config.model_settings.api_mode == "responses":
        return ResponsesClient(config)
    return ChatCompletionsClient(config)
