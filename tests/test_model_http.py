from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from sunus.config import ModelSettings, RuntimeConfig
from sunus.model import (
    DeepSeekClient,
    ModelCancelled,
    ModelError,
    ModelReply,
    ResponsesClient,
    create_model_client,
    normalize_dsml_message,
)


class Handler(BaseHTTPRequestHandler):
    last_payload = None

    def do_GET(self):
        if self.path != "/models":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"deepseek-flash"}]}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).last_payload = payload
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


class DroppingHandler(Handler):
    attempts = 0
    drop_attempts = 2

    def do_POST(self):
        type(self).attempts += 1
        if type(self).attempts <= type(self).drop_attempts:
            self.close_connection = True
            self.connection.close()
            return
        super().do_POST()


class StreamingHandler(Handler):
    chunks: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_payload = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in type(self).chunks:
            self.wfile.write(
                ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()
            )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class BlockingHandler(Handler):
    started = threading.Event()
    release = threading.Event()

    def do_POST(self):
        type(self).started.set()
        type(self).release.wait(5)


class ModelHttpTests(unittest.TestCase):
    def test_complete_retries_reasoning_output_exhausted_by_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RuntimeConfig(
                workspace=Path(temp),
                data_dir=Path(temp),
                api_key="sk-test",
                base_url="https://example.invalid",
                model="deepseek-flash",
            )
            client = DeepSeekClient(config)
            truncated = ModelReply(
                message={"role": "assistant", "content": None},
                usage={
                    "completion_tokens": 96,
                    "completion_tokens_details": {"reasoning_tokens": 96},
                },
                finish_reason="length",
            )
            completed = ModelReply(
                message={"role": "assistant", "content": '{"message":"好了"}'},
                usage={"completion_tokens": 80},
                finish_reason="stop",
            )
            with patch.object(
                client,
                "chat",
                side_effect=[truncated, completed],
            ) as chat:
                result = client.complete("system", "user", max_tokens=96)

            self.assertEqual(result, '{"message":"好了"}')
            self.assertEqual(chat.call_count, 2)
            self.assertEqual(chat.call_args_list[0].kwargs["max_tokens"], 96)
            self.assertEqual(chat.call_args_list[1].kwargs["max_tokens"], 1024)

    def test_complete_reports_finish_reason_and_reasoning_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RuntimeConfig(
                workspace=Path(temp),
                data_dir=Path(temp),
                api_key="sk-test",
                base_url="https://example.invalid",
                model="deepseek-flash",
            )
            client = DeepSeekClient(config)
            truncated = ModelReply(
                message={"role": "assistant", "content": None},
                usage={
                    "completion_tokens": 512,
                    "completion_tokens_details": {"reasoning_tokens": 512},
                },
                finish_reason="length",
            )
            with patch.object(
                client,
                "chat",
                return_value=truncated,
            ) as chat:
                with self.assertRaisesRegex(
                    ModelError,
                    r"finish_reason=length.*reasoning_tokens=512",
                ):
                    client.complete("system", "user", max_tokens=512)

            self.assertEqual(chat.call_count, 3)

    def test_complete_uses_reasoning_usage_to_size_third_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RuntimeConfig(
                workspace=Path(temp),
                data_dir=Path(temp),
                api_key="sk-test",
                base_url="https://example.invalid",
                model="deepseek-flash",
            )
            client = DeepSeekClient(config)
            truncated = ModelReply(
                message={"role": "assistant", "content": None},
                usage={
                    "completion_tokens": 830,
                    "completion_tokens_details": {"reasoning_tokens": 830},
                },
                finish_reason="length",
            )
            completed = ModelReply(
                message={"role": "assistant", "content": '{"message":"好了"}'},
                usage={"completion_tokens": 120},
                finish_reason="stop",
            )
            with patch.object(
                client,
                "chat",
                side_effect=[truncated, truncated, completed],
            ) as chat:
                result = client.complete("system", "user", max_tokens=256)

            self.assertEqual(result, '{"message":"好了"}')
            self.assertEqual(
                [
                    call.kwargs["max_tokens"]
                    for call in chat.call_args_list
                ],
                [256, 1598, 3196],
            )

    def test_chat_stream_emits_text_usage_and_final_message(self) -> None:
        StreamingHandler.chunks = [
            {
                "choices": [
                    {"delta": {"content": "你"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"delta": {"content": "好"}, "finish_reason": "stop"}
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        ]
        server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-test",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="deepseek-flash",
                )
                events = []
                reply = DeepSeekClient(config).chat_stream(
                    [{"role": "user", "content": "hello"}],
                    on_event=events.append,
                )
                self.assertEqual(reply.message["content"], "你好")
                self.assertEqual(reply.finish_reason, "stop")
                self.assertEqual(reply.usage["total_tokens"], 5)
                self.assertEqual(
                    [event.text for event in events if event.type == "text_delta"],
                    ["你", "好"],
                )
                self.assertTrue(StreamingHandler.last_payload["stream"])
        finally:
            server.shutdown()
            server.server_close()

    def test_chat_stream_assembles_fragmented_tool_call(self) -> None:
        StreamingHandler.chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '"a.txt"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
        server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-test",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="deepseek-flash",
                )
                reply = DeepSeekClient(config).chat_stream(
                    [{"role": "user", "content": "read"}],
                    tools=[],
                )
                self.assertEqual(reply.finish_reason, "tool_calls")
                call = reply.message["tool_calls"][0]
                self.assertEqual(call["id"], "call_1")
                self.assertEqual(call["function"]["name"], "read_file")
                self.assertEqual(
                    call["function"]["arguments"],
                    '{"path":"a.txt"}',
                )
        finally:
            server.shutdown()
            server.server_close()

    def test_normalizes_textual_dsml_tool_calls(self) -> None:
        content = (
            "<｜｜DSML｜｜ calls>\n"
            '<｜｜DSML｜｜ invoke name="submit_result">\n'
            '<｜｜DSML｜｜ parameter name="kind" string="true">'
            "answer</｜｜DSML｜｜ parameter>\n"
            '<｜｜DSML｜｜ parameter name="summary" string="true">'
            "你好。</｜｜DSML｜｜ parameter>\n"
            "</｜｜DSML｜｜ invoke>\n"
            "</｜｜DSML｜｜ calls>"
        )

        message, status = normalize_dsml_message(
            {"role": "assistant", "content": content},
            available_tool_names={"submit_result"},
        )

        self.assertEqual(status, "converted")
        self.assertIsNone(message["content"])
        self.assertEqual(len(message["tool_calls"]), 1)
        function = message["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "submit_result")
        self.assertEqual(
            json.loads(function["arguments"]),
            {"kind": "answer", "summary": "你好。"},
        )
        self.assertNotIn("DSML", json.dumps(message, ensure_ascii=False))

    def test_salvages_dsml_result_text_when_tools_are_unavailable(self) -> None:
        content = (
            "<｜｜DSML｜｜ calls>\n"
            '<｜｜DSML｜｜ invoke name="submit_result">\n'
            '<｜｜DSML｜｜ parameter name="summary" string="true">'
            "普通回复。</｜｜DSML｜｜ parameter>\n"
            "</｜｜DSML｜｜ invoke>\n"
            "</｜｜DSML｜｜ calls>"
        )

        message, status = normalize_dsml_message(
            {"role": "assistant", "content": content},
            available_tool_names=set(),
        )

        self.assertEqual(status, "salvaged")
        self.assertEqual(message["content"], "普通回复。")
        self.assertNotIn("tool_calls", message)
        self.assertNotIn("DSML", message["content"])

    def test_chat_and_models_against_local_server(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-test",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="deepseek-flash",
                )
                client = DeepSeekClient(config)
                self.assertEqual(client.list_models(), ["deepseek-flash"])
                reply = client.chat(
                    [{"role": "user", "content": "hello"}],
                    tools=None,
                )
                self.assertEqual(reply.message["content"], "OK")
        finally:
            server.shutdown()
            server.server_close()

    def test_turn_classifier_uses_model_and_returns_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RuntimeConfig(
                workspace=Path(temp),
                data_dir=Path(temp),
                api_key="sk-test",
                base_url="https://example.invalid",
                model="deepseek-flash",
            )
            client = DeepSeekClient(config)
            reply = ModelReply(
                message={"role": "assistant", "content": "TASK"},
                usage={"prompt_tokens": 12, "completion_tokens": 1},
                finish_reason="stop",
            )
            with patch.object(client, "chat", return_value=reply) as chat:
                mode, usage = client.classify_turn(
                    "加点卖萌的元素",
                    previous_assistant="上一轮的角色提示词。",
                )

            self.assertEqual(mode, "task")
            self.assertEqual(usage["prompt_tokens"], 12)
            kwargs = chat.call_args.kwargs
            self.assertIsNone(kwargs["tools"])
            self.assertEqual(kwargs["max_tokens"], 256)
            self.assertEqual(kwargs["temperature"], 0)
            messages = chat.call_args.args[0]
            self.assertIn("上一轮的角色提示词", messages[1]["content"])
            self.assertIn("加点卖萌的元素", messages[1]["content"])

    def test_remote_disconnect_retries_and_can_recover(self) -> None:
        DroppingHandler.attempts = 0
        DroppingHandler.drop_attempts = 2
        server = ThreadingHTTPServer(("127.0.0.1", 0), DroppingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-test",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="deepseek-flash",
                )
                with patch("sunus.model.time.sleep") as sleep:
                    reply = DeepSeekClient(config).chat(
                        [{"role": "user", "content": "hello"}],
                    )
                self.assertEqual(reply.message["content"], "OK")
                self.assertEqual(DroppingHandler.attempts, 3)
                self.assertEqual(sleep.call_count, 2)
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_disconnect_becomes_model_error_after_retries(self) -> None:
        DroppingHandler.attempts = 0
        DroppingHandler.drop_attempts = 3
        server = ThreadingHTTPServer(("127.0.0.1", 0), DroppingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-test",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="deepseek-flash",
                )
                with (
                    patch("sunus.model.time.sleep"),
                    self.assertRaises(ModelError) as raised,
                ):
                    DeepSeekClient(config).chat(
                        [{"role": "user", "content": "hello"}],
                    )
                self.assertIn("请求失败", str(raised.exception))
                self.assertRegex(
                    str(raised.exception),
                    "RemoteDisconnected|ConnectionResetError",
                )
                self.assertEqual(DroppingHandler.attempts, 3)
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_disconnected_class_is_converted_to_model_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RuntimeConfig(
                workspace=Path(temp),
                data_dir=Path(temp),
                api_key="sk-test",
                base_url="http://127.0.0.1:9",
                model="deepseek-flash",
            )
            client = DeepSeekClient(config)
            with (
                patch(
                    "sunus.model.DeepSeekClient._open_connection",
                    side_effect=http.client.RemoteDisconnected("closed"),
                ) as connection,
                patch("sunus.model.time.sleep"),
                self.assertRaises(ModelError) as raised,
            ):
                client._request("GET", "/models")
            self.assertEqual(connection.call_count, 3)
            self.assertIn("RemoteDisconnected", str(raised.exception))

    def test_cancel_closes_in_flight_request(self) -> None:
        BlockingHandler.started = threading.Event()
        BlockingHandler.release = threading.Event()
        server = ThreadingHTTPServer(("127.0.0.1", 0), BlockingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        errors: list[BaseException] = []
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-test",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="deepseek-flash",
                )
                client = DeepSeekClient(config)

                def request() -> None:
                    try:
                        client.chat([{"role": "user", "content": "wait"}])
                    except BaseException as error:
                        errors.append(error)

                request_thread = threading.Thread(target=request, daemon=True)
                request_thread.start()
                self.assertTrue(BlockingHandler.started.wait(2))
                client.cancel()
                request_thread.join(3)
                self.assertFalse(request_thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ModelCancelled)
        finally:
            BlockingHandler.release.set()
            server.shutdown()
            server.server_close()

    def test_connection_release_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = DeepSeekClient(
                RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-test",
                    base_url="https://example.invalid",
                    model="deepseek-flash",
                )
            )

            class Connection:
                def __init__(self) -> None:
                    self.close_count = 0

                def close(self) -> None:
                    self.close_count += 1

            connection = Connection()
            client._connection = connection  # type: ignore[assignment]

            client._release_connection(connection)  # type: ignore[arg-type]
            client._release_connection(connection)  # type: ignore[arg-type]

            self.assertEqual(connection.close_count, 1)

    def test_minimax_payload_uses_completion_tokens_and_reasoning(self) -> None:
        Handler.last_payload = None
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="mx-test",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="MiniMax-M3.1-Flash",
                    model_settings=ModelSettings(
                        provider="minimax",
                        token_parameter="max_completion_tokens",
                        reasoning_split=True,
                        options=(("reasoning_split", True),),
                    ),
                )
                reply = create_model_client(config).chat(
                    [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "provider reasoning",
                            "tool_calls": [],
                        },
                    ],
                )
                self.assertEqual(reply.message["content"], "OK")
                payload = Handler.last_payload
                self.assertIsNotNone(payload)
                self.assertEqual(payload["max_completion_tokens"], 16_384)
                self.assertNotIn("max_tokens", payload)
                self.assertNotIn("temperature", payload)
                self.assertTrue(payload["reasoning_split"])
                self.assertEqual(
                    payload["messages"][-1]["reasoning_content"],
                    "provider reasoning",
                )
        finally:
            server.shutdown()
            server.server_close()

    def test_openai_chat_payload_uses_compatible_parameters(self) -> None:
        Handler.last_payload = None
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = RuntimeConfig(
                    workspace=Path(temp),
                    data_dir=Path(temp),
                    api_key="sk-openai-test",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="test-model",
                    model_settings=ModelSettings(
                        provider="openai",
                        token_parameter="max_completion_tokens",
                    ),
                )
                reply = create_model_client(config).chat(
                    [{"role": "user", "content": "hello"}],
                )
                self.assertEqual(reply.message["content"], "OK")
                payload = Handler.last_payload
                self.assertIsNotNone(payload)
                self.assertEqual(payload["max_completion_tokens"], 16_384)
                self.assertNotIn("max_tokens", payload)
                self.assertNotIn("temperature", payload)
        finally:
            server.shutdown()
            server.server_close()

    def test_responses_api_converts_messages_tools_and_tool_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RuntimeConfig(
                workspace=Path(temp),
                data_dir=Path(temp),
                api_key="sk-openai-test",
                base_url="https://api.openai.com/v1",
                model="test-model",
                model_settings=ModelSettings(
                    provider="openai",
                    api_mode="responses",
                    token_parameter="max_completion_tokens",
                ),
            )
            client = create_model_client(config)
            self.assertIsInstance(client, ResponsesClient)
            response = {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "完成"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
            with patch.object(
                client,
                "_request",
                return_value=response,
            ) as request:
                reply = client.chat(
                    [
                        {"role": "system", "content": "系统规则"},
                        {"role": "user", "content": "读取文件"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.txt"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "file content",
                        },
                    ],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "read",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                )
            self.assertEqual(reply.message["content"], "完成")
            request.assert_called_once()
            method, path, payload = request.call_args.args
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/responses")
            self.assertEqual(payload["instructions"], "系统规则")
            self.assertEqual(payload["max_output_tokens"], 16_384)
            self.assertEqual(payload["tools"][0]["name"], "read_file")
            self.assertNotIn("function", payload["tools"][0])
            item_types = [item.get("type") for item in payload["input"]]
            self.assertIn("function_call", item_types)
            self.assertIn("function_call_output", item_types)

    def test_responses_api_normalizes_function_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = RuntimeConfig(
                workspace=Path(temp),
                data_dir=Path(temp),
                api_key="sk-openai-test",
                base_url="https://api.openai.com/v1",
                model="test-model",
                model_settings=ModelSettings(
                    provider="openai",
                    api_mode="responses",
                    token_parameter="max_completion_tokens",
                ),
            )
            client = ResponsesClient(config)
            response = {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path":"a.txt"}',
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
            with patch.object(client, "_request", return_value=response):
                reply = client.chat([{"role": "user", "content": "read"}])
            self.assertEqual(reply.finish_reason, "tool_calls")
            self.assertEqual(reply.message["tool_calls"][0]["id"], "call_1")
            self.assertEqual(
                reply.message["tool_calls"][0]["function"]["name"],
                "read_file",
            )


if __name__ == "__main__":
    unittest.main()

