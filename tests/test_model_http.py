from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from sonus.config import ModelSettings, RuntimeConfig
from sonus.model import (
    DeepSeekClient,
    ModelCancelled,
    ModelError,
    ResponsesClient,
    create_model_client,
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


class BlockingHandler(Handler):
    started = threading.Event()
    release = threading.Event()

    def do_POST(self):
        type(self).started.set()
        type(self).release.wait(5)


class ModelHttpTests(unittest.TestCase):
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
                with patch("sonus.model.time.sleep") as sleep:
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
                    patch("sonus.model.time.sleep"),
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
                    "sonus.model.DeepSeekClient._open_connection",
                    side_effect=http.client.RemoteDisconnected("closed"),
                ) as connection,
                patch("sonus.model.time.sleep"),
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
                self.assertEqual(payload["max_completion_tokens"], 4096)
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
                self.assertEqual(payload["max_completion_tokens"], 4096)
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
            self.assertEqual(payload["max_output_tokens"], 4096)
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

