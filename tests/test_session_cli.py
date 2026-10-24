from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sonus.cli import run_exec
from sonus.config import RuntimeConfig
from sonus.model import ModelReply
from sonus.storage import list_sessions


class ResumeAwareModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_first_submit",
                            "type": "function",
                            "function": {
                                "name": "submit_result",
                                "arguments": (
                                    '{"kind":"answer","status":"completed",'
                                    '"summary":"首轮完成。"}'
                                ),
                            },
                        }
                    ],
                },
                usage={"total_tokens": 5},
                finish_reason="tool_calls",
            )

        for index, message in enumerate(messages):
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                has_response = any(
                    item.get("role") == "tool"
                    and item.get("tool_call_id") == call_id
                    for item in messages[index + 1 :]
                )
                assert has_response, f"missing tool response for {call_id}"
        return ModelReply(
            message={"role": "assistant", "content": "恢复会话后继续完成。"},
            usage={"total_tokens": 5},
            finish_reason="stop",
        )


class SessionCliTests(unittest.TestCase):
    def config(self, root: Path) -> RuntimeConfig:
        return RuntimeConfig(
            workspace=root,
            data_dir=root / ".data",
            api_key="sk-test",
            base_url="https://example.invalid",
            model="fake",
            max_steps=4,
        )

    def test_exec_resume_keeps_valid_tool_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            model = ResumeAwareModel()
            with (
                patch("sonus.cli.create_model_client", return_value=model),
                redirect_stdout(io.StringIO()),
            ):
                first_code = run_exec(
                    config,
                    "第一轮",
                    json_mode=True,
                    verbose=False,
                )
                second_code = run_exec(
                    config,
                    "继续",
                    json_mode=True,
                    verbose=False,
                    resume_session="last",
                )

            self.assertEqual(first_code, 0)
            self.assertEqual(second_code, 0)
            self.assertEqual(model.calls, 2)
            self.assertEqual(len(list_sessions(root, root / ".data")), 1)

    def test_exec_fork_creates_independent_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            model = ResumeAwareModel()
            with (
                patch("sonus.cli.create_model_client", return_value=model),
                redirect_stdout(io.StringIO()),
            ):
                run_exec(config, "第一轮", json_mode=True, verbose=False)
                fork_code = run_exec(
                    config,
                    "分叉继续",
                    json_mode=True,
                    verbose=False,
                    fork_session="last",
                )

            self.assertEqual(fork_code, 0)
            sessions = list_sessions(root, root / ".data")
            self.assertEqual(len(sessions), 2)
            forked = next(
                row
                for row in sessions
                if '"type": "fork"' in Path(row["path"]).read_text(encoding="utf-8")
            )
            self.assertGreater(forked["turns"], 0)


if __name__ == "__main__":
    unittest.main()
