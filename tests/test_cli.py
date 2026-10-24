from __future__ import annotations

import io
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from sonus.cli import (
    RuntimeEventRenderer,
    ThinkingStateObserver,
    _thinking_interrupt_pressed,
    make_parser,
    print_banner,
)


class CliTests(unittest.TestCase):
    def test_parses_core_commands(self) -> None:
        parser = make_parser()
        self.assertEqual(parser.parse_args([]).command, None)
        args = parser.parse_args(["exec", "hello"])
        self.assertEqual(args.command, "exec")
        self.assertEqual(args.prompt, ["hello"])
        resumed = parser.parse_args(["exec", "--resume", "s_123", "继续"])
        self.assertEqual(resumed.resume, "s_123")
        forked = parser.parse_args(["exec", "--fork", "s_123", "换个方案"])
        self.assertEqual(forked.fork, "s_123")
        sessions = parser.parse_args(["sessions", "--limit", "5"])
        self.assertEqual(sessions.command, "sessions")
        self.assertEqual(sessions.limit, 5)
        resume = parser.parse_args(["resume", "--last"])
        self.assertTrue(resume.last)
        fork = parser.parse_args(["fork", "s_123"])
        self.assertEqual(fork.command, "fork")
        self.assertEqual(fork.session_id, "s_123")
        memory = parser.parse_args(["memory", "forget", "m_1", "--yes"])
        self.assertTrue(memory.forget_yes)
        context_metrics = parser.parse_args(
            [
                "memory",
                "context-metrics",
                "记忆 token",
                "--recall-limit",
                "3",
            ]
        )
        self.assertEqual(context_metrics.memory_action, "context-metrics")
        self.assertEqual(context_metrics.recall_limit, 3)
        proactive = parser.parse_args(["proactive", "status"])
        self.assertEqual(proactive.action, "status")
        ratios = parser.parse_args(
            [
                "proactive",
                "ratios",
                "日常=45",
                "工作=20",
                "成长=10",
                "性别=5",
                "癖好=20",
            ]
        )
        self.assertEqual(ratios.action, "ratios")
        self.assertEqual(ratios.ratio_values[0], "日常=45")
        provider = parser.parse_args(
            [
                "--provider",
                "openai",
                "--api-mode",
                "responses",
                "--model",
                "test-model",
            ]
        )
        self.assertEqual(provider.provider, "openai")
        self.assertEqual(provider.api_mode, "responses")
        self.assertEqual(provider.model, "test-model")
        budget = parser.parse_args(["--budget", "quick"])
        self.assertEqual(budget.budget, "quick")

    def test_banner_only_contains_brand_description(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            print_banner()
        text = output.getvalue()
        self.assertIn("本地记忆增强主动陪伴型 Agent", text)
        self.assertIn("她的边界，由你创造！", text)
        self.assertIn(
            "A local, memory-enhanced, proactive companion Agent",
            text,
        )
        self.assertIn("Her boundaries are shaped by you.", text)
        self.assertNotIn("工作区", text)
        self.assertNotIn("数据", text)
        self.assertNotIn("授权", text)
        self.assertNotIn("最近记忆", text)
        self.assertNotIn("deepseek", text)

    def test_runtime_renderer_shows_compact_operation_chain(self) -> None:
        output = io.StringIO()
        renderer = RuntimeEventRenderer(color_enabled=False, is_tty=True)
        with redirect_stdout(output):
            renderer(
                {
                    "type": "task_grant_started",
                    "risk_ceiling": "R1",
                }
            )
            renderer({"type": "model_request", "step": 1})
            renderer(
                {
                    "type": "assistant_progress",
                    "summary": "先检查说明文件，再确认目录职责。",
                }
            )
            renderer(
                {
                    "type": "tool_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "summary": "读取文件 · note.txt",
                }
            )
            renderer(
                {
                    "type": "tool_result",
                    "call_id": "call_1",
                    "name": "read_file",
                    "ok": True,
                    "changed_files": [],
                    "preview": "已读取 12 行",
                }
            )
            renderer({"type": "turn_end", "reason": "completed"})
        text = output.getvalue()
        self.assertIn("思考中…", text)
        self.assertIn("先检查说明文件", text)
        self.assertIn("读取文件 · note.txt", text)
        self.assertIn("✓ 已读取 12 行", text)
        self.assertIn("✓ 完成 · 1 次分析 · 1 个动作 · 读取 1", text)
        self.assertNotIn("sonus test", text)

    def test_runtime_renderer_streams_lines_without_tty(self) -> None:
        output = io.StringIO()
        renderer = RuntimeEventRenderer(color_enabled=False, is_tty=False)
        with redirect_stdout(output):
            renderer({"type": "model_request", "step": 1})
            renderer(
                {
                    "type": "tool_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "summary": "读取文件 · note.txt",
                }
            )
            renderer(
                {
                    "type": "tool_result",
                    "call_id": "call_1",
                    "name": "read_file",
                    "ok": True,
                    "preview": "已读取 1 行",
                }
            )
            renderer({"type": "turn_end", "reason": "completed"})
        text = output.getvalue()
        self.assertIn("思考中", text)
        self.assertIn("读取文件 · note.txt", text)
        self.assertIn("已读取 1 行", text)
        self.assertIn("完成 · 1 次分析", text)

    def test_thinking_state_observer(self) -> None:
        event = threading.Event()
        observer = ThinkingStateObserver(event)
        observer({"type": "model_request"})
        self.assertTrue(event.is_set())
        observer({"type": "tool_call"})
        self.assertFalse(event.is_set())
        observer({"type": "proactive_message_delivered"})
        self.assertFalse(event.is_set())

    def test_tab_and_escape_interrupts_but_arrow_does_not(self) -> None:
        with patch("sonus.cli.os.read", return_value=b"\t"):
            self.assertTrue(_thinking_interrupt_pressed(0))
        with (
            patch("sonus.cli.os.read", return_value=b"\x1b"),
            patch("sonus.cli.select.select", return_value=([], [], [])),
        ):
            self.assertTrue(_thinking_interrupt_pressed(0))
        with (
            patch(
                "sonus.cli.os.read",
                side_effect=[b"\x1b", b"[", b"A"],
            ),
            patch("sonus.cli.select.select", return_value=([0], [], [])),
        ):
            self.assertFalse(_thinking_interrupt_pressed(0))


if __name__ == "__main__":
    unittest.main()

