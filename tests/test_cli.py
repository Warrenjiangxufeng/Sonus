from __future__ import annotations

import io
import json
import re
import tempfile
import termios
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sunus.config import PersonaConfig, RuntimeConfig
from sunus.cli import (
    PendingTerminalInput,
    RuntimeEventRenderer,
    TerminalInputDock,
    ThinkingStateObserver,
    _enable_utf8_input,
    _input_with_initial,
    _read_active_turn_key,
    _remove_last_grapheme,
    _read_thinking_key,
    _thinking_interrupt_pressed,
    _wrap_input_lines,
    handle_slash,
    make_parser,
    persona_config_command,
    print_banner,
    render_terminal_markdown,
    relationship_config_command,
    residual_text,
)
from sunus.policy import ApprovalManager
from sunus.relationship import relationship_from_mapping


def _ansi_screen_lines(value: str) -> list[str]:
    lines = [""]
    row = 0
    column = 0
    index = 0
    while index < len(value):
        if value[index] == "\x1b":
            match = re.match(r"\x1b\[([0-9;?]*)([A-Za-z])", value[index:])
            if match:
                parameters, command = match.groups()
                if command == "A":
                    amount = int(parameters) if parameters.isdigit() else 1
                    row = max(0, row - amount)
                elif command == "C":
                    amount = int(parameters) if parameters.isdigit() else 1
                    column += amount
                elif command == "K":
                    while len(lines) <= row:
                        lines.append("")
                    lines[row] = ""
                index += len(match.group(0))
                continue
            index += 1
            continue
        character = value[index]
        if character == "\r":
            column = 0
        elif character == "\n":
            row += 1
            column = 0
            while len(lines) <= row:
                lines.append("")
        elif character >= " ":
            while len(lines) <= row:
                lines.append("")
            line = lines[row]
            if len(line) < column:
                line += " " * (column - len(line))
            lines[row] = line[:column] + character + line[column + 1 :]
            column += 1
        index += 1
    return [line.rstrip() for line in lines]


class CliTests(unittest.TestCase):
    def test_relationship_config_supports_inspect_update_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            config = RuntimeConfig(
                workspace=workspace,
                data_dir=root / "data",
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
            )
            args = make_parser().parse_args(
                [
                    "config",
                    "relationship",
                    "--label",
                    "恋人",
                    "--sunus-role",
                    "女朋友",
                    "--user-role",
                    "男朋友",
                    "--address",
                    "阿峰",
                ]
            )
            relationship_config_command(config, args)
            payload = json.loads(
                (config.data_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["persona"]["relationship"],
                {
                    "label": "恋人",
                    "sunus_role": "女朋友",
                    "user_role": "男朋友",
                    "address": "阿峰",
                },
            )

            clear = make_parser().parse_args(
                ["config", "relationship", "--clear"]
            )
            relationship_config_command(config, clear)
            payload = json.loads(
                (config.data_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["persona"]["relationship"],
                {
                    "label": "",
                    "sunus_role": "",
                    "user_role": "",
                    "address": "",
                },
            )

            sister = make_parser().parse_args(
                [
                    "config",
                    "relationship",
                    "--label",
                    "姐妹",
                    "--sunus-role",
                    "姐姐",
                    "--user-role",
                    "妹妹",
                ]
            )
            self.assertEqual(
                relationship_config_command(config, sister),
                0,
            )
            config = replace(
                config,
                persona=replace(
                    config.persona,
                    relationship=relationship_from_mapping(
                        {
                            "label": "姐妹",
                            "sunus_role": "姐姐",
                            "user_role": "妹妹",
                        }
                    ),
                ),
            )
            locked = make_parser().parse_args(
                [
                    "config",
                    "relationship",
                    "--label",
                    "兄弟",
                    "--sunus-role",
                    "哥哥",
                ]
            )
            self.assertEqual(
                relationship_config_command(config, locked),
                2,
            )
            payload = json.loads(
                (config.data_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["persona"]["relationship"]["label"],
                "姐妹",
            )

    def test_persona_config_is_written_through_dedicated_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            config = RuntimeConfig(
                workspace=workspace,
                data_dir=root / "data",
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                persona=PersonaConfig(name="旧名字"),
            )
            args = make_parser().parse_args(
                [
                    "config",
                    "persona",
                    "--name",
                    "苏诺",
                    "--prompt",
                    "默认短答。",
                    "--example",
                    "行，这样理解没问题。",
                    "--style",
                    "sweetness=0.35",
                    "--style",
                    "高冷=0.9",
                    "--conversation-max-chars",
                    "80",
                    "--stage-directions",
                    "--context",
                    "task=0.35",
                ]
            )

            output = io.StringIO()
            with redirect_stdout(output):
                result = persona_config_command(config, args)

            self.assertEqual(result, 0)
            payload = json.loads(
                (config.data_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["persona"]["name"], "苏诺")
            self.assertEqual(
                payload["persona"]["examples"],
                ["行，这样理解没问题。"],
            )
            self.assertEqual(payload["persona"]["style"]["sweetness"], 0.35)
            self.assertEqual(payload["persona"]["style"]["高冷"], 0.9)
            self.assertEqual(
                payload["persona"]["conversation_max_chars"],
                80,
            )
            self.assertTrue(payload["persona"]["stage_directions"])
            self.assertEqual(
                payload["persona"]["context_intensity"]["task"],
                0.35,
            )
            self.assertIn("已更新人物配置", output.getvalue())

    def test_enable_utf8_input_sets_terminal_flag(self) -> None:
        attributes = [0, 0, 0, 0, 0, 0, []]
        with (
            patch("sunus.cli.termios.tcgetattr", return_value=attributes),
            patch("sunus.cli.termios.tcsetattr") as set_attributes,
        ):
            self.assertTrue(_enable_utf8_input(0))
        self.assertTrue(attributes[0] & 0x4000)
        set_attributes.assert_called_once()

    def test_pending_terminal_input_edits_utf8_by_character(self) -> None:
        pending = PendingTerminalInput()
        pending.feed("你".encode("utf-8"))
        pending.feed("好".encode("utf-8"))
        pending.feed(b"\x7f")
        self.assertEqual(pending.text(), "你")
        pending.feed("呀".encode("utf-8"))
        pending.feed(b"\r")
        self.assertEqual(pending.text(), "")
        self.assertEqual(pending.take_submitted(), ["你呀"])

    def test_remove_last_grapheme_handles_unicode_clusters(self) -> None:
        self.assertEqual(_remove_last_grapheme("你好"), "你")
        self.assertEqual(_remove_last_grapheme("e\u0301"), "")
        self.assertEqual(_remove_last_grapheme("👨‍👩‍👧"), "")
        self.assertEqual(_remove_last_grapheme("🇨🇳"), "")

    def test_prefilled_input_falls_back_for_non_tty(self) -> None:
        with (
            patch("builtins.input", return_value="继续") as read_input,
            patch("sunus.cli.sys.stdin.isatty", return_value=False),
        ):
            result = _input_with_initial("你 › ", "前面的话")

        self.assertEqual(result, "前面的话继续")
        read_input.assert_called_once_with("你 › ")

    def test_bracketed_paste_is_read_as_one_multiline_buffer(self) -> None:
        payload = (
            b"\x1b[200~"
            + "第一段\n第二段".encode("utf-8")
            + b"\x1b[201~\n"
        )
        output = io.StringIO()
        with (
            patch("sunus.cli.sys.stdin.isatty", return_value=True),
            patch("sunus.cli.sys.stdin.fileno", return_value=0),
            patch("sunus.cli.termios.tcgetattr", return_value=[0, 0, 0, 0, 0, 0, []]),
            patch("sunus.cli.termios.tcsetattr"),
            patch("sunus.cli.tty.setcbreak") as set_cbreak,
            patch("sunus.cli.select.select", return_value=([0], [], [])),
            patch("sunus.cli.os.read", side_effect=[bytes([byte]) for byte in payload]),
            redirect_stdout(output),
        ):
            result = _input_with_initial("你 › ", "前缀")

        self.assertEqual(result, "前缀第一段\n第二段")
        self.assertNotIn("[200~", result)
        self.assertNotIn("[201~", result)
        set_cbreak.assert_called_once_with(0, when=termios.TCSADRAIN)
        self.assertIn("\x1b[?2004l", output.getvalue())

    def test_proactive_message_leaves_no_empty_prompt_above_it(self) -> None:
        output = io.StringIO()

        def on_idle(clear_input) -> bool:
            clear_input()
            print("Sunus 主动待办\n今天的日记已经写好。")
            return True

        with (
            patch("sunus.cli.sys.stdin.isatty", return_value=True),
            patch("sunus.cli.sys.stdin.fileno", return_value=0),
            patch(
                "sunus.cli.termios.tcgetattr",
                return_value=[0, 0, 0, 0, 0, 0, []],
            ),
            patch("sunus.cli.termios.tcsetattr"),
            patch("sunus.cli.tty.setcbreak"),
            patch(
                "sunus.cli.select.select",
                side_effect=[([], [], []), ([0], [], [])],
            ),
            patch("sunus.cli.os.read", return_value=b"\n"),
            redirect_stdout(output),
        ):
            result = _input_with_initial(
                "你 › ",
                "",
                on_idle=on_idle,
            )

        self.assertEqual(result, "")
        screen = [
            line
            for line in _ansi_screen_lines(output.getvalue())
            if line
        ]
        message_row = screen.index("Sunus 主动待办")
        self.assertNotIn("你 ›", screen[:message_row])
        self.assertIn("你 ›", screen[message_row + 1 :])

    def test_bracketed_paste_marker_is_not_queued_as_text(self) -> None:
        marker = b"\x1b[200~"
        with (
            patch("sunus.cli.select.select", return_value=([0], [], [])),
            patch(
                "sunus.cli.os.read",
                side_effect=[bytes([byte]) for byte in marker],
            ),
        ):
            action, content = _read_thinking_key(0)
        self.assertEqual(action, "ignore")
        self.assertEqual(content, b"")

    def test_active_turn_reader_detects_paste_markers(self) -> None:
        for marker, expected in (
            (b"\x1b[200~", "paste_start"),
            (b"\x1b[201~", "paste_end"),
        ):
            with (
                patch("sunus.cli.select.select", return_value=([0], [], [])),
                patch(
                    "sunus.cli.os.read",
                    side_effect=[bytes([byte]) for byte in marker],
                ),
            ):
                action, _ = _read_active_turn_key(0)
            self.assertEqual(action, expected)

    def test_wrapped_input_keeps_all_characters(self) -> None:
        text = "前面输入的完整内容不能缩略" * 20 + "\n最后一行"
        lines, cursor_row, _ = _wrap_input_lines("你 › ", text, len(text), 16)

        self.assertGreater(len(lines), 1)
        restored = "".join(
            line.replace("你 › ", "", 1).strip()
            for line in lines
        )
        self.assertEqual(restored, text.replace("\n", ""))
        self.assertNotIn("…", "".join(lines))
        self.assertEqual(cursor_row, len(lines) - 1)

    def test_input_dock_redraws_activity_and_input_together(self) -> None:
        pending = PendingTerminalInput()
        pending.feed("排队输入".encode("utf-8"))
        output = io.StringIO()
        dock = TerminalInputDock(
            "你 › ",
            pending,
            stream=output,
            is_tty=True,
            color_enabled=False,
        )

        dock.set_activity("  思考中")
        dock.refresh()

        text = output.getvalue()
        self.assertIn("思考中", text)
        self.assertIn("你 › 排队输入", text)
        self.assertIn("\x1b[1A", text)

    def test_repeated_dock_refresh_does_not_duplicate_activity(self) -> None:
        output = io.StringIO()
        now = [30.0]
        pending = PendingTerminalInput()
        dock = TerminalInputDock(
            "你 › ",
            pending,
            stream=output,
            is_tty=True,
            color_enabled=False,
        )
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
            input_dock=dock,
            clock=lambda: now[0],
        )
        renderer({"type": "model_request", "step": 1})
        renderer(
            {
                "type": "tool_call",
                "call_id": "call_1",
                "name": "list_files",
                "summary": "查看目录 · Desktop/sunus",
            }
        )
        renderer(
            {
                "type": "tool_result",
                "call_id": "call_1",
                "name": "list_files",
                "ok": True,
                "preview": "发现 6 项",
            }
        )

        for _ in range(18):
            now[0] += 1
            dock.refresh()

        screen = [
            line
            for line in _ansi_screen_lines(output.getvalue())
            if "查看目录 · Desktop/sunus" in line
        ]
        self.assertEqual(len(screen), 1)

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
        persona = parser.parse_args(["memory", "persona", "--distill"])
        self.assertEqual(persona.memory_action, "persona")
        self.assertTrue(persona.persona_distill)
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

    def test_permission_slash_commands_grant_revoke_and_reset(self) -> None:
        approvals = ApprovalManager(auto_approve=False, interactive=False)
        components = SimpleNamespace(approvals=approvals)

        with redirect_stdout(io.StringIO()):
            self.assertTrue(handle_slash("/approve off write", components))
        self.assertEqual(approvals.permission_state()["write"], "deny")

        with redirect_stdout(io.StringIO()):
            self.assertTrue(
                handle_slash("/approve on write,command", components)
            )
        self.assertEqual(approvals.permission_state()["write"], "allow")
        self.assertEqual(approvals.permission_state()["command"], "allow")

        with redirect_stdout(io.StringIO()):
            self.assertTrue(handle_slash("/permissions", components))
        with redirect_stdout(io.StringIO()):
            self.assertTrue(handle_slash("/approve reset write", components))
        self.assertEqual(approvals.permission_state()["write"], "auto")

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
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
        )
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
        renderer.commit_turn()
        renderer.activity_stream.write("最终回答。\n")
        text = output.getvalue()
        self.assertIn("思考中…", text)
        self.assertIn("先检查说明文件", text)
        self.assertIn("读取文件 · note.txt", text)
        self.assertIn("✓ 已读取 12 行", text)
        self.assertIn("✓ 完成 · 1 次分析 · 1 个动作 · 读取 1", text)
        self.assertNotIn("用时", text)
        self.assertLess(text.index("✓ 完成"), text.index("最终回答。"))
        self.assertNotIn("sunus test", text)

    def test_runtime_renderer_keeps_thinking_at_bottom_with_elapsed_time(
        self,
    ) -> None:
        output = io.StringIO()
        now = [100.0]
        pending = PendingTerminalInput()
        dock = TerminalInputDock(
            "你 › ",
            pending,
            stream=output,
            is_tty=True,
            color_enabled=False,
        )
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
            input_dock=dock,
            clock=lambda: now[0],
        )
        renderer({"type": "model_request", "step": 1})
        renderer(
            {
                "type": "assistant_progress",
                "summary": "先检查当前问题。",
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
                "preview": "已读取 1 行",
            }
        )
        now[0] += 12
        renderer._refresh_activity()
        active_lines = dock._activity.splitlines()
        self.assertEqual(len(active_lines), 1)
        self.assertIn("思考中", active_lines[0])
        self.assertIn("12秒", active_lines[0])
        pending_output = output.getvalue()
        self.assertEqual(pending_output.count("先检查当前问题"), 1)
        self.assertEqual(pending_output.count("读取文件 · note.txt"), 1)
        self.assertEqual(pending_output.count("已读取 1 行"), 1)

        renderer({"type": "turn_end", "reason": "completed"})
        renderer.commit_turn()
        final_text = output.getvalue()
        self.assertNotIn("思考中", final_text.splitlines()[-1])
        self.assertIn("✓ 完成", final_text)
        self.assertNotIn("用时", final_text)

    def test_thinking_timer_starts_when_turn_starts_before_model_request(
        self,
    ) -> None:
        output = io.StringIO()
        now = [40.0]
        pending = PendingTerminalInput()
        dock = TerminalInputDock(
            "你 › ",
            pending,
            stream=output,
            is_tty=True,
            color_enabled=False,
        )
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
            input_dock=dock,
            clock=lambda: now[0],
        )

        renderer({"type": "turn_started"})
        now[0] += 4
        renderer._refresh_activity()
        self.assertIn("思考中", dock._activity.splitlines()[-1])
        self.assertIn("4秒", dock._activity.splitlines()[-1])

        renderer({"type": "model_request", "step": 1})
        now[0] += 2
        renderer._refresh_activity()
        self.assertEqual(
            sum("思考中" in line for line in dock._activity.splitlines()),
            1,
        )
        self.assertIn("6秒", dock._activity.splitlines()[-1])

    def test_preprocessing_thinking_hides_for_conversation(self) -> None:
        output = io.StringIO()
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
        )
        renderer({"type": "turn_started"})
        self.assertTrue(renderer._thinking_active)

        renderer({"type": "response_mode", "mode": "conversation"})
        self.assertFalse(renderer._thinking_active)
        self.assertEqual(renderer._chain_lines, [])

    def test_terminal_screen_has_single_bottom_thinking_line(self) -> None:
        output = io.StringIO()
        now = [20.0]
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
            clock=lambda: now[0],
        )
        renderer({"type": "model_request", "step": 1})
        renderer(
            {
                "type": "assistant_progress",
                "summary": "先检查目录。",
            }
        )
        renderer(
            {
                "type": "tool_call",
                "call_id": "call_1",
                "name": "list_files",
                "summary": "查看目录 · Desktop/sunus",
            }
        )
        renderer(
            {
                "type": "tool_result",
                "call_id": "call_1",
                "name": "list_files",
                "ok": True,
                "preview": "发现 5 项",
            }
        )
        now[0] += 3
        renderer._refresh_activity()

        active_screen = [
            line
            for line in _ansi_screen_lines(output.getvalue())
            if line
        ]
        self.assertEqual(
            sum("思考中" in line for line in active_screen),
            1,
        )
        self.assertIn("思考中", active_screen[-1])
        self.assertIn("3秒", active_screen[-1])

        renderer({"type": "turn_end", "reason": "completed"})
        renderer.commit_turn()
        output.write("最终回答正文\n")
        final_screen = [
            line
            for line in _ansi_screen_lines(output.getvalue())
            if line
        ]
        self.assertNotIn("思考中", "\n".join(final_screen))
        self.assertIn("✓ 完成", final_screen[-2])
        self.assertNotIn("用时", "\n".join(final_screen))
        self.assertEqual(final_screen[-1], "最终回答正文")

    def test_turn_summary_and_answer_use_one_ordered_output_stream(
        self,
    ) -> None:
        activity_output = io.StringIO()
        completion_output = io.StringIO()
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=activity_output,
            completion_stream=completion_output,
        )
        renderer({"type": "model_request", "step": 1})
        renderer(
            {
                "type": "tool_call",
                "call_id": "call_1",
                "name": "read_file",
                "summary": "读取文件 · note.txt",
            }
        )
        renderer({"type": "turn_end", "reason": "completed"})
        renderer.commit_turn()

        self.assertNotIn("✓ 完成", activity_output.getvalue())
        completion_output.write("最终回答正文\n")
        final_screen = [
            line
            for line in _ansi_screen_lines(completion_output.getvalue())
            if line
        ]
        self.assertEqual(final_screen[-1], "最终回答正文")
        self.assertIn("✓ 完成", final_screen[-2])
        self.assertNotIn("用时", "\n".join(final_screen))

    def test_runtime_renderer_tracks_assistant_delta(self) -> None:
        output = io.StringIO()
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
        )
        renderer({"type": "model_request", "step": 1})
        renderer({"type": "assistant_delta", "text": "你好"})
        renderer({"type": "assistant_delta", "text": "，世界。"})
        renderer({"type": "turn_end", "reason": "completed"})
        renderer.commit_turn()
        text = output.getvalue()
        self.assertEqual(renderer.streamed_text, "")
        self.assertNotIn("你好，世界。", text)
        self.assertIn("✓ 完成", text)

    def test_runtime_renderer_hides_status_for_conversation(self) -> None:
        output = io.StringIO()
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=True,
            activity_stream=output,
        )

        renderer({"type": "response_mode", "mode": "conversation"})
        renderer({"type": "model_request", "step": 1})
        renderer({"type": "assistant_delta", "text": "好，我在。"})
        renderer({"type": "turn_end", "reason": "completed"})
        renderer.commit_turn()

        self.assertEqual(output.getvalue(), "")

    def test_runtime_renderer_is_silent_without_tty(self) -> None:
        output = io.StringIO()
        renderer = RuntimeEventRenderer(
            color_enabled=False,
            is_tty=False,
            activity_stream=output,
        )
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
        renderer.commit_turn()
        text = output.getvalue()
        self.assertEqual(text, "")

    def test_terminal_markdown_renders_answer_structure(self) -> None:
        raw = (
            "## 结论\n\n"
            "**重点** 是 `cli.py`。\n\n"
            "- 第一项\n"
            "- 第二项\n\n"
            "```python\nprint('ok')\n```"
        )

        plain = render_terminal_markdown(raw, color_enabled=False)

        self.assertIn("▌ 结论", plain)
        self.assertIn("重点 是 cli.py", plain)
        self.assertIn("• 第一项", plain)
        self.assertIn("┌─ python", plain)
        self.assertIn("└─", plain)
        self.assertNotIn("**", plain)
        self.assertNotIn("```", plain)
        self.assertNotIn("`cli.py`", plain)

        colored = render_terminal_markdown(raw, color_enabled=True)
        self.assertIn("\033[1m", colored)
        self.assertIn("\033[36m", colored)

    def test_thinking_state_observer(self) -> None:
        event = threading.Event()
        observer = ThinkingStateObserver(event)
        observer({"type": "turn_started"})
        self.assertTrue(event.is_set())
        observer({"type": "model_request"})
        self.assertTrue(event.is_set())
        observer({"type": "tool_call"})
        self.assertFalse(event.is_set())
        observer({"type": "proactive_message_delivered"})
        self.assertFalse(event.is_set())

    def test_tab_and_escape_interrupts_but_arrow_does_not(self) -> None:
        with patch("sunus.cli.os.read", return_value=b"\t"):
            self.assertTrue(_thinking_interrupt_pressed(0))
        with (
            patch("sunus.cli.os.read", return_value=b"\x1b"),
            patch("sunus.cli.select.select", return_value=([], [], [])),
        ):
            self.assertTrue(_thinking_interrupt_pressed(0))
        with (
            patch(
                "sunus.cli.os.read",
                side_effect=[b"\x1b", b"[", b"A"],
            ),
            patch("sunus.cli.select.select", return_value=([0], [], [])),
        ):
            self.assertFalse(_thinking_interrupt_pressed(0))

    def test_residual_text_does_not_reprint_streamed_text(self) -> None:
        # 完全相等：流式已经打完，不能再打一遍
        self.assertEqual(residual_text("你好，世界。", "你好，世界。"), "")
        # 终态是流式内容的前缀子集：同样不重复
        self.assertEqual(residual_text("你好，世界。", "你好，世界。以上是结论。"), "")
        # 终态以流式内容开头：只补尾部差量
        self.assertEqual(residual_text("你好，世界。还有补充", "你好，世界。"), "还有补充")
        # 终态是全新内容：整段输出
        self.assertEqual(residual_text("另一段结论", "你好，世界。"), "另一段结论")
        # 没有流式输出：照常输出
        self.assertEqual(residual_text("你好", ""), "你好")

    def test_residual_text_appends_render_result_extras(self) -> None:
        # 真实场景：正文流式打完，render_result 在其后追加 details / risks
        streamed = "备份完成，副本里改了补打逻辑。"
        rendered = (
            "备份完成，副本里改了补打逻辑。\n\n"
            "- 186 passed\n\n**限制**：主仓尚未改动"
        )
        self.assertEqual(
            residual_text(rendered, streamed),
            "\n\n- 186 passed\n\n**限制**：主仓尚未改动",
        )

    def test_residual_text_keeps_new_conclusion(self) -> None:
        # 终态是全新结论时不能被吞掉
        self.assertEqual(
            residual_text("另一段结论", "你好，世界。"),
            "另一段结论",
        )


if __name__ == "__main__":
    unittest.main()

