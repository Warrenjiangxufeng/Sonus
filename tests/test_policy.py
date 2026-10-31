from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sunus.policy import (
    ApprovalManager,
    PathGuard,
    PolicyError,
    TaskGrant,
    _choose_approval_option,
    _clear_approval_options,
    _read_menu_key,
    _render_approval_options,
    command_denial_reason,
    command_risk,
    normalize_capabilities,
)
from sunus.scope import classify_request


class PolicyTests(unittest.TestCase):
    def test_path_guard_blocks_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            guard = PathGuard(Path(temp) / "workspace")
            self.assertEqual(guard.resolve("src/file.py"), guard.workspace / "src/file.py")
            with self.assertRaises(PolicyError):
                guard.resolve("/tmp/outside-sunus-test")

    def test_path_guard_blocks_runtime_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            guard = PathGuard(Path(temp))
            with self.assertRaises(PolicyError):
                guard.assert_writable(Path(temp) / ".git" / "config")
            with self.assertRaises(PolicyError):
                guard.assert_writable(Path(temp) / ".sunus" / "state")

    def test_path_guard_allows_only_explicit_write_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diary = root / ".sunus" / "diary"
            guard = PathGuard(
                root / "workspace",
                write_allowlist=(diary,),
            )

            resolved = guard.resolve(
                diary / "2026-10-26.md",
                for_write=True,
            )
            guard.assert_writable(resolved)

            with self.assertRaises(PolicyError):
                guard.assert_writable(root / ".sunus" / "config.json")
            with self.assertRaises(PolicyError):
                guard.resolve(
                    root / ".sunus" / "other" / "file.txt",
                    for_write=True,
                )

    def test_current_request_can_authorize_an_outside_write_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "exports" / "result.md"
            guard = PathGuard(root / "workspace")
            guard.set_request_write_paths((target,))

            resolved = guard.resolve(target, for_write=True)
            guard.assert_writable(resolved)

            guard.clear_request_write_paths()
            with self.assertRaises(PolicyError):
                guard.resolve(target, for_write=True)

    def test_dangerous_command_rules(self) -> None:
        self.assertTrue(command_denial_reason("sudo rm file"))
        self.assertTrue(command_denial_reason("rm -rf /"))
        self.assertTrue(command_denial_reason("shutdown now"))
        self.assertTrue(command_denial_reason("cat ~/.codex/config.toml"))
        self.assertTrue(command_denial_reason("security find-generic-password -s deepseek"))
        self.assertEqual(command_denial_reason("git status --short"), "")
        self.assertEqual(command_risk("ls -la", workspace=Path.cwd()), "R0")
        self.assertEqual(
            command_risk(
                "find . -iname '*test*' 2>/dev/null | head -20",
                workspace=Path.cwd(),
            ),
            "R0",
        )
        self.assertEqual(command_risk("python3 -m unittest", workspace=Path.cwd()), "R1")
        self.assertEqual(
            command_risk("shasum -a 256 reports/summary.json", workspace=Path.cwd()),
            "R1",
        )
        self.assertEqual(
            command_risk("unknown-local-tool /etc/hosts", workspace=Path.cwd()),
            "R2",
        )
        self.assertEqual(command_risk("rm notes.txt", workspace=Path.cwd()), "R2")
        self.assertEqual(command_risk("curl https://example.com", workspace=Path.cwd()), "R2")
        self.assertEqual(
            command_risk('python3 -c "print(1)\nprint(2)"', workspace=Path.cwd()),
            "R1",
        )
        self.assertEqual(
            command_risk(
                "python3 - <<'PY'\nimport json\nprint(json.dumps({}))\nPY",
                workspace=Path.cwd(),
            ),
            "R1",
        )
        self.assertEqual(
            command_risk(
                "cd . && python3 - <<'PY'\n"
                "import json\n"
                "print(json.dumps({}))\n"
                "PY\n"
                'echo "EXIT=$?"',
                workspace=Path.cwd(),
            ),
            "R1",
        )
        self.assertEqual(command_risk("ls\nrm notes.txt", workspace=Path.cwd()), "R2")

    def test_task_grant_auto_allows_routine_workspace_actions(self) -> None:
        approvals = ApprovalManager(auto_approve=False, interactive=False)
        self.assertTrue(approvals.require("write", "file"))
        self.assertTrue(approvals.require_command("ls -la"))
        self.assertTrue(approvals.require_command("python3 -m unittest discover -s tests"))
        self.assertFalse(approvals.require("访问网络", "https://example.com", risk="R2"))
        auto = ApprovalManager(auto_approve=True, interactive=False)
        self.assertTrue(auto.require("访问网络", "https://example.com", risk="R2"))

    def test_safe_commands_are_auto_approved_but_high_risk_commands_are_not(self) -> None:
        approvals = ApprovalManager(auto_approve=False, interactive=False)
        self.assertTrue(approvals.require_command("git status --short"))
        self.assertTrue(approvals.require_command("ls -la"))
        self.assertTrue(approvals.require_command("rg Sunus ."))
        self.assertTrue(approvals.require_command("shasum -a 256 README.md"))
        self.assertTrue(
            approvals.require_command(
                "cd . && python3 -m unittest discover -s tests -v 2>&1 | tail -20"
            )
        )
        self.assertTrue(
            approvals.require_command(
                'PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover '
                '-s tests -v 2>&1 | tail -20; echo "---"; '
                "git status --short 2>/dev/null || true"
            )
        )
        self.assertTrue(
            approvals.require_command(
                "shasum -a 256 reports/summary.json; "
                "stat -f '%m %z' reports/summary.json; ls -la tests"
            )
        )
        self.assertFalse(approvals.require_command("git status; rm file"))
        self.assertFalse(approvals.require_command("rm notes.txt"))
        self.assertFalse(approvals.require_command("curl https://example.com"))
        self.assertFalse(approvals.require_command("ls /tmp"))

    def test_narrow_task_grant_can_be_restricted_to_read_only(self) -> None:
        approvals = ApprovalManager(
            auto_approve=False,
            interactive=False,
            default_grant=TaskGrant(risk_ceiling="R0"),
        )
        self.assertTrue(approvals.require_command("ls -la"))
        self.assertFalse(approvals.require("write", "file"))
        approvals.begin_task()
        approvals.end_task()
        self.assertFalse(approvals.require("write", "file"))

    def test_enter_accepts_approval_prompt(self) -> None:
        approvals = ApprovalManager(
            auto_approve=False,
            interactive=True,
            default_grant=TaskGrant(risk_ceiling="R0"),
        )
        with (
            patch("sunus.policy.sys.stdin.isatty", return_value=True),
            patch("sunus.policy._read_choice_key", return_value="ENTER"),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertTrue(
                approvals.require(
                    "访问网络",
                    "https://example.com",
                    risk="R2",
                )
            )

    def test_approval_menu_supports_arrows_and_quick_keys(self) -> None:
        with (
            patch("sunus.policy.os.read", side_effect=[b"\x1b", b"[", b"B"]),
            patch("sunus.policy.select.select", return_value=([0], [], [])),
        ):
            self.assertEqual(_read_menu_key(0), "DOWN")
        keys = iter(["DOWN", "ENTER"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                _choose_approval_option(key_reader=lambda: next(keys)),
                1,
            )
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                _choose_approval_option(key_reader=lambda: "y"),
                0,
            )
            self.assertEqual(
                _choose_approval_option(key_reader=lambda: "x"),
                2,
            )
            self.assertEqual(
                _choose_approval_option(key_reader=lambda: "TAB"),
                2,
            )
            self.assertEqual(
                _choose_approval_option(key_reader=lambda: "3"),
                2,
            )

    def test_interactive_denial_is_exposed_to_the_tool_runtime(self) -> None:
        approvals = ApprovalManager(
            auto_approve=False,
            interactive=True,
            workspace=Path.cwd(),
        )
        approvals.begin_task("R0")

        with patch("sunus.policy._choose_approval_option", return_value=2):
            allowed = approvals.require(
                "运行命令",
                "printf blocked",
                risk="R2",
                capability="command",
            )

        self.assertFalse(allowed)
        self.assertEqual(
            approvals.consume_interactive_decision(),
            "denied",
        )
        self.assertEqual(approvals.consume_interactive_decision(), "")

    def test_approval_menu_highlights_only_selected_option(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            _render_approval_options(
                0,
                first=True,
                color_enabled=True,
            )
        rendered = output.getvalue()
        self.assertIn(
            "\x1b[1;7;36m  ❯ 1. 允许本次（↑/↓ 选择 · Enter / y）",
            rendered,
        )
        self.assertIn(
            "\x1b[2m    2. 本会话全部允许（↑/↓ 选择 · 2 / a）\x1b[0m",
            rendered,
        )
        self.assertIn(
            "\x1b[2m    3. 拒绝（↑/↓ 选择 · 3 / x / Esc / Tab）\x1b[0m",
            rendered,
        )
        self.assertNotIn("Enter 确认 · y 允许 · x 拒绝", rendered)

        output = io.StringIO()
        with redirect_stdout(output):
            _render_approval_options(
                1,
                first=True,
                color_enabled=True,
            )
        rendered = output.getvalue()
        self.assertIn(
            "\x1b[1;7;36m  ❯ 2. 本会话全部允许（↑/↓ 选择 · 2 / a）",
            rendered,
        )
        self.assertIn(
            "\x1b[2m    1. 允许本次（↑/↓ 选择 · Enter / y）\x1b[0m",
            rendered,
        )

    def test_approval_menu_without_color_keeps_plain_text(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            _render_approval_options(
                0,
                first=True,
                color_enabled=False,
            )
        rendered = output.getvalue()
        self.assertIn("  ❯ 1. 允许本次（↑/↓ 选择 · Enter / y）", rendered)
        self.assertNotIn("Enter 确认 · y 允许 · x 拒绝", rendered)
        self.assertNotIn("\x1b[1;7;36m", rendered)
        self.assertNotIn("\x1b[2m", rendered)

    def test_approval_options_are_cleared_after_rendering(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            _render_approval_options(
                0,
                first=True,
                color_enabled=False,
            )
            _clear_approval_options()
        rendered = output.getvalue()
        self.assertIn("\x1b[3A", rendered)
        self.assertGreaterEqual(rendered.count("\x1b[2K"), 3)
        self.assertTrue(rendered.endswith("\r"))

    def test_session_allow_option_persists_for_session(self) -> None:
        approvals = ApprovalManager(
            auto_approve=False,
            interactive=True,
            default_grant=TaskGrant(risk_ceiling="R0"),
        )
        with (
            patch("sunus.policy._choose_approval_option", return_value=1),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertTrue(
                approvals.require(
                    "访问网络",
                    "https://example.com",
                    risk="R2",
                )
            )
        self.assertTrue(approvals.session_allow)

    def test_capability_overrides_are_granular_and_resettable(self) -> None:
        approvals = ApprovalManager(auto_approve=False, interactive=False)
        scope = classify_request("修改文件并运行测试")

        approvals.revoke_capabilities(["write"])
        denied = approvals.apply_scope(scope)
        self.assertFalse(denied.allows_mutation)
        self.assertTrue(denied.allows_commands)

        approvals.grant_capabilities(["write"])
        approvals.revoke_capabilities(["command"])
        mixed = approvals.apply_scope(scope)
        self.assertTrue(mixed.allows_mutation)
        self.assertFalse(mixed.allows_commands)

        approvals.reset_capabilities(["write", "command"])
        restored = approvals.apply_scope(scope)
        self.assertTrue(restored.allows_mutation)
        self.assertTrue(restored.allows_commands)
        self.assertEqual(
            approvals.permission_state(),
            {"command": "auto", "network": "auto", "write": "auto"},
        )

    def test_explicit_task_capability_expires_with_the_turn(self) -> None:
        approvals = ApprovalManager(auto_approve=False, interactive=False)
        approvals.begin_task()
        approvals.grant_task_capabilities(["command"])

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertTrue(
                approvals.require(
                    "运行命令",
                    "printf ok",
                    risk="R2",
                    capability="command",
                    use_task_grant=False,
                )
            )
            approvals.end_task()
            self.assertFalse(
                approvals.require(
                    "运行命令",
                    "printf blocked",
                    risk="R2",
                    capability="command",
                    use_task_grant=False,
                )
            )

    def test_network_grant_bypasses_popup_but_revoke_denies(self) -> None:
        approvals = ApprovalManager(
            auto_approve=False,
            interactive=False,
            default_grant=TaskGrant(risk_ceiling="R1"),
        )
        approvals.grant_capabilities(["network"])
        self.assertTrue(
            approvals.require(
                "访问网络",
                "https://example.com",
                risk="R2",
                capability="network",
            )
        )
        approvals.revoke_capabilities(["network"])
        self.assertFalse(
            approvals.require(
                "访问网络",
                "https://example.com",
                risk="R2",
                capability="network",
            )
        )

    def test_capability_aliases_are_normalized(self) -> None:
        self.assertEqual(
            normalize_capabilities(["写文件,shell", "网络", "all"]),
            {"write", "command", "network"},
        )

    def test_runtime_version_spoofing_is_hard_denied(self) -> None:
        self.assertTrue(
            command_denial_reason(
                "node -e "
                "'Object.defineProperty(process.versions,\"node\","
                "{value:\"20.11.0\"})'"
            )
        )
        self.assertTrue(
            command_denial_reason(
                "NODE_OPTIONS='--require /tmp/pw-node20-shim.js' "
                "node playwright-cli.js open"
            )
        )


if __name__ == "__main__":
    unittest.main()

