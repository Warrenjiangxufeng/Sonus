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
    _read_menu_key,
    command_denial_reason,
    command_risk,
)


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

    def test_dangerous_command_rules(self) -> None:
        self.assertTrue(command_denial_reason("sudo rm file"))
        self.assertTrue(command_denial_reason("rm -rf /"))
        self.assertTrue(command_denial_reason("shutdown now"))
        self.assertTrue(command_denial_reason("cat ~/.codex/config.toml"))
        self.assertTrue(command_denial_reason("security find-generic-password -s deepseek"))
        self.assertEqual(command_denial_reason("git status --short"), "")
        self.assertEqual(command_risk("ls -la", workspace=Path.cwd()), "R0")
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
                _choose_approval_option(key_reader=lambda: "3"),
                2,
            )

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


if __name__ == "__main__":
    unittest.main()

