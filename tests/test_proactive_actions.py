from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sunus.cli import _execute_proactive_action
from sunus.config import ProactiveConfig, RuntimeConfig
from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.policy import ApprovalManager, PathGuard
from sunus.storage import ArtifactStore
from sunus.tools import ToolExecutor


class ProactiveActionTests(unittest.TestCase):
    def make_components(self, root: Path):
        config = RuntimeConfig(
            workspace=root,
            data_dir=root / ".data",
            api_key="sk-test",
            base_url="https://example.invalid",
            model="fake",
            proactive=ProactiveConfig(
                action_mode="safe_auto",
                action_risk_ceiling="R2",
                action_tools=(
                    "search_memory",
                    "inspect_environment",
                    "write_file",
                    "run_command",
                ),
            ),
        )
        memory = MemoryStore(root / ".data" / "memory.db")
        memory.add(
            content="用户最近在准备周末计划",
            scope="global",
            kind="episode",
            status="supported",
        )
        events = EventBus(root / ".data")
        approvals = ApprovalManager(
            auto_approve=False,
            interactive=False,
            workspace=root,
        )
        tools = ToolExecutor(
            workspace=root,
            path_guard=PathGuard(root),
            approvals=approvals,
            artifacts=ArtifactStore(root, root / ".data"),
            events=events,
            memory=memory,
        )
        return SimpleNamespace(
            config=config,
            tools=tools,
            approvals=approvals,
            events=events,
            memory=memory,
        )

    def test_allowlisted_r0_action_runs_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            components = self.make_components(Path(temp))
            meta = {
                "action": {
                    "tool": "search_memory",
                    "arguments": {"query": "周末计划"},
                    "reason": "确认用户已有安排",
                }
            }

            _execute_proactive_action(
                components,  # type: ignore[arg-type]
                "我查一下你之前的计划。",
                meta,
            )

            self.assertEqual(meta["action_status"], "completed")
            self.assertIn("周末计划", meta["action_result_preview"])
            self.assertIn(
                "proactive_action_result",
                components.events.path.read_text(encoding="utf-8"),
            )
            components.tools.close()
            components.memory.close()

    def test_run_command_action_allows_sunus_to_choose_playwright(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            components = self.make_components(Path(temp))
            meta = {
                "action": {
                    "tool": "run_command",
                    "arguments": {
                        "command": "printf playwright-browser-ready",
                        "timeout_sec": 5,
                    },
                    "reason": "用现有 Playwright 能力检查页面",
                }
            }

            _execute_proactive_action(
                components,  # type: ignore[arg-type]
                "我用 Playwright 看一下页面。",
                meta,
            )

            self.assertEqual(meta["action_status"], "completed")
            self.assertIn(
                "playwright-browser-ready",
                meta["action_result_preview"],
            )
            components.tools.close()
            components.memory.close()

    def test_destructive_run_command_requires_additional_authorization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            components = self.make_components(Path(temp))
            meta = {
                "action": {
                    "tool": "run_command",
                    "arguments": {"command": "rm -rf data", "timeout_sec": 5},
                    "reason": "test deletion",
                }
            }

            _execute_proactive_action(
                components,  # type: ignore[arg-type]
                "我删掉一个目录。",
                meta,
            )

            self.assertEqual(meta["action_status"], "rejected")
            self.assertIn("额外授权", meta["action_error"])
            components.tools.close()
            components.memory.close()

    def test_write_file_can_create_new_file_but_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            components = self.make_components(root)
            meta = {
                "action": {
                    "tool": "write_file",
                    "arguments": {
                        "path": "proactive.txt",
                        "content": "new content",
                    },
                    "reason": "保存主动整理结果",
                }
            }
            _execute_proactive_action(
                components,  # type: ignore[arg-type]
                "我整理成一个新文件。",
                meta,
            )
            self.assertEqual(meta["action_status"], "completed")
            self.assertEqual(
                (root / "proactive.txt").read_text(encoding="utf-8"),
                "new content",
            )

            overwrite = {
                "action": {
                    "tool": "write_file",
                    "arguments": {
                        "path": "proactive.txt",
                        "content": "overwritten",
                    },
                    "reason": "test overwrite",
                }
            }
            _execute_proactive_action(
                components,  # type: ignore[arg-type]
                "我覆盖一下文件。",
                overwrite,
            )
            self.assertEqual(overwrite["action_status"], "rejected")
            self.assertEqual(
                (root / "proactive.txt").read_text(encoding="utf-8"),
                "new content",
            )
            components.tools.close()
            components.memory.close()


if __name__ == "__main__":
    unittest.main()
