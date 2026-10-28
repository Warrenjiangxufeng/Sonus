from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from sunus.cli import (
    _execute_proactive_action,
    _execute_proactive_action_with_task,
    due_message,
)
from sunus.config import ProactiveConfig, RuntimeConfig
from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.policy import ApprovalManager, PathGuard
from sunus.proactive import ProactiveScheduler
from sunus.proactive_thinking import ThoughtDecision
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

    def test_due_task_delivery_bypasses_social_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = [datetime(2026, 1, 1, 9, tzinfo=timezone.utc)]
            config = RuntimeConfig(
                workspace=root,
                data_dir=root / ".data",
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                proactive=ProactiveConfig(
                    action_mode="safe_auto",
                    action_risk_ceiling="R2",
                ),
            )
            (root / ".data").mkdir()
            scheduler = ProactiveScheduler(
                root / ".data" / "state.db",
                config.proactive,
                "s_due_task",
                clock=lambda: current[0],
            )
            scheduler.mark_sent()
            before = scheduler.status()
            self.assertGreater(scheduler.seconds_until_due(), 0)
            task = scheduler.create_task(
                name="daily_diary",
                instruction="写日记",
                target_path_template="diary.md",
                cadence="once",
                not_before="00:00",
            )

            class FakeProactive:
                def generate(self, *args, pending_task=None, **kwargs):
                    return "我把今天的日记写进目标文件。", {
                        "action": None,
                        "action_status": "completed",
                        "topic": "daily_emotion",
                        "proactive_task": {"id": pending_task.id},
                    }

                def record_delivery(self, *args, **kwargs):
                    return "m_task"

            events = EventBus(root / ".data")
            components = SimpleNamespace(
                config=config,
                scheduler=scheduler,
                proactive=FakeProactive(),
                events=events,
            )
            self.assertTrue(
                due_message(components)  # type: ignore[arg-type]
            )
            after = scheduler.status()
            self.assertEqual(
                after.stored_unanswered_streak,
                before.stored_unanswered_streak,
            )
            self.assertEqual(after.next_due_at, before.next_due_at)
            self.assertEqual(
                scheduler.list_tasks()[0].status,
                "completed",
            )
            self.assertIn(
                "proactive_task_delivered",
                events.path.read_text(encoding="utf-8"),
            )
            scheduler.close()

    def test_low_thought_score_defers_without_generating_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = [datetime(2026, 1, 1, 9, tzinfo=timezone.utc)]
            config = RuntimeConfig(
                workspace=root,
                data_dir=root / ".data",
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                proactive=ProactiveConfig(base_interval_sec=10),
            )
            (root / ".data").mkdir()
            scheduler = ProactiveScheduler(
                root / ".data" / "state.db",
                config.proactive,
                "s_thought_gate",
                clock=lambda: current[0],
            )
            current[0] += timedelta(seconds=10)
            events = EventBus(root / ".data")

            class FakeProactive:
                generated = 0

                def refresh_thoughts(self, **kwargs):
                    return True

                def think(self, **kwargs):
                    return ThoughtDecision(
                        action="do_nothing",
                        thought=None,
                        score=0.2,
                        wait_seconds=60,
                        reason="score_below_threshold",
                    )

                def generate(self, *args, **kwargs):
                    self.generated += 1
                    raise AssertionError("不应生成消息")

            proactive = FakeProactive()
            components = SimpleNamespace(
                config=config,
                scheduler=scheduler,
                proactive=proactive,
                events=events,
            )
            self.assertFalse(
                due_message(components)  # type: ignore[arg-type]
            )
            self.assertEqual(proactive.generated, 0)
            self.assertIn(
                "proactive_thought_deferred",
                events.path.read_text(encoding="utf-8"),
            )
            self.assertGreater(scheduler.seconds_until_due(), 0)
            scheduler.close()

    def test_casual_message_skips_unrelated_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            components = self.make_components(Path(temp))
            meta = {
                "action": {
                    "tool": "search_memory",
                    "arguments": {"query": "周末计划"},
                    "reason": "模型误附带了无关动作",
                }
            }

            _execute_proactive_action(
                components,  # type: ignore[arg-type]
                "这个点，你手边有橘子吗？",
                meta,
            )

            self.assertEqual(meta["action_status"], "skipped")
            self.assertIn("未明确表达", meta["action_error"])
            event_log = components.events.path.read_text(encoding="utf-8")
            self.assertIn("proactive_action_skipped", event_log)
            self.assertNotIn("proactive_action_result", event_log)
            components.tools.close()
            components.memory.close()

    def test_message_must_name_the_same_operation_as_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            components = self.make_components(Path(temp))
            meta = {
                "action": {
                    "tool": "search_memory",
                    "arguments": {"query": "周末计划"},
                    "reason": "检查历史安排",
                }
            }

            _execute_proactive_action(
                components,  # type: ignore[arg-type]
                "我看看你的环境配置。",
                meta,
            )

            self.assertEqual(meta["action_status"], "skipped")
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

    def test_completed_proactive_file_task_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            components = self.make_components(root)
            scheduler = ProactiveScheduler(
                root / ".data" / "memory.db",
                ProactiveConfig(),
                "task-test",
            )
            components.scheduler = scheduler
            target = root / "task-output.txt"
            task = scheduler.create_task(
                name="task_output",
                instruction="写入任务结果",
                target_path_template=str(target),
                cadence="once",
                source_memory_id="m_test",
            )
            meta = {
                "proactive_task": {
                    "id": task.id,
                    "name": task.name,
                    "path": str(target),
                },
                "action": {
                    "tool": "write_file",
                    "arguments": {
                        "path": str(target),
                        "content": "done",
                    },
                    "reason": "执行主动待办",
                },
            }

            completed = _execute_proactive_action_with_task(
                components,  # type: ignore[arg-type]
                "我把结果写进这个文件。",
                meta,
            )

            self.assertTrue(completed)
            self.assertEqual(meta["proactive_task_status"], "completed")
            self.assertEqual(
                scheduler.list_tasks(status="completed")[0].id,
                task.id,
            )
            scheduler.close()
            components.tools.close()
            components.memory.close()


if __name__ == "__main__":
    unittest.main()
