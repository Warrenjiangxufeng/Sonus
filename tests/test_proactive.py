from __future__ import annotations

import json
import inspect
import random
import tempfile
import unittest
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sunus.config import (
    PersonaConfig,
    ProactiveConfig,
)
from sunus.memory import MemoryStore
from sunus.model import ModelError
from sunus.proactive import (
    PresenceContext,
    ProactiveScheduler,
    ProactiveService,
    format_interval_demo,
    parse_topic_ratios,
    proactive_topic_schedule,
)
from sunus.relationship import RelationshipConfig


class FakeModel:
    def __init__(self) -> None:
        self.last_user_prompt = ""
        self.last_system_prompt = ""
        self.last_temperature = None
        self.last_max_tokens = None
        self.calls = 0

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        self.last_system_prompt = system
        self.last_user_prompt = user
        self.last_temperature = kwargs.get("temperature")
        self.last_max_tokens = kwargs.get("max_tokens")
        if "咖啡" in user:
            return "刚想到咖啡，今天喝了吗？"
        return (
            "今天想和你聊点轻松的。",
            "刚才那点小事，后来怎么样了？",
        )[(self.calls - 1) % 2]


class FailingModel(FakeModel):
    def complete(self, system: str, user: str, **kwargs) -> str:
        super().complete(system, user, **kwargs)
        raise ModelError("test model unavailable")


class VerboseModel(FakeModel):
    def complete(self, system: str, user: str, **kwargs) -> str:
        super().complete(system, user, **kwargs)
        return (
            "## 今日建议\n"
            "- 建议你先停下来分析一下今天的整体状态和任务安排。\n"
            "- 你可以尝试把每件事情拆成更小的步骤再逐项完成。"
        )


class AssistantToneModel(FakeModel):
    def complete(self, system: str, user: str, **kwargs) -> str:
        super().complete(system, user, **kwargs)
        return "作为一个AI，希望能帮到你，还有什么可以帮你？"


class ActionJsonModel(FakeModel):
    def __init__(self, action: dict | None = None) -> None:
        super().__init__()
        self.action = action

    def complete(self, system: str, user: str, **kwargs) -> str:
        super().complete(system, user, **kwargs)
        return json.dumps(
            {
                "message": "我去查一下。",
                "action": self.action,
            },
            ensure_ascii=False,
        )


class RoutineActionModel(FakeModel):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def complete(self, system: str, user: str, **kwargs) -> str:
        super().complete(system, user, **kwargs)
        return json.dumps(
            {
                "message": "我把今天写进日记。",
                "action": {
                    "tool": "write_file",
                    "arguments": {
                        "path": str(self.path),
                        "content": "今天过得还算踏实。",
                    },
                    "reason": "执行到期的每日日记事项",
                },
            },
            ensure_ascii=False,
        )


class ProactiveTests(unittest.TestCase):
    def test_request_has_enough_tokens_for_reasoning_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )

            service.generate(sequence_index=0)

            self.assertEqual(model.last_max_tokens, 256)
            memory.close()

    def test_default_schedule_is_five_minute_exponential(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(),
                "s_default",
                clock=lambda: current[0],
            )
            self.assertEqual(scheduler.status().next_delay_sec, 300)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 300)

            current[0] += timedelta(minutes=5)
            self.assertEqual(scheduler.mark_sent(), 600)
            self.assertEqual(scheduler.status().unanswered_streak, 1)

            current[0] += timedelta(minutes=10)
            self.assertEqual(scheduler.mark_sent(), 1200)
            current[0] += timedelta(minutes=20)
            self.assertEqual(scheduler.mark_sent(), 2400)

            scheduler.user_activity()
            self.assertEqual(scheduler.status().unanswered_streak, 0)
            self.assertEqual(scheduler.status().next_delay_sec, 300)
            scheduler.close()

    def test_presence_gate_requires_idle_window_and_can_defer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(
                    base_interval_sec=10,
                    min_idle_sec=8,
                    max_idle_sec=600,
                ),
                "s_presence",
                clock=lambda: current[0],
            )
            self.assertEqual(scheduler.presence_gate(), (True, None))
            scheduler.user_activity()
            current[0] += timedelta(seconds=5)
            self.assertEqual(scheduler.presence_gate(), (False, 5.0))
            current[0] += timedelta(seconds=5)
            allowed, idle = scheduler.presence_gate()
            self.assertTrue(allowed)
            self.assertEqual(idle, 10.0)
            current[0] += timedelta(seconds=700)
            self.assertTrue(scheduler.presence_gate()[0])
            self.assertEqual(scheduler.defer(30), 30)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 30)
            scheduler.close()

        self.assertEqual(
            format_interval_demo(300, 2, count=4),
            "5 分钟 → 10 分钟 → 20 分钟 → 40 分钟",
        )

    def test_interval_is_capped_and_shared_by_status_and_mark_sent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            local = timezone(timedelta(hours=8))
            current = [datetime(2026, 1, 1, 9, 0, tzinfo=local)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(
                    base_interval_sec=10,
                    backoff_factor=2,
                    max_interval_sec=60,
                ),
                "s_cap",
                clock=lambda: current[0],
            )

            delays = []
            for _ in range(5):
                delays.append(scheduler.mark_sent())
                current[0] += timedelta(seconds=delays[-1])

            self.assertEqual(delays, [20, 40, 60, 60, 60])
            signature = inspect.signature(scheduler._compute_delay)
            self.assertEqual(list(signature.parameters), ["streak"])
            scheduler.close()

    def test_new_reset_daypart_starts_a_fresh_streak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            local = timezone(timedelta(hours=8))
            current = [datetime(2026, 1, 1, 11, 40, tzinfo=local)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(base_interval_sec=300),
                "s_daypart",
                clock=lambda: current[0],
            )
            scheduler.mark_sent()
            current[0] += timedelta(minutes=5)
            scheduler.mark_sent()
            before = scheduler.status()
            self.assertEqual(before.stored_unanswered_streak, 2)

            current[0] = datetime(2026, 1, 1, 12, 5, tzinfo=local)
            reset = scheduler.status()
            self.assertTrue(reset.daypart_changed)
            self.assertEqual(reset.unanswered_streak, 0)
            self.assertEqual(reset.stored_unanswered_streak, 2)
            self.assertEqual(reset.next_delay_sec, 300)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 0)

            self.assertEqual(scheduler.mark_sent(), 600)
            after = scheduler.status()
            self.assertFalse(after.daypart_changed)
            self.assertEqual(after.stored_unanswered_streak, 1)
            scheduler.close()

    def test_old_proactive_state_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.db"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE proactive_state (
                    session_id TEXT PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    next_due_at REAL NOT NULL,
                    last_sent_at REAL,
                    last_user_at REAL,
                    unanswered_streak INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO proactive_state
                (session_id, next_due_at, last_sent_at, unanswered_streak, updated_at)
                VALUES ('old', 0, 0, 4, '2026-01-01T00:00:00+00:00')
                """
            )
            connection.commit()
            connection.close()

            scheduler = ProactiveScheduler(
                path,
                ProactiveConfig(),
                "old",
                clock=lambda: datetime(2026, 1, 1, 9, tzinfo=timezone.utc),
            )
            columns = {
                row["name"]
                for row in scheduler._connection.execute(
                    "PRAGMA table_info(proactive_state)"
                )
            }
            self.assertIn("last_daypart", columns)
            self.assertEqual(scheduler.status().stored_unanswered_streak, 4)
            scheduler.close()

    def test_daypart_review_can_be_acknowledged_without_sending(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            local = timezone(timedelta(hours=8))
            current = [datetime(2026, 1, 1, 11, 55, tzinfo=local)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(),
                "s_review",
                clock=lambda: current[0],
            )
            scheduler.mark_sent()
            current[0] = datetime(2026, 1, 1, 12, 5, tzinfo=local)
            self.assertTrue(scheduler.status().daypart_changed)
            scheduler.acknowledge_daypart()
            status = scheduler.status()
            self.assertFalse(status.daypart_changed)
            self.assertEqual(status.stored_unanswered_streak, 1)
            scheduler.close()

    def test_task_channel_does_not_move_social_streak_or_topic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            local = timezone(timedelta(hours=8))
            current = [datetime(2026, 1, 1, 9, 0, tzinfo=local)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(base_interval_sec=300),
                "s_task_channel",
                clock=lambda: current[0],
            )
            before = scheduler.status()
            message_id = scheduler.record_message(
                "日记已写入",
                channel="task",
                thought_id="th_diary",
            )
            self.assertTrue(message_id)
            self.assertEqual(scheduler.mark_sent(channel="task"), 0)
            after = scheduler.status()
            self.assertEqual(after.stored_unanswered_streak, 0)
            self.assertEqual(after.topic_index, before.topic_index)
            self.assertEqual(after.next_due_at, before.next_due_at)
            self.assertEqual(
                scheduler.recent_message_texts(channel="social"),
                [],
            )
            self.assertEqual(
                scheduler.recent_message_texts(channel="task"),
                ["日记已写入"],
            )
            scheduler.close()

    def test_due_task_is_leased_and_can_be_retried_after_defer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            local = timezone(timedelta(hours=8))
            current = [datetime(2026, 1, 1, 9, 0, tzinfo=local)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(),
                "s_task_lease",
                clock=lambda: current[0],
            )
            task = scheduler.create_task(
                name="daily_diary",
                instruction="写今天的日记",
                target_path_template="diary.md",
                cadence="once",
                not_before="00:00",
            )
            self.assertTrue(scheduler.has_due_task())
            claimed = scheduler.claim_due_task()
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.id, task.id)
            self.assertFalse(scheduler.has_due_task())
            self.assertIsNone(scheduler.claim_due_task())
            scheduler.defer_task(task.id, 60)
            current[0] += timedelta(seconds=61)
            retried = scheduler.claim_due_task()
            self.assertIsNotNone(retried)
            self.assertEqual(retried.id, task.id)
            scheduler.defer_task(task.id, 60)
            scheduler.close()

    def test_schedule_persists_across_scheduler_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
            db_path = Path(temp) / "state.db"
            first = ProactiveScheduler(
                db_path,
                ProactiveConfig(),
                "s_resume",
                clock=lambda: current[0],
            )
            expected_due = first.status().next_due_at
            current[0] += timedelta(minutes=1)
            first.close()

            resumed = ProactiveScheduler(
                db_path,
                ProactiveConfig(),
                "s_resume",
                clock=lambda: current[0],
            )
            self.assertEqual(resumed.status().next_due_at, expected_due)
            self.assertAlmostEqual(resumed.seconds_until_due(), 240)
            resumed.close()

    def test_backoff_and_reply_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]

            def clock() -> datetime:
                return current[0]

            config = ProactiveConfig(base_interval_sec=10, backoff_factor=2)
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                config,
                "s_test",
                clock=clock,
            )
            self.assertEqual(scheduler.status().next_delay_sec, 10)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 10)

            current[0] += timedelta(seconds=10)
            self.assertEqual(scheduler.seconds_until_due(), 0)
            self.assertEqual(scheduler.mark_sent(), 20)
            self.assertEqual(scheduler.status().unanswered_streak, 1)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 20)

            current[0] += timedelta(seconds=20)
            self.assertEqual(scheduler.mark_sent(), 40)
            self.assertEqual(scheduler.status().unanswered_streak, 2)

            scheduler.user_activity()
            self.assertEqual(scheduler.status().unanswered_streak, 0)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 10)
            scheduler.close()

    def test_paused_state_does_not_become_due(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(base_interval_sec=5),
                "s_pause",
                clock=lambda: current[0],
            )
            scheduler.pause()
            current[0] += timedelta(seconds=100)
            self.assertTrue(scheduler.status().paused)
            self.assertGreater(scheduler.seconds_until_due(), 0)
            scheduler.close()

    def test_failed_delivery_retries_without_incrementing_streak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(base_interval_sec=10),
                "s_failure",
                clock=lambda: current[0],
            )
            current[0] += timedelta(seconds=10)
            scheduler.mark_failed()
            status = scheduler.status()
            self.assertEqual(status.unanswered_streak, 0)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 10)
            scheduler.close()

    def test_message_generation_reads_recent_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.add(
                content="用户偏好：最近喜欢咖啡和轻音乐",
                scope="global",
                kind="preference",
                status="supported",
                tags=("preference",),
            )
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )
            text, meta = service.generate()
            self.assertIn("咖啡", text)
            self.assertIn("咖啡", model.last_user_prompt)
            self.assertEqual(meta["preference_count"], 1)
            self.assertEqual(meta["generation_mode"], "model")
            memory.close()

    def test_model_failure_does_not_send_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = FailingModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )

            with self.assertRaises(ModelError) as raised:
                service.generate(sequence_index=0)

            self.assertIn("本次不发送", str(raised.exception))
            self.assertIn("test model unavailable", str(raised.exception))
            memory.close()

    def test_message_generation_uses_configured_persona(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
                persona=PersonaConfig(
                    name="苏诺",
                    prompt="苏诺声音软软的，会自然撒娇，也很暖心。",
                    relationship=RelationshipConfig(
                        label="恋人",
                        address="阿峰",
                    ),
                    examples=("这句夸我收下啦，正事也会认真做。",),
                ),
            )

            service.generate()

            self.assertIn("你是苏诺的主动陪伴模块", model.last_system_prompt)
            self.assertIn("声音软软的", model.last_system_prompt)
            self.assertIn("语气示例", model.last_system_prompt)
            self.assertIn("陪伴情境和最近对话", model.last_system_prompt)
            self.assertIn("不要退回固定问候", model.last_system_prompt)
            self.assertIn("延续人物设定", model.last_system_prompt)
            self.assertIn("禁止客服腔", model.last_system_prompt)
            self.assertIn("固定安慰句式", model.last_system_prompt)
            self.assertIn("最近对话已经收束时只写一句", model.last_system_prompt)
            self.assertIn("双方关系：恋人", model.last_system_prompt)
            self.assertIn("Sunus 对用户的称呼：阿峰", model.last_system_prompt)
            self.assertIn("运行时核心设定", model.last_system_prompt)
            self.assertIn("稳定、连续的拟人身体与内在感受", model.last_system_prompt)
            self.assertIn("身体自主权", model.last_system_prompt)
            memory.close()

    def test_proactive_reasoning_uses_relationship_evidence_without_presets(
        self,
    ) -> None:
        cases = (
            (
                RelationshipConfig(
                    label="情侣",
                    sunus_role="女朋友",
                    user_role="男朋友",
                    address="阿峰",
                ),
                "情侣关系：阿峰和女朋友约定周末一起试一家新开的店",
            ),
            (
                RelationshipConfig(
                    label="父女",
                    sunus_role="女儿",
                    user_role="父亲",
                    address="爸爸",
                ),
                "父女关系：爸爸和女儿上周一起修了自行车",
            ),
            (
                RelationshipConfig(
                    label="兄弟",
                    sunus_role="哥哥",
                    user_role="弟弟",
                    address="老弟",
                ),
                "兄弟关系：哥哥和老弟最近在讨论一场球赛",
            ),
        )
        for relationship, evidence in cases:
            with self.subTest(relationship=relationship.label):
                with tempfile.TemporaryDirectory() as temp:
                    memory = MemoryStore(Path(temp) / "memory.db")
                    memory.add(
                        content=evidence,
                        scope="/project/example",
                        kind="episode",
                        status="supported",
                        tags=("conversation", "recent"),
                    )
                    model = FakeModel()
                    service = ProactiveService(
                        model=model,  # type: ignore[arg-type]
                        memory=memory,
                        scope="/project/example",
                        config=ProactiveConfig(),
                        persona=PersonaConfig(
                            relationship=relationship,
                            style=(("sweetness", 0.95), ("initiative", 0.85)),
                        ),
                    )

                    _text, meta = service.generate(sequence_index=0)

                    self.assertIn("关系语境", model.last_user_prompt)
                    self.assertIn(relationship.label, model.last_user_prompt)
                    self.assertIn(evidence.split("：", 1)[1], model.last_user_prompt)
                    self.assertIn("轻量选题判断", model.last_system_prompt)
                    self.assertIn("不要套用预设的关系类型模板", model.last_user_prompt)
                    self.assertIn("「甜度」很强", model.last_system_prompt)
                    self.assertIn("「主动性」很强", model.last_system_prompt)
                    self.assertNotIn("关系类型：情侣", model.last_user_prompt)
                    self.assertTrue(meta["relationship_memory_ids"])
                    self.assertEqual(
                        meta["proactive_reasoning"],
                        "relationship_memory_chain",
                    )
                    self.assertEqual(model.calls, 1)
                    memory.close()

    def test_proactive_json_can_carry_one_allowlisted_read_only_action(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            action = {
                "tool": "search_memory",
                "arguments": {"query": "最近的计划"},
                "reason": "确认是否已有可延续的计划",
            }
            model = ActionJsonModel(action)
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
                action_catalog=(
                    {
                        "name": "search_memory",
                        "description": "Search long-term memory.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                ),
            )

            text, meta = service.generate(sequence_index=0)

            self.assertEqual(text, "我去查一下。")
            self.assertEqual(meta["action"], action)
            self.assertEqual(meta["action_status"], "proposed")
            self.assertEqual(meta["response_format"], "json")
            self.assertEqual(model.calls, 1)
            memory.close()

    def test_self_directed_write_is_optional_without_a_scheduled_task(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "diary" / "today.txt"
            memory = MemoryStore(root / "memory.db")
            model = ActionJsonModel(
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": str(target),
                        "content": "今天这件事值得留下来。",
                    },
                    "reason": "此刻确有想记住的内容",
                }
            )
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope=str(root),
                config=ProactiveConfig(),
                action_catalog=(
                    {
                        "name": "write_file",
                        "description": "Write a new file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                ),
                writable_paths=(str(root / "diary"),),
            )

            _text, meta = service.generate(sequence_index=0)

            self.assertIsNone(meta["proactive_task"])
            self.assertEqual(meta["action"]["tool"], "write_file")
            self.assertIn("不是日程", model.last_user_prompt)
            self.assertIn(str(root / "diary"), model.last_user_prompt)
            self.assertIn("动作不是义务", model.last_system_prompt)
            self.assertIn("今天还没写日记", model.last_system_prompt)
            memory.close()

    def test_scheduler_does_not_seed_a_fixed_diary_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(),
                "no_fixed_diary",
            )

            self.assertEqual(scheduler.list_tasks(), [])
            scheduler.close()

    def test_due_task_generates_diary_message_and_write_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / f"{datetime.now().astimezone().date().isoformat()}.txt"
            memory = MemoryStore(root / "memory.db")
            scheduler = ProactiveScheduler(
                root / "memory.db",
                ProactiveConfig(),
                "task-test",
                clock=lambda: datetime.now().astimezone(),
            )
            task = scheduler.create_task(
                name="daily_diary",
                instruction="写今天的日记",
                target_path_template=f"{root}/{{date}}.txt",
                cadence="once",
            )
            model = RoutineActionModel(target)
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope=str(root),
                config=ProactiveConfig(),
                action_catalog=(
                    {
                        "name": "write_file",
                        "description": "写入文件",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                ),
            )

            due_task = scheduler.due_task()
            self.assertIsNotNone(due_task)
            text, meta = service.generate(
                sequence_index=0,
                pending_task=due_task,
            )

            self.assertEqual(text, "我把今天写进日记。")
            self.assertEqual(meta["proactive_task"]["id"], task.id)
            self.assertEqual(meta["proactive_task"]["name"], "daily_diary")
            self.assertEqual(
                meta["proactive_task"]["path"],
                str(target.resolve()),
            )
            self.assertEqual(meta["action"]["tool"], "write_file")
            self.assertIn("待执行主动待办", model.last_user_prompt)
            self.assertIn(str(target), model.last_user_prompt)
            scheduler.complete_task(task.id)
            self.assertEqual(
                scheduler.list_tasks(status="completed")[0].id,
                task.id,
            )
            memory.close()
            scheduler.close()

    def test_existing_daily_task_file_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / f"{datetime.now().astimezone().date().isoformat()}.txt"
            target.write_text("already written", encoding="utf-8")
            memory = MemoryStore(root / "memory.db")
            scheduler = ProactiveScheduler(
                root / "memory.db",
                ProactiveConfig(),
                "task-test",
                clock=lambda: datetime.now().astimezone(),
            )
            scheduler.create_task(
                name="daily_diary",
                instruction="写今天的日记",
                target_path_template=f"{root}/{{date}}.txt",
                cadence="daily",
            )
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope=str(root),
                config=ProactiveConfig(),
            )

            _text, meta = service.generate(
                sequence_index=0,
                pending_task=scheduler.due_task(),
            )

            self.assertIsNone(meta["proactive_task"])
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "already written",
            )
            tasks = scheduler.list_tasks(status="active")
            self.assertEqual(len(tasks), 1)
            self.assertGreater(
                tasks[0].next_due_at,
                datetime.now().astimezone().timestamp(),
            )
            memory.close()
            scheduler.close()

    def test_proactive_action_outside_allowlist_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = ActionJsonModel(
                {
                    "tool": "write_file",
                    "arguments": {"path": "x.txt", "content": "x"},
                    "reason": "test",
                }
            )
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
                action_catalog=(),
            )

            text, meta = service.generate(sequence_index=0)

            self.assertEqual(text, "我去查一下。")
            self.assertIsNone(meta["action"])
            self.assertIn("白名单", meta["action_error"])
            memory.close()

    def test_repeated_model_output_is_not_sent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )

            with self.assertRaises(ModelError) as raised:
                service.generate(
                    recent_messages=["今天想和你聊点轻松的。"],
                    sequence_index=0,
                )

            self.assertIn("与最近内容重复", str(raised.exception))
            memory.close()

    def test_generation_uses_presence_intent_presentation_and_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
                random_source=random.Random(7),
            )

            text, meta = service.generate(
                sequence_index=0,
                presence=PresenceContext(
                    idle_seconds=42,
                    foreground_app="Terminal",
                    activity="刚结束一段任务",
                ),
            )

            self.assertLessEqual(len(text), 120)
            self.assertIn(meta["intent"], {"care", "curious", "playful"})
            self.assertIn("本轮参考表达动作", model.last_user_prompt)
            self.assertIn("Terminal", model.last_user_prompt)
            self.assertIn("刚结束一段任务", model.last_user_prompt)
            self.assertEqual(meta["presence"]["idle_seconds"], 42)
            self.assertEqual(
                meta["presentation"],
                {"characters_per_second": 28, "bubble_seconds": 15},
            )
            self.assertEqual(model.last_temperature, 0.85)
            memory.close()

    def test_default_topic_schedule_matches_requested_weights(self) -> None:
        schedule = proactive_topic_schedule()

        self.assertEqual(len(schedule), 100)
        self.assertEqual(
            Counter(schedule),
            {
                "daily_emotion": 20,
                "work_coding": 20,
                "self_evolution": 20,
                "user_gender": 20,
                "user_custom": 20,
            },
        )
        self.assertEqual(schedule[0], "daily_emotion")
        self.assertGreater(len(set(schedule[:10])), 3)

    def test_topic_ratios_accept_aliases_and_require_100(self) -> None:
        weights = parse_topic_ratios(
            [
                "日常=45",
                "工作=20",
                "成长=10",
                "性别=5",
                "癖好=20",
            ]
        )
        self.assertEqual(
            dict(weights),
            {
                "daily_emotion": 45,
                "work_coding": 20,
                "self_evolution": 10,
                "user_gender": 5,
                "user_custom": 20,
            },
        )
        with self.assertRaisesRegex(ValueError, "合计必须为 100"):
            parse_topic_ratios(["daily=50"])

    def test_topic_index_persists_increments_and_resets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(),
                "s_topics",
            )
            self.assertEqual(scheduler.current_topic_index(), 0)
            scheduler.record_message("第一条")
            self.assertEqual(scheduler.current_topic_index(), 1)
            scheduler.close()

            resumed = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(),
                "s_topics",
            )
            self.assertEqual(resumed.current_topic_index(), 1)
            resumed.reset()
            self.assertEqual(resumed.current_topic_index(), 0)
            resumed.close()

    def test_existing_state_table_migrates_topic_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "state.db"
            connection = sqlite3.connect(db_path)
            connection.execute(
                """
                CREATE TABLE proactive_state (
                    session_id TEXT PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    next_due_at REAL NOT NULL,
                    last_sent_at REAL,
                    last_user_at REAL,
                    unanswered_streak INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
            connection.close()

            scheduler = ProactiveScheduler(
                db_path,
                ProactiveConfig(),
                "s_legacy",
            )
            self.assertEqual(scheduler.current_topic_index(), 0)
            scheduler.record_message("迁移后的消息")
            self.assertEqual(scheduler.current_topic_index(), 1)
            scheduler.close()

    def test_proactive_prompt_uses_weighted_topic_gender_and_custom_focus(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.add(
                content="用户明确设置性别：男；沟通偏好简洁直接",
                scope="global",
                kind="preference",
                status="supported",
                tags=("preference",),
            )
            model = FakeModel()
            config = ProactiveConfig(
                user_gender="男",
                custom_topics=("摄影", "周末散步"),
            )
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=config,
            )

            _text, meta = service.generate(sequence_index=0)

            self.assertEqual(meta["topic"], "daily_emotion")
            self.assertEqual(meta["topic_weight"], 25)
            self.assertEqual(meta["scheduled_topic_weight"], 20)
            self.assertIn("日常与情绪 20%", model.last_user_prompt)
            self.assertIn("用户明确设定的性别：男", model.last_user_prompt)
            self.assertIn("摄影、周末散步", model.last_user_prompt)
            self.assertIn("不得因为某类内容", model.last_system_prompt)
            memory.close()

    def test_missing_gender_and_custom_topics_are_learned_from_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.add(
                content="用户：蒋旭峰，男，34 岁，沟通偏好简洁直接",
                scope="/project/example",
                kind="semantic",
                status="supported",
            )
            memory.add(
                content="用户喜欢摄影、散步和轻音乐",
                scope="/project/example",
                kind="preference",
                status="supported",
                tags=("preference",),
            )
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )
            _text, meta = service.generate(sequence_index=2)

            self.assertEqual(meta["topic"], "user_gender")
            self.assertFalse(meta["gender_configured"])
            self.assertFalse(meta["custom_topics_configured"])
            self.assertTrue(meta["learned_gender_memory_ids"])
            self.assertTrue(meta["learned_custom_memory_ids"])
            self.assertIn("男", model.last_user_prompt)
            self.assertIn("用户性别与称呼证据", model.last_user_prompt)
            self.assertIn("用户自定义癖好", model.last_user_prompt)
            memory.close()

    def test_daily_and_work_topics_recall_matching_recent_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.add(
                content="用户昨晚睡眠不好，今天情绪有点低落",
                scope="/project/example",
                kind="episode",
                status="supported",
            )
            memory.add(
                content="Sunus 项目还在修复测试和代码执行链路，尚未完成",
                scope="/project/example",
                kind="episode",
                status="supported",
            )
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )
            schedule = proactive_topic_schedule()

            _text, daily_meta = service.generate(
                sequence_index=schedule.index("daily_emotion")
            )
            self.assertGreaterEqual(daily_meta["topic_recall_count"], 1)
            self.assertIn("睡眠不好", model.last_user_prompt)

            _text, work_meta = service.generate(
                sequence_index=schedule.index("work_coding")
            )
            self.assertEqual(work_meta["topic"], "work_coding")
            self.assertEqual(work_meta["scheduled_topic"], "work_coding")
            self.assertFalse(work_meta["work_rerouted"])
            self.assertTrue(work_meta["unresolved_project_memory_ids"])
            self.assertGreaterEqual(work_meta["topic_recall_count"], 1)
            self.assertIn("修复测试", model.last_user_prompt)
            memory.close()

    def test_work_category_is_excluded_without_unresolved_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.add(
                content="Sunus 项目已经完成，测试全部通过",
                scope="/project/example",
                kind="episode",
                status="supported",
            )
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )
            schedule = proactive_topic_schedule()

            text, meta = service.generate(
                sequence_index=schedule.index("work_coding")
            )

            self.assertEqual(meta["scheduled_topic"], "work_coding")
            self.assertEqual(meta["topic"], "self_evolution")
            self.assertEqual(meta["base_topic"], "self_evolution")
            self.assertTrue(meta["work_rerouted"])
            self.assertEqual(meta["unresolved_project_memory_ids"], [])
            self.assertIn("禁止发送工作提醒", model.last_user_prompt)
            self.assertLessEqual(len(text), 120)
            memory.close()

    def test_partial_work_memory_allows_work_topic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.record_turn(
                user_input="继续修复 Sunus 项目的测试链路",
                assistant_text="本轮只完成了前半部分。",
                scope="/project/example",
                receipt_id="r_proactive_partial",
                turn_status="partial_result",
            )
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )
            schedule = proactive_topic_schedule()

            _text, meta = service.generate(
                sequence_index=schedule.index("work_coding")
            )

            self.assertEqual(meta["topic"], "work_coding")
            self.assertFalse(meta["work_rerouted"])
            self.assertTrue(meta["unresolved_project_memory_ids"])
            self.assertIn("已确认一条最近的项目记忆仍未完成", model.last_user_prompt)
            memory.close()

    def test_task_like_casual_output_is_not_sent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = VerboseModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )
            schedule = proactive_topic_schedule()

            with self.assertRaises(ModelError) as raised:
                service.generate(sequence_index=schedule.index("daily_emotion"))

            self.assertIn("任务式内容", str(raised.exception))
            self.assertIn("除工作门槛", model.last_system_prompt)
            self.assertIn("长度和结构由具体话题", model.last_system_prompt)
            memory.close()

    def test_assistant_tone_is_not_sent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            service = ProactiveService(
                model=AssistantToneModel(),  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )
            schedule = proactive_topic_schedule()

            with self.assertRaises(ModelError) as raised:
                service.generate(sequence_index=schedule.index("daily_emotion"))

            self.assertIn("助手腔", str(raised.exception))
            memory.close()

    def test_delivered_message_becomes_self_share_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )

            text, meta = service.generate(sequence_index=0)
            self.assertEqual(
                memory.recent_by_kind(
                    scope="/project/example",
                    kind="self_share",
                ),
                [],
            )

            memory_id = service.record_delivery(
                text,
                meta,
                message_id="p_msg_1",
                trigger="test_delivery",
            )

            rows = memory.recent_by_kind(
                scope="/project/example",
                kind="self_share",
            )
            self.assertEqual([row["id"] for row in rows], [memory_id])
            self.assertEqual(rows[0]["scope"], "global")
            self.assertIn("proactive", rows[0]["tags"].split(","))
            self.assertIn("self_share", rows[0]["tags"].split(","))
            self.assertIn(meta["topic"], rows[0]["tags"].split(","))
            self.assertIn('"message_id": "p_msg_1"', rows[0]["metadata"])
            self.assertIn('"workspace": "/project/example"', rows[0]["metadata"])
            self.assertEqual(meta["persona_traits_updated"], [])

            service.record_delivery(
                "记得慢慢来呀。",
                meta,
                message_id="p_msg_2",
                trigger="test_delivery",
            )
            service.record_delivery(
                "今天有没有开心一点呢？",
                meta,
                message_id="p_msg_3",
                trigger="test_delivery",
            )
            service.generate(sequence_index=1)
            self.assertIn("Sunus 最近主动分享", model.last_user_prompt)
            self.assertIn("记得慢慢来呀", model.last_user_prompt)
            self.assertNotIn(
                "会留意并关心对方的日常与情绪",
                model.last_user_prompt,
            )

            opted_in = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
                persona=PersonaConfig(learn_from_self_share=True),
            )
            opted_in.record_delivery(
                "记得慢慢来呀。",
                meta,
                message_id="p_msg_opt_in",
                trigger="test_opt_in",
            )
            self.assertTrue(meta["persona_traits_updated"])
            memory.close()

    def test_five_memory_categories_follow_eligible_weighted_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            service = ProactiveService(
                model=FakeModel(),  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(),
            )

            without_work = Counter(
                service.generate(sequence_index=index)[1]["topic"]
                for index in range(20)
            )
            self.assertEqual(
                without_work,
                Counter(
                    {
                        "daily_emotion": 5,
                        "self_evolution": 5,
                        "user_gender": 5,
                        "user_custom": 5,
                    }
                ),
            )

            memory.record_turn(
                user_input="继续修复 Sunus 项目的测试链路",
                assistant_text="还没完成。",
                scope="/project/example",
                receipt_id="r_weighted_partial",
                turn_status="partial_result",
            )
            with_work = Counter(
                service.generate(sequence_index=index)[1]["topic"]
                for index in range(20)
            )
            self.assertEqual(set(with_work), set(proactive_topic_schedule()))
            self.assertEqual(set(with_work.values()), {4})
            memory.close()

    def test_gender_category_uses_gender_memory_and_accumulated_persona(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.record_turn(
                user_input="以后回答先给结论，尽量简洁",
                assistant_text="明白。",
                scope="/project/example",
                receipt_id="r_persona",
            )
            model = FakeModel()
            service = ProactiveService(
                model=model,  # type: ignore[arg-type]
                memory=memory,
                scope="/project/example",
                config=ProactiveConfig(
                    sunus_gender="女",
                    user_gender="男",
                    custom_quirks=("收集钢笔", "雨天散步"),
                ),
            )
            _text, first = service.generate(sequence_index=2)
            self.assertEqual(first["base_topic"], "user_gender")
            self.assertTrue(first["gender_style_layer"])
            self.assertIn("自身性别：女", model.last_user_prompt)
            self.assertIn("用户明确设定的性别：男", model.last_user_prompt)

            _text, second = service.generate(sequence_index=6)
            self.assertEqual(second["base_topic"], "user_gender")

            _text, persona_meta = service.generate(
                sequence_index=1
            )
            self.assertTrue(persona_meta["persona_memory_ids"])
            self.assertIn("先给结论", model.last_user_prompt)

            _text, custom_meta = service.generate(
                sequence_index=3
            )
            self.assertTrue(custom_meta["custom_quirks_configured"])
            self.assertIn("收集钢笔、雨天散步", model.last_user_prompt)
            memory.close()


if __name__ == "__main__":
    unittest.main()

