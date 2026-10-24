from __future__ import annotations

import tempfile
import unittest
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sonus.config import ProactiveConfig
from sonus.memory import MemoryStore
from sonus.proactive import (
    ProactiveScheduler,
    ProactiveService,
    format_interval_demo,
    parse_topic_ratios,
    proactive_topic_schedule,
)


class FakeModel:
    def __init__(self) -> None:
        self.last_user_prompt = ""
        self.last_system_prompt = ""

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.last_system_prompt = system
        self.last_user_prompt = user
        return "最近你提到喜欢咖啡，工作间隙可以休息一下。"


class ProactiveTests(unittest.TestCase):
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

        self.assertEqual(
            format_interval_demo(300, 2, count=4),
            "5 分钟 → 10 分钟 → 20 分钟 → 40 分钟",
        )

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
            self.assertEqual(meta["topic_weight"], 20)
            self.assertIn("日常与情绪 20%", model.last_user_prompt)
            self.assertIn("用户明确设定的性别：男", model.last_user_prompt)
            self.assertIn("摄影、周末散步", model.last_user_prompt)
            self.assertIn("不得因为 coding", model.last_system_prompt)
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
            schedule = proactive_topic_schedule()
            gender_index = schedule.index("user_gender")

            _text, meta = service.generate(sequence_index=gender_index)

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
                content="Sonus 项目正在修复测试和代码执行链路",
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
            self.assertGreaterEqual(work_meta["topic_recall_count"], 1)
            self.assertIn("修复测试", model.last_user_prompt)
            memory.close()

    def test_gender_fusion_uses_daily_then_work_and_accumulated_persona(self) -> None:
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
                    sonus_gender="女",
                    user_gender="男",
                    custom_quirks=("收集钢笔", "雨天散步"),
                ),
            )
            schedule = proactive_topic_schedule()
            gender_indices = [
                index
                for index, topic in enumerate(schedule)
                if topic == "user_gender"
            ]

            _text, first = service.generate(
                sequence_index=gender_indices[0]
            )
            self.assertEqual(first["base_topic"], "daily_emotion")
            self.assertTrue(first["gender_style_layer"])
            self.assertIn("自身性别：女", model.last_user_prompt)
            self.assertIn("用户明确设定的性别：男", model.last_user_prompt)

            _text, second = service.generate(
                sequence_index=gender_indices[1]
            )
            self.assertEqual(second["base_topic"], "work_coding")

            _text, persona_meta = service.generate(
                sequence_index=schedule.index("self_evolution")
            )
            self.assertTrue(persona_meta["persona_memory_ids"])
            self.assertIn("先给结论", model.last_user_prompt)

            _text, custom_meta = service.generate(
                sequence_index=schedule.index("user_custom")
            )
            self.assertTrue(custom_meta["custom_quirks_configured"])
            self.assertIn("收集钢笔、雨天散步", model.last_user_prompt)
            memory.close()


if __name__ == "__main__":
    unittest.main()

