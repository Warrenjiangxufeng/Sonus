from __future__ import annotations

import random
import tempfile
import unittest
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sunus.config import PersonaConfig, ProactiveConfig
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


class FakeModel:
    def __init__(self) -> None:
        self.last_user_prompt = ""
        self.last_system_prompt = ""
        self.last_temperature = None
        self.calls = 0

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        self.last_system_prompt = system
        self.last_user_prompt = user
        self.last_temperature = kwargs.get("temperature")
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
            self.assertFalse(scheduler.presence_gate()[0])
            self.assertEqual(scheduler.defer(30), 30)
            self.assertAlmostEqual(scheduler.seconds_until_due(), 30)
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

            self.assertLessEqual(len(text), 40)
            self.assertIn(meta["intent"], {"care", "curious", "playful"})
            self.assertIn("本轮表达意图", model.last_user_prompt)
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
            self.assertLessEqual(len(text), 40)
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
            self.assertIn("最多两句", model.last_system_prompt)
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

