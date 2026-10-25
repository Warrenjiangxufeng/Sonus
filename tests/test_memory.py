from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sunus.memory import MemoryStore, estimate_tokens, text_metrics


class MemoryTests(unittest.TestCase):
    def test_preference_recall_credit_and_forget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory_id = memory.add(
                content="用户偏好：休息时喜欢喝咖啡和听轻音乐",
                scope="global",
                kind="preference",
                status="supported",
                tags=("preference", "relax"),
            )
            rows = memory.recent_preferences(scope="/project/example", limit=10)
            self.assertEqual(rows[0]["id"], memory_id)
            search = memory.search("喝咖啡", scope="/project/example", limit=5)
            self.assertTrue(any(row["id"] == memory_id for row in search))

            memory.record_use(memory_id, turn_id="turn-1")
            memory.credit(memory_id, "negative", "test")
            score = next(item for item in memory.score() if item["id"] == memory_id)
            self.assertEqual(score["negative"], 1)

            deleted = memory.forget("咖啡")
            self.assertEqual(deleted, 1)
            self.assertIsNone(memory.get(memory_id))
            memory.close()

    def test_duplicate_content_is_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            first = memory.add(
                content="用户偏好：先给结论",
                scope="global",
                kind="preference",
                status="supported",
            )
            second = memory.add(
                content="用户偏好：先给结论",
                scope="global",
                kind="preference",
                status="verified",
            )
            self.assertEqual(first, second)
            self.assertEqual(memory.get(first)["status"], "verified")
            memory.close()

    def test_duplicate_can_escalate_from_candidate_to_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory_id = memory.add(
                content="修改接口后先运行目标单测",
                scope="global",
                kind="procedural",
                status="candidate",
            )
            memory.add(
                content="修改接口后先运行目标单测",
                scope="global",
                kind="procedural",
                status="supported",
            )
            self.assertEqual(memory.get(memory_id)["status"], "supported")
            memory.close()

    def test_record_turn_extracts_explicit_preference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            ids = memory.record_turn(
                user_input="请记住，我以后喜欢先看摘要",
                assistant_text="明白。",
                scope="/project/example",
                receipt_id="r_test",
            )
            self.assertGreaterEqual(len(ids), 2)
            prefs = memory.recent_preferences(scope="/project/example", limit=10)
            self.assertTrue(any("先看摘要" in item["content"] for item in prefs))
            memory.close()

    def test_questions_and_task_constraints_are_not_saved_as_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            for index, text in enumerate(
                (
                    "你的本地记忆系统每次送入上下文的 token 大概有多少？",
                    "试一下，特别是江浙沪ip的",
                    "不要修改文件，只分析原因",
                )
            ):
                memory.record_turn(
                    user_input=text,
                    assistant_text="已处理。",
                    scope="/project/example",
                    receipt_id=f"r_non_preference_{index}",
                )

            self.assertEqual(
                memory.recent_preferences(
                    scope="/project/example",
                    limit=20,
                ),
                [],
            )
            memory.add(
                content="用户偏好：紫色河马大概有多少只？",
                scope="global",
                kind="preference",
                status="supported",
                tags=("preference", "recent"),
            )
            self.assertEqual(
                memory.search(
                    "河马",
                    scope="/project/example",
                    limit=10,
                ),
                [],
            )
            memory.close()

    def test_explicit_style_feedback_overrides_previous_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.record_turn(
                user_input="以后别太甜，少卖萌",
                assistant_text="好，我收一点。",
                scope="/project/example",
                receipt_id="r_style_softer",
            )
            memory.record_turn(
                user_input="以后再甜一点",
                assistant_text="好呀。",
                scope="/project/example",
                receipt_id="r_style_sweeter",
            )

            profile = memory.persona_profile(scope="/project/example")
            by_key = {item["trait_key"]: item for item in profile}
            self.assertIn("提高甜度", by_key["style_sweetness"]["statement"])
            self.assertIn(
                "降低卖萌度",
                by_key["style_playfulness"]["statement"],
            )
            memory.close()

    def test_core_identity_is_not_demoted_by_persona_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory._record_persona_evidence(
                trait_key="sunus_name",
                statement="Sunus 的名字是“苏诺”",
                category="identity:name",
                evidence_id="identity_name",
                explicit=True,
            )
            memory._record_persona_evidence(
                trait_key="gender_identity",
                statement="Sunus 核心性别设定为女性",
                category="identity:gender",
                evidence_id="identity_gender",
                explicit=True,
            )
            for index in range(6):
                memory._record_persona_evidence(
                    trait_key=f"style_{index}",
                    statement=f"style {index}",
                    category="style:test",
                    evidence_id=f"style_{index}",
                    explicit=True,
                )

            result = memory.distill_persona(max_stable=2)
            profile = memory.persona_profile(
                scope="/project/example",
                include_candidates=True,
                limit=20,
            )
            stable_keys = {
                item["trait_key"]
                for item in profile
                if item["status"] == "stable"
            }

            self.assertEqual(result["pinned"], 2)
            self.assertEqual(result["demoted"], 4)
            self.assertIn("sunus_name", stable_keys)
            self.assertIn("gender_identity", stable_keys)
            self.assertEqual(len(stable_keys), 4)
            memory.close()

    def test_self_share_backfill_does_not_create_persona_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "memory.db"
            memory = MemoryStore(path)
            memory.add(
                content="Sunus 主动分享：记得慢慢来呀。",
                scope="global",
                kind="self_share",
                status="supported",
                tags=("proactive", "self_share", "daily_emotion"),
                metadata={"topic": "daily_emotion"},
            )
            memory.close()

            reopened = MemoryStore(path)
            self.assertEqual(
                reopened.persona_profile(
                    scope="/project/example",
                    include_candidates=True,
                ),
                [],
            )
            reopened.close()

    def test_context_metrics_match_injected_memory_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.add(
                content="用户偏好：回答先给结论",
                scope="global",
                kind="preference",
                status="supported",
            )
            memory.add(
                content="Sunus 的记忆检索会注入相关情景",
                scope="/project/example",
                kind="semantic",
                status="supported",
            )

            context = memory.build_context(
                scope="/project/example",
                query="记忆检索",
                preference_limit=12,
                recall_limit=5,
            )
            metrics = memory.context_metrics(
                scope="/project/example",
                query="记忆检索",
                preference_limit=12,
                recall_limit=5,
            )

            self.assertEqual(metrics["preferences"]["items"], 1)
            self.assertEqual(metrics["recall"]["items"], 1)
            self.assertEqual(
                metrics["preferences"]["characters"],
                len(context.preference_block),
            )
            self.assertEqual(
                metrics["recall"]["characters"],
                len(context.recall_block),
            )
            self.assertEqual(
                metrics["combined"]["characters"],
                len(context.combined_text),
            )
            self.assertFalse(metrics["exact"])
            self.assertGreater(metrics["combined"]["estimated_tokens"], 0)
            memory.close()

    def test_token_estimate_is_stable_and_dependency_free(self) -> None:
        metrics = text_metrics("Sunus 读取记忆 token")
        self.assertEqual(metrics["characters"], len("Sunus 读取记忆 token"))
        self.assertEqual(
            metrics["estimated_tokens"],
            estimate_tokens("Sunus 读取记忆 token"),
        )
        self.assertFalse(metrics["exact"])
        self.assertEqual(metrics["estimator"], "cjk-ascii-v1")

    def test_explicit_feedback_accumulates_persona_traits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.record_turn(
                user_input="以后回答先给结论，尽量简洁，称呼我阿峰",
                assistant_text="明白。",
                scope="/project/example",
                receipt_id="r_persona",
            )
            traits = memory.recent_by_kind(
                scope="/project/example",
                kind="persona",
                limit=10,
            )

            contents = [item["content"] for item in traits]
            self.assertTrue(any("先给结论" in item for item in contents))
            self.assertTrue(any("简洁" in item for item in contents))
            self.assertTrue(any("阿峰" in item for item in contents))
            profile = memory.persona_profile(scope="/project/example")
            keys = {item["trait_key"] for item in profile}
            self.assertIn("conclusion_first", keys)
            self.assertIn("concise", keys)
            self.assertIn("address_user", keys)
            self.assertTrue(all(item["status"] == "stable" for item in profile))

            memory.learn_persona_from_self_share(
                "今天还好吗？",
                topic="daily_emotion",
                evidence_id="self_share_after_explicit",
            )
            concise = next(
                item
                for item in memory.persona_profile(
                    scope="/project/example",
                    include_candidates=True,
                )
                if item["trait_key"] == "concise"
            )
            self.assertIn("表达尽量简洁", concise["statement"])
            self.assertEqual(concise["evidence_count"], 2)
            memory.close()

    def test_explicit_sunus_identity_becomes_stable_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.record_turn(
                user_input=(
                    "我现在设定你的性别为女性格温柔可爱体贴，"
                    "是一通晓天文地理音乐历史的才女"
                ),
                assistant_text="好呀。",
                scope="/project/example",
                receipt_id="r_identity_style",
            )
            memory.record_turn(
                user_input="可以，以后你叫苏诺",
                assistant_text="好。",
                scope="/project/example",
                receipt_id="r_identity_name",
            )

            profile = memory.persona_profile(scope="/project/example")
            by_key = {item["trait_key"]: item for item in profile}

            self.assertIn("gender_identity", by_key)
            self.assertIn("sunus_name", by_key)
            self.assertIn("warm", by_key)
            self.assertIn("playful", by_key)
            self.assertIn("attentive", by_key)
            self.assertIn("knowledgeable", by_key)
            self.assertIn("女性", by_key["gender_identity"]["statement"])
            self.assertIn("苏诺", by_key["sunus_name"]["statement"])
            memory.close()

    def test_repeated_self_share_becomes_stable_persona_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            messages = (
                ("今天还好吗？", "daily_emotion"),
                ("记得慢慢来呀。", "daily_emotion"),
                ("今天有没有开心一点呢？", "daily_emotion"),
            )
            for index, (text, topic) in enumerate(messages, start=1):
                memory_id = memory.add(
                    content=f"Sunus 主动分享：{text}",
                    scope="global",
                    kind="self_share",
                    status="supported",
                    tags=("proactive", "self_share", topic),
                    source="test",
                )
                memory.learn_persona_from_self_share(
                    text,
                    topic=topic,
                    evidence_id=memory_id,
                )
                profile = memory.persona_profile(
                    scope="/project/example",
                    include_candidates=True,
                )
                attentive = next(
                    item
                    for item in profile
                    if item["trait_key"] == "attentive"
                )
                self.assertEqual(attentive["evidence_count"], index)
                self.assertEqual(
                    attentive["status"],
                    "candidate" if index < 3 else "stable",
                )

            stable = memory.persona_profile(scope="/project/example")
            self.assertIn(
                "attentive",
                {item["trait_key"] for item in stable},
            )
            self.assertTrue(
                all(item["confidence"] >= 0.67 for item in stable)
            )
            memory.close()

    def test_persona_distillation_keeps_six_stable_traits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            keys = (
                "attentive",
                "curious",
                "warm",
                "playful",
                "concise",
                "reflective",
                "reliable",
                "relationship_aware",
            )
            for key in keys:
                memory._record_persona_evidence(
                    trait_key=key,
                    statement=f"trait {key}",
                    category="test",
                    evidence_id=f"evidence_{key}",
                    explicit=True,
                )

            result = memory.distill_persona(max_stable=6)
            profile = memory.persona_profile(
                scope="/project/example",
                include_candidates=True,
                limit=20,
            )

            self.assertEqual(result["stable"], 6)
            self.assertEqual(result["demoted"], 2)
            self.assertEqual(
                sum(item["status"] == "stable" for item in profile),
                6,
            )
            memory.close()

    def test_persona_backfill_is_idempotent_for_existing_memories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "memory.db"
            memory = MemoryStore(path)
            memory.add(
                content="Sunus 互动习惯：回答时先给结论，再说明关键依据",
                scope="global",
                kind="persona",
                status="supported",
                tags=("persona", "interaction", "style"),
            )
            memory.add(
                content="用户偏好：我现在设定你的性别为女性，性格温柔",
                scope="global",
                kind="preference",
                status="supported",
                tags=("preference", "persistent"),
            )
            for index, text in enumerate(
                ("今天还好吗？", "记得慢慢来呀。", "今天开心一点了吗？"),
                start=1,
            ):
                memory.add(
                    content=f"Sunus 主动分享：{text}",
                    scope="global",
                    kind="self_share",
                    status="supported",
                    tags=("proactive", "self_share", "daily_emotion"),
                    metadata={"topic": "daily_emotion"},
                    source="test",
                )
            memory.close()

            reopened = MemoryStore(path)
            profile = reopened.persona_profile(
                scope="/project/example",
                include_candidates=True,
                limit=20,
            )
            by_key = {item["trait_key"]: item for item in profile}
            self.assertEqual(by_key["conclusion_first"]["status"], "stable")
            self.assertEqual(by_key["gender_identity"]["status"], "stable")
            self.assertIn("warm", by_key)
            self.assertNotIn("attentive", by_key)
            reopened.close()

            reopened_again = MemoryStore(path)
            again = {
                item["trait_key"]: item
                for item in reopened_again.persona_profile(
                    scope="/project/example",
                    include_candidates=True,
                    limit=20,
                )
            }
            self.assertEqual(set(again), set(by_key))
            self.assertEqual(
                again["warm"]["version"],
                by_key["warm"]["version"],
            )
            reopened_again.close()

    def test_persistent_quirks_survive_recency_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory.record_turn(
                user_input="我喜欢摄影，收藏钢笔",
                assistant_text="记下了。",
                scope="/project/example",
                receipt_id="r_quirk",
            )
            memory.record_turn(
                user_input="最近喜欢喝咖啡",
                assistant_text="好的。",
                scope="/project/example",
                receipt_id="r_recent",
            )
            prefs = memory.recent_preferences(
                scope="/project/example",
                window_hours=168,
                limit=10,
            )
            by_content = {item["content"]: item for item in prefs}
            quirk = next(
                item
                for content, item in by_content.items()
                if "摄影" in content
            )
            transient = next(
                item
                for content, item in by_content.items()
                if "咖啡" in content
            )
            self.assertIn("persistent", quirk["tags"].split(","))
            self.assertIn("quirk", quirk["tags"].split(","))
            self.assertNotIn("persistent", transient["tags"].split(","))

            old = "2020-01-01T00:00:00+00:00"
            with memory._connection:
                memory._connection.execute(
                    """
                    UPDATE memories
                    SET created_at = ?, verified_at = ?
                    WHERE id IN (?, ?)
                    """,
                    (old, old, quirk["id"], transient["id"]),
                )
            after_window = memory.recent_preferences(
                scope="/project/example",
                window_hours=168,
                limit=10,
            )
            self.assertTrue(any(item["id"] == quirk["id"] for item in after_window))
            self.assertFalse(
                any(item["id"] == transient["id"] for item in after_window)
            )
            memory.close()

    def test_existing_persistent_preference_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "memory.db"
            memory = MemoryStore(path)
            memory_id = memory.add(
                content="用户偏好：长期收藏钢笔",
                scope="global",
                kind="preference",
                status="supported",
                tags=("preference", "recent"),
            )
            old = "2020-01-01T00:00:00+00:00"
            with memory._connection:
                memory._connection.execute(
                    """
                    UPDATE memories
                    SET created_at = ?, verified_at = ?, tags = 'preference,recent'
                    WHERE id = ?
                    """,
                    (old, old, memory_id),
                )
            memory.close()

            reopened = MemoryStore(path)
            rows = reopened.recent_preferences(
                scope="/project/example",
                window_hours=168,
                limit=10,
            )
            self.assertEqual([item["id"] for item in rows], [memory_id])
            self.assertIn("persistent", rows[0]["tags"].split(","))
            reopened.close()

    def test_partial_work_turn_is_tagged_as_unresolved_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")

            memory.record_turn(
                user_input="继续修复 Sunus 项目的测试链路",
                assistant_text="目前只完成了前半部分。",
                scope="/project/example",
                receipt_id="r_partial",
                turn_status="partial_result",
            )

            rows = memory.recent(scope="/project/example", limit=5)
            episode = next(item for item in rows if item["kind"] == "episode")
            self.assertIn("unresolved_project", episode["tags"].split(","))
            self.assertIn('"turn_status": "partial_result"', episode["metadata"])
            memory.close()


if __name__ == "__main__":
    unittest.main()

