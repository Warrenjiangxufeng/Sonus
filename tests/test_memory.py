from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sonus.memory import MemoryStore, estimate_tokens, text_metrics


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
                content="Sonus 的记忆检索会注入相关情景",
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
        metrics = text_metrics("Sonus 读取记忆 token")
        self.assertEqual(metrics["characters"], len("Sonus 读取记忆 token"))
        self.assertEqual(
            metrics["estimated_tokens"],
            estimate_tokens("Sonus 读取记忆 token"),
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
            memory.close()

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


if __name__ == "__main__":
    unittest.main()

