from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sunus.config import ProactiveConfig
from sunus.memory import MemoryStore
from sunus.proactive import ProactiveScheduler, ProactiveService
from sunus.proactive_thinking import ThoughtSeed


class ProactiveThinkingTests(unittest.TestCase):
    def test_upsert_deduplicates_open_subject_and_merges_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, 9, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(base_interval_sec=10),
                "s_thought",
                clock=lambda: current[0],
            )
            store = scheduler.thoughts
            first = store.upsert(
                ThoughtSeed(
                    question="确认失败是否仍然存在",
                    source_kind="project",
                    source_ref="m1",
                    confidence=0.6,
                    importance=0.8,
                    action_hint="verify",
                    evidence_refs=("m1",),
                )
            )
            second = store.upsert(
                ThoughtSeed(
                    question="确认失败是否仍然存在",
                    source_kind="project",
                    source_ref="m1",
                    confidence=0.9,
                    importance=0.7,
                    action_hint="verify",
                    evidence_refs=("m2",),
                )
            )

            self.assertEqual(first.id, second.id)
            self.assertEqual(second.evidence_refs, ("m1", "m2"))
            self.assertEqual(second.confidence, 0.9)
            self.assertEqual(len(store.open_thoughts()), 1)
            scheduler.close()

    def test_selection_allows_no_action_and_verifies_before_speaking(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, 9, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(
                    base_interval_sec=10,
                    thought_min_score=0.5,
                    thought_cooldown_sec=60,
                ),
                "s_select",
                clock=lambda: current[0],
            )
            store = scheduler.thoughts
            empty = store.select()
            self.assertEqual(empty.action, "do_nothing")

            thought = store.upsert(
                ThoughtSeed(
                    question="先确认失败记录是否已经变化",
                    source_kind="project",
                    source_ref="failure",
                    confidence=0.55,
                    importance=0.9,
                    action_hint="verify",
                    evidence_refs=("episode-1",),
                )
            )
            verify = store.select()
            self.assertEqual(verify.action, "verify")
            self.assertEqual(verify.thought.id, thought.id)

            store.defer(thought.id, 60, "verify_before_speaking")
            current[0] += timedelta(seconds=61)
            message = store.select()
            self.assertEqual(message.action, "message")
            self.assertEqual(message.reason, "verification_window_elapsed")
            scheduler.close()

    def test_response_and_no_response_change_priority_without_treating_silence_as_rejection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, 9, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(
                    base_interval_sec=10,
                    thought_no_response_sec=60,
                ),
                "s_feedback",
                clock=lambda: current[0],
            )
            store = scheduler.thoughts
            thought = store.upsert(
                ThoughtSeed(
                    question="是否适合继续这个话题",
                    source_kind="curiosity",
                    source_ref="topic",
                    confidence=0.8,
                    importance=0.8,
                    prediction="低负担，不要求回复",
                )
            )
            message_id = scheduler.record_message(
                "轻轻问一句",
                thought_id=thought.id,
            )
            self.assertTrue(message_id)
            store.observe_response("continued")
            row = scheduler._connection.execute(
                "SELECT response_outcome FROM proactive_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            self.assertEqual(row["response_outcome"], "continued")
            continued = store.open_thoughts()[0]
            self.assertGreater(continued.importance, 0.8)

            current[0] += timedelta(seconds=61)
            scheduler.record_message(
                "另一条消息",
                thought_id=thought.id,
            )
            current[0] += timedelta(seconds=61)
            updated = store.refresh_outcomes()
            self.assertIn(thought.id, updated)
            row = scheduler._connection.execute(
                """
                SELECT response_outcome FROM proactive_messages
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
            self.assertEqual(row["response_outcome"], "no_response")
            no_response = store.open_thoughts()[0]
            self.assertLess(no_response.importance, continued.importance)
            self.assertEqual(no_response.prediction_match, "matched")
            scheduler.close()

    def test_repeated_delivery_enters_dormant_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = [datetime(2026, 1, 1, 9, tzinfo=timezone.utc)]
            scheduler = ProactiveScheduler(
                Path(temp) / "state.db",
                ProactiveConfig(
                    base_interval_sec=10,
                    thought_cooldown_sec=30,
                ),
                "s_rumination",
                clock=lambda: current[0],
            )
            thought = scheduler.thoughts.upsert(
                ThoughtSeed(
                    question="同一件事反复想",
                    source_kind="curiosity",
                    source_ref="repeat",
                    confidence=0.8,
                    importance=0.9,
                )
            )
            for _ in range(3):
                scheduler.thoughts.record_delivery(thought.id)
                current[0] += timedelta(hours=1)

            stats = scheduler.thoughts.stats()
            self.assertEqual(stats.get("dormant"), 1)
            self.assertEqual(scheduler.thoughts.open_thoughts(), [])
            scheduler.close()

    def test_service_seeds_daypart_review_and_unresolved_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = MemoryStore(root / "memory.db")
            item_id = memory.add(
                content="用户还有一个未完成的同步问题，需要下次确认。",
                scope=str(root),
                kind="episode",
                status="supported",
                tags=("pending",),
            )
            scheduler = ProactiveScheduler(
                root / "state.db",
                ProactiveConfig(base_interval_sec=10),
                "s_seed",
            )
            service = ProactiveService(
                model=None,  # type: ignore[arg-type]
                memory=memory,
                scope=str(root),
                config=scheduler.config,
                thoughts=scheduler.thoughts,
            )
            service.refresh_thoughts(daypart_changed=True, force=True)
            thoughts = scheduler.thoughts.open_thoughts()
            kinds = {item.source_kind for item in thoughts}
            self.assertIn("daypart_review", kinds)
            self.assertIn("memory_gap", kinds)
            self.assertTrue(
                any(item_id in item.evidence_refs for item in thoughts)
            )
            scheduler.close()
            memory.close()


if __name__ == "__main__":
    unittest.main()
