from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sunus.affect import (
    DESIRE_EXPRESSIONS,
    DESIRE_LABELS,
    DESIRES,
    EMOTION_EXPRESSIONS,
    EMOTION_LABELS,
    EMOTIONS,
    SunusAffectState,
    lexical_perception,
    lexical_events,
)
from sunus.body_runtime import HostTelemetry
from sunus.embodiment import BODY_ONTOLOGY


class SunusAffectTests(unittest.TestCase):
    def test_every_emotion_and_desire_has_an_expression_direction(self) -> None:
        self.assertEqual(set(EMOTIONS), set(EMOTION_LABELS))
        self.assertEqual(set(EMOTIONS), set(EMOTION_EXPRESSIONS))
        self.assertEqual(set(DESIRES), set(DESIRE_LABELS))
        self.assertEqual(set(DESIRES), set(DESIRE_EXPRESSIONS))

    def test_positive_and_negative_input_change_persisted_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = SunusAffectState(path)
            before = state.snapshot(now=1_000.0)

            positive = state.observe(
                "谢谢你，抱抱，今天看见你真开心",
                now=1_010.0,
            )
            self.assertGreater(
                positive.body["closeness"],
                before.body["closeness"],
            )
            self.assertGreater(positive.emotions["affection"], 0.0)

            restored = SunusAffectState(path).snapshot(now=1_011.0)
            self.assertEqual(
                restored.interaction_count,
                positive.interaction_count,
            )
            self.assertGreater(restored.body["closeness"], 0.50)

            negative = state.observe("滚，闭嘴，你真烦", now=1_020.0)
            self.assertGreater(negative.body["tension"], positive.body["tension"])
            self.assertLess(negative.body["closeness"], positive.body["closeness"])
            self.assertGreater(negative.emotions["anger"], 0.0)
            self.assertGreater(negative.desires["distance"], 0.0)

            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("今天看见你真开心", raw)
            self.assertNotIn("你真烦", raw)

    def test_support_and_repair_reduce_tension(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = SunusAffectState(Path(temp) / "state.json")
            upset = state.observe("滚，别烦我", now=2_000.0)
            repaired = state.observe(
                "对不起，别担心，我在，慢慢来",
                now=2_060.0,
            )
            self.assertLess(repaired.body["tension"], upset.body["tension"])
            self.assertGreater(repaired.body["safety"], upset.body["safety"])
            self.assertGreater(repaired.desires["repair"], 0.0)

    def test_state_decays_toward_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = SunusAffectState(Path(temp) / "state.json")
            upset = state.observe("我恨你，滚", now=3_000.0)
            later = state.snapshot(now=3_000.0 + 24 * 3600)
            self.assertLess(later.body["tension"], upset.body["tension"])
            self.assertLess(later.emotions["anger"], upset.emotions["anger"])

    def test_protective_negation_does_not_create_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = SunusAffectState(Path(temp) / "state.json")
            before = state.snapshot(now=4_000.0)
            after = state.observe(
                "我不想让你担心，也不想离开你",
                now=4_001.0,
            )
            self.assertLess(after.desires["distance"], 0.05)
            self.assertLess(after.emotions["anger"], 0.05)
            self.assertLess(abs(after.body["tension"] - before.body["tension"]), 0.05)

    def test_prompt_block_is_explicit_but_not_raw_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = SunusAffectState(path)
            state.observe("谢谢你，我有点好奇为什么", now=5_000.0)
            prompt = state.prompt_block(now=5_001.0)
            self.assertIn("[Sunus 持久内在状态", prompt)
            self.assertIn("身体：", prompt)
            self.assertIn("情绪趋向：", prompt)
            self.assertIn("欲望趋向：", prompt)
            self.assertIn("行动倾向：", prompt)
            self.assertIn("追问准备度：", prompt)
            self.assertNotIn("谢谢你，我有点好奇为什么", prompt)

    def test_question_readiness_is_an_increment_above_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = SunusAffectState(Path(temp) / "state.json")
            baseline = state.snapshot(now=5_500.0)
            self.assertEqual(baseline.question_readiness, 0.0)
            warmed = state.observe(
                "谢谢你，抱抱，我今天很开心",
                now=5_501.0,
            )
            self.assertGreater(warmed.question_readiness, 0.08)

    def test_semantic_appraisal_overrides_keyword_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = SunusAffectState(path)
            snapshot = state.observe(
                "你别这样想，别问我为什么",
                now=5_700.0,
                semantic_events=["安抚"],
                semantic_confidence=0.92,
            )
            self.assertEqual(snapshot.events, ("安抚",))
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["last_appraisal"]["source"], "semantic")
            self.assertFalse(snapshot.questions_inhibited)

    def test_lexical_fallback_resolves_question_conflict(self) -> None:
        events = lexical_events("别问我为什么")
        self.assertIn("追问疲劳", events)
        self.assertNotIn("探索", events)

    def test_question_desire_is_gated_by_inner_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = SunusAffectState(Path(temp) / "state.json")
            self.assertIn(
                "追问欲望未达到阈值",
                state.prompt_block(now=6_000.0),
            )

            state.observe("谢谢你，抱抱，我今天很开心", now=6_001.0)
            warmed_prompt = state.prompt_block(now=6_002.0)
            self.assertIn("可自然问一个具体问题", warmed_prompt)
            self.assertIn("喜欢/依恋", warmed_prompt)
            self.assertIn("关心", warmed_prompt)

            tired = state.observe("别问，天天问问问", now=6_003.0)
            self.assertIn("追问疲劳", tired.events)
            self.assertLess(tired.desires["understanding"], 0.2)
            self.assertIn(
                "追问被当前关系信号暂时压下",
                state.prompt_block(now=6_004.0),
            )
            state.observe("谢谢你，抱抱，我还是想听你说", now=6_003.0 + 46 * 60)
            self.assertIn(
                "可自然问一个具体问题",
                state.prompt_block(now=6_004.0 + 46 * 60),
            )

    def test_perception_separates_observation_inference_and_alternatives(self) -> None:
        perception = lexical_perception(
            "别碰我，我现在有点难受",
            previous_assistant="我刚才追问得太多。",
        )
        self.assertIn("边界", perception.observed_cues)
        self.assertGreaterEqual(
            perception.inferred_user_state["boundary"],
            0.5,
        )
        self.assertEqual(perception.relationship_signal, "boundary")
        self.assertTrue(perception.alternative_explanations)
        self.assertIn("对方拉开距离", perception.response_feedback)

    def test_boundary_arbitration_suppresses_closeness_and_intimacy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = SunusAffectState(Path(temp) / "state.json")
            state.observe("不要碰我，离我远点", now=7_000.0)
            prompt = state.prompt_block(now=7_001.0)
            snapshot = state.snapshot(now=7_001.0)

            self.assertIn("先尊重边界和自主", prompt)
            self.assertNotIn("语气可以放软和靠近", prompt)
            self.assertNotIn("亲密", prompt)
            kinds = [item["kind"] for item in snapshot.action_intents]
            self.assertIn("boundary", kinds)
            self.assertNotIn("closeness", kinds)
            self.assertNotIn("intimacy", kinds)

    def test_response_feedback_updates_next_turn_without_storing_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = SunusAffectState(path)
            after = state.observe(
                "不对，你没懂我的意思",
                now=8_000.0,
            )
            self.assertIn("回应被纠正", after.response_feedback)
            self.assertGreater(after.emotions["shame"], 0.1)
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("你没懂我的意思", raw)

    def test_body_channels_match_ontology_and_record_host_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            state = SunusAffectState(path)
            telemetry = HostTelemetry(
                sampled_at=9_000.0,
                local_hour=1.0,
                load_ratio=0.95,
                disk_used_ratio=0.91,
                observed=("late_hours", "host_load_high", "disk_pressure"),
            )
            snapshot = state.observe(
                "继续处理这个长任务",
                now=9_000.0,
                host_telemetry=telemetry,
            )

            self.assertEqual(
                set(snapshot.body_channels),
                set(BODY_ONTOLOGY.channels),
            )
            self.assertEqual(
                snapshot.host_telemetry["observed"],
                ["late_hours", "host_load_high", "disk_pressure"],
            )
            self.assertEqual(
                snapshot.body_channels["host_sensory_input"]["input_density"],
                round(len("继续处理这个长任务") / 240, 3),
            )

    def test_runtime_activity_updates_action_and_expression_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = SunusAffectState(Path(temp) / "state.json")
            snapshot = state.record_runtime_activity(
                action_names=["read_file", "run_command"],
                failure_count=2,
                changed_file_count=1,
                expression_text="已处理并验证。" * 30,
                mode="task",
                elapsed_seconds=180,
                now=10_000.0,
            )

            action = snapshot.body_channels[
                "proprioceptive_action_state"
            ]
            expression = snapshot.body_channels["expression_and_voice"]
            self.assertEqual(action["status"], "needs_recovery")
            self.assertEqual(action["last_actions"], ["read_file", "run_command"])
            self.assertEqual(expression["mode"], "task")
            self.assertIn(expression["pace"], {"compact", "steady", "expanded"})


if __name__ == "__main__":
    unittest.main()
