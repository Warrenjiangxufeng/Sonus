from __future__ import annotations

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
)


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


if __name__ == "__main__":
    unittest.main()
