from __future__ import annotations

import unittest

from sunus.reasoning import build_reasoning_plan
from sunus.scope import classify_request
from sunus.thinking_operators import (
    MAX_OPERATORS,
    OPERATOR_LIBRARY,
    select_operators,
)


class ThinkingOperatorTests(unittest.TestCase):
    def test_library_contains_all_eighteen_operators(self) -> None:
        self.assertEqual(len(OPERATOR_LIBRARY), 18)
        self.assertEqual(len(set(OPERATOR_LIBRARY)), 18)

    def test_mode_routes_are_bounded_and_distinct(self) -> None:
        cases = {
            "根因诊断": {"溯因", "因果", "反事实", "批判"},
            "修改交付": {"发散", "反事实", "决策优化", "元认知"},
            "一般任务": {"归纳", "类比", "概率", "元认知"},
            "简单状态查询": {"演绎", "批判"},
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode):
                selection = select_operators(
                    mode=mode,
                    complexity="高" if mode != "简单状态查询" else "低",
                )
                self.assertTrue(2 <= len(selection.operators) <= MAX_OPERATORS)
                self.assertTrue(expected.issubset(set(selection.operators)))
                self.assertTrue(selection.contrast_checks)

    def test_casual_prompt_skips_operator_overhead(self) -> None:
        plan = build_reasoning_plan(
            "今天有点累",
            classify_request("今天有点累"),
            conversation=True,
        )
        self.assertNotIn("思维算子", plan.prompt())

    def test_complex_prompt_adds_library_but_stays_compact(self) -> None:
        text = "排查上次为什么崩溃并修复"
        plan = build_reasoning_plan(text, classify_request(text))
        prompt = plan.prompt()

        self.assertIn("当前首选", prompt)
        self.assertIn("反向复核", prompt)
        self.assertIn("备用算子", prompt)
        self.assertLess(len(prompt), 1400)


if __name__ == "__main__":
    unittest.main()
