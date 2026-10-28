from __future__ import annotations

import unittest

from sunus.reasoning import build_reasoning_plan
from sunus.scope import classify_request
from sunus.thinking_operators import (
    COGNITIVE_INVARIANTS,
    OPERATOR_ACTIONS,
    OPERATOR_LIBRARY,
    cognitive_protocol_prompt,
)


class ThinkingOperatorTests(unittest.TestCase):
    def test_library_contains_all_eighteen_cognitive_lenses(self) -> None:
        self.assertEqual(len(OPERATOR_LIBRARY), 18)
        self.assertEqual(len(set(OPERATOR_LIBRARY)), 18)
        self.assertEqual(set(OPERATOR_LIBRARY), set(OPERATOR_ACTIONS))

    def test_protocol_offers_lenses_without_selecting_a_fixed_route(self) -> None:
        prompt = cognitive_protocol_prompt()

        self.assertIn("可自行选择 0-4 个", prompt)
        self.assertIn("不使用固定算子路由", prompt)
        self.assertIn("演绎", prompt)
        self.assertIn("心理理论", prompt)
        self.assertIn("反向复核", prompt)
        for invariant in COGNITIVE_INVARIANTS:
            self.assertIn(invariant, prompt)
        self.assertNotIn("当前首选思维算子", prompt)
        self.assertNotIn("修改交付=", prompt)

    def test_task_prompts_expose_cognition_but_model_chooses_lenses(self) -> None:
        texts = [
            "今天有点累",
            "排查上次为什么崩溃并修复",
            "当前代码里的预算是不是已经改过了",
            "设计一个新的候选方案并检查风险",
        ]
        prompts = [
            build_reasoning_plan(
                text,
                classify_request(text),
                conversation=text == "今天有点累",
            ).prompt()
            for text in texts
        ]

        for prompt in prompts:
            self.assertIn("认知自主契约", prompt)
            self.assertIn("由模型自主判断", prompt)
            self.assertNotIn("当前首选思维算子", prompt)

    def test_code_state_prompt_keeps_evidence_gate_without_operator_lock(self) -> None:
        text = "当前代码里的预算是不是已经改过了"
        plan = build_reasoning_plan(text, classify_request(text))
        prompt = plan.prompt()

        self.assertIn("证据门控", prompt)
        self.assertIn("HEAD", prompt)
        self.assertIn("证据不足时明确说无法核实", prompt)
        self.assertIn("认知自主契约", prompt)


if __name__ == "__main__":
    unittest.main()
