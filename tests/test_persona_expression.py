from __future__ import annotations

import unittest

from sunus.config import PersonaConfig
from sunus.persona_expression import build_persona_expression


class PersonaExpressionTests(unittest.TestCase):
    def test_high_sweet_styles_become_behavior_rules(self) -> None:
        persona = PersonaConfig(
            style=(
                ("sweetness", 0.95),
                ("playfulness", 0.8),
                ("initiative", 0.85),
            ),
            context_intensity=(("chat", 1.0), ("task", 0.6)),
        )
        profile = build_persona_expression(
            persona,
            context="chat",
            interaction_heat=0.7,
            relationship_stage="couple",
        )
        prompt = profile.prompt()

        self.assertIn("甜度: 95%", prompt)
        self.assertIn("「甜度」很强", prompt)
        self.assertIn("「卖萌度」很强", prompt)
        self.assertIn("「主动性」很强", prompt)
        self.assertGreater(profile.temperature_bonus, 0)
        self.assertGreater(profile.dynamic_delta, 0)

    def test_task_context_reduces_expression_intensity(self) -> None:
        persona = PersonaConfig(
            style=(("sweetness", 0.9),),
            context_intensity=(("chat", 1.0), ("task", 0.3)),
        )
        profile = build_persona_expression(
            persona,
            context="task",
            interaction_heat=0.4,
        )
        self.assertIn("甜度: 59%", profile.style_text)
        self.assertNotIn("高甜", profile.prompt())

    def test_interaction_heat_changes_only_bounded_expression(self) -> None:
        persona = PersonaConfig(
            style=(("sweetness", 0.7),),
            context_intensity=(("chat", 1.0),),
        )
        cold = build_persona_expression(persona, interaction_heat=0.0)
        hot = build_persona_expression(
            persona,
            interaction_heat=1.0,
            relationship_stage="couple",
        )

        self.assertLess(cold.dynamic_delta, 0)
        self.assertGreater(hot.dynamic_delta, 0)
        self.assertLessEqual(abs(hot.dynamic_delta), 0.1)
        self.assertIn("动态调节", hot.prompt())

    def test_custom_trait_names_are_interpreted_without_archetype_branches(
        self,
    ) -> None:
        persona = PersonaConfig(
            style=(("高冷", 0.9), ("甜度", 0.1)),
            context_intensity=(("chat", 1.0),),
        )
        prompt = build_persona_expression(persona).prompt()

        self.assertIn("高冷: 90%", prompt)
        self.assertIn("「高冷」很强", prompt)
        self.assertIn("「甜度」很低", prompt)
        self.assertIn("多个维度必须组合解释", prompt)

    def test_empty_style_has_no_prompt_overhead(self) -> None:
        profile = build_persona_expression(PersonaConfig())
        self.assertEqual(profile.prompt(), "")
        self.assertEqual(profile.temperature_bonus, 0)


if __name__ == "__main__":
    unittest.main()
