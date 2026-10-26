from __future__ import annotations

import unittest

from sunus.relationship import (
    RelationshipConfig,
    extract_relationship_update,
    relationship_from_mapping,
    relationship_prompt,
    relationship_stage,
    relationship_trait,
    transition_relationship,
)


class RelationshipTests(unittest.TestCase):
    def test_extracts_sunus_role_and_relationship_label(self) -> None:
        update = extract_relationship_update(
            "以后你就是我的女朋友"
        )
        self.assertEqual(
            update,
            {
                "label": "恋人",
                "sunus_role": "女朋友",
            },
        )

    def test_extracts_user_role_and_address(self) -> None:
        update = extract_relationship_update(
            "从今天起你是我姐姐，叫我阿峰"
        )
        self.assertEqual(
            update,
            {
                "label": "家人",
                "sunus_role": "姐姐",
                "address": "阿峰",
            },
        )

    def test_relationship_questions_and_roleplay_are_not_persisted(self) -> None:
        for text in (
            "你是我女朋友吗？",
            "如果你是我女朋友会怎样",
            "假装你是我女朋友",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_relationship_update(text))

    def test_relationship_can_be_cleared(self) -> None:
        self.assertEqual(
            extract_relationship_update("以后你不是我女朋友了"),
            {},
        )
        self.assertEqual(
            extract_relationship_update("取消人物关系设定"),
            {},
        )
        self.assertEqual(
            extract_relationship_update("以后我们升级成夫妻"),
            {"label": "夫妻"},
        )
        self.assertEqual(
            extract_relationship_update("我们结婚吧"),
            {"label": "夫妻"},
        )

    def test_relationship_prompt_sets_context_without_overriding_boundaries(
        self,
    ) -> None:
        relationship = relationship_from_mapping(
            {
                "label": "恋人",
                "sunus_role": "女朋友",
                "user_role": "男朋友",
                "address": "阿峰",
            }
        )
        prompt = relationship_prompt(relationship)
        self.assertIn("双方关系：恋人", prompt)
        self.assertIn("Sunus 的角色：女朋友", prompt)
        self.assertIn("用户的角色：男朋友", prompt)
        self.assertIn("称呼：阿峰", prompt)
        self.assertIn("默认按这套关系的亲疏、称呼和边界", prompt)
        self.assertIn("不能改变事实准确性、安全边界、工具权限", prompt)
        self.assertEqual(
            relationship_trait(relationship),
            "Sunus 与用户的人物关系设定为“恋人”；"
            "Sunus 的角色是“女朋友”；用户的角色是“男朋友”",
        )

    def test_empty_relationship_has_no_prompt(self) -> None:
        self.assertEqual(relationship_prompt(RelationshipConfig()), "")

    def test_non_couple_relationship_locks_after_initial_setup(self) -> None:
        initial = transition_relationship(
            RelationshipConfig(),
            {
                "label": "父女",
                "sunus_role": "女儿",
                "user_role": "父亲",
            },
        )
        self.assertTrue(initial.allowed)
        self.assertEqual(initial.relationship.label, "父女")
        self.assertEqual(relationship_stage(initial.relationship), "locked")

        denied = transition_relationship(
            initial.relationship,
            {"label": "兄弟", "sunus_role": "妹妹"},
        )
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.relationship, initial.relationship)
        self.assertIn("已锁定", denied.reason)

    def test_couple_can_update_clear_or_upgrade_to_spouse(self) -> None:
        couple = RelationshipConfig(
            label="恋人",
            sunus_role="女朋友",
            user_role="男朋友",
            address="阿峰",
        )
        self.assertEqual(relationship_stage(couple), "couple")
        updated = transition_relationship(couple, {"address": "亲爱的"})
        self.assertTrue(updated.allowed)
        self.assertEqual(updated.relationship.address, "亲爱的")

        spouse = transition_relationship(
            updated.relationship,
            {"label": "夫妻", "sunus_role": "妻子", "user_role": "丈夫"},
        )
        self.assertTrue(spouse.allowed)
        self.assertEqual(relationship_stage(spouse.relationship), "spouse")
        self.assertIn("升级为夫妻", spouse.reason)

        natural_upgrade = transition_relationship(
            couple,
            extract_relationship_update("以后我们升级成夫妻") or {},
        )
        self.assertTrue(natural_upgrade.allowed)
        self.assertEqual(
            relationship_stage(natural_upgrade.relationship),
            "spouse",
        )

        locked = transition_relationship(spouse.relationship, {})
        self.assertFalse(locked.allowed)
        self.assertEqual(locked.relationship, spouse.relationship)

        cleared = transition_relationship(couple, {})
        self.assertTrue(cleared.allowed)
        self.assertFalse(cleared.relationship.configured)


if __name__ == "__main__":
    unittest.main()
