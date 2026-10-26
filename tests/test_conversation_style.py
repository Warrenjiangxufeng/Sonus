from __future__ import annotations

import unittest

from sunus.conversation_style import plan_conversation_turn


class ConversationStyleTests(unittest.TestCase):
    def test_reply_budget_mirrors_input_turn_length(self) -> None:
        cases = (
            ("嗯", 32),
            ("今天有点累", 72),
            ("今天开会的时候被临时加了一个很麻烦的需求", 120),
            (
                "今天发生了很多事情，我想从头跟你说一下具体情况，包括上午开会时"
                "突然增加的需求、下午和同事沟通出现的问题，以及晚上还要继续处理的工作。",
                180,
            ),
        )
        for text, limit in cases:
            with self.subTest(text=text):
                plan = plan_conversation_turn(text)
                self.assertEqual(plan.max_chars, limit)
                self.assertIn("用户本轮约", plan.prompt())

    def test_explicit_detail_request_can_expand(self) -> None:
        plan = plan_conversation_turn("详细说说为什么")
        self.assertEqual(plan.max_chars, 260)
        self.assertIn("明确要求展开", plan.target_style)


if __name__ == "__main__":
    unittest.main()
