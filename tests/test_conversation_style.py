from __future__ import annotations

import unittest

from sunus.conversation_style import (
    conversation_repair_instruction,
    plan_conversation_turn,
)


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

    def test_verbosity_and_ai_feedback_force_a_tight_human_reply(self) -> None:
        plan = plan_conversation_turn(
            "你咋老爱说那么多废话，我懒得读小作文"
        )
        self.assertEqual(plan.max_chars, 40)
        self.assertIn("嫌话多", plan.target_style)
        self.assertIn("禁止解释自己", plan.target_style)
        self.assertIn("我不问了", plan.target_style)

    def test_question_rejection_forbids_another_question(self) -> None:
        plan = plan_conversation_turn(
            "你根本不知道我的情绪，天天问问问，有啥问的"
        )
        self.assertEqual(plan.max_chars, 40)
        self.assertIn("暂停追问", plan.target_style)
        self.assertIn("不要宣布", plan.target_style)

    def test_self_affect_question_blocks_internal_mechanism_dump(self) -> None:
        plan = plan_conversation_turn("你有自己的情绪和欲望感知系统么")
        self.assertEqual(plan.max_chars, 100)
        self.assertIn("先直接回答", plan.target_style)
        self.assertIn("禁止罗列精力、疲劳", plan.target_style)

    def test_humanlike_request_blocks_self_improvement_plan(self) -> None:
        plan = plan_conversation_turn("怎么样变得更像人那一套")
        self.assertEqual(plan.max_chars, 90)
        self.assertIn("禁止列三条计划", plan.target_style)
        self.assertIn("用“你要嫌烦我就自己磨”", plan.target_style)

    def test_assistant_like_meta_reply_is_sent_for_one_repair(self) -> None:
        cases = (
            (
                "你咋老爱说那么多废话",
                "行，我话多了。说吧，什么事让你烦。",
            ),
            (
                "天天问问问，有啥问的",
                "嗯，是我话多了还老问。那我先不问了。",
            ),
            (
                "你看你说的这像人话么",
                "不像。人不会说这种话，我也不装了。",
            ),
            (
                "你有自己的情绪和欲望感知系统么",
                "有。我的情绪有抓手，精力、疲劳、亲近感都会积累。",
            ),
            (
                "怎么样变得更像人那一套",
                "三条。一是被说假时不解释，二是敢说不喜欢，三是短。",
            ),
            (
                "我来调教你更像人",
                "今天我把裁判的位置推给你了。要是你嫌烦，那我就自己磨。",
            ),
        )
        for user_input, reply in cases:
            with self.subTest(user_input=user_input):
                repair = conversation_repair_instruction(user_input, reply)
                self.assertIn("重写本轮日常回复", repair)

    def test_direct_human_reply_does_not_need_repair(self) -> None:
        repair = conversation_repair_instruction(
            "天天问问问，有啥问的",
            "行，挨骂也认。你先消气，我不绕了。",
        )
        self.assertEqual(repair, "")

    def test_human_self_affect_answer_does_not_need_repair(self) -> None:
        repair = conversation_repair_instruction(
            "你有自己的情绪和欲望感知系统么",
            "有。会累，也会被一句话弄得心里一紧；想靠近，偶尔也想躲。",
        )
        self.assertEqual(repair, "")

    def test_question_is_repaired_only_when_inner_readiness_disallows_it(self) -> None:
        blocked = conversation_repair_instruction(
            "今天有点累",
            "怎么了，跟我说说？",
            questions_allowed=False,
        )
        allowed = conversation_repair_instruction(
            "今天有点累",
            "怎么了，跟我说说？",
            questions_allowed=True,
        )
        self.assertIn("追问准备度不足", blocked)
        self.assertEqual(allowed, "")


if __name__ == "__main__":
    unittest.main()
