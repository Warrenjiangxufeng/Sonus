from __future__ import annotations

import unittest

from sunus.scope import (
    classify_request,
    classify_request_with_context,
    conversation_hint,
    is_casual_conversation,
)


class RequestScopeTests(unittest.TestCase):
    def test_explicit_actions_are_allowed(self) -> None:
        scope = classify_request("请修复问题并运行测试")
        self.assertTrue(scope.allows_mutation)
        self.assertTrue(scope.allows_commands)
        self.assertFalse(scope.allows_network)

    def test_mutation_implicitly_allows_local_verification_commands(self) -> None:
        scope = classify_request("修复记忆系统的写入逻辑")
        self.assertTrue(scope.allows_mutation)
        self.assertTrue(scope.allows_commands)
        self.assertFalse(scope.allows_network)

    def test_colloquial_install_command_is_recognized(self) -> None:
        scope = classify_request("你怎么可能没有shell工具？没有就装啊！")
        self.assertTrue(scope.allows_commands)

    def test_questions_and_negative_examples_do_not_authorize_actions(self) -> None:
        for text in (
            "为什么会报错？",
            "如何修复这个问题？",
            "能否修改这个文件？",
            "不能修改文件，只告诉我原因。",
            "不要修改，也不要运行测试，只分析原因。",
        ):
            with self.subTest(text=text):
                scope = classify_request(text)
                self.assertFalse(scope.allows_mutation)
                self.assertFalse(scope.allows_commands)

    def test_confirmation_allows_previously_described_plan(self) -> None:
        scope = classify_request("可以，做吧")
        self.assertTrue(scope.allows_mutation)
        self.assertTrue(scope.allows_commands)
        self.assertTrue(scope.allows_network)

    def test_suggestion_request_allows_suggestions_but_not_side_effects(self) -> None:
        scope = classify_request("你觉得有哪些优化建议？")
        self.assertTrue(scope.allows_suggestions)
        self.assertFalse(scope.allows_mutation)
        self.assertFalse(scope.allows_commands)

    def test_casual_conversation_is_separate_from_project_questions(self) -> None:
        for text in (
            "哈喽在干嘛",
            "聊天别这么呆板嘛，你是女孩吗",
            "是有点冷淡了",
            "怎么追一个不太喜欢你的相亲对象啊",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_casual_conversation(text))

        for text in (
            "codex CLI 能同时开两个窗口吗",
            "修复登录接口并运行测试",
            "分析当前项目架构",
            "读取 Documents/Sunus/README.md",
            "读取 note.txt",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_casual_conversation(text))

    def test_short_followup_is_marked_ambiguous_for_model_review(self) -> None:
        self.assertEqual(conversation_hint("哈喽在干嘛"), "casual")
        self.assertEqual(conversation_hint("加点卖萌的元素"), "ambiguous")
        self.assertEqual(conversation_hint("分析当前项目架构"), "task")

    def test_referential_followup_inherits_only_proposed_capability(self) -> None:
        previous = (
            "我手里确实有联网工具，只是默认关着。"
            "你要是愿意，我现在试一次，去搜搜看。"
        )
        scope = classify_request_with_context(
            "试一下，特别是江浙沪ip的",
            previous_assistant=previous,
        )
        self.assertTrue(scope.allows_network)
        self.assertFalse(scope.allows_mutation)
        self.assertFalse(scope.allows_commands)
        self.assertFalse(
            is_casual_conversation(
                "试一下，特别是江浙沪ip的",
                previous_assistant=previous,
            )
        )
        without_network_offer = classify_request_with_context(
            "试一下",
            previous_assistant="我刚才只是在解释概念。",
        )
        self.assertFalse(without_network_offer.allows_network)


if __name__ == "__main__":
    unittest.main()
