from __future__ import annotations

import unittest

from sonus.scope import classify_request


class RequestScopeTests(unittest.TestCase):
    def test_explicit_actions_are_allowed(self) -> None:
        scope = classify_request("请修复问题并运行测试")
        self.assertTrue(scope.allows_mutation)
        self.assertTrue(scope.allows_commands)
        self.assertFalse(scope.allows_network)

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


if __name__ == "__main__":
    unittest.main()
