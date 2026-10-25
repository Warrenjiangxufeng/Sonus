from __future__ import annotations

import unittest

from sunus.reasoning import (
    diagnostic_protocol_prompt,
    is_diagnostic_request,
    is_status_query_request,
    status_query_prompt,
)


class ReasoningTests(unittest.TestCase):
    def test_diagnostic_requests_are_detected(self) -> None:
        for text in (
            "为什么它查错了目录？",
            "查一下 why 找不到今天的记录",
            "它说没有 shell 工具，这是怎么回事",
            "帮我排查 pytest 无法运行的问题",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_diagnostic_request(text))

    def test_normal_mutation_is_not_forced_into_diagnostic_protocol(self) -> None:
        self.assertFalse(is_diagnostic_request("修复记忆系统的写入逻辑"))

    def test_status_lookup_does_not_trigger_diagnostic_protocol(self) -> None:
        for text in (
            "查一下本地sunus代码有没有提交到git",
            "检查 Git 状态",
            "看看备份目录有没有改动",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_diagnostic_request(text))
                self.assertTrue(is_status_query_request(text))

    def test_causal_status_question_remains_diagnostic(self) -> None:
        text = "查一下为什么代码没有提交到 git"
        self.assertTrue(is_diagnostic_request(text))
        self.assertFalse(is_status_query_request(text))

    def test_protocol_requires_competing_hypotheses_and_evidence_gates(self) -> None:
        prompt = diagnostic_protocol_prompt()
        self.assertIn("至少形成两个竞争假设", prompt)
        self.assertIn("优先尝试证伪", prompt)
        self.assertIn("inspect_environment", prompt)
        self.assertIn("单个失败入口不能证明", prompt)
        self.assertIn("不输出思维链", prompt)
        self.assertIn("最多 3 条关键依据", prompt)
        self.assertIn("不展示完整代码、diff", prompt)

    def test_status_prompt_keeps_answer_inside_requested_scope(self) -> None:
        prompt = status_query_prompt()
        self.assertIn("只检查用户明确点名的仓库", prompt)
        self.assertIn("备份目录、worktree 和副本不主动审计", prompt)
        self.assertIn("submit_result.details 留空", prompt)


if __name__ == "__main__":
    unittest.main()
