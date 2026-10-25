from __future__ import annotations

import unittest

from sunus.reasoning import (
    complex_task_prompt,
    conversation_thinking_prompt,
    diagnostic_protocol_prompt,
    is_complex_request,
    is_diagnostic_request,
    is_simple_lookup_request,
    is_status_query_request,
    lookup_query_prompt,
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

    def test_complex_tasks_get_a_planning_checkpoint(self) -> None:
        self.assertTrue(is_complex_request("排查上次为什么崩溃并修复"))
        self.assertFalse(is_complex_request("检查 Git 状态"))
        prompt = complex_task_prompt()
        self.assertIn("第一次工具调用前", prompt)
        self.assertIn("失败后先归因", prompt)
        self.assertIn("立即收束", prompt)

    def test_relational_conversation_has_a_thought_checkpoint(self) -> None:
        prompt = conversation_thinking_prompt()
        self.assertIn("陪伴、被理解、一个真实评价、建议", prompt)
        self.assertIn("只选择一个主要回应动作", prompt)
        self.assertIn("不过度心理分析", prompt)
        self.assertIn("模板化、咨询师化、客服化", prompt)

    def test_read_only_lookup_requests_are_compact(self) -> None:
        self.assertTrue(
            is_simple_lookup_request(
                "只读任务：检查当前浏览器页面，只给结论"
            )
        )
        self.assertFalse(
            is_simple_lookup_request("检查为什么这个文件报错")
        )
        prompt = lookup_query_prompt()
        self.assertIn("只读查证任务", prompt)
        self.assertIn("submit_result.details 留空", prompt)


if __name__ == "__main__":
    unittest.main()
