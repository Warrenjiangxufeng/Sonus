from __future__ import annotations

import unittest

from sunus.reasoning import diagnostic_protocol_prompt, is_diagnostic_request


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

    def test_protocol_requires_competing_hypotheses_and_evidence_gates(self) -> None:
        prompt = diagnostic_protocol_prompt()
        self.assertIn("至少形成两个竞争假设", prompt)
        self.assertIn("优先尝试证伪", prompt)
        self.assertIn("inspect_environment", prompt)
        self.assertIn("单个失败入口不能证明", prompt)
        self.assertIn("不输出思维链", prompt)


if __name__ == "__main__":
    unittest.main()
