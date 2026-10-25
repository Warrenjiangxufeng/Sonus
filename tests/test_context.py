from __future__ import annotations

import unittest

from sunus.context import (
    compact_messages,
    estimate_messages_tokens,
    repair_tool_pairs,
)


class ContextTests(unittest.TestCase):
    def test_repair_inserts_missing_tool_response_before_next_user(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_missing",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "继续"},
        ]

        repaired, inserted, dropped = repair_tool_pairs(messages)

        self.assertEqual(inserted, 1)
        self.assertEqual(dropped, 0)
        self.assertEqual([item["role"] for item in repaired], [
            "assistant",
            "tool",
            "user",
        ])
        self.assertEqual(repaired[1]["tool_call_id"], "call_missing")

    def test_repair_drops_orphan_tool_message(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "orphan", "content": "{}"},
            {"role": "user", "content": "保留我"},
        ]

        repaired, inserted, dropped = repair_tool_pairs(messages)

        self.assertEqual(inserted, 0)
        self.assertEqual(dropped, 1)
        self.assertEqual(repaired, [{"role": "user", "content": "保留我"}])

    def test_compaction_preserves_early_requirements_and_recent_turns(self) -> None:
        messages = [
            {"role": "user", "content": "必须保留：只修改 src/，不要动配置。"},
            {"role": "assistant", "content": "已记录边界。"},
        ]
        for index in range(8):
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": f"第 {index} 轮要求：" + ("检查详细证据" * 80),
                    },
                    {
                        "role": "assistant",
                        "content": f"第 {index} 轮结论：" + ("已完成检查" * 100),
                    },
                ]
            )
        messages.append({"role": "user", "content": "最新任务：整理最终结论"})

        before = estimate_messages_tokens(messages)
        result = compact_messages(
            messages,
            existing_summary="",
            soft_tokens=1_000,
            hard_tokens=2_000,
            keep_recent=6,
            summary_max_chars=8_000,
        )

        self.assertTrue(result.compacted)
        self.assertIn("只修改 src/", result.summary)
        self.assertIn("最新任务", result.messages[-1]["content"])
        self.assertLess(result.after_tokens, before)
        self.assertGreater(result.summarized_messages, 0)

    def test_provider_prompt_usage_can_trigger_single_user_compaction(self) -> None:
        messages = [{"role": "user", "content": "分析本地记忆架构"}]
        for index in range(6):
            messages.extend(
                [
                    {"role": "assistant", "content": f"步骤 {index}"},
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{index}",
                        "content": '{"ok":true,"content":"结果"}',
                    },
                ]
            )

        result = compact_messages(
            messages,
            existing_summary="",
            soft_tokens=100,
            hard_tokens=120,
            keep_recent=2,
            summary_max_chars=2_000,
            observed_tokens=200,
        )

        self.assertTrue(result.compacted)
        self.assertEqual(result.messages[0]["role"], "user")
        self.assertLess(result.after_tokens, 200)


if __name__ == "__main__":
    unittest.main()
