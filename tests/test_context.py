from __future__ import annotations

import unittest

from sunus.context import (
    compact_conversation_reply,
    compact_messages,
    estimate_messages_tokens,
    repair_tool_pairs,
    strip_parenthetical_stage_directions,
)


class ContextTests(unittest.TestCase):
    def test_conversation_reply_compacts_on_sentence_boundary(self) -> None:
        text = "第一句说清楚。第二句补充一点。第三句继续展开很多没必要的内容。"
        compacted = compact_conversation_reply(text, limit=18)

        self.assertEqual(compacted, "第一句说清楚。第二句补充一点。")
        self.assertLessEqual(len(compacted), 18)

    def test_conversation_reply_keeps_short_text_unchanged(self) -> None:
        self.assertEqual(
            compact_conversation_reply("先这样。", limit=100),
            "先这样。",
        )

    def test_conversation_reply_removes_parenthetical_stage_directions(self) -> None:
        text = (
            "（没挡，只是吸了口气，手撑在你肩上）\n\n"
            "……轻点。\n\n"
            "（声音闷了点）你这样，我可没法好好站在原地。"
        )

        self.assertEqual(
            compact_conversation_reply(text, limit=100),
            "……轻点。 你这样，我可没法好好站在原地。",
        )

    def test_conversation_reply_keeps_factual_parenthetical_text(self) -> None:
        self.assertEqual(
            compact_conversation_reply("开会时间还是（明天下午）。", limit=100),
            "开会时间还是（明天下午）。",
        )

    def test_conversation_reply_keeps_stage_directions_when_enabled(self) -> None:
        text = "（轻轻靠近）你回来了。"

        self.assertEqual(
            compact_conversation_reply(
                text,
                limit=100,
                allow_stage_directions=True,
            ),
            text,
        )

    def test_stage_direction_cleanup_handles_inline_actions(self) -> None:
        self.assertEqual(
            strip_parenthetical_stage_directions(
                "（抬眼看你，手在衣襟上停了一下）你倒是张口就来。"
            ),
            "你倒是张口就来。",
        )

    def test_stage_direction_cleanup_unwraps_corner_quotes(self) -> None:
        self.assertEqual(
            strip_parenthetical_stage_directions(
                "没躲。\n\n「……一大早就这么不客气。」手还搭着没松。"
            ),
            "没躲。\n……一大早就这么不客气。手还搭着没松。",
        )

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
