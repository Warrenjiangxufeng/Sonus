from __future__ import annotations

import re
from dataclasses import dataclass


_DETAIL_REQUEST = re.compile(
    r"(?:详细|展开|具体说|多说|解释一下|讲清楚|为什么|怎么做)"
)


@dataclass(frozen=True)
class ConversationTurnPlan:
    input_chars: int
    target_style: str
    max_chars: int

    def prompt(self) -> str:
        return (
            f"用户本轮约 {self.input_chars} 个字符。{self.target_style}"
            f"回复硬上限约 {self.max_chars} 字，但不要为了凑满或截短而改变自然语气。"
        )


def plan_conversation_turn(text: str) -> ConversationTurnPlan:
    cleaned = " ".join(str(text or "").split())
    length = len(cleaned)
    if _DETAIL_REQUEST.search(cleaned):
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "对方明确要求展开，可以分点或分段，但先直接回答，再补必要解释。"
            ),
            max_chars=260,
        )
    if length <= 4:
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "这是很短的接话或情绪感叹；通常回应一句，别把一句话扩成小作文。"
            ),
            max_chars=32,
        )
    if length <= 16:
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "这是简短聊天；通常一两句话，围绕一个具体点接住即可。"
            ),
            max_chars=72,
        )
    if length <= 48:
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "对方提供了较多信息；可回应两三个重点，但保持真实聊天节奏。"
            ),
            max_chars=120,
        )
    return ConversationTurnPlan(
        input_chars=length,
        target_style=(
            "对方输入较长；先抓住最重要的一点回应，必要时再补充，不要逐句复述。"
        ),
        max_chars=180,
    )
