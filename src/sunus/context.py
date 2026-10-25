from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sunus.memory import estimate_tokens


@dataclass(frozen=True)
class CompressionResult:
    messages: list[dict[str, Any]]
    summary: str
    compacted: bool
    before_tokens: int
    after_tokens: int
    summarized_messages: int


def message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    content = message.get("content")
    if content:
        parts.append(str(content))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        parts.append(str(function.get("name") or ""))
        parts.append(str(function.get("arguments") or ""))
    if message.get("reasoning_content"):
        parts.append(str(message["reasoning_content"]))
    if message.get("reasoning_details"):
        parts.append(str(message["reasoning_details"]))
    return "\n".join(part for part in parts if part)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(message_text(message)) for message in messages)


def repair_tool_pairs(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Make assistant tool calls and their tool responses valid for chat APIs."""
    repaired: list[dict[str, Any]] = []
    inserted = 0
    dropped = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        role = str(message.get("role") or "")
        if role == "tool":
            dropped += 1
            index += 1
            continue
        repaired.append(message)
        tool_calls = message.get("tool_calls") or []
        if role != "assistant" or not tool_calls:
            index += 1
            continue

        expected = [
            str(call.get("id") or "")
            for call in tool_calls
            if call.get("id")
        ]
        found: set[str] = set()
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            tool_message = messages[cursor]
            call_id = str(tool_message.get("tool_call_id") or "")
            if call_id in expected and call_id not in found:
                repaired.append(tool_message)
                found.add(call_id)
            else:
                dropped += 1
            cursor += 1
        for call_id in expected:
            if call_id in found:
                continue
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "ok": False,
                            "content": (
                                "运行时恢复：此工具调用未收到原始结果，"
                                "继续对话时按中断处理。"
                            ),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            inserted += 1
        index = cursor
    return repaired, inserted, dropped


def summarize_messages(
    messages: list[dict[str, Any]],
    *,
    existing: str = "",
    max_chars: int = 12_000,
) -> str:
    lines: list[str] = []
    if existing.strip():
        lines.append(existing.strip())
    for message in messages:
        role = str(message.get("role") or "")
        content = " ".join(str(message.get("content") or "").split())
        if role == "user":
            if content:
                lines.append(f"- 用户要求：{content[:800]}")
        elif role == "assistant":
            if content:
                lines.append(f"- Agent 结论：{content[:500]}")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = str(function.get("name") or "unknown")
                arguments = " ".join(str(function.get("arguments") or "").split())
                if name != "submit_result":
                    lines.append(f"- 已调用工具 {name}：{arguments[:300]}")
        elif role == "tool":
            detail = _tool_summary(content)
            if detail:
                lines.append(f"- 工具结果：{detail}")
    summary = "\n".join(dict.fromkeys(line for line in lines if line))
    if len(summary) <= max_chars:
        return summary
    head_chars = max(0, max_chars - 2_500)
    omitted = len(summary) - head_chars - 2_000
    return (
        summary[:head_chars]
        + f"\n- …省略 {omitted} 字符…\n"
        + summary[-2_000:]
    )


def _tool_summary(content: str) -> str:
    if not content:
        return ""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content[:400]
    if not isinstance(payload, dict):
        return content[:400]
    ok = payload.get("ok")
    body = " ".join(str(payload.get("content") or "").split())
    artifact = payload.get("artifact")
    changed = payload.get("changed_files") or []
    parts = [f"ok={ok}"]
    if changed:
        parts.append("changed=" + ",".join(str(item) for item in changed[:8]))
    if artifact:
        parts.append(f"artifact={artifact}")
    if body:
        parts.append(body[:300])
    return "；".join(parts)


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    existing_summary: str,
    soft_tokens: int,
    hard_tokens: int,
    keep_recent: int,
    summary_max_chars: int,
    observed_tokens: int = 0,
) -> CompressionResult:
    repaired, _, _ = repair_tool_pairs(messages)
    before_tokens = estimate_messages_tokens(repaired)
    trigger_tokens = max(before_tokens, max(0, observed_tokens))
    if trigger_tokens <= soft_tokens:
        return CompressionResult(
            messages=repaired,
            summary=existing_summary,
            compacted=False,
            before_tokens=trigger_tokens,
            after_tokens=trigger_tokens,
            summarized_messages=0,
        )

    user_indices = [
        index
        for index, message in enumerate(repaired)
        if message.get("role") == "user"
    ]
    if len(user_indices) <= 1:
        pinned = repaired[user_indices[0]] if user_indices else None
        candidates = [
            index
            for index in range(2 if pinned else 1, len(repaired))
            if repaired[index].get("role") in {"assistant", "user"}
        ]
        if not candidates:
            return CompressionResult(
                messages=repaired,
                summary=existing_summary,
                compacted=False,
                before_tokens=trigger_tokens,
                after_tokens=trigger_tokens,
                summarized_messages=0,
            )
        desired = max(2 if pinned else 1, len(repaired) - keep_recent)
        early_candidates = [index for index in candidates if index <= desired]
        cut = early_candidates[-1] if early_candidates else candidates[0]
        while cut < len(repaired):
            kept = ([pinned] if pinned else []) + repaired[cut:]
            if estimate_messages_tokens(kept) <= hard_tokens:
                break
            later = [index for index in candidates if index > cut]
            if not later:
                break
            cut = later[0]
        old_messages = repaired[1 if pinned else 0 : cut]
        if pinned:
            old_messages = [pinned, *old_messages]
        kept = ([pinned] if pinned else []) + repaired[cut:]
        summary = summarize_messages(
            old_messages,
            existing=existing_summary,
            max_chars=summary_max_chars,
        )
        return CompressionResult(
            messages=kept,
            summary=summary,
            compacted=True,
            before_tokens=trigger_tokens,
            after_tokens=estimate_messages_tokens(kept),
            summarized_messages=len(old_messages),
        )

    desired = max(1, len(repaired) - keep_recent)
    candidates = [index for index in user_indices if index <= desired]
    cut = candidates[-1] if candidates else user_indices[0]
    while cut > user_indices[0]:
        tail_tokens = estimate_messages_tokens(repaired[cut:])
        if tail_tokens <= hard_tokens:
            break
        later = [index for index in user_indices if index > cut]
        if not later:
            break
        cut = later[0]

    old_messages = repaired[:cut]
    kept = repaired[cut:]
    summary = summarize_messages(
        old_messages,
        existing=existing_summary,
        max_chars=summary_max_chars,
    )
    after_tokens = estimate_messages_tokens(kept)
    return CompressionResult(
        messages=kept,
        summary=summary,
        compacted=True,
        before_tokens=trigger_tokens,
        after_tokens=after_tokens,
        summarized_messages=len(old_messages),
    )
