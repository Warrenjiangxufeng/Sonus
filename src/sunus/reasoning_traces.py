from __future__ import annotations

import json
from typing import Any

from sunus.memory import MemoryStore


def _string_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        text = " ".join(str(item or "").split())[:300]
        if text and text not in values:
            values.append(text)
    return values[:limit]


def _lesson_for(
    *,
    mode: str,
    reason: str,
    blocker: str,
    tool_names: list[str],
) -> str:
    if reason == "completed":
        return (
            "从当前可用能力中选择能产生新证据且风险可接受的动作；"
            "若新证据已足够回答，立即停止扩展。"
        )
    if reason == "circuit_open":
        return (
            f"{mode} 中跨动作仍没有产生足够新证据；"
            "下次更早比较信息增益，并在无证据增益时收口。"
        )
    if blocker == "authorization":
        return "权限拒绝是确定性阻塞；保留已确认事实，并说明缺少的具体能力。"
    if blocker == "missing_path":
        return "点名路径不存在时直接报告，不遍历相邻目录碰运气。"
    return "保留已确认事实，先归因再换路径，不重复同一假设。"


def record_reasoning_trace(
    memory: MemoryStore,
    *,
    scope: str,
    receipt_id: str,
    objective: str,
    mode: str,
    capabilities: tuple[str, ...],
    preferred_tools: tuple[str, ...],
    tool_traces: list[dict[str, Any]],
    blocker: str,
    reason: str,
    operators: tuple[str, ...] = (),
    contrast_checks: tuple[str, ...] = (),
    operator_checkpoints: tuple[str, ...] = (),
    perception: dict[str, Any] | None = None,
) -> str | None:
    if not objective:
        return None
    tool_names = [
        str(trace.get("name") or "")
        for trace in tool_traces
        if trace.get("name") and trace.get("name") != "submit_result"
    ]
    lesson = _lesson_for(
        mode=mode,
        reason=reason,
        blocker=blocker,
        tool_names=tool_names,
    )
    content = (
        "策略轨迹："
        f"轨迹={receipt_id}；"
        f"模式={mode}；目标={objective[:300]}；"
        f"能力={','.join(capabilities) or '无'}；"
        f"首选={','.join(preferred_tools) or '直接回答'}；"
        f"算子={','.join(operators) or '无'}；"
        f"检查点={','.join(operator_checkpoints) or '无'}；"
        f"复核={','.join(contrast_checks) or '无'}；"
        f"感知={json.dumps(perception or {}, ensure_ascii=False, sort_keys=True)}；"
        f"实际动作={'→'.join(tool_names[:8]) or '无'}；"
        f"阻塞={blocker or '无'}；结果={reason}；"
        f"策略={lesson}"
    )
    tags = ["strategy", "reasoning_trace", mode]
    if reason == "completed":
        tags.append("completed")
        outcome = "pending"
    elif reason in {"circuit_open", "model_error", "internal_error"}:
        tags.append("failed")
        outcome = "negative"
    else:
        tags.append("partial")
        outcome = "unknown"
    return memory.add(
        content=content,
        scope=scope,
        kind="strategy",
        status="candidate",
        tags=tuple(tags),
        source="reasoning_trace",
        evidence_kind="strategy_trace",
        receipt_id=receipt_id,
        metadata={
            "reason": reason,
            "blocker": blocker,
            "tool_sequence": tool_names,
            "operators": list(operators),
            "contrast_checks": list(contrast_checks),
            "operator_checkpoints": list(operator_checkpoints),
            "outcome": outcome,
            "perception": perception or {},
        },
    )


def record_strategy_reflection(
    memory: MemoryStore,
    *,
    scope: str,
    receipt_id: str,
    trace_id: str,
    objective: str,
    reason: str,
    reflection: dict[str, Any],
) -> str | None:
    """Store a model-derived strategy as a candidate, never as an established rule."""
    strategy = " ".join(str(reflection.get("strategy") or "").split())[:1200]
    if not strategy:
        return None
    when_to_use = _string_list(reflection.get("when_to_use"))
    avoid_when = _string_list(reflection.get("avoid_when"))
    decisive = _string_list(reflection.get("decisive_evidence"))
    unproductive = _string_list(reflection.get("unproductive_patterns"))
    try:
        confidence = min(
            1.0,
            max(0.0, float(reflection.get("confidence") or 0.0)),
        )
    except (TypeError, ValueError):
        confidence = 0.0
    content = (
        "候选策略复盘："
        f"目标={objective[:300]}；"
        f"策略={strategy}；"
        f"适用条件={'；'.join(when_to_use) or '未明确'}；"
        f"不适用条件={'；'.join(avoid_when) or '未明确'}；"
        f"决定性证据={'；'.join(decisive) or '未明确'}；"
        f"低信息增益模式={'；'.join(unproductive) or '未明确'}；"
        f"置信度={confidence:.2f}"
    )
    tags = ["strategy", "reflection", "candidate"]
    if reason == "completed":
        tags.append("promising")
    elif reason in {"circuit_open", "model_error", "internal_error"}:
        tags.append("failed")
    else:
        tags.append("partial")
    return memory.add(
        content=content,
        scope=scope,
        kind="strategy",
        status="candidate",
        tags=tuple(tags),
        source="model_strategy_reflection",
        evidence_kind="strategy_reflection",
        receipt_id=receipt_id,
        metadata={
            "trace_id": trace_id,
            "reason": reason,
            "outcome": "untested",
            "when_to_use": when_to_use,
            "avoid_when": avoid_when,
            "decisive_evidence": decisive,
            "unproductive_patterns": unproductive,
            "confidence": confidence,
        },
    )


REASONING_EXEMPLARS = (
    (
        "复杂问题先固定目标、约束和成功判据，再选择信息增益最高的一步；"
        "证据足够或候选无法区分时停止。"
    ),
    (
        "工具失败先归因为权限、路径、环境、模型或验证问题，再选择不同路径；"
        "同因重试不改变结果。"
    ),
    (
        "复合目标先明确缺少的证据，再比较当前可用能力能提供的证据类型；"
        "不要把某个工具名当成唯一入口。"
    ),
    (
        "只读查证先给结论，不把命令、工具日志和内部推理混进最终回答。"
    ),
    (
        "运行时版本不兼容时停止并请求用户决定安装或改用替代路径；"
        "禁止伪造 runtime 版本绕过依赖检查。"
    ),
)


def seed_reasoning_exemplars(
    memory: MemoryStore,
) -> list[str]:
    ids: list[str] = []
    for lesson in REASONING_EXEMPLARS:
        ids.append(
            memory.add(
                content=f"通用策略：{lesson}",
                scope="global",
                kind="strategy",
                status="supported",
                tags=("strategy", "reasoning", "exemplar", "successful"),
                source="codex_reasoning_exemplar",
                evidence_kind="curated_exemplar",
            )
        )
    return ids
