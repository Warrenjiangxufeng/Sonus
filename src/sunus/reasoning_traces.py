from __future__ import annotations

from typing import Any

from sunus.memory import MemoryStore


def _lesson_for(
    *,
    mode: str,
    reason: str,
    blocker: str,
    tool_names: list[str],
) -> str:
    if reason == "completed":
        if "inspect_browser" in tool_names:
            return "现有页面优先用结构化浏览器读取；登录或权限边界出现时立即停止。"
        if "inspect_runtime_history" in tool_names:
            return "历史故障先按时间锚点取候选，再只下钻一个 turn。"
        if "run_command" in tool_names:
            return "先验证环境和真实命令结果，再基于输出决定下一步。"
        return "用最少只读检查拿到证据，证据足够后直接收束。"
    if reason == "circuit_open":
        return f"同类路径在 {mode} 中达到预算或重复失败；下次更早归因并向用户收口。"
    if blocker == "authorization":
        return "权限拒绝是确定性阻塞；第一次出现就申请对应能力或说明缺少的授权。"
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
        f"模式={mode}；目标={objective[:300]}；"
        f"能力={','.join(capabilities) or '无'}；"
        f"首选={','.join(preferred_tools) or '直接回答'}；"
        f"实际动作={'→'.join(tool_names[:8]) or '无'}；"
        f"阻塞={blocker or '无'}；结果={reason}；"
        f"策略={lesson}"
    )
    tags = ["strategy", "reasoning_trace", mode]
    if reason == "completed":
        tags.append("successful")
    elif reason in {"circuit_open", "model_error", "internal_error"}:
        tags.append("failed")
    else:
        tags.append("partial")
    return memory.add(
        content=content,
        scope=scope,
        kind="strategy",
        status="supported" if reason == "completed" else "candidate",
        tags=tuple(tags),
        source="reasoning_trace",
        evidence_kind="strategy_trace",
        receipt_id=receipt_id,
        metadata={
            "reason": reason,
            "blocker": blocker,
            "tool_sequence": tool_names,
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
        "复合目标先映射能力组合，例如读取已打开网页需要浏览器读取能力，"
        "外部抓取才需要网络能力。"
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
