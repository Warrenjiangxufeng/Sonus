from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


RESULT_KINDS = {"answer", "change_report", "analysis", "document", "blocked"}
RESULT_STATUSES = {"completed", "partial", "blocked"}
CHANGE_ACTIONS = {"created", "modified", "deleted", "renamed"}
VERIFICATION_STATUSES = {"passed", "failed", "not_run"}


@dataclass(frozen=True)
class ChangeItem:
    path: str
    action: str
    purpose: str = ""


@dataclass(frozen=True)
class VerificationItem:
    command: str
    status: str
    summary: str = ""


@dataclass(frozen=True)
class ResultEnvelope:
    kind: str
    status: str
    summary: str
    details: tuple[str, ...] = ()
    changes: tuple[ChangeItem, ...] = ()
    verification: tuple[VerificationItem, ...] = ()
    risks: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    next_action: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ResultEnvelope":
        if not isinstance(value, dict):
            raise ValueError("结果必须是 JSON object")
        allowed = {
            "kind",
            "status",
            "summary",
            "details",
            "changes",
            "verification",
            "risks",
            "artifacts",
            "next_action",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"结果包含未知字段: {', '.join(unknown)}")

        kind = str(value.get("kind") or "").strip()
        status = str(value.get("status") or "").strip()
        summary = str(value.get("summary") or "").strip()
        if kind not in RESULT_KINDS:
            raise ValueError(f"kind 必须是: {', '.join(sorted(RESULT_KINDS))}")
        if status not in RESULT_STATUSES:
            raise ValueError(
                f"status 必须是: {', '.join(sorted(RESULT_STATUSES))}"
            )
        if not summary:
            raise ValueError("summary 不能为空")
        if kind == "blocked" and status != "blocked":
            raise ValueError("kind=blocked 时 status 必须是 blocked")
        if kind == "document" and not value.get("artifacts"):
            raise ValueError("kind=document 时必须提供 artifacts")
        if status == "blocked" and not value.get("next_action"):
            raise ValueError("status=blocked 时必须提供 next_action")

        changes: list[ChangeItem] = []
        for item in value.get("changes") or []:
            if not isinstance(item, dict):
                raise ValueError("changes 的元素必须是 object")
            path = str(item.get("path") or "").strip()
            action = str(item.get("action") or "").strip()
            if not path:
                raise ValueError("change 缺少 path")
            if action == "read":
                continue
            if action not in CHANGE_ACTIONS:
                raise ValueError(
                    f"change 需要 path，action 必须是 "
                    f"{', '.join(sorted(CHANGE_ACTIONS))}"
                )
            changes.append(
                ChangeItem(
                    path=path,
                    action=action,
                    purpose=str(item.get("purpose") or "").strip(),
                )
            )

        verification: list[VerificationItem] = []
        for item in value.get("verification") or []:
            if not isinstance(item, dict):
                raise ValueError("verification 的元素必须是 object")
            command = str(item.get("command") or "").strip()
            verification_status = str(item.get("status") or "").strip()
            if not command or verification_status not in VERIFICATION_STATUSES:
                raise ValueError(
                    "verification 需要 command，status 必须是 "
                    f"{', '.join(sorted(VERIFICATION_STATUSES))}"
                )
            verification.append(
                VerificationItem(
                    command=command,
                    status=verification_status,
                    summary=str(item.get("summary") or "").strip(),
                )
            )

        return cls(
            kind=kind,
            status=status,
            summary=summary[:6000],
            details=_text_items(value.get("details"), 3, 600),
            changes=tuple(changes[:20]),
            verification=tuple(verification[:5]),
            risks=_text_items(value.get("risks"), 2, 400),
            artifacts=_text_items(value.get("artifacts"), 20, 500),
            next_action=str(value.get("next_action") or "").strip()[:1200],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "summary": self.summary,
            "details": list(self.details),
            "changes": [
                {
                    "path": item.path,
                    "action": item.action,
                    "purpose": item.purpose,
                }
                for item in self.changes
            ],
            "verification": [
                {
                    "command": item.command,
                    "status": item.status,
                    "summary": item.summary,
                }
                for item in self.verification
            ],
            "risks": list(self.risks),
            "artifacts": list(self.artifacts),
            "next_action": self.next_action,
        }


def _text_items(value: Any, limit: int, item_limit: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("字段必须是字符串数组")
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text[:item_limit])
        if len(result) >= limit:
            break
    return tuple(result)


def fallback_result(
    text: str,
    *,
    changed_files: list[str],
    reason: str,
) -> ResultEnvelope:
    completed = reason == "completed"
    blocked = reason in {"blocked", "circuit_open", "model_error", "internal_error"}
    kind = "change_report" if changed_files else "blocked" if blocked else "answer"
    return ResultEnvelope(
        kind=kind,
        status="blocked" if blocked else "completed" if completed else "partial",
        summary=(text.strip() or "任务未产生可展示结果。")[:6000],
        changes=tuple(
            ChangeItem(path=path, action="modified", purpose="")
            for path in changed_files[:20]
        ),
        next_action="请提供缺失信息或调整任务后重试。" if blocked else "",
    )


def render_result(envelope: ResultEnvelope) -> str:
    lines = [envelope.summary]
    if envelope.kind in {"answer", "analysis"}:
        if envelope.details:
            lines.append("")
            lines.extend(f"- {item}" for item in envelope.details[:3])
        if envelope.risks:
            lines.extend(["", f"**限制**：{envelope.risks[0]}"])
        return "\n".join(lines).strip()

    if envelope.kind == "blocked":
        if envelope.details:
            lines.append("")
            lines.extend(f"- {item}" for item in envelope.details[:2])
        if envelope.changes:
            lines.extend(["", "**改动**"])
            for item in envelope.changes:
                suffix = f"：{item.purpose}" if item.purpose else ""
                lines.append(f"- `{item.path}` · {item.action}{suffix}")
        if envelope.risks:
            lines.extend(["", f"**限制**：{envelope.risks[0]}"])
        if envelope.next_action:
            lines.extend(["", f"**恢复条件**：{envelope.next_action}"])
        return "\n".join(lines).strip()

    if envelope.details:
        lines.append("")
        lines.extend(f"- {item}" for item in envelope.details[:3])
    if envelope.changes:
        lines.extend(["", "**改动**"])
        for item in envelope.changes:
            suffix = f"：{item.purpose}" if item.purpose else ""
            lines.append(f"- `{item.path}` · {item.action}{suffix}")
    if envelope.verification:
        lines.extend(["", "**验证**"])
        marker = {"passed": "✓", "failed": "✗", "not_run": "○"}
        for item in envelope.verification:
            detail = f" · {item.summary}" if item.summary else ""
            lines.append(
                f"- {marker[item.status]} `{item.command}`{detail}"
            )
    if envelope.risks:
        lines.extend(["", "**风险**"])
        lines.extend(f"- {item}" for item in envelope.risks[:2])
    if envelope.artifacts:
        lines.extend(["", "**产物**"])
        lines.extend(f"- `{item}`" for item in envelope.artifacts)
    if envelope.next_action:
        lines.extend(["", f"**下一步**：{envelope.next_action}"])
    return "\n".join(lines).strip()


def normalize_result(
    envelope: ResultEnvelope,
    *,
    tool_traces: list[dict[str, Any]],
    changed_files: list[str],
) -> ResultEnvelope:
    actual_changes = set(changed_files)
    mutation_items = tuple(
        item
        for item in envelope.changes
        if item.path in actual_changes and item.action in CHANGE_ACTIONS
    )
    verification_by_command: dict[str, bool] = {}
    for trace in tool_traces:
        if trace.get("name") != "run_command":
            continue
        arguments = trace.get("arguments") or {}
        command = str(arguments.get("command") or "").strip()
        if command:
            verification_by_command[command] = bool(trace.get("ok"))
    actual_verification = tuple(
        replace(
            item,
            status=(
                "passed"
                if verification_by_command[item.command]
                else "failed"
            ),
        )
        for item in envelope.verification
        if item.command in verification_by_command
    )

    if envelope.kind in {"answer", "analysis"}:
        return replace(
            envelope,
            details=envelope.details[:3],
            changes=(),
            verification=(),
            risks=envelope.risks[:1],
        )
    if envelope.kind == "blocked":
        return replace(
            envelope,
            details=envelope.details[:2],
            changes=mutation_items,
            verification=(),
            risks=envelope.risks[:1],
        )
    if envelope.kind == "change_report" and not mutation_items:
        return replace(
            envelope,
            kind="answer",
            details=envelope.details[:3],
            changes=(),
            verification=(),
            artifacts=(),
            risks=envelope.risks[:1],
        )
    return replace(
        envelope,
        details=envelope.details[:3],
        changes=mutation_items,
        verification=actual_verification,
        risks=envelope.risks[:2],
    )


def result_tool_schema() -> dict[str, Any]:
    def string_array(description: str, max_items: int) -> dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string"},
            "description": description,
            "maxItems": max_items,
        }

    return {
        "type": "function",
        "function": {
            "name": "submit_result",
            "description": (
                "Submit the final structured result. Use this exactly once when "
                "the task is complete, partial, or blocked. Do not execute side effects."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": sorted(RESULT_KINDS)},
                    "status": {"type": "string", "enum": sorted(RESULT_STATUSES)},
                    "summary": {
                        "type": "string",
                        "description": (
                            "The direct user-facing answer or change summary. "
                            "Do not put internal telemetry or tool audit logs here."
                        ),
                    },
                    "details": string_array(
                        "Optional key evidence, at most 3 concise items.",
                        3,
                    ),
                    "changes": {
                        "type": "array",
                        "description": (
                            "Only real file mutations. Never include reads, "
                            "directory listings, searches, or metrics."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "path": {"type": "string"},
                                "action": {
                                    "type": "string",
                                    "enum": sorted(CHANGE_ACTIONS),
                                },
                                "purpose": {
                                    "type": "string",
                                    "description": "One short reason for the mutation.",
                                },
                            },
                            "required": ["path", "action"],
                        },
                    },
                    "verification": {
                        "type": "array",
                        "description": (
                            "Only verification commands actually executed in this "
                            "turn, and only when verification is relevant."
                        ),
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "command": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": sorted(VERIFICATION_STATUSES),
                                },
                                "summary": {"type": "string"},
                            },
                            "required": ["command", "status"],
                        },
                    },
                    "risks": string_array(
                        "Only confirmed material limitations or risks; omit "
                        "unverified speculation.",
                        2,
                    ),
                    "artifacts": string_array(
                        "Files generated by this task; omit otherwise.",
                        20,
                    ),
                    "next_action": {"type": "string"},
                },
                "required": ["kind", "status", "summary"],
            },
        },
    }
