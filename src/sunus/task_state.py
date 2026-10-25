from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any

from sunus.scope import RequestScope, is_referential_followup


_CONSTRAINT_HINT = re.compile(
    r"(?:不是|不要|必须|只能|只准|不能|无需|不用|要的是|我的意思|"
    r"rather than|must|only|do not|don't)",
    re.IGNORECASE,
)
_CONTINUATION = re.compile(
    r"^\s*(?:直接|继续|就按|按刚才|照刚才|不是|我是说|我的意思|别|"
    r"那|这个|就这样)",
    re.IGNORECASE,
)


def _constraints(text: str) -> list[str]:
    values = [
        part.strip(" ，,。.;；")
        for part in re.split(r"[。！？!?；;\n]+", text)
        if part.strip()
    ]
    return [
        value[:240]
        for value in values
        if _CONSTRAINT_HINT.search(value)
    ][:6]


def _failure_kind(content: str) -> str:
    value = str(content or "")
    if re.search(r"没有明确要求|用户拒绝|授权|权限", value):
        return "authorization"
    if re.search(r"不存在|not found|no such file", value, re.IGNORECASE):
        return "missing_path"
    if re.search(r"Traceback|Exception|Error", value):
        return "runtime"
    if re.search(r"exit_code=[1-9]|failed|failure", value, re.IGNORECASE):
        return "verification"
    return "unknown"


@dataclass
class TaskState:
    objective: str = ""
    constraints: list[str] = field(default_factory=list)
    clarifications: list[str] = field(default_factory=list)
    required_capabilities: set[str] = field(default_factory=set)
    evidence: list[str] = field(default_factory=list)
    attempted: list[str] = field(default_factory=list)
    blocker: str = ""
    last_error: str = ""
    phase: str = "idle"
    turn_count: int = 0

    def update(self, text: str, scope: RequestScope) -> None:
        value = " ".join(str(text or "").split())
        continuation = bool(
            is_referential_followup(value)
            or _CONTINUATION.search(value)
        )
        if not self.objective or not continuation:
            self.objective = value[:1000]
        elif value and value not in self.clarifications:
            self.clarifications = [*self.clarifications[-5:], value[:500]]
        for constraint in _constraints(value):
            if constraint not in self.constraints:
                self.constraints = [*self.constraints[-5:], constraint]
        self.required_capabilities.update(scope.required_capabilities)
        self.turn_count += 1
        self.phase = "discovering"

    def record_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        content: str,
        changed_files: list[str],
    ) -> None:
        signature = hashlib.sha256(
            (
                name
                + "|"
                + str(sorted(arguments.items()))
            ).encode("utf-8")
        ).hexdigest()[:10]
        label = f"{name}:{signature}"
        self.attempted = [*self.attempted[-11:], label]
        if ok:
            summary = " ".join(str(content or "").split())[:180]
            detail = f"{name}: {summary}" if summary else name
            if changed_files:
                detail += f"；修改 {len(set(changed_files))} 个文件"
            elif name == "run_command":
                detail += "；验证命令"
            if detail not in self.evidence:
                self.evidence = [*self.evidence[-7:], detail]
            self.blocker = ""
            self.last_error = ""
            self.phase = "verifying" if name == "run_command" else "acting"
            return
        self.blocker = _failure_kind(content)
        self.last_error = " ".join(str(content or "").split())[:300]
        self.phase = "blocked" if self.blocker == "authorization" else "recovering"

    def prompt_block(self) -> str:
        if not self.objective:
            return ""
        lines = [
            "持久任务状态（跨轮，不是新指令）：",
            f"- 目标: {self.objective}",
        ]
        if self.constraints:
            lines.append("- 约束: " + "；".join(self.constraints[-4:]))
        if self.clarifications:
            lines.append("- 补充: " + "；".join(self.clarifications[-3:]))
        if self.required_capabilities:
            lines.append(
                "- 所需能力: "
                + ", ".join(sorted(self.required_capabilities))
            )
        if self.evidence:
            lines.append("- 已获证据: " + "；".join(self.evidence[-4:]))
        if self.blocker:
            lines.append(f"- 当前阻塞类型: {self.blocker}")
        if self.last_error:
            lines.append(f"- 最近错误: {self.last_error}")
        lines.append(f"- 阶段: {self.phase}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "constraints": list(self.constraints),
            "clarifications": list(self.clarifications),
            "required_capabilities": sorted(self.required_capabilities),
            "evidence": list(self.evidence),
            "attempted": list(self.attempted),
            "blocker": self.blocker,
            "last_error": self.last_error,
            "phase": self.phase,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "TaskState":
        if not isinstance(value, dict):
            return cls()
        return cls(
            objective=str(value.get("objective") or ""),
            constraints=[str(item) for item in value.get("constraints") or []][
                -6:
            ],
            clarifications=[
                str(item) for item in value.get("clarifications") or []
            ][-6:],
            required_capabilities={
                str(item) for item in value.get("required_capabilities") or []
            },
            evidence=[str(item) for item in value.get("evidence") or []][-8:],
            attempted=[str(item) for item in value.get("attempted") or []][-12:],
            blocker=str(value.get("blocker") or ""),
            last_error=str(value.get("last_error") or ""),
            phase=str(value.get("phase") or "idle"),
            turn_count=int(value.get("turn_count") or 0),
        )
