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
    success_criteria: list[str] = field(default_factory=list)
    current_subgoal: str = ""
    working_hypothesis: str = ""
    evidence_gap: str = ""
    evidence_layers: list[str] = field(default_factory=list)
    decision_log: list[str] = field(default_factory=list)
    stalled_steps: int = 0
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
            self.current_subgoal = value[:500]
            self.working_hypothesis = ""
            self.evidence_gap = "尚未取得足以推进当前子目标的证据"
            self.stalled_steps = 0
            if not self.success_criteria:
                self.success_criteria = ["取得与目标一致且可验证的结果"]
        elif value and value not in self.clarifications:
            self.clarifications = [*self.clarifications[-5:], value[:500]]
        for constraint in _constraints(value):
            if constraint not in self.constraints:
                self.constraints = [*self.constraints[-5:], constraint]
        self.required_capabilities.update(scope.required_capabilities)
        self.turn_count += 1
        self.phase = "discovering"

    def update_cognition(
        self,
        *,
        current_subgoal: str = "",
        working_hypothesis: str = "",
        evidence_gap: str = "",
        success_criteria: list[str] | None = None,
        decision: str = "",
    ) -> None:
        if current_subgoal.strip():
            self.current_subgoal = current_subgoal.strip()[:500]
        if working_hypothesis.strip():
            self.working_hypothesis = working_hypothesis.strip()[:800]
        if evidence_gap.strip():
            self.evidence_gap = evidence_gap.strip()[:800]
        if success_criteria:
            values = [
                " ".join(str(item).split())[:300]
                for item in success_criteria
                if str(item).strip()
            ]
            if values:
                self.success_criteria = list(dict.fromkeys(values))[-4:]
        if decision.strip():
            self.decision_log = [
                *self.decision_log[-11:],
                "cognition: " + " ".join(decision.split())[:300],
            ]
        self.stalled_steps = 0
        self.phase = "planning"

    def record_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        content: str,
        changed_files: list[str],
        state_changed: bool = False,
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
        if name not in self.evidence_layers:
            self.evidence_layers = [*self.evidence_layers[-7:], name]
        summary = " ".join(str(content or "").split())[:140]
        outcome = "state_changed" if state_changed else "ok" if ok else "failed"
        self.decision_log = [
            *self.decision_log[-11:],
            f"{name}: {outcome}; {summary}",
        ]
        cognitive_update = name == "update_reasoning_state"
        if cognitive_update:
            self.stalled_steps = 0
        elif changed_files or state_changed:
            self.stalled_steps = 0
            self.evidence_gap = "目标状态已变化，检查是否满足成功判据"
        else:
            self.stalled_steps = min(99, self.stalled_steps + 1)
            self.evidence_gap = "判断新证据是否真正改变当前假设"
        if ok:
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
        if self.success_criteria:
            lines.append("- 成功判据: " + "；".join(self.success_criteria[-3:]))
        if self.current_subgoal:
            lines.append(f"- 当前子目标: {self.current_subgoal}")
        if self.working_hypothesis:
            lines.append(f"- 当前假设: {self.working_hypothesis}")
        if self.evidence_gap:
            lines.append(f"- 证据缺口: {self.evidence_gap}")
        if self.evidence_layers:
            lines.append(
                "- 已尝试证据层: " + "；".join(self.evidence_layers[-5:])
            )
        if self.stalled_steps:
            lines.append(f"- 连续无状态变化步骤: {self.stalled_steps}")
        if self.decision_log:
            lines.append(
                "- 最近决策记录: " + "；".join(self.decision_log[-3:])
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
            "success_criteria": list(self.success_criteria),
            "current_subgoal": self.current_subgoal,
            "working_hypothesis": self.working_hypothesis,
            "evidence_gap": self.evidence_gap,
            "evidence_layers": list(self.evidence_layers),
            "decision_log": list(self.decision_log),
            "stalled_steps": self.stalled_steps,
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
            success_criteria=[
                str(item) for item in value.get("success_criteria") or []
            ][-4:],
            current_subgoal=str(value.get("current_subgoal") or ""),
            working_hypothesis=str(value.get("working_hypothesis") or ""),
            evidence_gap=str(value.get("evidence_gap") or ""),
            evidence_layers=[
                str(item) for item in value.get("evidence_layers") or []
            ][-8:],
            decision_log=[
                str(item) for item in value.get("decision_log") or []
            ][-12:],
            stalled_steps=max(0, int(value.get("stalled_steps") or 0)),
            evidence=[str(item) for item in value.get("evidence") or []][-8:],
            attempted=[str(item) for item in value.get("attempted") or []][-12:],
            blocker=str(value.get("blocker") or ""),
            last_error=str(value.get("last_error") or ""),
            phase=str(value.get("phase") or "idle"),
            turn_count=int(value.get("turn_count") or 0),
        )
