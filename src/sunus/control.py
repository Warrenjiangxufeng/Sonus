from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from sunus.config import LoopBudgetConfig
from sunus.events import EventBus


class CircuitOpen(RuntimeError):
    def __init__(self, reason: str, snapshot: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.snapshot = snapshot


@dataclass
class _ErrorRecord:
    label: str
    family: str
    count: int
    tool_signature: str
    tool_label: str


class TurnController:
    def __init__(
        self,
        budget: LoopBudgetConfig,
        *,
        events: EventBus | None = None,
        turn_id: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.budget = budget
        self.events = events
        self.turn_id = turn_id
        self.clock = clock
        self.started_at: float | None = None
        self.model_calls = 0
        self.tool_calls = 0
        self.total_tokens = 0
        self.no_progress_rounds = 0
        self.round_has_progress = False
        self.consecutive_failures = 0
        self.recovery_count = 0
        self.evidence_seen: set[str] = set()
        self.errors: dict[str, _ErrorRecord] = {}
        self.error_counts_at_recovery: dict[str, int] = {}
        self.blocked_tool_signatures: set[str] = set()
        self.blocked_tools: dict[str, str] = {}
        self.search_calls = 0
        self.search_checkpoint_sent = False
        self._warned: set[str] = set()
        self._started = False
        self.finalizing = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.started_at = self.clock()
        self._emit(
            "budget_started",
            profile=self.budget.profile,
            max_model_calls=self.budget.max_model_calls,
            max_tool_calls=self.budget.max_tool_calls,
            max_wall_seconds=self.budget.max_wall_seconds,
            max_total_tokens=self.budget.max_total_tokens,
            finalization_reserve=self.finalization_reserve,
            summary=f"任务预算 {self.budget.profile}",
        )

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.clock() - self.started_at)

    def snapshot(self) -> dict[str, Any]:
        return {
            "profile": self.budget.profile,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "total_tokens": self.total_tokens,
            "no_progress_rounds": self.no_progress_rounds,
            "consecutive_failures": self.consecutive_failures,
            "recovery_count": self.recovery_count,
            "blocked_tools": len(self.blocked_tool_signatures),
            "search_calls": self.search_calls,
            "error_fingerprints": len(self.errors),
            "finalizing": self.finalizing,
            "finalization_reserve": self.finalization_reserve,
        }

    @property
    def finalization_reserve(self) -> int:
        return min(
            max(0, self.budget.finalization_reserve),
            max(0, self.budget.max_model_calls - 1),
        )

    @property
    def tool_model_limit(self) -> int:
        return max(1, self.budget.max_model_calls - self.finalization_reserve)

    def before_model_call(self) -> str | None:
        self.start()
        self._finalize_previous_round()
        if (
            self.finalization_reserve > 0
            and not self.finalizing
            and (
                self.model_calls >= self.tool_model_limit
                or self.total_tokens
                >= self.budget.max_total_tokens * self.budget.warning_ratio
            )
        ):
            self._check_hard_limits(include_model=False)
            self.finalizing = True
            token_limit_reached = (
                self.total_tokens
                >= self.budget.max_total_tokens * self.budget.warning_ratio
            )
            self._emit(
                "finalization_started",
                remaining_model_calls=max(
                    0,
                    self.budget.max_model_calls - self.model_calls,
                ),
                reason=(
                    "Token 预算进入收口区间"
                    if token_limit_reached
                    else "模型轮次进入收口区间"
                ),
                summary="进入终态阶段，停止执行新动作",
            )
            recovery = self._finalization_prompt(
                "Token 预算已使用超过 "
                f"{self.budget.warning_ratio:.0%}"
                if token_limit_reached
                else "模型轮次接近上限"
            )
        else:
            self._check_hard_limits()
            recovery = None if self.finalizing else self._recovery_if_needed()
            if (
                recovery is None
                and not self.finalizing
                and self.search_calls >= 4
                and not self.search_checkpoint_sent
            ):
                self.search_checkpoint_sent = True
                recovery = (
                    "检索收束检查点：本轮已完成至少 4 次全文搜索。"
                    "停止继续更换关键词；先使用已有结果按文件聚合、缩小到具体候选，"
                    "或改用 inspect_runtime_history。若仍无法区分“上次/最近”的对象，"
                    "直接向用户询问一个时间或任务锚点。"
                )
                self._emit(
                    "search_checkpoint",
                    search_calls=self.search_calls,
                    summary="搜索达到 4 次，要求收束",
                )
        self._emit_warnings()
        self.model_calls += 1
        return recovery

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)):
            self.total_tokens += max(0, int(total))
        else:
            for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    self.total_tokens += max(0, int(value))
        self._emit_warnings()

    def ensure_within_budget(self) -> None:
        self._check_hard_limits(include_model=False)

    def before_tool(self, name: str, arguments: dict[str, Any]) -> str | None:
        self._check_hard_limits(include_model=False)
        if self.finalizing:
            reason = "已进入终态阶段；只能提交结果，不能执行新的工具动作。"
            self._emit(
                "finalization_tool_blocked",
                name=name,
                summary=reason,
            )
            return reason
        if name == "search_text":
            self.search_calls += 1
            if self.search_calls > 6:
                reason = (
                    "本轮全文搜索已超过 6 次。停止更换关键词，"
                    "改用文件聚合、局部读取或 inspect_runtime_history；"
                    "无法区分对象时向用户询问时间锚点。"
                )
                self._emit(
                    "search_budget_blocked",
                    search_calls=self.search_calls,
                    summary=reason,
                )
                return reason
        signature = self._tool_signature(name, arguments)
        self.tool_calls += 1
        self._emit_warnings()
        if signature in self.blocked_tool_signatures:
            reason = "运行控制拒绝了重复失败的动作；请改用不同参数或方案。"
            self._emit(
                "repeat_tool_blocked",
                name=name,
                summary=reason,
            )
            return reason
        return None

    def record_tool_result(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        content: str,
        changed_files: list[str],
    ) -> None:
        signature = self._tool_signature(name, arguments)
        normalized = self._normalize(content)
        evidence = self._hash(
            json.dumps(
                [name, arguments, ok, normalized, sorted(changed_files)],
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        )
        if evidence not in self.evidence_seen:
            self.evidence_seen.add(evidence)
            self.round_has_progress = True
            self.no_progress_rounds = 0
            self._emit(
                "progress_marked",
                kind="files" if changed_files else "evidence",
                summary="发现新证据" if not changed_files else "文件状态已变化",
            )

        if ok:
            self.consecutive_failures = 0
            return

        self.consecutive_failures += 1
        error_family, error_detail, error_label = self._error_fingerprint(
            name,
            content,
        )
        error_fingerprint = self._hash(
            f"{name}|{error_family}|{error_detail.lower()}"
        )
        tool_label = self._tool_label(name, arguments)
        record = self.errors.get(error_fingerprint)
        if record is None:
            record = _ErrorRecord(
                label=error_label,
                family=error_family,
                count=0,
                tool_signature=signature,
                tool_label=tool_label,
            )
            self.errors[error_fingerprint] = record
        record.count += 1
        if record.count >= self.budget.same_error_soft:
            self.blocked_tool_signatures.add(signature)
            self.blocked_tools[signature] = tool_label
        self._emit(
            "tool_error",
            name=name,
            count=record.count,
            summary=error_label,
        )
        self._check_hard_limits()

    def _finalize_previous_round(self) -> None:
        if self.model_calls == 0:
            return
        if self.round_has_progress:
            self.no_progress_rounds = 0
        else:
            self.no_progress_rounds += 1
        self.round_has_progress = False

    def _recovery_if_needed(self) -> str | None:
        unrecovered_errors = {
            fingerprint: record
            for fingerprint, record in self.errors.items()
            if record.count >= self.budget.same_error_soft
            and record.count > self.error_counts_at_recovery.get(fingerprint, 0)
        }
        newest_error = max(
            unrecovered_errors.values(),
            key=lambda record: record.count,
            default=None,
        )
        reason = ""
        if self.no_progress_rounds >= self.budget.no_progress_soft:
            reason = f"连续 {self.no_progress_rounds} 轮没有发现新证据"
        elif newest_error is not None:
            reason = f"同类错误已重复 {newest_error.count} 次"
        if not reason:
            return None
        if self.recovery_count >= self.budget.max_recoveries:
            self._trip(f"恢复次数已达上限：{reason}")
        self.recovery_count += 1
        for fingerprint, record in unrecovered_errors.items():
            self.error_counts_at_recovery[fingerprint] = record.count
        prompt = self._recovery_prompt(reason)
        self._emit(
            "recovery_requested",
            attempt=self.recovery_count,
            reason=reason,
            summary=reason,
        )
        return prompt

    def _recovery_prompt(self, reason: str) -> str:
        errors = sorted(
            self.errors.values(),
            key=lambda item: item.count,
            reverse=True,
        )[:4]
        error_lines = [
            f"- {item.tool_label}: {item.label[:240]}（{item.count} 次）"
            for item in errors
        ]
        blocked = sorted(self.blocked_tools.values())[:6]
        return (
            "运行控制恢复检查点："
            f"{reason}。停止重复已经失败的步骤，重新评估假设。\n"
            "最近失败：\n"
            + ("\n".join(error_lines) or "- 无明确错误，但缺少新证据")
            + "\n禁止重复的工具调用指纹：\n"
            + ("\n".join(f"- {value}" for value in blocked) or "- 无")
            + "\n请给出一个新的、最小可验证动作，并明确成功判据。"
            "如果缺少依赖、tokenizer 或工具，不要自行安装；"
            "直接说明能力边界，并用现有证据给出部分结论或阻塞原因。"
        )

    def _finalization_prompt(self, reason: str) -> str:
        remaining = min(
            self.finalization_reserve,
            max(0, self.budget.max_model_calls - self.model_calls),
        )
        return (
            "运行控制已进入终态阶段："
            f"{reason}，最多还剩 {remaining} 次模型调用。"
            "禁止调用 submit_result 之外的任何工具，不得再读取、执行、安装或修改。"
            "立即基于已有证据提交结构化终态；证据不足时必须使用 partial 或 "
            "blocked，明确写出已验证事实、能力缺口和未确认部分。"
        )

    def _check_hard_limits(self, *, include_model: bool = True) -> None:
        if include_model and self.model_calls >= self.budget.max_model_calls:
            self._trip(f"模型调用达到上限 {self.budget.max_model_calls}")
        if self.tool_calls >= self.budget.max_tool_calls:
            self._trip(f"工具调用达到上限 {self.budget.max_tool_calls}")
        if self.elapsed_seconds() >= self.budget.max_wall_seconds:
            self._trip(f"执行时间达到上限 {self.budget.max_wall_seconds} 秒")
        if self.total_tokens >= self.budget.max_total_tokens:
            self._trip(f"Token 达到上限 {self.budget.max_total_tokens}")
        if self.no_progress_rounds >= self.budget.no_progress_hard:
            self._trip(f"连续 {self.no_progress_rounds} 轮无新证据")
        max_error_count = max(
            (record.count for record in self.errors.values()),
            default=0,
        )
        if max_error_count >= self.budget.same_error_hard:
            self._trip(f"同类错误重复达到 {max_error_count} 次")
        if self.consecutive_failures >= self.budget.consecutive_failure_hard:
            self._trip(f"连续失败达到 {self.consecutive_failures} 次")

    def _emit_warnings(self) -> None:
        ratios = {
            "model_calls": self.model_calls / self.budget.max_model_calls,
            "tool_calls": self.tool_calls / self.budget.max_tool_calls,
            "wall_time": self.elapsed_seconds() / self.budget.max_wall_seconds,
            "tokens": self.total_tokens / self.budget.max_total_tokens,
        }
        for name, ratio in ratios.items():
            if ratio >= self.budget.warning_ratio and name not in self._warned:
                self._warned.add(name)
                self._emit(
                    "budget_warning",
                    dimension=name,
                    ratio=round(ratio, 3),
                    summary=f"{name} 已使用 {ratio:.0%}",
                )

    def _trip(self, reason: str) -> None:
        snapshot = self.snapshot()
        self._emit(
            "circuit_open",
            reason=reason,
            snapshot=snapshot,
            summary=reason,
        )
        raise CircuitOpen(reason, snapshot)

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.events is None:
            return
        self.events.emit(
            event_type,
            turn_id=self.turn_id,
            **payload,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        text = re.sub(r"\x1b\[[0-9;]*m", "", str(value))
        text = re.sub(
            r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2})?\b",
            "<timestamp>",
            text,
        )
        text = re.sub(r"(?:/private)?/tmp/[^\s:,'\"]+", "<tmp-path>", text)
        text = re.sub(r"/var/folders/[^\s:,'\"]+", "<tmp-path>", text)
        text = re.sub(
            r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b",
            "<uuid>",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b(?:pid|PID)[=: ]+\d+\b", "pid=<pid>", text)
        text = re.sub(r"\b0x[0-9a-f]+\b", "<address>", text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d+\.\d{3}s\b", "<duration>", text)
        text = re.sub(r"\b\d{10,}\b", "<n>", text)
        return " ".join(text.split())[:4000]

    @classmethod
    def _error_fingerprint(
        cls,
        tool_name: str,
        value: str,
    ) -> tuple[str, str, str]:
        text = re.sub(r"\x1b\[[0-9;]*m", "", str(value))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        exit_match = re.search(r"(?m)^exit_code=(\-?\d+)$", text)
        exit_code = int(exit_match.group(1)) if exit_match else None

        signal_lines: list[str] = []
        for line in lines:
            normalized = cls._normalize_error_line(line)
            if re.search(
                r"(?i)(error|exception|failed|failure|denied|timeout|"
                r"not found|no such file|invalid|unable to)",
                normalized,
            ):
                signal_lines.append(normalized)
        if not signal_lines and lines:
            signal_lines.append(cls._normalize_error_line(lines[-1]))
        if not signal_lines:
            signal_lines.append("unknown tool failure")
        signal_lines = list(dict.fromkeys(signal_lines))
        detail = " | ".join(signal_lines[-3:])[:1200]
        label = signal_lines[-1][:300]

        exception = re.search(
            r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b",
            detail,
        )
        if exit_code is not None:
            family = f"exit:{exit_code}"
        elif exception:
            family = exception.group(1)
        else:
            family = f"{tool_name}:tool-failure"
        return family, detail, label

    @staticmethod
    def _normalize_error_line(value: str) -> str:
        text = re.sub(r"\s+", " ", value.strip())
        text = re.sub(
            r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2})?\b",
            "<timestamp>",
            text,
        )
        text = re.sub(
            r"(?:/private)?/tmp/[^\s:,'\"]+",
            "<tmp-path>",
            text,
        )
        text = re.sub(
            r"/var/folders/[^\s:,'\"]+",
            "<tmp-path>",
            text,
        )
        text = re.sub(
            r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b",
            "<uuid>",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b(?:pid|PID)[=: ]+\d+\b", "pid=<pid>", text)
        text = re.sub(r"\b0x[0-9a-f]+\b", "<address>", text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d+\.\d{3}s\b", "<duration>", text)
        return text[:600]

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _tool_signature(self, name: str, arguments: dict[str, Any]) -> str:
        return self._hash(
            json.dumps(
                [name, arguments],
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        )

    @staticmethod
    def _tool_label(name: str, arguments: dict[str, Any]) -> str:
        raw = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        return f"{name}({raw[:180]})"
