from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
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


@dataclass
class _FailureStrategyRecord:
    key: str
    label: str
    count: int
    examples: list[str] = field(default_factory=list)


_RUNTIME_COMPATIBILITY_PATTERN = re.compile(
    r"(?i)(?:"
    r"unexpected token ['\"]?\?\?=|"
    r"url\.canparse is not a function|"
    r"requires (?:node(?:\.js)?|python|java|ruby|go)\s+\d+(?:\.\d+)*"
    r"\s+or (?:higher|newer)|"
    r"cannot find module ['\"][^'\"]*(?:playwright|node_modules)"
    r")"
)

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
        self.granted_model_call_extensions = 0
        self.budget_extension_requests = 0
        self.reviewing_budget = False
        self.tool_calls = 0
        self.total_tokens = 0
        self.no_progress_rounds = 0
        self.round_has_progress = False
        self.steps_without_state_change = 0
        self.last_cognitive_checkpoint_tool = 0
        self.cognitive_checkpoint_count = 0
        self.consecutive_failures = 0
        self.recovery_count = 0
        self.evidence_seen: set[str] = set()
        self.errors: dict[str, _ErrorRecord] = {}
        self.error_counts_at_recovery: dict[str, int] = {}
        self.failure_strategies: dict[str, _FailureStrategyRecord] = {}
        self.strategy_counts_at_recovery: dict[str, int] = {}
        self.blocked_tool_signatures: set[str] = set()
        self.blocked_tools: dict[str, str] = {}
        self.failed_tool_signatures: set[str] = set()
        self.failed_tool_labels: dict[str, str] = {}
        self.recovery_blocked_signatures: set[str] = set()
        self.duplicate_evidence_signatures: set[str] = set()
        self.search_calls = 0
        self.search_checkpoint_sent = False
        self._warned: set[str] = set()
        self._started = False
        self.finalizing = False
        self.last_block_outcome = ""

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
            "base_model_calls": self.base_model_limit,
            "effective_model_limit": self.effective_model_limit,
            "granted_model_call_extensions": self.granted_model_call_extensions,
            "budget_extension_requests": self.budget_extension_requests,
            "reviewing_budget": self.reviewing_budget,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "total_tokens": self.total_tokens,
            "no_progress_rounds": self.no_progress_rounds,
            "steps_without_state_change": self.steps_without_state_change,
            "cognitive_checkpoint_count": self.cognitive_checkpoint_count,
            "consecutive_failures": self.consecutive_failures,
            "recovery_count": self.recovery_count,
            "blocked_tools": len(self.blocked_tool_signatures),
            "duplicate_evidence_actions": len(
                self.duplicate_evidence_signatures
            ),
            "search_calls": self.search_calls,
            "error_fingerprints": len(self.errors),
            "failure_strategies": len(self.failure_strategies),
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
    def base_model_limit(self) -> int:
        return min(
            max(1, self.budget.base_model_calls),
            max(1, self.budget.max_model_calls),
        )

    @property
    def effective_model_limit(self) -> int:
        return min(
            max(1, self.budget.max_model_calls - self.finalization_reserve),
            self.base_model_limit + self.granted_model_call_extensions,
        )

    @property
    def tool_model_limit(self) -> int:
        return self.effective_model_limit

    def before_model_call(self) -> str | None:
        self.start()
        self._finalize_previous_round()
        if (
            not self.finalizing
            and not self.reviewing_budget
            and self.model_calls >= self.effective_model_limit
            and self.effective_model_limit
            < max(1, self.budget.max_model_calls - self.finalization_reserve)
        ):
            self.reviewing_budget = True
            self._emit(
                "budget_review_started",
                model_calls=self.model_calls,
                base_model_calls=self.base_model_limit,
                effective_model_limit=self.effective_model_limit,
                hard_model_limit=self.budget.max_model_calls,
                summary="基础模型调用预算已用完，等待显式扩容申请",
            )
            return self._budget_review_prompt()
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
            if (
                recovery is None
                and not self.finalizing
                and self.steps_without_state_change
                >= self.budget.cognitive_checkpoint_after
                and self.tool_calls - self.last_cognitive_checkpoint_tool
                >= self.budget.cognitive_checkpoint_repeat
            ):
                recovery = self._cognitive_checkpoint_prompt()
                self.last_cognitive_checkpoint_tool = self.tool_calls
                self.cognitive_checkpoint_count += 1
                self._emit(
                    "cognitive_checkpoint",
                    count=self.cognitive_checkpoint_count,
                    steps_without_state_change=self.steps_without_state_change,
                    summary="连续动作没有改变目标状态，要求重新评估假设",
                )
        self._emit_warnings()
        self.model_calls += 1
        return recovery

    def _budget_review_prompt(self) -> str:
        return (
            "基础模型调用预算已经用完。现在只能二选一："
            "1. 调用 request_budget_extension，说明尚未完成的具体子目标、"
            "已排除的假设、下一批动作和所需额外模型调用数；"
            "2. 直接调用 submit_result 提交当前已完成、部分完成或阻塞结果。"
            "不得继续调用其他工具，也不得仅因为“可能还需要检查”申请扩容。"
        )

    def request_budget_extension(
        self,
        additional_calls: int,
        *,
        reason: str,
    ) -> tuple[bool, str]:
        self.budget_extension_requests += 1
        limit = max(
            1,
            self.budget.max_model_calls - self.finalization_reserve,
        )
        available = max(0, limit - self.effective_model_limit)
        requested = max(1, min(20, int(additional_calls)))
        granted = min(requested, available)
        if not self.reviewing_budget:
            self._emit(
                "budget_extension_denied",
                requested_calls=requested,
                reason="尚未进入预算复核阶段",
                summary="预算扩容请求被拒绝：尚未进入复核阶段",
            )
            return False, "当前未进入预算复核阶段，不能申请扩容。"
        if not str(reason).strip():
            self._emit(
                "budget_extension_denied",
                requested_calls=requested,
                reason="缺少具体扩容理由",
                summary="预算扩容请求被拒绝：缺少具体理由",
            )
            return False, "扩容申请必须说明尚未完成的子目标和下一批动作。"
        if granted <= 0:
            self._emit(
                "budget_extension_denied",
                requested_calls=requested,
                reason="已经达到硬上限",
                summary="预算扩容请求被拒绝：已经达到硬上限",
            )
            return False, "已经达到模型调用硬上限，请提交当前结果。"
        self.granted_model_call_extensions += granted
        self.reviewing_budget = False
        self._emit(
            "budget_extension_granted",
            requested_calls=requested,
            granted_calls=granted,
            effective_model_limit=self.effective_model_limit,
            hard_model_limit=self.budget.max_model_calls,
            reason=str(reason).strip()[:1000],
            summary=f"预算扩容 +{granted} 次模型调用",
        )
        return True, (
            f"已批准额外 {granted} 次模型调用；当前可用上限 "
            f"{self.effective_model_limit}。优先执行能一次补足证据缺口的批量检查。"
        )

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

    def record_auxiliary_model_call(
        self,
        usage: dict[str, Any] | None,
        *,
        purpose: str,
    ) -> None:
        self.start()
        self.model_calls += 1
        self.record_usage(usage)
        self._emit(
            "auxiliary_model_call",
            purpose=purpose,
            model_calls=self.model_calls,
            usage=usage or {},
            summary=f"辅助模型调用: {purpose}",
        )

    def ensure_within_budget(self) -> None:
        self._check_hard_limits(include_model=False)

    def before_tool(self, name: str, arguments: dict[str, Any]) -> str | None:
        self.last_block_outcome = ""
        self._check_hard_limits(include_model=False)
        if self.finalizing:
            reason = "已进入终态阶段；只能提交结果，不能执行新的工具动作。"
            self.last_block_outcome = "finalization_blocked"
            self._emit(
                "finalization_tool_blocked",
                name=name,
                summary=reason,
            )
            return reason
        if self.reviewing_budget and name not in {
            "request_budget_extension",
            "submit_result",
        }:
            reason = (
                "基础模型调用预算已用完；只能申请一次有明确计划的扩容，"
                "或提交当前结果。"
            )
            self.last_block_outcome = "budget_review_blocked"
            self._emit(
                "budget_review_tool_blocked",
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
        if signature in self.recovery_blocked_signatures:
            reason = (
                "恢复阶段禁止原样重复已失败动作；请基于当前证据选择不同参数、"
                "不同工具、不同证据源或无工具的回答方案。"
            )
            self._emit(
                "recovery_tool_signature_blocked",
                name=name,
                summary=reason,
            )
            return reason
        if signature in self.duplicate_evidence_signatures:
            reason = (
                "相同动作已经返回过完全相同的证据，继续重复不会增加信息。"
                "请改变参数、改变证据来源，或基于已有证据回答。"
            )
            self._emit(
                "repeat_evidence_blocked",
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
        state_changed: bool = False,
        control_outcome: str = "",
    ) -> None:
        if (
            control_outcome == "budget_review_blocked"
            and not ok
        ):
            self._emit(
                "budget_review_noncompliant",
                name=name,
                summary=(
                    "预算复核阶段返回了非白名单工具；不计入失败或无进展，"
                    "下一轮进入终态收口"
                ),
            )
            return
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
        new_evidence = evidence not in self.evidence_seen
        if new_evidence:
            self.evidence_seen.add(evidence)
            self.duplicate_evidence_signatures.discard(signature)
        elif ok and not changed_files and not state_changed:
            self.duplicate_evidence_signatures.add(signature)

        if ok and new_evidence:
            self.round_has_progress = True
            self.no_progress_rounds = 0
            self._emit(
                "progress_marked",
                kind="files" if changed_files else "evidence",
                summary="发现新证据" if not changed_files else "文件状态已变化",
            )

        cognitive_update = name == "update_reasoning_state"
        if changed_files or state_changed or cognitive_update:
            self.steps_without_state_change = 0
        else:
            self.steps_without_state_change += 1

        if ok:
            self.consecutive_failures = 0
            if (changed_files or state_changed) and not cognitive_update:
                self._clear_failure_state(
                    reason=(
                        "文件状态已变化"
                        if changed_files
                        else "目标系统状态已变化"
                    )
                )
            return
        if new_evidence:
            self._emit(
                "diagnostic_evidence",
                name=name,
                summary="失败提供了新证据，但不算解决进展",
            )

        self.consecutive_failures += 1
        error_family, error_detail, error_label = self._error_fingerprint(
            name,
            content,
        )
        error_fingerprint = self._hash(
            f"{name}|{error_family}|{error_detail.lower()}"
        )
        tool_label = self._tool_label(name, arguments)
        self.failed_tool_signatures.add(signature)
        self.failed_tool_labels[signature] = tool_label
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
        strategy_key = self._failure_strategy_key(
            name,
            error_family,
            error_detail,
            content,
        )
        strategy = self.failure_strategies.get(strategy_key)
        if strategy is None:
            strategy = _FailureStrategyRecord(
                key=strategy_key,
                label=self._failure_strategy_label(strategy_key),
                count=0,
            )
            self.failure_strategies[strategy_key] = strategy
        strategy.count += 1
        if error_label not in strategy.examples:
            strategy.examples.append(error_label)
            strategy.examples = strategy.examples[-3:]
        self._emit(
            "tool_error",
            name=name,
            count=record.count,
            strategy=strategy_key,
            strategy_count=strategy.count,
            summary=error_label,
        )
        self._check_hard_limits()

    def _cognitive_checkpoint_prompt(self) -> str:
        return (
            "认知收束检查点："
            f"本轮已有 {self.steps_without_state_change} 个连续动作"
            "没有产生文件或目标状态变化。\n"
            "不要把“工具返回了不同内容”自动当成目标进展。"
            "下一步前在内部重新确认：\n"
            "- 当前子目标、成功判据和证据缺口是什么；\n"
            "- 最近结果支持、否定，还是仍未触及当前假设；\n"
            "- 若仍依赖同一假设，停止继续更换同类参数或查询；\n"
            "- 若假设、子目标或证据缺口确实改变，可用 update_reasoning_state "
            "写回结构化状态；\n"
            "- 选择一种能区分剩余假设的不同证据来源。"
            "如果没有新的可区分检查，应直接回答、提问或如实部分收口。"
        )

    def _clear_failure_state(self, *, reason: str) -> None:
        """A real state change invalidates previous retry thresholds."""
        had_failures = bool(
            self.errors
            or self.failure_strategies
            or self.blocked_tool_signatures
        )
        self.consecutive_failures = 0
        self.errors.clear()
        self.error_counts_at_recovery.clear()
        self.failure_strategies.clear()
        self.strategy_counts_at_recovery.clear()
        self.blocked_tool_signatures.clear()
        self.blocked_tools.clear()
        self.failed_tool_signatures.clear()
        self.failed_tool_labels.clear()
        self.recovery_blocked_signatures.clear()
        self.duplicate_evidence_signatures.clear()
        if had_failures:
            self._emit(
                "failure_state_reset",
                reason=reason,
                summary=f"{reason}，旧失败阈值已重置",
            )

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
        unrecovered_strategies = {
            key: record
            for key, record in self.failure_strategies.items()
            if record.count >= self.budget.same_error_soft
            and record.count > self.strategy_counts_at_recovery.get(key, 0)
        }
        newest_strategy = max(
            unrecovered_strategies.values(),
            key=lambda record: record.count,
            default=None,
        )
        reason = ""
        if self.no_progress_rounds >= self.budget.no_progress_soft:
            reason = f"连续 {self.no_progress_rounds} 轮没有发现新证据"
        elif newest_strategy is not None:
            reason = (
                f"同类失败策略「{newest_strategy.label}」"
                f"已重复 {newest_strategy.count} 次"
            )
        elif newest_error is not None:
            reason = f"同类错误已重复 {newest_error.count} 次"
        if not reason:
            return None
        if self.recovery_count >= self.budget.max_recoveries:
            self._trip(f"恢复次数已达上限：{reason}")
        self.recovery_count += 1
        self.recovery_blocked_signatures.update(self.failed_tool_signatures)
        for fingerprint, record in unrecovered_errors.items():
            self.error_counts_at_recovery[fingerprint] = record.count
        for strategy_key, record in unrecovered_strategies.items():
            self.strategy_counts_at_recovery[strategy_key] = record.count
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
            f"- [{item.family}] {item.tool_label}: "
            f"{item.label[:240]}（{item.count} 次）"
            for item in errors
        ]
        strategies = sorted(
            self.failure_strategies.values(),
            key=lambda item: item.count,
            reverse=True,
        )[:4]
        strategy_lines = [
            f"- {item.label}（{item.count} 次）: "
            + "；".join(example[:160] for example in item.examples)
            for item in strategies
        ]
        blocked = sorted(self.failed_tool_labels.values())[:8]
        attempted = sorted(self.failed_tool_signatures)
        return (
            "运行控制恢复契约："
            f"{reason}。停止原样重复，重新判断目标和证据缺口。\n"
            "最近失败：\n"
            + ("\n".join(error_lines) or "- 无明确错误，但缺少新证据")
            + "\n重复失败策略：\n"
            + ("\n".join(strategy_lines) or "- 无")
            + "\n禁止重复的工具调用指纹：\n"
            + ("\n".join(f"- {value}" for value in blocked) or "- 无")
            + "\n本轮已失败动作的内部签名数量："
            + str(len(attempted))
            + "\n恢复要求：\n"
            "- 由你结合用户目标、已确认证据、当前实际可用工具和记忆召回自行提出下一步；"
            "不得机械套用固定工具替代表。\n"
            "- 新动作必须与原失败动作签名不同，并说明它预期补上什么证据缺口。\n"
            "- 可以换参数、换工具、换证据源、换抽象层级，或判断现有证据足以直接回答。\n"
            "- 如果确实缺少能力，先判断是否有合法的替代路径；没有时直接提交部分或阻塞结果，"
            "不要假装成功。\n"
            "- 只执行一个信息增益最高、风险可接受的下一步；得到新证据后重新评估。"
        )

    def recovery_query_terms(self) -> tuple[str, ...]:
        terms = [
            f"family:{record.family}"
            for record in self.errors.values()
        ]
        terms.extend(
            f"strategy:{record.label}"
            for record in self.failure_strategies.values()
        )
        terms.extend(
            f"failed_action:{record.tool_label}"
            for record in self.errors.values()
        )
        return tuple(dict.fromkeys(terms))[:12]

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
        max_strategy_count = max(
            (record.count for record in self.failure_strategies.values()),
            default=0,
        )
        if max_strategy_count >= self.budget.same_error_hard:
            self._trip(f"同类失败策略重复达到 {max_strategy_count} 次")
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
        if re.search(r"没有明确要求|用户拒绝|授权|permission denied|forbidden", detail):
            family = "authorization"
        elif re.search(r"not found|no such file|路径不存在", detail, re.IGNORECASE):
            family = "path"
        elif re.search(r"timeout|timed out|超时", detail, re.IGNORECASE):
            family = "timeout"
        elif re.search(r"modelerror|model error|模型调用失败", detail, re.IGNORECASE):
            family = "model"
        elif exit_code is not None:
            family = f"exit:{exit_code}"
            if exit_code != 0:
                family = "verification"
        elif exception:
            family = exception.group(1)
        else:
            family = f"{tool_name}:tool-failure"
        return family, detail, label

    @classmethod
    def _failure_strategy_key(
        cls,
        tool_name: str,
        family: str,
        detail: str,
        content: str,
    ) -> str:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            error_code = str(payload.get("error") or "unknown").strip()
            target = str(
                payload.get("browser")
                or payload.get("path")
                or payload.get("target")
                or ""
            ).strip()
            error_detail = str(payload.get("detail") or detail)
            reason = cls._structured_failure_reason(
                error_code,
                error_detail,
            )
            return (
                f"{tool_name}|structured|{target or 'target'}|"
                f"{error_code}|{reason}"
            )
        if _RUNTIME_COMPATIBILITY_PATTERN.search(detail):
            return "runtime-compatibility"
        if family in {"authorization", "path", "timeout", "model"}:
            return f"{tool_name}|{family}"
        return f"{tool_name}|{family}|{cls._hash(detail.lower())}"

    @classmethod
    def _structured_failure_reason(
        cls,
        error_code: str,
        detail: str,
    ) -> str:
        value = str(detail or "")
        if re.search(
            r"javascript.*(?:已关闭|disabled)|allow javascript",
            value,
            re.IGNORECASE,
        ):
            return "javascript-disabled"
        if re.search(
            r"(?:不能获得|invalid index|invalid.*window|window\s+1)",
            value,
            re.IGNORECASE,
        ):
            return "no-window"
        if re.search(
            r"not authorized|not permitted|permission denied|权限",
            value,
            re.IGNORECASE,
        ):
            return "permission"
        if error_code in {
            "network_timeout",
            "network_error",
            "dns_error",
            "tls_error",
            "http_error",
        }:
            return error_code.replace("_", "-")
        if error_code == "browser_not_running":
            return "not-running"
        normalized = cls._normalize_error_line(value).lower()
        return f"detail-{cls._hash(normalized)}"

    @staticmethod
    def _failure_strategy_label(key: str) -> str:
        if key == "runtime-compatibility":
            return "运行时或依赖版本不兼容"
        parts = key.split("|")
        tool_name = parts[0]
        if len(parts) >= 5 and parts[1] == "structured":
            return (
                f"{tool_name} / {parts[2]} / {parts[3]} / {parts[4]}"
            )
        family = parts[1] if len(parts) > 1 else "unknown"
        return f"{tool_name} / {family or 'unknown'}"

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
