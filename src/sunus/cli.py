from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import select
import shutil
import sqlite3
import sys
import termios
import threading
import tty
import unicodedata
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

from sunus import __version__
from sunus.agent import Agent, AgentResult
from sunus.config import (
    ConfigError,
    RuntimeConfig,
    config_summary,
    redact_text,
    resolve_config,
    save_user_config,
)
from sunus.environment import discover_environment
from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.memory_import import import_session_memory
from sunus.model import ModelClient, ModelError, create_model_client
from sunus.policy import CAPABILITIES, ApprovalManager, PathGuard
from sunus.proactive import (
    PresenceContext,
    ProactiveScheduler,
    ProactiveService,
    format_interval_demo,
    parse_topic_ratios,
    proactive_topic_schedule,
)
from sunus.storage import (
    ArtifactStore,
    SessionLogger,
    SessionTranscript,
    list_sessions,
    load_session,
    resolve_session_path,
)
from sunus.tools import ToolExecutor, find_ripgrep


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
_IUTF8_FLAG = getattr(termios, "IUTF8", 0x4000)


def color(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def _terminal_inline_markdown(line: str, *, color_enabled: bool) -> str:
    parts = re.split(r"(`[^`\n]+`)", line)
    rendered: list[str] = []
    for part in parts:
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            content = part[1:-1]
            rendered.append(color(content, CYAN, color_enabled))
            continue
        if color_enabled:
            part = re.sub(
                r"\*\*(.+?)\*\*",
                lambda match: color(match.group(1), BOLD, True),
                part,
            )
        else:
            part = re.sub(r"\*\*(.+?)\*\*", r"\1", part)
        rendered.append(part)
    return "".join(rendered)


def render_terminal_markdown(text: str, *, color_enabled: bool) -> str:
    """Render the small Markdown subset used by ResultEnvelope in a terminal."""
    lines: list[str] = []
    in_code = False
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if in_code:
                lines.append(color("  └─", DIM, color_enabled))
                in_code = False
            else:
                in_code = True
                language = stripped[3:].strip()
                label = f"  ┌─ {language}" if language else "  ┌─ code"
                lines.append(color(label, DIM, color_enabled))
            continue
        if in_code:
            lines.append(color(f"  {raw_line}", DIM, color_enabled))
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", raw_line)
        if heading:
            title = _terminal_inline_markdown(
                heading.group(2),
                color_enabled=color_enabled,
            )
            marker = "▌ " if len(heading.group(1)) <= 2 else "• "
            lines.append(color(marker + title, BOLD + CYAN, color_enabled))
            continue
        bullet = re.match(r"^(\s*)[-*]\s+(.*)$", raw_line)
        if bullet:
            body = _terminal_inline_markdown(
                bullet.group(2),
                color_enabled=color_enabled,
            )
            lines.append(f"{bullet.group(1)}• {body}")
            continue
        quote = re.match(r"^\s*>\s?(.*)$", raw_line)
        if quote:
            body = _terminal_inline_markdown(
                quote.group(1),
                color_enabled=color_enabled,
            )
            lines.append(color(f"│ {body}", DIM, color_enabled))
            continue
        lines.append(
            _terminal_inline_markdown(raw_line, color_enabled=color_enabled)
        )
    return "\n".join(lines)


def print_terminal_markdown(
    text: str,
    *,
    stream: TextIO | None = None,
    color_enabled: bool | None = None,
) -> None:
    stream = stream or sys.stdout
    enabled = stream.isatty() if color_enabled is None else color_enabled
    print(render_terminal_markdown(text, color_enabled=enabled), file=stream)


def _enable_utf8_input(descriptor: int) -> bool:
    try:
        attributes = termios.tcgetattr(descriptor)
        if attributes[0] & _IUTF8_FLAG:
            return True
        attributes[0] |= _IUTF8_FLAG
        termios.tcsetattr(descriptor, termios.TCSADRAIN, attributes)
        return True
    except (OSError, ValueError, termios.error):
        return False


@dataclass
class PendingTerminalInput:
    raw: bytearray = field(default_factory=bytearray)
    submitted: deque[str] = field(default_factory=deque)

    def feed(self, value: bytes, *, submit_newline: bool = True) -> None:
        if not value:
            return
        if value in {b"\x7f", b"\x08"}:
            while self.raw and self.raw[-1] & 0xC0 == 0x80:
                self.raw.pop()
            if self.raw:
                self.raw.pop()
            return
        if value in {b"\r", b"\n"}:
            if submit_newline:
                text = self.text().strip()
                if text:
                    self.submitted.append(text)
                self.clear()
            else:
                self.raw.extend(b"\n")
            return
        first = value[0]
        if first >= 0x20 or first >= 0x80:
            self.raw.extend(value)

    def text(self) -> str:
        return self.raw.decode("utf-8", errors="ignore")

    def clear(self) -> None:
        self.raw.clear()

    def take_submitted(self) -> list[str]:
        values = list(self.submitted)
        self.submitted.clear()
        return values


def _wrap_input_lines(
    prompt: str,
    text: str,
    cursor: int,
    width: int,
) -> tuple[list[str], int, int]:
    """Wrap one editable input buffer without dropping its leading text."""
    prompt_width = _display_width(prompt)
    continuation = " " * prompt_width
    hard_width = max(prompt_width + 8, width - 1)
    cursor = max(0, min(cursor, len(text)))
    rows = [prompt]
    row = 0
    column = prompt_width
    cursor_row = 0
    cursor_column = prompt_width

    for index, character in enumerate(text):
        if index == cursor:
            cursor_row, cursor_column = row, column
        if character == "\n":
            rows.append(continuation)
            row += 1
            column = prompt_width
            continue
        character_width = _display_width(character)
        if column + character_width > hard_width and column > prompt_width:
            rows.append(continuation)
            row += 1
            column = prompt_width
        rows[row] += character
        column += character_width
    if cursor >= len(text):
        cursor_row, cursor_column = row, column
    return rows, cursor_row, cursor_column


class TerminalInputDock:
    """Keep editable input visible below the live activity line."""

    def __init__(
        self,
        prompt: str,
        pending: PendingTerminalInput,
        *,
        stream: TextIO | None = None,
        is_tty: bool | None = None,
        color_enabled: bool | None = None,
    ) -> None:
        self.prompt = prompt
        self.pending = pending
        self.stream = stream or sys.stdout
        self.is_tty = self.stream.isatty() if is_tty is None else is_tty
        self.color_enabled = (
            self.stream.isatty() if color_enabled is None else color_enabled
        )
        self._activity = ""
        self._rendered_rows = 0
        self._cursor_row = 0
        self._lock = threading.RLock()

    def _erase(self) -> None:
        if not self.is_tty or self._rendered_rows <= 0:
            return
        up = self._rendered_rows - 1 - self._cursor_row
        self.stream.write("\r")
        if up:
            self.stream.write(f"\x1b[{up}A")
        for index in range(self._rendered_rows):
            self.stream.write("\x1b[2K")
            if index < self._rendered_rows - 1:
                self.stream.write("\n")
        if self._rendered_rows > 1:
            self.stream.write(f"\x1b[{self._rendered_rows - 1}A")
        self.stream.write("\r")
        self.stream.flush()
        self._rendered_rows = 0

    def refresh(self) -> None:
        if not self.is_tty:
            return
        with self._lock:
            self._erase()
            width = shutil.get_terminal_size((80, 24)).columns
            activity_lines = self._activity.splitlines() or []
            input_lines, cursor_row, cursor_column = _wrap_input_lines(
                self.prompt,
                self.pending.text(),
                len(self.pending.text()),
                width,
            )
            lines = [*activity_lines, *input_lines]
            for index, line in enumerate(lines):
                if index:
                    self.stream.write("\n")
                self.stream.write("\x1b[2K")
                self.stream.write(line)
            total_rows = len(lines)
            target_row = len(activity_lines) + cursor_row
            up = total_rows - 1 - target_row
            self.stream.write("\r")
            if up:
                self.stream.write(f"\x1b[{up}A")
            if cursor_column:
                self.stream.write(f"\x1b[{cursor_column}C")
            self.stream.flush()
            self._rendered_rows = total_rows
            self._cursor_row = target_row

    def set_activity(self, text: str) -> None:
        self._activity = str(text or "")
        self.refresh()

    def clear(self) -> None:
        with self._lock:
            self._erase()
            self._activity = ""


class RuntimeEventRenderer:
    _TOOL_GROUPS = {
        "read_file": "读取",
        "list_files": "目录",
        "search_text": "搜索",
        "run_command": "命令",
        "write_file": "写入",
        "write_document": "文档",
        "patch_file": "修改",
        "fetch_url": "网络",
        "list_skills": "技能",
        "read_skill": "技能",
        "submit_result": "结果",
    }

    def __init__(
        self,
        *,
        color_enabled: bool | None = None,
        is_tty: bool | None = None,
        activity_stream: TextIO | None = None,
        input_dock: TerminalInputDock | None = None,
    ) -> None:
        self.color_enabled = (
            sys.stdout.isatty() if color_enabled is None else color_enabled
        )
        self.is_tty = sys.stdout.isatty() if is_tty is None else is_tty
        self.activity_stream = activity_stream or sys.stderr
        self.input_dock = input_dock
        self._line_active = False
        self._analysis_count = 0
        self._tool_counts: dict[str, int] = {}
        self.streamed_text = ""
        self._turn_active = False

    def _status(self, text: str, code: str = "") -> None:
        if self.input_dock is not None:
            self.input_dock.set_activity(
                color(text.strip(), code, self.color_enabled)
            )
            self._line_active = True
            return
        if not self.is_tty:
            return
        self.activity_stream.write(
            "\r\x1b[2K" + color(text, code, self.color_enabled)
        )
        self.activity_stream.flush()
        self._line_active = True

    def _finish(self, text: str, code: str = "") -> None:
        if self.input_dock is not None:
            self.input_dock.set_activity(
                color(text.strip(), code, self.color_enabled)
            )
            self._line_active = False
            return
        if not self.is_tty:
            return
        if self._line_active:
            self.activity_stream.write("\r\x1b[2K")
        print(
            color(text, code, self.color_enabled),
            file=self.activity_stream,
            flush=True,
        )
        self._line_active = False

    def _summary(self) -> str:
        total_tools = sum(self._tool_counts.values())
        parts = [
            f"{self._analysis_count} 次分析",
            f"{total_tools} 个动作",
        ]
        groups = [
            f"{name} {count}"
            for name, count in self._tool_counts.items()
            if count
        ]
        if groups:
            parts.append(" · ".join(groups))
        return "  ✓ 完成 · " + " · ".join(parts)

    def __call__(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type.startswith("task_grant") or event_type.startswith("proactive_"):
            return
        if event_type == "model_request":
            if not self._turn_active:
                self.streamed_text = ""
                self._turn_active = True
            self._analysis_count += 1
            self._status("  ◇ 思考中…（Tab/Esc 打断）", DIM)
            return
        if event_type == "assistant_delta":
            text = str(event.get("text") or "")
            if not text:
                return
            self.streamed_text += text
            return
        if event_type == "assistant_progress":
            summary = str(event.get("summary") or "").strip()
            if summary:
                summary = " ".join(summary.split())
                if len(summary) > 100:
                    summary = summary[:100] + "…"
                self._status(f"  ◇ {summary}", DIM)
            return
        if event_type == "recovery_requested":
            reason = str(event.get("reason") or "需要调整策略")
            self._status(f"  ◇ 切换策略 · {reason}", YELLOW)
            return
        if event_type == "finalization_started":
            self._status("  ◇ 进入终态 · 停止新动作并汇总结论", YELLOW)
            return
        if event_type == "finalization_tool_blocked":
            self._status("  ⚠ 终态阶段已阻止新动作", YELLOW)
            return
        if event_type == "budget_warning":
            dimension = str(event.get("dimension") or "budget")
            ratio = float(event.get("ratio") or 0)
            self._status(f"  ⚠ {dimension} 预算已使用 {ratio:.0%}", YELLOW)
            return
        if event_type == "repeat_tool_blocked":
            self._status("  ⚠ 已阻止重复失败动作", YELLOW)
            return
        if event_type == "tool_call":
            summary = str(event.get("summary") or event.get("name") or "调用工具")
            name = str(event.get("name") or "")
            group = self._TOOL_GROUPS.get(name, name or "工具")
            self._tool_counts[group] = self._tool_counts.get(group, 0) + 1
            self._status(f"  ⚡ {summary}", CYAN)
            return
        if event_type == "tool_result":
            ok = bool(event.get("ok"))
            marker = "✓" if ok else "✗"
            preview = str(event.get("preview") or ("完成" if ok else "失败"))
            self._status(
                f"  {marker} {preview}",
                GREEN if ok else RED,
            )
            return
        if event_type == "turn_end":
            reason = str(event.get("reason") or "completed")
            if reason == "completed":
                self._finish(self._summary(), GREEN)
            else:
                labels = {
                    "max_steps": "达到工具步数上限",
                    "user_interrupt": "已中断",
                    "model_error": "模型调用失败",
                    "output_truncated": "模型正文达到输出上限",
                    "empty_model_response": "模型返回空正文",
                    "internal_error": "内部错误",
                    "circuit_open": "任务已熔断",
                    "partial_result": "部分完成",
                    "blocked": "任务阻塞",
                }
                total_tools = sum(self._tool_counts.values())
                self._finish(
                    f"  ⚠ {labels.get(reason, reason)} · "
                    f"{self._analysis_count} 次分析 · {total_tools} 个动作",
                    YELLOW,
                )
            self._analysis_count = 0
            self._tool_counts.clear()
            self._turn_active = False
            return
        if event_type.endswith("_error"):
            summary = str(event.get("summary") or event_type)
            self._status(f"  ⚠ {summary}", RED)


class ThinkingStateObserver:
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type") or "")
        if event_type.startswith("task_grant") or event_type.startswith("proactive_"):
            return
        if event_type == "model_request":
            self.event.set()
        else:
            self.event.clear()


def session_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"s_{stamp}_{uuid.uuid4().hex[:6]}"


def load_selected_session(
    config: RuntimeConfig,
    session_name: str | None,
    *,
    last: bool,
) -> SessionTranscript:
    _, path = resolve_session_path(
        config.workspace,
        config.data_dir,
        session_name,
        last=last,
    )
    return load_session(path)


def copy_transcript_messages(
    session: SessionLogger,
    transcript: SessionTranscript,
    *,
    forked_from: str,
) -> None:
    for message in transcript.messages:
        session.append(
            "message",
            {
                "turn_id": "fork",
                "message": message,
            },
        )
    session.append(
        "fork",
        {
            "source_session": forked_from,
            "source_path": str(transcript.path),
            "copied_messages": len(transcript.messages),
        },
    )


def update_proactive_ratios(
    config: RuntimeConfig,
    values: list[str],
) -> tuple[RuntimeConfig, Path | None]:
    weights = parse_topic_ratios(values, config.proactive.topic_weights)
    if not values:
        return config, None
    path = save_user_config(
        config.data_dir,
        {
            "proactive": {
                "topic_weights": dict(weights),
            }
        },
    )
    proactive = replace(config.proactive, topic_weights=weights)
    return replace(config, proactive=proactive), path


def proactive_ratios_summary(config: RuntimeConfig) -> dict[str, Any]:
    weights = dict(config.proactive.topic_weights)
    schedule = proactive_topic_schedule(config.proactive.topic_weights)
    counts = {
        topic: schedule.count(topic)
        for topic, _ in config.proactive.topic_weights
    }
    return {
        "weights": weights,
        "per_100_messages": counts,
        "first_10": list(schedule[:10]),
    }


@dataclass
class Components:
    config: RuntimeConfig
    memory: MemoryStore
    events: EventBus
    approvals: ApprovalManager
    tools: ToolExecutor
    session: SessionLogger
    scheduler: ProactiveScheduler
    model: ModelClient | None
    proactive: ProactiveService | None
    agent: Agent | None

    def close(self) -> None:
        try:
            self.scheduler.close()
        finally:
            try:
                self.tools.close()
            finally:
                self.finalize_session_memory()
                self.memory.close()

    def finalize_session_memory(self) -> None:
        """退出前把会话摘要落进记忆库，失败不阻断关闭流程。"""
        agent = self.agent
        if agent is None:
            return
        try:
            agent.finalize_session_memory()
        except Exception:
            pass


def build_components(
    config: RuntimeConfig,
    *,
    session_name: str,
    interactive: bool,
    verbose: bool = False,
    require_model: bool = True,
    renderer: Callable[[dict[str, Any]], None] | None = None,
    observer: Callable[[dict[str, Any]], None] | None = None,
    approval_event: threading.Event | None = None,
    approval_before_prompt: Callable[[], None] | None = None,
    approval_after_prompt: Callable[[], None] | None = None,
    session_path: Path | None = None,
) -> Components:
    memory = MemoryStore(config.data_dir / "memory.db")
    events = EventBus(
        config.data_dir,
        secret=config.api_key,
        verbose=verbose,
        renderer=renderer,
        observer=observer,
    )
    approvals = ApprovalManager(
        auto_approve=config.auto_approve,
        interactive=interactive,
        workspace=config.workspace,
        prompt_event=approval_event,
        before_prompt=approval_before_prompt,
        after_prompt=approval_after_prompt,
    )
    path_guard = PathGuard(config.workspace, allow_outside=config.allow_outside)
    artifacts = ArtifactStore(config.workspace, config.data_dir, config.api_key)
    tools = ToolExecutor(
        workspace=config.workspace,
        path_guard=path_guard,
        approvals=approvals,
        artifacts=artifacts,
        events=events,
        data_dir=config.data_dir,
        secret=config.api_key,
        memory=memory,
        mcp_servers=config.mcp_servers,
        max_parallel_tools=config.max_parallel_tools,
    )
    session = SessionLogger(
        session_name,
        config.workspace,
        config.data_dir,
        config.api_key,
        path=session_path,
    )
    tools.current_session_id = session.session_id
    session.append(
        "meta",
        {
            "workspace": str(config.workspace.resolve()),
            "model": config.model,
            "provider": config.model_settings.provider,
            "session_version": 2,
        },
    )
    scheduler = ProactiveScheduler(
        config.data_dir / "memory.db",
        config.proactive,
        session_name,
    )
    model: ModelClient | None = None
    proactive: ProactiveService | None = None
    agent: Agent | None = None
    if require_model:
        model = create_model_client(config)
        proactive = ProactiveService(
            model=model,
            memory=memory,
            scope=str(config.workspace),
            config=config.proactive,
            persona=config.persona,
        )
        agent = Agent(
            config=config,
            model=model,
            memory=memory,
            tools=tools,
            events=events,
            session=session,
        )
    return Components(
        config=config,
        memory=memory,
        events=events,
        approvals=approvals,
        tools=tools,
        session=session,
        scheduler=scheduler,
        model=model,
        proactive=proactive,
        agent=agent,
    )


def print_banner() -> None:
    tty = sys.stdout.isatty()
    print(color("Sunus", CYAN, tty))
    print("本地记忆增强主动陪伴型 Agent · 她的边界，由你创造！")
    print(
        "A local, memory-enhanced, proactive companion Agent · "
        "Her boundaries are shaped by you."
    )
    print(color("输入 /help 查看命令；直接输入任务即可。", DIM, tty))
    print()


def _remember_proactive_delivery(
    components: Components,
    text: str,
    meta: dict[str, Any],
    *,
    message_id: str,
    trigger: str,
) -> str:
    if components.proactive is None:
        return ""
    try:
        return components.proactive.record_delivery(
            text,
            meta,
            message_id=message_id,
            trigger=trigger,
        )
    except Exception as error:
        components.events.emit(
            "proactive_memory_write_failed",
            message_id=message_id,
            summary=f"主动分享记忆写入失败: {error}",
        )
        return ""


def due_message(components: Components) -> None:
    if (
        not components.config.proactive.enabled
        or components.proactive is None
    ):
        return
    status = components.scheduler.status()
    if status.paused or components.scheduler.seconds_until_due() > 0:
        return
    allowed, idle_seconds = components.scheduler.presence_gate()
    if not allowed:
        delay = components.scheduler.defer(30)
        components.events.emit(
            "proactive_presence_deferred",
            delay_sec=delay,
            idle_seconds=idle_seconds,
            summary="主动发言时机不合适，已延后",
        )
        return
    tty = sys.stdout.isatty()
    try:
        text, meta = components.proactive.generate(
            components.scheduler.recent_message_texts(5),
            sequence_index=components.scheduler.current_topic_index(),
            presence=PresenceContext(idle_seconds=idle_seconds),
        )
        message_id = components.scheduler.record_message(text)
        memory_id = _remember_proactive_delivery(
            components,
            text,
            meta,
            message_id=message_id,
            trigger="scheduled",
        )
        delay = components.scheduler.mark_sent()
        components.events.emit(
            "proactive_message_delivered",
            message_id=message_id,
            memory_id=memory_id,
            delay_sec=delay,
            topic=meta.get("topic"),
            topic_weight=meta.get("topic_weight"),
            generation_mode=meta.get("generation_mode"),
            summary=f"主动消息已送达 · {meta.get('topic_label', '主动陪伴')}",
        )
        print("\n" + color("Sunus 主动消息", GREEN, tty) + "\n" + text + "\n")
    except Exception as error:
        components.scheduler.mark_failed()
        components.events.emit(
            "proactive_message_failed",
            summary=f"主动消息失败: {error}",
        )


def _display_width(text: str) -> int:
    clean = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    width = 0
    for character in clean:
        if unicodedata.combining(character):
            continue
        width += (
            2
            if unicodedata.east_asian_width(character) in {"W", "F"}
            else 1
        )
    return width


def _visible_input_window(
    text: str,
    cursor: int,
    available_width: int,
) -> tuple[str, int]:
    flat = text.replace("\r", "").replace("\n", "↵")
    cursor = max(0, min(cursor, len(flat)))
    room = max(8, available_width - 2)
    start = cursor
    used = 0
    while start > 0:
        width = _display_width(flat[start - 1])
        if used + width > room // 2:
            break
        start -= 1
        used += width
    end = cursor
    while end < len(flat):
        width = _display_width(flat[end])
        if used + width > room:
            break
        end += 1
        used += width
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    visible = prefix + flat[start:end] + suffix
    cursor_column = _display_width(prefix + flat[start:cursor])
    return visible, cursor_column


def _read_escape_sequence(descriptor: int) -> bytes:
    sequence = bytearray(b"\x1b")
    while len(sequence) < 16 and select.select([descriptor], [], [], 0.03)[0]:
        sequence.extend(os.read(descriptor, 1))
        if sequence[-1:].isalpha() or sequence.endswith(b"~"):
            break
    return bytes(sequence)


def _input_with_initial(
    prompt: str,
    initial: str,
    *,
    on_idle: Callable[[], bool] | None = None,
) -> str:
    if not sys.stdin.isatty():
        return initial + input(prompt)
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    buffer = initial
    cursor = len(buffer)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    def redraw() -> None:
        terminal_width = shutil.get_terminal_size((80, 24)).columns
        visible, cursor_column = _visible_input_window(
            buffer,
            cursor,
            terminal_width - _display_width(prompt),
        )
        sys.stdout.write("\r\x1b[2K" + prompt + visible)
        trailing = _display_width(visible) - cursor_column
        if trailing > 0:
            sys.stdout.write(f"\x1b[{trailing}D")
        sys.stdout.flush()

    def delete_word() -> None:
        nonlocal buffer, cursor
        start = cursor
        while start > 0 and buffer[start - 1].isspace():
            start -= 1
        while start > 0 and not buffer[start - 1].isspace():
            start -= 1
        buffer = buffer[:start] + buffer[cursor:]
        cursor = start

    def remove_character() -> None:
        nonlocal buffer, cursor
        if cursor <= 0:
            return
        prefix = _remove_last_grapheme(buffer[:cursor])
        buffer = prefix + buffer[cursor:]
        cursor = len(prefix)

    try:
        _enable_utf8_input(descriptor)
        tty.setcbreak(descriptor, when=termios.TCSADRAIN)
        attributes = termios.tcgetattr(descriptor)
        attributes[3] &= ~termios.ECHO
        termios.tcsetattr(descriptor, termios.TCSADRAIN, attributes)
        sys.stdout.write("\x1b[?2004h")
        redraw()
        in_paste = False
        while True:
            if not select.select([descriptor], [], [], 0.25)[0]:
                if on_idle is not None and on_idle():
                    redraw()
                continue
            raw = os.read(descriptor, 1)
            if raw == b"":
                if buffer:
                    return buffer
                raise EOFError
            if raw == b"\x1b":
                sequence = _read_escape_sequence(descriptor)
                if sequence == b"\x1b[200~":
                    in_paste = True
                elif sequence == b"\x1b[201~":
                    in_paste = False
                    redraw()
                elif not in_paste and sequence == b"\x1b[D":
                    cursor = max(0, cursor - 1)
                    redraw()
                elif not in_paste and sequence == b"\x1b[C":
                    cursor = min(len(buffer), cursor + 1)
                    redraw()
                elif not in_paste and sequence in {
                    b"\x1b[H",
                    b"\x1b[1~",
                    b"\x1b[7~",
                }:
                    cursor = 0
                    redraw()
                elif not in_paste and sequence in {
                    b"\x1b[F",
                    b"\x1b[4~",
                    b"\x1b[8~",
                }:
                    cursor = len(buffer)
                    redraw()
                elif not in_paste and sequence == b"\x1b[3~":
                    if cursor < len(buffer):
                        buffer = buffer[:cursor] + _remove_first_grapheme(
                            buffer[cursor:]
                        )
                        redraw()
                continue
            if raw in {b"\r", b"\n"} and not in_paste:
                sys.stdout.write("\n")
                sys.stdout.flush()
                decoder.reset()
                return buffer.rstrip("\r\n")
            if raw == b"\x03" and not in_paste:
                raise KeyboardInterrupt
            if raw == b"\x04" and not in_paste:
                if not buffer:
                    raise EOFError
                continue
            if raw in {b"\x7f", b"\x08"} and not in_paste:
                remove_character()
                redraw()
                continue
            if raw == b"\x15" and not in_paste:
                buffer = ""
                cursor = 0
                redraw()
                continue
            if raw == b"\x17" and not in_paste:
                delete_word()
                redraw()
                continue
            if raw == b"\x01" and not in_paste:
                cursor = 0
                redraw()
                continue
            if raw == b"\x05" and not in_paste:
                cursor = len(buffer)
                redraw()
                continue
            if raw == b"\x0b" and not in_paste:
                buffer = buffer[:cursor]
                redraw()
                continue
            if raw == b"\x0c" and not in_paste:
                redraw()
                continue
            if in_paste and raw == b"\r":
                raw = b"\n"
            piece = decoder.decode(raw, final=False)
            if piece and (
                (in_paste and piece in {"\n", "\t"})
                or all(character.isprintable() for character in piece)
            ):
                buffer = buffer[:cursor] + piece + buffer[cursor:]
                cursor += len(piece)
                if not in_paste:
                    redraw()
    finally:
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _remove_last_grapheme(text: str) -> str:
    if not text:
        return text
    end = len(text)
    while end > 0 and (
        unicodedata.combining(text[end - 1])
        or text[end - 1] in {"\ufe0e", "\ufe0f"}
        or 0x1F3FB <= ord(text[end - 1]) <= 0x1F3FF
    ):
        end -= 1
    if (
        end >= 2
        and 0x1F1E6 <= ord(text[end - 1]) <= 0x1F1FF
        and 0x1F1E6 <= ord(text[end - 2]) <= 0x1F1FF
    ):
        end -= 2
    elif end > 0:
        end -= 1
    while end > 0 and text[end - 1] == "\u200d":
        end -= 1
        if end > 0:
            end -= 1
    return text[:end]


def _remove_first_grapheme(text: str) -> str:
    if not text:
        return text
    end = 1
    while end < len(text) and (
        unicodedata.combining(text[end])
        or text[end] in {"\ufe0e", "\ufe0f"}
        or 0x1F3FB <= ord(text[end]) <= 0x1F3FF
        or text[end - 1] == "\u200d"
    ):
        end += 1
    while end < len(text) and text[end] == "\u200d":
        end += 2
    return text[end:]


def read_with_proactive(
    components: Components,
    prompt: str,
    *,
    initial: str = "",
) -> str | None:
    if not sys.stdin.isatty():
        try:
            return initial + input(prompt)
        except EOFError:
            return None
    _enable_utf8_input(sys.stdin.fileno())

    def on_idle() -> bool:
        if (
            components.proactive is None
            or not components.config.proactive.enabled
            or components.scheduler.status().paused
            or components.scheduler.seconds_until_due() > 0
        ):
            return False
        due_message(components)
        return True

    try:
        return _input_with_initial(
            prompt,
            initial,
            on_idle=on_idle,
        )
    except EOFError:
        return None


def print_help(config: RuntimeConfig) -> None:
    first_delay = format_interval_demo(
        config.proactive.base_interval_sec,
        config.proactive.backoff_factor,
        count=1,
    )
    print(
        f"""可用命令：
  /help                       显示帮助
  /new                        清空当前对话并重置主动陪伴为 {first_delay}
  /status                     查看模型、工作区、记忆与主动陪伴状态
  /memory                     查看最近记忆摘要
  /proactive status|ratios|pause|resume|reset|trigger
  /approve status             查看按能力划分的授权
  /approve on [write,command,network|all]
  /approve off [write,command,network|all]
  /approve reset [write,command,network|all]
  /tools                      查看可用工具
  /exit                       退出

分析等待中可按 Tab 或单独 Esc 打断，已完成的文件修改不会自动回滚。

命令行模式：
  sunus exec "任务"
  sunus memory recent|search|score|distill|persona
  sunus proactive status|ratios|pause|resume|reset
  sunus doctor [--network]
"""
    )


def handle_slash(command: str, components: Components) -> bool:
    parts = command.strip().split()
    name = parts[0].lower()
    if name in {"/exit", "/quit"}:
        return False
    if name == "/help":
        print_help(components.config)
    elif name == "/new":
        if components.agent:
            components.agent.reset()
        components.scheduler.reset()
        print(
            "已开始新对话，主动陪伴已复位为 "
            + format_interval_demo(
                components.config.proactive.base_interval_sec,
                components.config.proactive.backoff_factor,
                count=1,
            )
            + "。"
        )
    elif name == "/status":
        summary = config_summary(components.config)
        status = components.scheduler.status()
        summary["proactive_state"] = status.__dict__
        summary["approval"] = components.approvals.describe()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif name == "/memory":
        digest = components.memory.digest(scope=str(components.config.workspace))
        print(
            json.dumps(
                {
                    "preferences": digest["preferences"],
                    "recent": digest["recent"],
                    "persona": digest["persona"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif name == "/tools":
        print(
            "\n".join(
                f"{item['function']['name']}: {item['function']['description']}"
                for item in components.tools.schemas()
            )
        )
    elif name in {"/approve", "/permissions"}:
        value = parts[1].lower() if len(parts) > 1 else ""
        capability_values = parts[2:]
        if name == "/permissions" and not value:
            value = "status"
        if value in {"on", "grant"}:
            selected = components.approvals.grant_capabilities(
                capability_values or CAPABILITIES
            )
            if not capability_values:
                components.approvals.session_allow = True
            print(
                "已允许本会话能力："
                + ("、".join(sorted(selected)) or "无")
            )
        elif value in {"off", "revoke"}:
            selected = components.approvals.revoke_capabilities(
                capability_values or CAPABILITIES
            )
            components.approvals.session_allow = False
            print(
                "已拒绝本会话能力："
                + ("、".join(sorted(selected)) or "无")
            )
        elif value in {"reset", "auto"}:
            selected = components.approvals.reset_capabilities(
                capability_values or CAPABILITIES
            )
            if not capability_values:
                components.approvals.session_allow = False
            print(
                "已恢复按请求意图判断："
                + ("、".join(sorted(selected)) or "当前能力")
            )
        else:
            state = components.approvals.permission_state()
            labels = {
                "write": "写文件/修改",
                "command": "本地命令",
                "network": "网络访问",
            }
            print("当前会话授权：")
            for capability in sorted(state):
                print(
                    f"  {labels.get(capability, capability)}: "
                    f"{state[capability]}"
                )
            print(
                "  session_allow: "
                + ("on" if components.approvals.session_allow else "off")
            )
    elif name == "/proactive":
        action = parts[1].lower() if len(parts) > 1 else "status"
        if action == "status":
            print(json.dumps(components.scheduler.status().__dict__, ensure_ascii=False, indent=2))
        elif action == "ratios":
            try:
                updated, path = update_proactive_ratios(
                    components.config,
                    parts[2:],
                )
            except ValueError as error:
                print(str(error))
            else:
                components.config = updated
                components.scheduler.config = updated.proactive
                if components.proactive is not None:
                    components.proactive.config = updated.proactive
                print(
                    json.dumps(
                        proactive_ratios_summary(updated),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                if path is not None:
                    components.events.emit(
                        "proactive_ratios_updated",
                        weights=dict(updated.proactive.topic_weights),
                        summary="主动消息比例已更新",
                    )
                    print(f"已保存到 {path}")
        elif action == "pause":
            components.scheduler.pause()
            print("主动陪伴已暂停。")
        elif action == "resume":
            components.scheduler.resume()
            print(
                "主动陪伴已恢复，"
                + format_interval_demo(
                    components.config.proactive.base_interval_sec,
                    components.config.proactive.backoff_factor,
                    count=1,
                )
                + "后开始。"
            )
        elif action == "reset":
            components.scheduler.reset()
            print(
                "主动陪伴已复位为 "
                + format_interval_demo(
                    components.config.proactive.base_interval_sec,
                    components.config.proactive.backoff_factor,
                    count=1,
                )
                + "。"
            )
        elif action == "trigger":
            if components.proactive is None:
                print("模型未初始化。")
            else:
                try:
                    text, meta = components.proactive.generate(
                        sequence_index=components.scheduler.current_topic_index(),
                    )
                except Exception as error:
                    components.scheduler.mark_failed()
                    components.events.emit(
                        "proactive_message_failed",
                        trigger="slash_trigger",
                        summary=f"主动消息生成失败，本次未发送: {error}",
                    )
                    print("主动消息生成失败，本次未发送。")
                    return True
                message_id = components.scheduler.record_message(text)
                memory_id = _remember_proactive_delivery(
                    components,
                    text,
                    meta,
                    message_id=message_id,
                    trigger="slash_trigger",
                )
                components.events.emit(
                    "proactive_message_generated",
                    message_id=message_id,
                    memory_id=memory_id,
                    summary="手动触发主动消息",
                    **meta,
                )
                print(text)
        else:
            print(
                "用法：/proactive status|ratios|pause|resume|reset|trigger"
            )
    elif name == "/pwd":
        print(components.config.workspace)
    else:
        print(f"未知命令：{name}。输入 /help 查看命令。")
    return True


def _thinking_interrupt_pressed(descriptor: int) -> bool:
    action, _ = _read_thinking_key(descriptor)
    return action == "interrupt"


def _read_thinking_key(descriptor: int) -> tuple[str, bytes]:
    try:
        character = os.read(descriptor, 1)
    except OSError:
        return "ignore", b""
    if character == b"\t":
        return "interrupt", character
    if character != b"\x1b":
        return "text", character
    sequence = _read_escape_sequence(descriptor)
    if sequence == b"\x1b":
        return "interrupt", character
    return "ignore", b""


def _read_active_turn_key(descriptor: int) -> tuple[str, bytes]:
    """Read a task-time keypress while preserving paste and input events."""
    try:
        character = os.read(descriptor, 1)
    except OSError:
        return "ignore", b""
    if character == b"\t":
        return "interrupt", character
    if character != b"\x1b":
        return "text", character
    sequence = _read_escape_sequence(descriptor)
    if sequence == b"\x1b":
        return "interrupt", character
    if sequence == b"\x1b[200~":
        return "paste_start", sequence
    if sequence == b"\x1b[201~":
        return "paste_end", sequence
    return "ignore", b""


def run_turn_with_interrupt(
    components: Components,
    user_input: str,
    thinking_event: threading.Event,
    pending_input: PendingTerminalInput | None = None,
    input_dock: TerminalInputDock | None = None,
    approval_event: threading.Event | None = None,
) -> tuple[AgentResult, list[str]]:
    agent = components.agent
    if agent is None:
        raise RuntimeError("模型未初始化")
    if not sys.stdin.isatty():
        return agent.run_turn(user_input), []

    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["result"] = agent.run_turn(user_input)
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=worker, name="sunus-turn", daemon=True)
    thread.start()
    descriptor = sys.stdin.fileno()
    _enable_utf8_input(descriptor)
    previous: list[Any] | None = termios.tcgetattr(descriptor)
    cancelled = False
    in_paste = False
    try:
        tty.setcbreak(descriptor, when=termios.TCSADRAIN)
        attributes = termios.tcgetattr(descriptor)
        attributes[3] &= ~termios.ECHO
        termios.tcsetattr(descriptor, termios.TCSADRAIN, attributes)
        sys.stdout.write("\x1b[?2004h")
        sys.stdout.flush()
        while thread.is_alive():
            if approval_event is not None and approval_event.is_set():
                thread.join(0.05)
                continue
            if select.select([descriptor], [], [], 0.05)[0]:
                action, character = _read_active_turn_key(descriptor)
                if action == "interrupt" and thinking_event.is_set() and not cancelled:
                    cancelled = True
                    cancel = getattr(components.model, "cancel", None)
                    if callable(cancel):
                        cancel()
                elif action == "paste_start":
                    in_paste = True
                elif action == "paste_end":
                    in_paste = False
                    if input_dock is not None:
                        input_dock.refresh()
                elif action == "text" and pending_input is not None:
                    if in_paste and character == b"\r":
                        character = b"\n"
                    submit = character in {b"\r", b"\n"} and not in_paste
                    pending_input.feed(
                        character,
                        submit_newline=submit,
                    )
                    if input_dock is not None:
                        if submit:
                            input_dock.set_activity(
                                color(
                                    "  ⏎ 已排队 · 当前任务结束后发送",
                                    DIM,
                                    input_dock.color_enabled,
                                )
                            )
                        else:
                            input_dock.refresh()
            thread.join(0.05)
    except KeyboardInterrupt:
        cancel = getattr(components.model, "cancel", None)
        if callable(cancel):
            cancel()
        raise
    finally:
        if previous is not None:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
        if input_dock is not None:
            input_dock.clear()
        thread.join()

    if "error" in outcome:
        raise outcome["error"]
    result = outcome.get("result")
    if not isinstance(result, AgentResult):
        raise RuntimeError("Agent 回合未返回结果")
    submitted = pending_input.take_submitted() if pending_input is not None else []
    return result, submitted


def residual_text(result_text: str, streamed_text: str) -> str:
    """返回仍需补打到终端的正文片段，避免同一段话显示两遍。

    流式阶段已经把 streamed_text 逐字写到终端了，终态文本只补差量：
    - 终态文本以流式文本开头：只补尾部（render_result 追加 details/risks 的情形）
    - 终态文本已被流式文本包含（含完全相等）：不补
    - 其余情况（终态文本确实是新内容）：整段补打

    只做前缀与子串判断，不做相似度估算：措辞被改写过时判不准，
    而误判的代价（吞掉真实结论）比多打一遍更高。
    """
    if not streamed_text:
        return result_text
    if result_text.startswith(streamed_text):
        return result_text[len(streamed_text) :]
    if result_text and result_text in streamed_text:
        return ""
    return result_text


def run_repl(
    config: RuntimeConfig,
    *,
    verbose: bool,
    transcript: SessionTranscript | None = None,
    forked_from: str | None = None,
) -> int:
    identifier = transcript.session_id if transcript and forked_from is None else session_id()
    thinking_event = threading.Event()
    approval_event = threading.Event()
    pending_input = PendingTerminalInput()
    queued_inputs: deque[str] = deque()
    input_dock = (
        TerminalInputDock(
            color("你 › ", CYAN, sys.stdout.isatty()),
            pending_input,
        )
        if not verbose and sys.stdin.isatty()
        else None
    )
    renderer = (
        RuntimeEventRenderer(input_dock=input_dock)
        if not verbose
        else None
    )

    components = build_components(
        config,
        session_name=identifier,
        interactive=True,
        verbose=verbose,
        renderer=renderer,
        observer=ThinkingStateObserver(thinking_event),
        approval_event=approval_event,
        approval_before_prompt=input_dock.clear if input_dock is not None else None,
        approval_after_prompt=input_dock.refresh if input_dock is not None else None,
        session_path=transcript.path if transcript and forked_from is None else None,
    )
    try:
        if transcript is not None:
            if components.agent is None:
                print("模型未初始化。")
                return 2
            components.agent.restore_messages(
                transcript.messages,
                summary=transcript.summary,
                task_state=transcript.task_state,
            )
            if forked_from is not None:
                copy_transcript_messages(
                    components.session,
                    transcript,
                    forked_from=forked_from,
                )
        print_banner()
        if transcript is not None:
            action = "已分叉" if forked_from else "已恢复"
            print(f"{action}会话 {transcript.session_id}，共 {transcript.turns} 轮。")
            print()
        while True:
            if queued_inputs:
                user_input = queued_inputs.popleft()
            else:
                try:
                    user_input = read_with_proactive(
                        components,
                        color("你 › ", CYAN, sys.stdout.isatty()),
                        initial=pending_input.text(),
                    )
                    pending_input.clear()
                except KeyboardInterrupt:
                    pending_input.clear()
                    print("\n已取消当前输入。")
                    continue
            if user_input is None:
                print("\n再见。")
                break
            if not user_input.strip():
                continue
            components.scheduler.user_activity()
            if user_input.strip().startswith("/"):
                if not handle_slash(user_input, components):
                    print("再见。")
                    break
                continue
            if components.agent is None:
                print("模型未初始化。")
                continue
            print(color("Sunus", GREEN, sys.stdout.isatty()), flush=True)
            try:
                result, submitted = run_turn_with_interrupt(
                    components,
                    user_input,
                    thinking_event,
                    pending_input,
                    input_dock,
                    approval_event,
                )
            except KeyboardInterrupt:
                print("\n操作已中断。")
                continue
            queued_inputs.extend(submitted)
            print_terminal_markdown(result.text)
            if renderer is not None:
                renderer.streamed_text = ""
            if result.reason not in {"completed"}:
                print(
                    color(
                        f"[turn: {result.reason}; receipt: {result.receipt_id}]",
                        YELLOW,
                        sys.stdout.isatty(),
                    )
                )
            print()
    finally:
        components.close()
    return 0


def run_exec(
    config: RuntimeConfig,
    prompt: str,
    *,
    json_mode: bool,
    verbose: bool,
    resume_session: str | None = None,
    fork_session: str | None = None,
) -> int:
    transcript: SessionTranscript | None = None
    forked_from: str | None = None
    identifier = session_id()
    session_path: Path | None = None
    if resume_session and fork_session:
        raise ValueError("--resume 和 --fork 不能同时使用")
    if resume_session:
        transcript = load_selected_session(
            config,
            None if resume_session == "last" else resume_session,
            last=resume_session == "last",
        )
        identifier = transcript.session_id
        session_path = transcript.path
    elif fork_session:
        transcript = load_selected_session(
            config,
            None if fork_session == "last" else fork_session,
            last=fork_session == "last",
        )
        forked_from = transcript.session_id
    components = build_components(
        config,
        session_name=identifier,
        interactive=False,
        verbose=verbose,
        session_path=session_path,
    )
    try:
        if components.agent is None:
            print("模型未初始化。", file=sys.stderr)
            return 2
        if transcript is not None:
            components.agent.restore_messages(
                transcript.messages,
                summary=transcript.summary,
                task_state=transcript.task_state,
            )
            if forked_from is not None:
                copy_transcript_messages(
                    components.session,
                    transcript,
                    forked_from=forked_from,
                )
        result = components.agent.run_turn(prompt)
        if json_mode:
            print(
                json.dumps(
                    {
                        "text": result.text,
                        "reason": result.reason,
                        "receipt_id": result.receipt_id,
                        "tool_calls": result.tool_traces,
                        "usage": result.usage,
                        "changed_files": result.changed_files,
                        "metrics": result.metrics,
                        "structured_result": result.structured_result,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print_terminal_markdown(result.text)
        return 0 if result.reason == "completed" else 1
    finally:
        components.close()


def sessions_command(config: RuntimeConfig, *, limit: int) -> int:
    rows = list_sessions(config.workspace, config.data_dir, limit=limit)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def proactive_command(
    config: RuntimeConfig,
    action: str,
    ratio_values: list[str],
) -> int:
    if action == "ratios":
        try:
            updated, path = update_proactive_ratios(config, ratio_values)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        print(
            json.dumps(
                proactive_ratios_summary(updated),
                ensure_ascii=False,
                indent=2,
            )
        )
        if path is not None:
            print(f"已保存到 {path}")
        return 0

    no_model = action in {"status", "pause", "resume", "reset"}
    components = build_components(
        config,
        session_name="proactive-cli",
        interactive=False,
        require_model=not no_model,
    )
    try:
        if action == "status":
            print(json.dumps(components.scheduler.status().__dict__, ensure_ascii=False, indent=2))
        elif action == "pause":
            components.scheduler.pause()
            print("主动陪伴已暂停。")
        elif action == "resume":
            components.scheduler.resume()
            print(
                "主动陪伴已恢复，"
                + format_interval_demo(
                    config.proactive.base_interval_sec,
                    config.proactive.backoff_factor,
                    count=1,
                )
                + "后开始。"
            )
        elif action == "reset":
            components.scheduler.reset()
            print(
                "主动陪伴已复位为 "
                + format_interval_demo(
                    config.proactive.base_interval_sec,
                    config.proactive.backoff_factor,
                    count=1,
                )
                + "。"
            )
        elif action == "trigger":
            if components.proactive is None:
                raise ModelError("模型未初始化")
            try:
                text, meta = components.proactive.generate(
                    sequence_index=components.scheduler.current_topic_index(),
                )
            except Exception as error:
                components.scheduler.mark_failed()
                components.events.emit(
                    "proactive_message_failed",
                    trigger="cli_trigger",
                    summary=f"主动消息生成失败，本次未发送: {error}",
                )
                print(f"主动消息生成失败，本次未发送：{error}", file=sys.stderr)
                return 1
            message_id = components.scheduler.record_message(text)
            memory_id = _remember_proactive_delivery(
                components,
                text,
                meta,
                message_id=message_id,
                trigger="cli_trigger",
            )
            meta["message_id"] = message_id
            meta["memory_id"] = memory_id
            print(text)
            print(json.dumps(meta, ensure_ascii=False), file=sys.stderr)
        return 0
    finally:
        components.close()


def memory_command(config: RuntimeConfig, args: argparse.Namespace) -> int:
    memory = MemoryStore(config.data_dir / "memory.db")
    scope = str(config.workspace)
    try:
        if args.memory_action == "import-sessions":
            result = import_session_memory(
                memory,
                data_dir=config.data_dir,
                date=args.date,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.memory_action == "recent":
            rows = memory.recent(scope=scope, limit=args.limit)
        elif args.memory_action == "search":
            rows = memory.search(args.query, scope=scope, limit=args.limit)
        elif args.memory_action == "add":
            memory_id = memory.add(
                content=args.content,
                scope="global" if args.global_scope else scope,
                kind=args.kind,
                status=args.status,
                tags=args.tags.split(",") if args.tags else [],
                source="cli",
                evidence_kind="user_statement",
            )
            if args.kind == "persona":
                memory.learn_explicit_persona(
                    args.content,
                    evidence_id=memory_id,
                )
            print(memory_id)
            return 0
        elif args.memory_action == "credit":
            memory.credit(args.id, args.outcome, args.reason or "")
            print("已记录信用反馈。")
            return 0
        elif args.memory_action == "score":
            rows = memory.score()
        elif args.memory_action == "context-metrics":
            metrics = memory.context_metrics(
                scope=scope,
                query=args.query,
                preference_limit=args.preference_limit,
                recall_limit=args.recall_limit,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0
        elif args.memory_action == "distill":
            print(json.dumps(memory.distill(scope=scope), ensure_ascii=False, indent=2))
        elif args.memory_action == "persona":
            distilled = (
                memory.distill_persona()
                if args.persona_distill
                else None
            )
            rows = memory.persona_profile(
                scope=scope,
                limit=50,
                include_candidates=True,
            )
            print(
                json.dumps(
                    {
                        "profile": rows,
                        "distill": distilled,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        elif args.memory_action == "forget":
            if not (config.auto_approve or getattr(args, "forget_yes", False)):
                print("归档记忆需要显式使用 --yes。", file=sys.stderr)
                return 2
            count = memory.forget(args.identifier)
            print(f"已归档 {count} 条记忆，原始内容和审计记录已保留。")
            return 0
        else:
            raise ValueError(f"未知 memory 操作: {args.memory_action}")
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    finally:
        memory.close()


def _latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    files = [path for path in directory.glob(pattern) if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime, default=None)


def doctor(config: RuntimeConfig, *, network: bool) -> int:
    checks: list[dict[str, Any]] = []
    checks.append({"name": "python", "ok": sys.version_info >= (3, 10), "value": sys.version.split()[0]})
    checks.append(
        {
            "name": "workspace",
            "ok": config.workspace.is_dir(),
            "value": str(config.workspace),
        }
    )
    checks.append(
        {
            "name": "data_dir",
            "ok": config.data_dir.is_dir(),
            "value": str(config.data_dir),
        }
    )
    memory_path = config.data_dir / "memory.db"
    memory_status = "missing"
    memory_ok = False
    if memory_path.is_file():
        try:
            connection = sqlite3.connect(
                f"file:{memory_path.as_posix()}?mode=ro",
                uri=True,
            )
            try:
                row = connection.execute("PRAGMA quick_check").fetchone()
                memory_status = str(row[0] if row else "unknown")
            finally:
                connection.close()
            memory_ok = memory_status == "ok"
        except sqlite3.Error as error:
            memory_status = str(error)
    checks.append(
        {
            "name": "memory_db",
            "ok": memory_ok,
            "value": f"{memory_path}: {memory_status}",
        }
    )
    latest_event = _latest_file(config.data_dir / "events", "*.jsonl")
    checks.append(
        {
            "name": "latest_event",
            "ok": latest_event is not None,
            "value": str(latest_event or "missing"),
        }
    )
    latest_session = _latest_file(config.data_dir / "sessions", "s_*.jsonl")
    checks.append(
        {
            "name": "latest_session",
            "ok": latest_session is not None,
            "value": str(latest_session or "missing"),
        }
    )
    legacy_dir = Path.home() / ".sonus"
    checks.append(
        {
            "name": "legacy_sonus",
            "ok": True,
            "value": (
                f"{legacy_dir} (deprecated compatibility data; not canonical)"
                if legacy_dir.exists()
                else "absent"
            ),
        }
    )
    environment = discover_environment()
    path_report = environment["path"]
    checks.append(
        {
            "name": "path_inventory",
            "ok": path_report["command_count"] > 0,
            "value": (
                f"{path_report['command_count']} commands across "
                f"{len(path_report['path_roots'])} PATH roots"
            ),
        }
    )
    for category in (
        "shell",
        "python",
        "test_runner",
        "node",
        "node_package_manager",
        "vcs",
        "search",
        "database",
        "http",
    ):
        entries = environment["categories"].get(category, [])
        checks.append(
            {
                "name": f"available:{category}",
                "ok": bool(entries),
                "value": ", ".join(
                    f"{item.get('command', '')}"
                    for item in entries
                )
                or "not found on PATH",
            }
        )
    checks.append(
        {
            "name": "test_command",
            "ok": bool(environment["recommended_test_command"]),
            "value": environment["recommended_test_command"] or "not found",
        }
    )
    checks.append({"name": "api_key", "ok": bool(config.api_key), "value": "loaded" if config.api_key else "missing"})
    rg_path = find_ripgrep()
    checks.append(
        {
            "name": "ripgrep",
            "ok": bool(rg_path),
            "value": "loaded" if rg_path else "missing (bounded Python fallback available)",
        }
    )
    checks.append(
        {
            "name": "token_metrics",
            "ok": True,
            "value": "estimated only; use `sunus memory context-metrics`",
        }
    )
    try:
        probe = sqlite3.connect(":memory:")
        probe.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
        probe.close()
        checks.append({"name": "sqlite_fts5", "ok": True, "value": "available"})
    except sqlite3.Error as error:
        checks.append({"name": "sqlite_fts5", "ok": False, "value": str(error)})
    if network:
        try:
            models = create_model_client(config).list_models()
            checks.append(
                {
                    "name": f"provider:{config.model_settings.provider}",
                    "ok": True,
                    "value": ",".join(models),
                }
            )
        except Exception as error:
            checks.append(
                {
                    "name": f"provider:{config.model_settings.provider}",
                    "ok": False,
                    "value": redact_text(error, config.api_key),
                }
            )
    print(json.dumps({"checks": checks, "config": config_summary(config)}, ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in checks) else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sunus",
        description="Sunus local coding agent with memory and proactive companionship.",
    )
    parser.add_argument("--version", action="version", version=f"sunus {__version__}")
    parser.add_argument("-C", "--cwd", default=".", help="Workspace directory")
    parser.add_argument("--provider", help="Model provider")
    parser.add_argument(
        "--api-mode",
        choices=["chat_completions", "responses"],
        help="Provider API mode",
    )
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--base-url", help="Override API base URL")
    parser.add_argument(
        "--reasoning-effort",
        help="Provider-specific reasoning effort",
    )
    parser.add_argument(
        "--budget",
        choices=["quick", "standard", "deep"],
        help="Task budget profile",
    )
    parser.add_argument("--yes", action="store_true", help="Auto approve writes and commands")
    parser.add_argument("--allow-outside", action="store_true", help="Allow tools outside workspace")
    parser.add_argument("--no-proactive", action="store_true", help="Disable proactive messages")
    parser.add_argument("--verbose", action="store_true", help="Print runtime events")

    sub = parser.add_subparsers(dest="command")

    exec_parser = sub.add_parser("exec", help="Run one non-interactive task")
    exec_parser.add_argument("prompt", nargs="*", help="Task prompt; reads stdin when omitted")
    exec_parser.add_argument("--json", action="store_true", help="Print structured JSON result")
    exec_parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        metavar="SESSION",
        help="Resume a session before running the task; omit ID for the latest",
    )
    exec_parser.add_argument(
        "--fork",
        nargs="?",
        const="last",
        metavar="SESSION",
        help="Fork a session before running the task; omit ID for the latest",
    )

    sessions_parser = sub.add_parser("sessions", help="List recoverable sessions")
    sessions_parser.add_argument("--limit", type=int, default=20)

    resume_parser = sub.add_parser("resume", help="Resume an interactive session")
    resume_parser.add_argument("session_id", nargs="?")
    resume_parser.add_argument("--last", action="store_true")

    fork_parser = sub.add_parser("fork", help="Fork an interactive session")
    fork_parser.add_argument("session_id", nargs="?")
    fork_parser.add_argument("--last", action="store_true")

    sub.add_parser("models", help="List models available to the configured key")

    doctor_parser = sub.add_parser("doctor", help="Check local runtime and optional API connectivity")
    doctor_parser.add_argument("--network", action="store_true")

    proactive = sub.add_parser("proactive", help="Manage proactive companionship")
    proactive.add_argument(
        "action",
        choices=["status", "ratios", "pause", "resume", "reset", "trigger"],
    )
    proactive.add_argument(
        "ratio_values",
        nargs="*",
        help="Ratio updates, e.g. daily_emotion=45 work_coding=20 ...",
    )

    memory = sub.add_parser("memory", help="Manage Sunus memory")
    memory_sub = memory.add_subparsers(dest="memory_action", required=True)
    import_sessions = memory_sub.add_parser(
        "import-sessions",
        help="Import one local date of session receipts into memory",
    )
    import_sessions.add_argument(
        "--date",
        default=datetime.now().astimezone().strftime("%Y-%m-%d"),
    )
    recent = memory_sub.add_parser("recent")
    recent.add_argument("--limit", type=int, default=20)
    search = memory_sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    add = memory_sub.add_parser("add")
    add.add_argument("content")
    add.add_argument("--kind", default="semantic")
    add.add_argument("--status", default="supported", choices=["candidate", "supported", "verified"])
    add.add_argument("--tags", default="")
    add.add_argument("--global-scope", action="store_true")
    credit = memory_sub.add_parser("credit")
    credit.add_argument("id")
    credit.add_argument("outcome", choices=["positive", "negative", "unknown"])
    credit.add_argument("--reason", default="")
    context_metrics = memory_sub.add_parser(
        "context-metrics",
        help="Measure memory text injected into a query context",
    )
    context_metrics.add_argument("query")
    context_metrics.add_argument("--preference-limit", type=int, default=12)
    context_metrics.add_argument("--recall-limit", type=int, default=5)
    memory_sub.add_parser("score")
    memory_sub.add_parser("distill")
    persona = memory_sub.add_parser(
        "persona",
        help="Inspect Sunus's evolving personality profile",
    )
    persona.add_argument(
        "--distill",
        action="store_true",
        dest="persona_distill",
        help="Keep at most six stable traits",
    )
    forget = memory_sub.add_parser("forget")
    forget.add_argument("identifier")
    forget.add_argument("--yes", action="store_true", dest="forget_yes")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(
            Path(args.cwd),
            provider=args.provider,
            api_mode=args.api_mode,
            model=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
            budget_profile=args.budget,
            auto_approve=args.yes,
            allow_outside=args.allow_outside,
            proactive_enabled=False if args.no_proactive else None,
        )
    except ConfigError as error:
        print(f"配置错误：{error}", file=sys.stderr)
        return 2

    try:
        if args.command == "exec":
            prompt = " ".join(args.prompt).strip()
            if not prompt:
                prompt = sys.stdin.read().strip()
            if not prompt:
                print("缺少任务内容。", file=sys.stderr)
                return 2
            return run_exec(
                config,
                prompt,
                json_mode=args.json,
                verbose=args.verbose,
                resume_session=args.resume,
                fork_session=args.fork,
            )
        if args.command == "sessions":
            return sessions_command(config, limit=args.limit)
        if args.command in {"resume", "fork"}:
            if args.last and args.session_id:
                raise ValueError("--last 与 session_id 不能同时使用")
            transcript = load_selected_session(
                config,
                args.session_id,
                last=args.last or not args.session_id,
            )
            return run_repl(
                config,
                verbose=args.verbose,
                transcript=transcript,
                forked_from=(
                    transcript.session_id if args.command == "fork" else None
                ),
            )
        if args.command == "models":
            print("\n".join(create_model_client(config).list_models()))
            return 0
        if args.command == "doctor":
            return doctor(config, network=args.network)
        if args.command == "proactive":
            return proactive_command(config, args.action, args.ratio_values)
        if args.command == "memory":
            return memory_command(config, args)
        return run_repl(config, verbose=args.verbose)
    except (ConfigError, ModelError) as error:
        print(redact_text(error, config.api_key), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

