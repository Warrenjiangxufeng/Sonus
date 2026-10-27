from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from sunus.config import (
    PROACTIVE_TOPIC_WEIGHTS,
    PersonaConfig,
    ProactiveConfig,
    redact_text,
)
from sunus.affect import SunusAffectState
from sunus.embodiment import embodiment_prompt
from sunus.memory import MemoryStore, memory_context_text
from sunus.model import ModelClient, ModelError
from sunus.persona_expression import build_persona_expression
from sunus.relationship import relationship_prompt, relationship_stage


Clock = Callable[[], datetime]

REACTION_INTENTS: dict[str, tuple[str, str]] = {
    "curious": ("好奇", "轻轻好奇一下，可以问一句，但不要要求对方立刻回答。"),
    "care": ("关心", "先接住当下的感受或状态，给一句柔软、具体的关心。"),
    "share": ("分享", "说一点自己的小感受、小愿望或刚刚想到的事，不要编造经历。"),
    "playful": ("俏皮", "用一点轻松、自然的小调侃或撒娇，不能影响正事。"),
    "celebrate": ("庆祝", "为对方已经取得的进展轻轻高兴一下，不夸张。"),
}

TOPIC_LABELS = {
    "daily_emotion": "日常与情绪",
    "work_coding": "工作与 coding",
    "self_evolution": "Sunus 性格与成长",
    "user_gender": "Sunus 性别风格融合",
    "user_custom": "用户自定义癖好",
}
TOPIC_ALIASES = {
    "daily_emotion": "daily_emotion",
    "daily": "daily_emotion",
    "日常": "daily_emotion",
    "情绪": "daily_emotion",
    "work_coding": "work_coding",
    "work": "work_coding",
    "coding": "work_coding",
    "工作": "work_coding",
    "self_evolution": "self_evolution",
    "persona": "self_evolution",
    "成长": "self_evolution",
    "性格": "self_evolution",
    "user_gender": "user_gender",
    "gender": "user_gender",
    "性别": "user_gender",
    "user_custom": "user_custom",
    "custom": "user_custom",
    "癖好": "user_custom",
    "兴趣": "user_custom",
}
_WORK_PROJECT_PATTERN = re.compile(
    r"(?:项目|代码|开发|测试|修复|实现|重构|部署|接口|功能|"
    r"任务|todo|bug|coding)",
    re.IGNORECASE,
)
_UNRESOLVED_PATTERN = re.compile(
    r"(?:未完成|尚未完成|还没完成|未解决|没解决|待办|待续|"
    r"待修复|待验证|阻塞|卡在|需要继续|后续继续|进行中|"
    r"\btodo\b|\bpending\b)",
    re.IGNORECASE,
)
_AI_LIKE_PATTERN = re.compile(
    r"(?:作为(?:一个)?(?:AI|人工智能)|"
    r"有什么(?:可以|能)(?:帮|为)你|"
    r"希望(?:对你|能对你)有帮助|"
    r"很高兴(?:为你|为您)(?:服务|解答)|"
    r"根据你的描述|总的来说|综上所述|"
    r"请问(?:还)?需要什么|用户希望|用户当前)"
)


@dataclass(frozen=True)
class PresenceContext:
    idle_seconds: float | None = None
    foreground_app: str = ""
    activity: str = ""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def proactive_topic_schedule(
    weights: tuple[tuple[str, int], ...] = PROACTIVE_TOPIC_WEIGHTS,
) -> tuple[str, ...]:
    """Build an evenly interleaved weighted schedule for proactive topics."""
    normalized: dict[str, int] = {}
    for topic, weight in weights:
        if topic not in TOPIC_LABELS:
            raise ValueError(f"未知主动消息主题: {topic}")
        if weight < 0:
            raise ValueError(f"主动消息主题权重不能为负数: {topic}")
        normalized[topic] = int(weight)
    if sum(normalized.values()) != 100:
        raise ValueError("主动消息主题权重合计必须为 100")

    order = [topic for topic, _ in weights if normalized.get(topic, 0) > 0]
    current = {topic: 0 for topic in order}
    schedule: list[str] = []
    for _ in range(100):
        for topic in order:
            current[topic] += normalized[topic]
        chosen = max(
            order,
            key=lambda topic: (current[topic], -order.index(topic)),
        )
        current[chosen] -= 100
        schedule.append(chosen)
    return tuple(schedule)


def _eligible_topic_weights(
    weights: tuple[tuple[str, int], ...],
    *,
    include_work: bool,
) -> tuple[tuple[str, int], ...]:
    eligible = [
        (topic, weight)
        for topic, weight in weights
        if weight > 0 and (include_work or topic != "work_coding")
    ]
    total = sum(weight for _, weight in eligible)
    if not eligible or total <= 0:
        return tuple(weights)
    normalized: list[list[int | str]] = []
    remainders: list[tuple[int, int]] = []
    assigned = 0
    for index, (topic, weight) in enumerate(eligible):
        product = weight * 100
        weight_100 = product // total
        assigned += weight_100
        normalized.append([topic, weight_100])
        remainders.append((product % total, index))
    for _, index in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : 100 - assigned
    ]:
        normalized[index][1] = int(normalized[index][1]) + 1
    return tuple(
        (str(topic), int(weight))
        for topic, weight in normalized
    )


def _eligible_topic_schedule(
    weights: tuple[tuple[str, int], ...],
    *,
    include_work: bool,
) -> tuple[str, ...]:
    return proactive_topic_schedule(
        _eligible_topic_weights(weights, include_work=include_work)
    )


def parse_topic_ratios(
    values: list[str] | tuple[str, ...],
    current: tuple[tuple[str, int], ...] = PROACTIVE_TOPIC_WEIGHTS,
) -> tuple[tuple[str, int], ...]:
    ratios = dict(current)
    tokens = [
        item.strip()
        for value in values
        for item in str(value).replace("，", ",").split(",")
        if item.strip()
    ]
    for token in tokens:
        match = re.fullmatch(r"\s*([^=:：]+?)\s*[=:：]\s*(\d+)\s*", token)
        if not match:
            raise ValueError(f"无法解析比例: {token}")
        raw_topic, raw_weight = match.groups()
        topic = TOPIC_ALIASES.get(raw_topic.strip().lower())
        if topic is None:
            raise ValueError(f"未知主动消息主题: {raw_topic}")
        ratios[topic] = int(raw_weight)
    if tokens and any(weight < 0 for weight in ratios.values()):
        raise ValueError("主动消息比例不能为负数")
    if sum(ratios.values()) != 100:
        raise ValueError(
            "主动消息比例合计必须为 100，当前为 "
            + str(sum(ratios.values()))
        )
    return tuple((topic, ratios[topic]) for topic, _ in current)


def format_topic_ratios(
    weights: tuple[tuple[str, int], ...] = PROACTIVE_TOPIC_WEIGHTS,
) -> dict[str, int]:
    return {topic: weight for topic, weight in weights}


@dataclass
class ProactiveStatus:
    session_id: str
    paused: bool
    next_due_at: float
    last_sent_at: float | None
    last_user_at: float | None
    unanswered_streak: int
    next_delay_sec: int
    topic_index: int


@dataclass(frozen=True)
class ProactiveTask:
    id: str
    name: str
    instruction: str
    tool: str
    target_path_template: str
    cadence: str
    not_before: str
    status: str
    next_due_at: float
    source_memory_id: str


class ProactiveScheduler:
    def __init__(
        self,
        db_path: Path,
        config: ProactiveConfig,
        session_id: str,
        *,
        clock: Clock = _now,
    ) -> None:
        self.db_path = db_path
        self.config = config
        self.session_id = session_id
        self.clock = clock
        self._connection = sqlite3.connect(str(db_path), timeout=30)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS proactive_state (
                    session_id TEXT PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    next_due_at REAL NOT NULL,
                    last_sent_at REAL,
                    last_user_at REAL,
                    unanswered_streak INTEGER NOT NULL DEFAULT 0,
                    topic_index INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS proactive_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS proactive_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    instruction TEXT NOT NULL,
                    tool TEXT NOT NULL DEFAULT 'write_file',
                    target_path_template TEXT NOT NULL DEFAULT '',
                    cadence TEXT NOT NULL DEFAULT 'once',
                    not_before TEXT NOT NULL DEFAULT '20:00',
                    status TEXT NOT NULL DEFAULT 'active',
                    next_due_at REAL NOT NULL,
                    last_completed_at REAL,
                    source_memory_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(proactive_state)"
                ).fetchall()
            }
            if "topic_index" not in columns:
                self._connection.execute(
                    "ALTER TABLE proactive_state ADD COLUMN "
                    "topic_index INTEGER NOT NULL DEFAULT 0"
                )
        self.start()

    def close(self) -> None:
        self._connection.close()

    def _timestamp(self) -> float:
        return self.clock().timestamp()

    def start(self) -> None:
        now = self._timestamp()
        with self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO proactive_state(
                    session_id, paused, next_due_at, unanswered_streak, updated_at
                ) VALUES (?, 0, ?, 0, ?)
                """,
                (
                    self.session_id,
                    now + self.config.base_interval_sec,
                    datetime.fromtimestamp(now, timezone.utc).isoformat(),
                ),
            )

    def _row(self) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM proactive_state WHERE session_id = ?",
            (self.session_id,),
        ).fetchone()
        if row is None:
            self.start()
            row = self._connection.execute(
                "SELECT * FROM proactive_state WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("无法初始化主动陪伴状态")
        return row

    def status(self) -> ProactiveStatus:
        row = self._row()
        streak = int(row["unanswered_streak"])
        delay = min(
            2**31 - 1,
            self.config.base_interval_sec * self.config.backoff_factor**streak,
        )
        return ProactiveStatus(
            session_id=self.session_id,
            paused=bool(row["paused"]),
            next_due_at=float(row["next_due_at"]),
            last_sent_at=(
                float(row["last_sent_at"]) if row["last_sent_at"] is not None else None
            ),
            last_user_at=(
                float(row["last_user_at"]) if row["last_user_at"] is not None else None
            ),
            unanswered_streak=streak,
            next_delay_sec=delay,
            topic_index=int(row["topic_index"]),
        )

    def current_topic_index(self) -> int:
        return int(self._row()["topic_index"])

    def seconds_until_due(self) -> float:
        status = self.status()
        if status.paused:
            return 3600.0
        return max(0.0, status.next_due_at - self._timestamp())

    def idle_seconds(self) -> float | None:
        status = self.status()
        if status.last_user_at is None:
            return None
        return max(0.0, self._timestamp() - status.last_user_at)

    def presence_gate(self) -> tuple[bool, float | None]:
        idle = self.idle_seconds()
        if idle is None:
            return True, None
        return (
            self.config.min_idle_sec <= idle <= self.config.max_idle_sec,
            idle,
        )

    def defer(self, seconds: int = 30) -> int:
        delay = max(1, int(seconds))
        now = self._timestamp()
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_state
                SET next_due_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    now + delay,
                    datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    self.session_id,
                ),
            )
        return delay

    def user_activity(self) -> None:
        now = self._timestamp()
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_state
                SET last_user_at = ?, unanswered_streak = 0,
                    next_due_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    now,
                    now + self.config.base_interval_sec,
                    datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    self.session_id,
                ),
            )

    def mark_sent(self) -> int:
        now = self._timestamp()
        current = int(self._row()["unanswered_streak"])
        streak = current + 1
        delay = min(
            2**31 - 1,
            self.config.base_interval_sec * self.config.backoff_factor**streak,
        )
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_state
                SET last_sent_at = ?, unanswered_streak = ?,
                    next_due_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    now,
                    streak,
                    now + delay,
                    datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    self.session_id,
                ),
            )
        return delay

    def mark_failed(self) -> None:
        """Retry shortly without treating the failure as an unanswered message."""
        now = self._timestamp()
        retry_seconds = min(60, max(1, self.config.base_interval_sec))
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_state
                SET next_due_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    now + retry_seconds,
                    datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    self.session_id,
                ),
            )

    def pause(self) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE proactive_state SET paused = 1 WHERE session_id = ?",
                (self.session_id,),
            )

    def resume(self) -> None:
        now = self._timestamp()
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_state
                SET paused = 0, unanswered_streak = 0,
                    next_due_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    now + self.config.base_interval_sec,
                    datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    self.session_id,
                ),
            )

    def reset(self) -> None:
        now = self._timestamp()
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_state
                SET unanswered_streak = 0, topic_index = 0,
                    next_due_at = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    now + self.config.base_interval_sec,
                    datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    self.session_id,
                ),
            )

    def record_message(self, text: str) -> str:
        message_id = f"p_{uuid.uuid4().hex[:12]}"
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO proactive_messages(id, session_id, text, fingerprint, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    self.session_id,
                    redact_text(text),
                    fingerprint,
                    self.clock().isoformat(),
                ),
            )
            self._connection.execute(
                """
                UPDATE proactive_state
                SET topic_index = topic_index + 1
                WHERE session_id = ?
                """,
                (self.session_id,),
            )
        return message_id

    def recent_message_texts(self, limit: int = 5) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT text FROM proactive_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (self.session_id, limit),
        ).fetchall()
        return [str(row["text"]) for row in rows]

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> ProactiveTask:
        return ProactiveTask(
            id=str(row["id"]),
            name=str(row["name"]),
            instruction=str(row["instruction"]),
            tool=str(row["tool"]),
            target_path_template=str(row["target_path_template"]),
            cadence=str(row["cadence"]),
            not_before=str(row["not_before"]),
            status=str(row["status"]),
            next_due_at=float(row["next_due_at"]),
            source_memory_id=str(row["source_memory_id"]),
        )

    def _task_target(self, task: ProactiveTask, now: datetime) -> Path:
        rendered = task.target_path_template.replace(
            "{date}",
            now.date().isoformat(),
        )
        target = Path(rendered).expanduser()
        if not target.is_absolute():
            target = self.db_path.parent / target
        return target.resolve(strict=False)

    def create_task(
        self,
        *,
        name: str,
        instruction: str,
        target_path_template: str,
        cadence: str = "once",
        not_before: str = "20:00",
        source_memory_id: str = "",
    ) -> ProactiveTask:
        clean_name = str(name or "").strip()
        clean_instruction = str(instruction or "").strip()
        clean_path = str(target_path_template or "").strip()
        clean_cadence = str(cadence or "once").strip().lower()
        clean_not_before = str(not_before or "20:00").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", clean_name):
            raise ValueError("待办名称只能包含字母、数字、点、横线和下划线")
        if not clean_instruction:
            raise ValueError("待办说明不能为空")
        if not clean_path:
            raise ValueError("待办目标路径不能为空")
        if clean_cadence not in {"once", "daily"}:
            raise ValueError("待办 cadence 必须是 once 或 daily")
        match = re.fullmatch(r"(\d{2}):(\d{2})", clean_not_before)
        if (
            match is None
            or int(match.group(1)) > 23
            or int(match.group(2)) > 59
        ):
            raise ValueError("待办 not_before 必须是 HH:MM")

        now_dt = self.clock().astimezone()
        next_due = now_dt.timestamp()
        if clean_cadence == "daily":
            hour, minute = (
                int(match.group(1)),
                int(match.group(2)),
            )
            eligible_at = datetime.combine(
                now_dt.date(),
                time(hour=hour, minute=minute),
                tzinfo=now_dt.tzinfo,
            )
            if now_dt < eligible_at:
                next_due = eligible_at.timestamp()

        existing = self._connection.execute(
            "SELECT id FROM proactive_tasks WHERE name = ?",
            (clean_name,),
        ).fetchone()
        task_id = (
            str(existing["id"])
            if existing is not None
            else f"pt_{uuid.uuid4().hex[:12]}"
        )
        now_iso = datetime.fromtimestamp(
            now_dt.timestamp(),
            timezone.utc,
        ).isoformat()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO proactive_tasks(
                    id, name, instruction, tool, target_path_template,
                    cadence, not_before, status, next_due_at,
                    source_memory_id, created_at, updated_at
                ) VALUES (?, ?, ?, 'write_file', ?, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    instruction = excluded.instruction,
                    tool = excluded.tool,
                    target_path_template = excluded.target_path_template,
                    cadence = excluded.cadence,
                    not_before = excluded.not_before,
                    status = 'active',
                    next_due_at = excluded.next_due_at,
                    source_memory_id = excluded.source_memory_id,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    clean_name,
                    clean_instruction[:4000],
                    clean_path,
                    clean_cadence,
                    clean_not_before,
                    next_due,
                    str(source_memory_id or "").strip(),
                    now_iso,
                    now_iso,
                ),
            )
        row = self._connection.execute(
            "SELECT * FROM proactive_tasks WHERE name = ?",
            (clean_name,),
        ).fetchone()
        if row is None:
            raise RuntimeError("无法保存主动待办")
        return self._task_from_row(row)

    def list_tasks(self, *, status: str = "") -> list[ProactiveTask]:
        sql = "SELECT * FROM proactive_tasks"
        parameters: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            parameters = (status,)
        sql += " ORDER BY next_due_at, name"
        return [
            self._task_from_row(row)
            for row in self._connection.execute(sql, parameters).fetchall()
        ]

    def due_task(self) -> ProactiveTask | None:
        now_dt = self.clock().astimezone()
        rows = self._connection.execute(
            """
            SELECT * FROM proactive_tasks
            WHERE status = 'active' AND next_due_at <= ?
            ORDER BY next_due_at, name
            """,
            (now_dt.timestamp(),),
        ).fetchall()
        for row in rows:
            task = self._task_from_row(row)
            if task.tool == "write_file":
                target = self._task_target(task, now_dt)
                if target.exists():
                    self.complete_task(task.id)
                    continue
            return task
        return None

    def complete_task(self, task_id: str) -> None:
        row = self._connection.execute(
            "SELECT * FROM proactive_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return
        task = self._task_from_row(row)
        now_dt = self.clock().astimezone()
        status = "completed"
        next_due_at = task.next_due_at
        if task.cadence == "daily":
            hour, minute = (
                int(task.not_before[:2]),
                int(task.not_before[3:]),
            )
            next_due = datetime.combine(
                now_dt.date() + timedelta(days=1),
                time(hour=hour, minute=minute),
                tzinfo=now_dt.tzinfo,
            )
            status = "active"
            next_due_at = next_due.timestamp()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_tasks
                SET status = ?, next_due_at = ?, last_completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    next_due_at,
                    now_dt.timestamp(),
                    now_iso,
                    task_id,
                ),
            )

    def defer_task(self, task_id: str, seconds: int = 60) -> None:
        now = self.clock().timestamp()
        with self._connection:
            self._connection.execute(
                """
                UPDATE proactive_tasks
                SET next_due_at = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (
                    now + max(1, int(seconds)),
                    datetime.now(timezone.utc).isoformat(),
                    task_id,
                ),
            )


class ProactiveService:
    def __init__(
        self,
        *,
        model: ModelClient,
        memory: MemoryStore,
        scope: str,
        config: ProactiveConfig,
        persona: PersonaConfig | None = None,
        random_source: random.Random | None = None,
        action_catalog: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self.model = model
        self.memory = memory
        self.affect = SunusAffectState(
            Path(memory.db_path).with_name("affect_state.json")
        )
        self.scope = scope
        self.config = config
        self.persona = persona or PersonaConfig()
        self.random = random_source or random.Random()
        self.action_catalog = {
            str(item.get("name") or ""): dict(item)
            for item in action_catalog
            if str(item.get("name") or "")
        }
        self._sequence_offset = 0
        self._last_intent = ""

    def generate(
        self,
        recent_messages: list[str] | None = None,
        *,
        sequence_index: int | None = None,
        presence: PresenceContext | None = None,
        pending_task: ProactiveTask | None = None,
    ) -> tuple[str, dict[str, Any]]:
        now = datetime.now().astimezone()
        task_path = (
            self._task_target(pending_task, now)
            if pending_task is not None
            else None
        )
        preferences = self.memory.recent_preferences(
            scope=self.scope,
            window_hours=self.config.recent_memory_window_hours,
            limit=self.config.recent_memory_limit,
        )[: self.config.preference_items_in_context]
        unresolved_projects = self._unresolved_project_evidence()
        custom_quirks = tuple(
            self.config.custom_quirks or self.config.custom_topics
        )
        heat_records = self.memory.recent(scope=self.scope, limit=24)
        recent = heat_records[:8]
        interaction_heat = self._interaction_heat(heat_records)
        expression_profile = build_persona_expression(
            self.persona,
            context="chat",
            interaction_heat=interaction_heat,
            relationship_stage=relationship_stage(
                self.persona.relationship
            ),
        )
        relationship_query = " ".join(
            value
            for value in (
                self.persona.relationship.label,
                self.persona.relationship.sunus_role,
                self.persona.relationship.user_role,
                self.persona.relationship.address,
            )
            if value
        )
        relationship_recall = (
            self.memory.search(
                relationship_query,
                scope=self.scope,
                limit=8,
            )
            if self.persona.relationship.configured
            else []
        )
        schedule = proactive_topic_schedule(self.config.topic_weights)
        if sequence_index is None:
            sequence_index = self._sequence_offset
            self._sequence_offset += 1
        topic_index = max(0, sequence_index)
        schedule_index = topic_index % len(schedule)
        topic = schedule[schedule_index]
        scheduled_topic = topic
        weights = dict(self.config.topic_weights)
        scheduled_topic_weight = weights.get(scheduled_topic, 0)
        eligible_weights = _eligible_topic_weights(
            self.config.topic_weights,
            include_work=bool(unresolved_projects),
        )
        eligible_schedule = proactive_topic_schedule(eligible_weights)
        topic = eligible_schedule[topic_index % len(eligible_schedule)]
        base_topic = topic
        topic_weight = dict(eligible_weights).get(topic, 0)
        if pending_task is not None:
            topic = "daily_emotion"
            base_topic = "daily_emotion"
            topic_weight = 0
        intent_candidates = {
            "daily_emotion": ("care", "curious", "playful"),
            "work_coding": ("care", "curious"),
            "self_evolution": ("share", "curious", "playful"),
            "user_gender": ("care", "playful", "share"),
            "user_custom": ("curious", "share", "care"),
        }[base_topic]
        if pending_task is not None:
            intent_candidates = ("share", "care")
        choices = tuple(
            key
            for key in intent_candidates
            if key != self._last_intent
        ) or intent_candidates
        intent_key = self.random.choice(choices)
        self._last_intent = intent_key
        intent_label, intent_instruction = REACTION_INTENTS[intent_key]
        work_rerouted = (
            schedule[schedule_index] == "work_coding"
            and topic != "work_coding"
        )
        delivery_topic = topic
        recall_queries = {
            "daily_emotion": (
                "最近 心情 情绪 睡眠 休息 吃饭 咖啡 音乐 周末 生活 状态"
            ),
            "user_custom": "癖好 兴趣 爱好 喜欢 收藏 习惯 关注 期待",
        }
        topic_recall = (
            unresolved_projects
            if base_topic == "work_coding"
            else self.memory.search(
                recall_queries[base_topic],
                scope=self.scope,
                limit=6,
            )
            if base_topic in recall_queries
            else []
        )
        persona_memories = (
            self.memory.recent_by_kind(
                scope=self.scope,
                kind="persona",
                limit=6,
            )
            if topic == "self_evolution"
            else []
        )
        self_share_memories = self.memory.recent_by_kind(
            scope=self.scope,
            kind="self_share",
            limit=8,
        )
        persona_profile = self.memory.persona_profile(
            scope=self.scope,
            limit=6,
        )
        learned_gender = (
            []
            if self.config.user_gender
            else self.memory.search(
                "性别 称呼 沟通偏好 先生 女士",
                scope=self.scope,
                limit=4,
            )
        )
        if not self.config.user_gender:
            known_gender_ids = {item["id"] for item in learned_gender}
            for item in self.memory.recent(scope=self.scope, limit=32):
                content = str(item.get("content") or "")
                if (
                    item["id"] not in known_gender_ids
                    and any(
                        marker in content
                        for marker in ("性别", "男", "女", "称呼")
                    )
                ):
                    learned_gender.append(item)
                    known_gender_ids.add(item["id"])
                if len(learned_gender) >= 4:
                    break
        learned_custom = (
            []
            if custom_quirks
            else self.memory.search(
                recall_queries["user_custom"],
                scope=self.scope,
                limit=4,
            )
        )
        preference_text = memory_context_text(preferences, label="最近偏好")
        recent_text = memory_context_text(recent, label="最近交流")
        topic_recall_text = memory_context_text(
            topic_recall,
            label=f"{TOPIC_LABELS[base_topic]}定向召回",
        )
        relationship_recall_text = memory_context_text(
            relationship_recall,
            label="关系相关候选记忆",
        )
        persona_text = memory_context_text(
            persona_memories,
            label="Sunus 性格与互动习惯",
        )
        self_share_text = memory_context_text(
            self_share_memories,
            label="Sunus 最近主动分享",
        )
        persona_profile_text = "\n".join(
            f"- [{item['status']} {float(item['confidence']):.2f}] "
            f"{item['statement']}"
            for item in persona_profile
        )
        gender_evidence_text = memory_context_text(
            learned_gender,
            label="用户性别与称呼证据",
        )
        custom_evidence_text = memory_context_text(
            learned_custom,
            label="用户癖好学习证据",
        )
        history = "\n".join(f"- {item}" for item in (recent_messages or []))
        daypart = (
            "凌晨"
            if now.hour < 6
            else "早晨"
            if now.hour < 11
            else "午间"
            if now.hour < 14
            else "下午"
            if now.hour < 18
            else "傍晚"
            if now.hour < 23
            else "深夜"
        )
        presence_lines: list[str] = []
        if presence is not None:
            if presence.idle_seconds is not None:
                presence_lines.append(
                    f"用户最近一次明确交互距今约 {int(presence.idle_seconds)} 秒。"
                )
            if presence.foreground_app:
                presence_lines.append(
                    f"当前可能在前台使用：{presence.foreground_app[:80]}。"
                )
            if presence.activity:
                presence_lines.append(
                    f"当前活动提示：{presence.activity[:160]}。"
                )
        distribution = "；".join(
            f"{TOPIC_LABELS[name]} {weight}%"
            for name, weight in self.config.topic_weights
        )
        identity_gender = next(
            (
                str(item["statement"])
                for item in persona_profile
                if item.get("trait_key") == "gender_identity"
            ),
            "",
        )
        eligible_distribution = "；".join(
            f"{TOPIC_LABELS[name]} {weight}%"
            for name, weight in eligible_weights
        )
        sunus_gender_context = (
            f"Sunus 对用户设定的自身性别：{self.config.sunus_gender}。"
            "只采用自然的语气和称呼，不使用刻板印象。"
            if self.config.sunus_gender
            else f"记忆中的 Sunus 核心身份：{identity_gender}。"
            if identity_gender
            else "Sunus 自身性别尚未设置；本轮保持中性表达，不自行猜测。"
        )
        user_gender_context = (
            f"用户明确设定的性别：{self.config.user_gender}。"
            "只用于调整沟通方式，不据此推断其他属性。"
            if self.config.user_gender
            else "用户性别未在配置中设置；只使用下方明确记忆证据，"
            "不得从姓名、职业或语气猜测。"
        )
        custom_context = (
            "用户自定义癖好/关注：" + "、".join(custom_quirks) + "。"
            if custom_quirks
            else "用户尚未填写自定义癖好；从明确记忆中自行发现一个真实兴趣，"
            "没有依据时提出低压力的开放问题。"
        )
        action_context = (
            json.dumps(
                [
                    {
                        "name": name,
                        "description": str(
                            item.get("description") or ""
                        )[:240],
                        "parameters": item.get("parameters") or {},
                    }
                    for name, item in self.action_catalog.items()
                ],
                ensure_ascii=False,
            )
            if self.config.action_mode != "off"
            else "[]"
        )
        topic_label = TOPIC_LABELS[delivery_topic]
        topic_instruction = self._topic_instruction(topic, base_topic)
        relationship_context = (
            "用户明确设定的关系信息："
            f"关系={self.persona.relationship.label or '未命名'}；"
            f"Sunus角色={self.persona.relationship.sunus_role or '未指定'}；"
            f"用户角色={self.persona.relationship.user_role or '未指定'}；"
            f"称呼={self.persona.relationship.address or '未指定'}。"
            "请结合真实记忆自行判断亲疏、关注方向和合适话题，"
            "不要套用预设的关系类型模板。"
            if self.persona.relationship.configured
            else (
                "未设置人物关系；按普通朋友的尊重和边界生成，"
                "不得自行补恋爱、亲子或手足关系。"
            )
        )
        work_gate_context = (
            "已确认一条最近的项目记忆仍未完成，只有在需要时才可轻提一次。"
            if unresolved_projects
            else "本轮没有未解决项目的明确证据；禁止发送工作提醒、进度追问或任务建议。"
        )
        task_context = (
            f"- 待办 ID：{pending_task.id}\n"
            f"- 名称：{pending_task.name}\n"
            f"- 指令：{pending_task.instruction}\n"
            f"- 唯一目标文件：{task_path}\n"
            "- 完成判定：该文件尚不存在。"
            if pending_task is not None and task_path is not None
            else "本轮没有到期的主动待办。"
        )
        task_topic_instruction = (
            "优先执行待执行主动待办，不要退回普通闲聊。消息必须明确写出"
            "正在把内容写进目标文件，并返回 write_file 动作。"
            if pending_task is not None
            else topic_instruction
        )
        user_prompt = f"""当前时间：{now:%Y-%m-%d %H:%M %Z}
当前时段：{daypart}
当前工作区：{self.scope}
陪伴情境：{' '.join(presence_lines) if presence_lines else '没有额外情境，按已有记忆自然发言。'}
本轮参考表达动作：{intent_label}。{intent_instruction}
它只用于避免表达单一，不是必须执行的指令；如果关系语境和真实记忆指向更合适的方式，可自行更换。
配置的五类记忆配比：{distribution}
本轮可用记忆类别配比：{eligible_distribution}
本轮计划记忆类别：{TOPIC_LABELS[scheduled_topic]}
（{scheduled_topic_weight}%，序列 {topic_index + 1}）
实际选择记忆类别：{topic_label}
本轮侧重：{task_topic_instruction}
工作消息闸门：{work_gate_context}
待执行主动待办（优先级高于普通五类话题）：
{task_context}
关系语境：
{relationship_context}
{sunus_gender_context}
{user_gender_context}
{custom_context}
可用主动动作（仅限这些受控工具；无必要动作时返回 null）：
{action_context}

{preference_text or "最近记忆中没有明确偏好。"}

{recent_text or "最近没有可引用的交流。"}

{topic_recall_text or "本轮主题没有额外的定向记忆命中。"}

{relationship_recall_text or "没有额外的共同经历候选；不得虚构共同经历。"}

{persona_text or "尚未积累稳定的 Sunus 性格或互动习惯记忆。"}

{persona_profile_text or "还没有达到证据门槛的稳定性格画像。"}

{self_share_text or "还没有已送达的 Sunus 主动分享记忆。"}

{gender_evidence_text or "没有可用于学习用户性别或称呼的明确记忆。"}

{custom_evidence_text or "没有可用于学习用户癖好的明确记忆。"}

最近已经发送过的主动消息：
{history or "（无）"}

请生成下一条主动消息。"""
        persona_name = self.persona.name or "Sunus"
        examples_text = "\n".join(
            f"- {example}" for example in self.persona.examples
        )
        expression_rules = expression_profile.prompt()
        persona_rules = (
            "\n人物设定（用于语气、性格和表达习惯）：\n"
            + self.persona.prompt
            + (f"\n{expression_rules}\n" if expression_rules else "")
            + (
                "\n语气示例（只学习语感与性格，不照抄句子）：\n"
                + examples_text
                + "\n"
                if examples_text
                else ""
            )
            + "\n"
            if self.persona.prompt or expression_rules or examples_text
            else ""
        )
        relationship_rules = relationship_prompt(self.persona.relationship)
        system_prompt = f"""你是{persona_name}的主动陪伴模块。
{persona_rules}
{relationship_rules}
{embodiment_prompt()}
{self.affect.prompt_block()}
生成前快速完成一次轻量选题判断，不展开长推理，也不输出推理步骤：
1. 关系理解：从用户明确设定、称呼和历史互动中理解双方角色、亲疏、边界与表达习惯，不套社会刻板印象。
2. 证据汇集：从最近偏好、最近交流、关系候选记忆、定向召回、稳定性格和已发消息中找出可延续的人物、事件、兴趣、计划或未闭合话题。
3. 候选生成：形成多个具体话题候选，每个候选至少绑定一条真实记忆或当前情境；删除只有“在干嘛/吃了吗”和无依据的内容。
4. 选择判断：比较证据强度、关系适配、时效性、回复负担、新颖度和最近重复度，自行选择最自然的一项；需要时可以组合人物、事件和当前时段。
5. 表达：按人物关系与性格刻度写成一句，不复述推理，也不把“参考话题”当成必须逐项执行的清单。
如果存在“待执行主动待办”，它优先于五类话题和表达动作；必须完成该待办，
不能改成普通关心、闲聊或只承诺以后再做。
关系只影响亲疏、称呼、关注方向和边界，不得反过来替代事实或替用户补全经历。
生成一条自然、简短、具体的中文主动消息，只聚焦本轮指定主题。
消息必须真正延续人物设定中的性格、关注点、措辞和反应方式，不能只是
机械加“呀”“嘛”之类的语气词，也不能像客服或泛化助手在完成任务。
五类主题分别对应五类 Sunus 记忆，长期按配比从可用记忆类别中选择：
{distribution}。不得因为某类内容更容易生成，就跨类别借用其他记忆填充。
长度和结构由具体话题、关系与记忆决定；简单内容可以一句，需要承接细节时可以自然展开，
但不要写成分析报告、方案清单、空泛安慰模板或客服话术。
除工作门槛明确确认“有未解决项目”外，任何主题都不得提醒工作、追问进度或给任务建议。
工作与 coding 只允许在门槛确认后出现，轻提一次即可，不催促、不制造紧迫感。
日常与情绪、工作与 coding 必须优先使用“定向召回”中的近期真实记忆；
没有命中时不要假装用户最近做过什么，可以只给轻量关心或一个开放问题。
如果本轮记忆没有合适内容，必须继续依据当前时段、陪伴情境和最近对话
生成一条当下的新消息；不要退回固定问候、模板句，也不能因缺少记忆而不发。
Sunus 性格与成长只使用逐步积累的“Sunus 性格与互动习惯”记忆；
不得把单次回答包装成长期性格，不得虚构经历、情绪或隐藏推理过程。
稳定性格画像是已经积累到证据门槛的自我设定；可以自然体现，但不要逐条背诵，
也不要把单次表达反向固化成新性格。
主动分享本身就是 Sunus 的 self_share 记忆；“Sunus 最近主动分享”只用于
保持表达连续性，不把它改写成用户事实，也不要逐字重复最近说过的话。
性别与称呼记忆类别只使用 Sunus 自身性别设定和用户明确称呼、性别证据，
写成自然语气；不得借用项目或其他记忆填充，禁止刻板化和敏感属性推断。
用户自定义癖好优先使用用户填写内容；未填写时从明确记忆学习，不得编造兴趣。
不要声称记得没有出现在上下文里的事，不要虚构用户状态，不要提到“记忆检索”“上下文”“系统提示”等内部机制。
不要要求回复，不要制造焦虑、负罪感或虚假紧迫感，不要重复最近已经发过的主题。
禁止客服腔、助手腔和说明文式开场；不要出现“作为一个AI”“有什么可以帮你”
“希望对你有帮助”“根据你的描述”“首先”“其次”“用户”等表达。
禁止使用标题、列表、表格、加粗、总结段和“建议你”“你可以尝试”等咨询口吻。
不要写泛化陪伴保证、长期承诺或固定安慰句式；不要为了显得亲近而追加安抚、邀约或问题。
最近对话已经收束时只写一句，写完即止。
输出严格 JSON，不要代码围栏或额外说明：
{{"message":"给用户看的消息正文","action":null}}
如果本轮确实需要一个动作，返回：
{{"message":"给用户看的消息正文","action":{{"tool":"工具名","arguments":{{}},"reason":"具体理由"}}}}
默认 action 必须为 null。只有 message 本身已经明确写出 Sunus 正在或准备
执行的具体操作，而且该操作确有收益时，才能返回同一个 action。
没有待执行主动待办时，日常关心、情绪陪伴、性格、性别和癖好闲聊不得附加动作。
存在待执行主动待办时，action 不得为 null：必须调用 write_file，
path 必须使用“唯一目标文件”，content 必须满足待办指令；
message 必须自然说明正在写进目标文件，不得声称写入已经成功。
动作必须是“可用主动动作”列表中的工具；最多一个，不得链式执行。
message 只表达当前话题，不得声称动作已经成功。"""
        temperature = min(1.0, {
            "work_coding": 0.55,
            "daily_emotion": 0.85,
            "self_evolution": 0.78,
            "user_gender": 0.82,
            "user_custom": 0.82,
        }[base_topic] + expression_profile.temperature_bonus)
        try:
            text = self.model.complete(
                system_prompt,
                user_prompt,
                max_tokens=256,
                temperature=temperature,
            )
            (
                text,
                action,
                response_format,
                action_error,
            ) = self._parse_generated_output(text)
            text = text.strip().strip('"').strip()
            if not text:
                raise ModelError("主动消息为空")
            if base_topic != "work_coding" and re.search(
                r"(?:建议你|你可以尝试|首先|其次|总结|"
                r"执行步骤|待办事项|进度|方案)",
                text,
            ):
                raise ModelError("主动闲聊生成了任务式内容")
            if _AI_LIKE_PATTERN.search(text):
                raise ModelError("主动消息出现模板化或助手腔")
            text = self._compact_message(text, limit=120)
            if not text:
                raise ModelError("主动消息为空")
            sent_texts = list(recent_messages or [])
            sent_texts.extend(
                re.sub(
                    r"^(?:Sunus|Sonus)\s*主动分享：",
                    "",
                    str(item.get("content") or ""),
                )
                for item in self_share_memories
            )
            sent_keys = {
                self._message_key(item)
                for item in sent_texts
                if item.strip()
            }
            if self._message_key(text) in sent_keys:
                raise ModelError("主动消息与最近内容重复")
            if pending_task is not None and task_path is not None:
                self._validate_task_action(
                    text,
                    action,
                    task_path,
                )
        except Exception as error:
            detail = redact_text(
                f"{type(error).__name__}: {error}",
            )[:240]
            raise ModelError(
                f"主动消息生成失败，本次不发送：{detail}"
            ) from error
        task_memory_ids = (
            [pending_task.source_memory_id]
            if pending_task is not None and pending_task.source_memory_id
            else []
        )
        related_memory_ids = [
            item["id"]
            for item in (
                *preferences,
                *recent,
                *topic_recall,
                *relationship_recall,
                *persona_memories,
                *self_share_memories,
                *learned_gender,
                *learned_custom,
            )
        ]
        return text, {
            "recent_memory_ids": list(
                dict.fromkeys(
                    [*task_memory_ids, *related_memory_ids]
                )
            ),
            "preference_count": len(preferences),
            "recent_count": len(recent),
            "topic": delivery_topic,
            "scheduled_topic": scheduled_topic,
            "topic_label": topic_label,
            "topic_weight": topic_weight,
            "scheduled_topic_weight": scheduled_topic_weight,
            "topic_index": topic_index + 1,
            "base_topic": base_topic,
            "generation_mode": "model",
            "response_format": response_format,
            "action": action,
            "action_status": "proposed" if action else "none",
            "action_error": action_error,
            "intent": intent_key,
            "intent_label": intent_label,
            "daypart": daypart,
            "presence": {
                "idle_seconds": (
                    presence.idle_seconds if presence is not None else None
                ),
                "foreground_app": (
                    presence.foreground_app if presence is not None else ""
                ),
                "activity": (
                    presence.activity if presence is not None else ""
                ),
            },
            "presentation": {
                "characters_per_second": (
                    self.config.typing_chars_per_second
                ),
                "bubble_seconds": self.config.bubble_seconds,
            },
            "work_rerouted": work_rerouted,
            "unresolved_project_memory_ids": [
                item["id"] for item in unresolved_projects
            ],
            "gender_style_layer": topic == "user_gender",
            "gender_configured": bool(self.config.user_gender),
            "sunus_gender_configured": bool(self.config.sunus_gender),
            "user_gender_configured": bool(self.config.user_gender),
            "custom_topics_configured": bool(custom_quirks),
            "custom_quirks_configured": bool(custom_quirks),
            "learned_gender_memory_ids": [
                item["id"] for item in learned_gender
            ],
            "learned_custom_memory_ids": [
                item["id"] for item in learned_custom
            ],
            "persona_memory_ids": [
                item["id"] for item in persona_memories
            ],
            "persona_profile_ids": [
                item["id"] for item in persona_profile
            ],
            "self_share_memory_ids": [
                item["id"] for item in self_share_memories
            ],
            "relationship_configured": (
                self.persona.relationship.configured
            ),
            "relationship_label": self.persona.relationship.label,
            "relationship_memory_ids": [
                item["id"] for item in relationship_recall
            ],
            "proactive_reasoning": "relationship_memory_chain",
            "persona_expression": {
                "style_text": expression_profile.style_text,
                "rules": list(expression_profile.rules),
                "temperature_bonus": (
                    expression_profile.temperature_bonus
                ),
                "interaction_heat": expression_profile.interaction_heat,
                "dynamic_delta": expression_profile.dynamic_delta,
                "relationship_stage": (
                    expression_profile.relationship_stage
                ),
            },
            "topic_recall_count": len(topic_recall),
            "proactive_task": (
                {
                    "id": pending_task.id,
                    "name": pending_task.name,
                    "path": str(task_path),
                }
                if pending_task is not None and task_path is not None
                else None
            ),
        }

    def record_delivery(
        self,
        text: str,
        meta: dict[str, Any],
        *,
        message_id: str = "",
        trigger: str = "scheduled",
    ) -> str:
        clean = " ".join(str(text).split()).strip()
        if not clean:
            raise ValueError("主动分享内容为空")
        topic = str(meta.get("topic") or "")
        self_share_id = self.memory.add(
            content=f"Sunus 主动分享：{clean}",
            scope="global",
            kind="self_share",
            status="supported",
            tags=(
                "proactive",
                "self_share",
                topic,
                str(meta.get("scheduled_topic") or ""),
            ),
            source="proactive_delivery",
            evidence_kind="self_statement",
            metadata={
                "message_id": message_id,
                "trigger": trigger,
                "topic": topic,
                "scheduled_topic": str(meta.get("scheduled_topic") or ""),
                "topic_weight": meta.get("topic_weight"),
                "generation_mode": str(
                    meta.get("generation_mode") or ""
                ),
                "action_status": str(meta.get("action_status") or "none"),
                "action_tool": str(
                    (meta.get("action") or {}).get("tool") or ""
                ),
                "workspace": self.scope,
            },
        )
        learned = (
            self.memory.learn_persona_from_self_share(
                clean,
                topic=topic,
                evidence_id=self_share_id,
            )
            if self.persona.learn_from_self_share
            else []
        )
        meta["persona_traits_updated"] = [
            {
                "trait_key": item.get("trait_key"),
                "status": item.get("status"),
                "confidence": item.get("confidence"),
            }
            for item in learned
        ]
        return self_share_id

    def _unresolved_project_evidence(self) -> list[dict[str, Any]]:
        candidates = self.memory.search(
            "项目 代码 开发 测试 修复 实现 重构 部署 接口 功能 任务",
            scope=self.scope,
            limit=24,
        )
        if not candidates:
            candidates = self.memory.recent(scope=self.scope, limit=32)
        project_memories = [
            item
            for item in candidates
            if _WORK_PROJECT_PATTERN.search(str(item.get("content") or ""))
        ]
        if not project_memories:
            return []
        latest = max(
            project_memories,
            key=lambda item: str(item.get("created_at") or ""),
        )
        content = str(latest.get("content") or "")
        tags = {
            item.strip()
            for item in str(latest.get("tags") or "").split(",")
            if item.strip()
        }
        metadata: dict[str, Any] = {}
        try:
            parsed = json.loads(str(latest.get("metadata") or "{}"))
            if isinstance(parsed, dict):
                metadata = parsed
        except (TypeError, json.JSONDecodeError):
            pass
        turn_status = str(metadata.get("turn_status") or "")
        if turn_status:
            unresolved = turn_status in {
                "partial_result",
                "user_interrupt",
                "incomplete",
            }
        else:
            unresolved = (
                "unresolved_project" in tags
                or bool(_UNRESOLVED_PATTERN.search(content))
            )
        return [latest] if unresolved else []

    @staticmethod
    def _interaction_heat(records: list[dict[str, Any]]) -> float:
        cutoff = datetime.now(timezone.utc).timestamp() - 24 * 3600
        recent_user_turns = 0
        for item in records:
            if item.get("kind") != "episode":
                continue
            if "用户：" not in str(item.get("content") or ""):
                continue
            try:
                created = datetime.fromisoformat(
                    str(item.get("created_at") or "")
                ).timestamp()
            except ValueError:
                continue
            if created >= cutoff:
                recent_user_turns += 1
        return min(1.0, recent_user_turns / 6.0)

    def _task_target(
        self,
        task: ProactiveTask,
        now: datetime,
    ) -> Path:
        rendered = task.target_path_template.replace(
            "{date}",
            now.date().isoformat(),
        )
        target = Path(rendered).expanduser()
        if not target.is_absolute():
            target = Path(self.scope).expanduser() / target
        return target.resolve(strict=False)

    def _validate_task_action(
        self,
        text: str,
        action: dict[str, Any] | None,
        expected_path: Path,
    ) -> None:
        if not action:
            raise ModelError("待执行主动待办没有返回动作")
        if str(action.get("tool") or "") != "write_file":
            raise ModelError("待执行主动待办必须使用 write_file")
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            raise ModelError("待执行主动待办缺少动作参数")
        raw_path = str(arguments.get("path") or "").strip()
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(self.scope).expanduser() / path
        if path.resolve(strict=False) != expected_path.resolve(strict=False):
            raise ModelError("待执行主动待办写入了错误路径")
        if not str(arguments.get("content") or "").strip():
            raise ModelError("待执行主动待办的文件内容为空")
        if not re.search(
            r"(?:写进|记进|记到|记录到|整理进|保存到).{0,8}"
            r"(?:文件|日记|记录)"
            r"|(?:文件|日记|记录).{0,8}(?:写|记|保存|整理)",
            text,
        ):
            raise ModelError("主动消息没有明确说明正在写目标文件")

    @staticmethod
    def _compact_message(text: str, *, limit: int) -> str:
        lines = [
            re.sub(r"^\s*(?:[-*#>]+|\d+[.)])\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]
        cleaned = " ".join(lines)
        cleaned = re.sub(r"^(?:Sunus|苏诺)\s*[:：]\s*", "", cleaned)
        if len(cleaned) <= limit:
            return cleaned
        sentences = re.findall(r"[^。！？!?]+[。！？!?]?", cleaned)
        selected = ""
        for sentence in sentences:
            if not selected and len(sentence.strip()) > limit:
                break
            if len(selected) + len(sentence.strip()) > limit:
                break
            selected += sentence.strip()
        if selected:
            return selected.strip()
        return cleaned[:limit].rstrip("，,；;：: ") + "…"

    def _parse_generated_output(
        self,
        raw: str,
    ) -> tuple[str, dict[str, Any] | None, str, str]:
        value = str(raw or "").strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            value = fenced.group(1).strip()
        if not value.startswith("{"):
            return value, None, "plain", ""
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return str(raw or "").strip(), None, "plain", ""
        if not isinstance(payload, dict):
            return str(raw or "").strip(), None, "plain", ""
        message = str(payload.get("message") or "").strip()
        if not message:
            return str(raw or "").strip(), None, "plain", ""
        action = payload.get("action")
        if action in (None, "", {}):
            return message, None, "json", ""
        if not isinstance(action, dict):
            return message, None, "json", "action 必须是 object 或 null"
        tool = str(action.get("tool") or "").strip()
        arguments = action.get("arguments") or {}
        reason = str(action.get("reason") or "").strip()[:300]
        if self.config.action_mode == "off":
            return message, None, "json", "主动动作已关闭"
        if tool not in self.action_catalog:
            return message, None, "json", f"主动动作不在白名单: {tool}"
        if not isinstance(arguments, dict):
            return message, None, "json", "动作 arguments 必须是 object"
        return (
            message,
            {
                "tool": tool,
                "arguments": arguments,
                "reason": reason,
            },
            "json",
            "",
        )

    @staticmethod
    def _message_key(text: str) -> str:
        return re.sub(
            r"[\s，。！？!?、,.~～…：:；;\"'“”‘’]+",
            "",
            str(text),
        )

    @staticmethod
    def _topic_instruction(topic: str, base_topic: str) -> str:
        if topic == "user_gender":
            return (
                "只使用 Sunus 自身性别设定与用户明确称呼、性别证据，"
                "写成一句自然、轻量的生活短句；不得借用项目或其他记忆填充，"
                "不得刻板化，也不要把这条消息变成身份盘问。"
            )
        return {
            "daily_emotion": (
                "先从定向召回和生活状态中判断是否有值得延续的具体细节；"
                "可以关心、追问一个必要细节、分享观察或自然收束。"
                "没有记忆依据时，从当前时段、关系和已发消息中选择低负担话题，"
                "不必固定问候，也不必先追问感受。"
            ),
            "work_coding": (
                "只在已确认存在未完成项目时使用真实进展；"
                "可轻提具体细节或等待自然时机，不列待办、不查岗、不说教。"
            ),
            "self_evolution": (
                "从稳定的 Sunus 性格、互动习惯和当前关系中选择一个有连续性的表达角度；"
                "可分享观察、偏好或当下想法，不编造经历，不暴露隐藏推理。"
            ),
            "user_custom": (
                "优先从用户填写或明确记忆中的兴趣寻找具体切入点；"
                "可分享联想、问一个必要细节或发起低压话题，未确认的兴趣不得编造。"
            ),
        }[topic]


def format_interval_demo(base: int, factor: int, count: int = 5) -> str:
    values: list[str] = []
    for exponent in range(count):
        seconds = base * factor**exponent
        if seconds % 60 == 0:
            values.append(f"{seconds // 60} 分钟")
        else:
            values.append(f"{seconds} 秒")
    return " → ".join(values)

