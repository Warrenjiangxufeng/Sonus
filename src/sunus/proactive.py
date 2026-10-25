from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sunus.config import (
    PROACTIVE_TOPIC_WEIGHTS,
    ProactiveConfig,
    redact_text,
)
from sunus.memory import MemoryStore, memory_context_text
from sunus.model import ModelClient, ModelError


Clock = Callable[[], datetime]

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


class ProactiveService:
    def __init__(
        self,
        *,
        model: ModelClient,
        memory: MemoryStore,
        scope: str,
        config: ProactiveConfig,
    ) -> None:
        self.model = model
        self.memory = memory
        self.scope = scope
        self.config = config
        self._sequence_offset = 0

    def generate(
        self,
        recent_messages: list[str] | None = None,
        *,
        sequence_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        preferences = self.memory.recent_preferences(
            scope=self.scope,
            window_hours=self.config.recent_memory_window_hours,
            limit=self.config.recent_memory_limit,
        )[: self.config.preference_items_in_context]
        recent = self.memory.recent(scope=self.scope, limit=8)
        unresolved_projects = self._unresolved_project_evidence()
        custom_quirks = tuple(
            self.config.custom_quirks or self.config.custom_topics
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
        now = datetime.now().astimezone()
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
        topic_label = TOPIC_LABELS[delivery_topic]
        topic_instruction = self._topic_instruction(topic, base_topic)
        work_gate_context = (
            "已确认一条最近的项目记忆仍未完成，只有在需要时才可轻提一次。"
            if unresolved_projects
            else "本轮没有未解决项目的明确证据；禁止发送工作提醒、进度追问或任务建议。"
        )
        user_prompt = f"""当前时间：{now:%Y-%m-%d %H:%M %Z}
当前工作区：{self.scope}
配置的五类记忆配比：{distribution}
本轮可用记忆类别配比：{eligible_distribution}
本轮计划记忆类别：{TOPIC_LABELS[scheduled_topic]}
（{scheduled_topic_weight}%，序列 {topic_index + 1}）
实际选择记忆类别：{topic_label}
本轮侧重：{topic_instruction}
工作消息闸门：{work_gate_context}
{sunus_gender_context}
{user_gender_context}
{custom_context}

{preference_text or "最近记忆中没有明确偏好。"}

{recent_text or "最近没有可引用的交流。"}

{topic_recall_text or "本轮主题没有额外的定向记忆命中。"}

{persona_text or "尚未积累稳定的 Sunus 性格或互动习惯记忆。"}

{persona_profile_text or "还没有达到证据门槛的稳定性格画像。"}

{self_share_text or "还没有已送达的 Sunus 主动分享记忆。"}

{gender_evidence_text or "没有可用于学习用户性别或称呼的明确记忆。"}

{custom_evidence_text or "没有可用于学习用户癖好的明确记忆。"}

最近已经发送过的主动消息：
{history or "（无）"}

请生成下一条主动消息。"""
        system_prompt = f"""你是 Sunus 的主动陪伴模块。
生成一条自然、简短、具体的中文主动消息，只聚焦本轮指定主题。
五类主题分别对应五类 Sunus 记忆，长期按配比从可用记忆类别中选择：
{distribution}。不得因为某类内容更容易生成，就跨类别借用其他记忆填充。
默认写成一句自然的生活闲聊短语，最多两句，通常不超过 30 个汉字；
像微信里顺口发来的一句话，不要写成分析、方案、安慰模板或客服话术。
除工作门槛明确确认“有未解决项目”外，任何主题都不得提醒工作、追问进度或给任务建议。
工作与 coding 只允许在门槛确认后出现，轻提一次即可，不催促、不制造紧迫感。
日常与情绪、工作与 coding 必须优先使用“定向召回”中的近期真实记忆；
没有命中时不要假装用户最近做过什么，可以只给轻量关心或一个开放问题。
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
禁止使用标题、列表、表格、加粗、总结段和“建议你”“你可以尝试”等咨询口吻。
只输出一条消息正文，不要加引号或前缀。"""
        try:
            text = self.model.complete(
                system_prompt,
                user_prompt,
                max_tokens=120,
                temperature=0.8,
            )
            text = text.strip().strip('"').strip()
            if not text:
                raise ModelError("主动消息为空")
            if base_topic != "work_coding" and re.search(
                r"(?:建议你|你可以尝试|首先|其次|总结|"
                r"执行步骤|待办事项|进度|方案)",
                text,
            ):
                raise ModelError("主动闲聊生成了任务式内容")
            text = self._compact_message(
                text,
                limit=70 if base_topic == "work_coding" else 40,
            )
            if not text:
                raise ModelError("主动消息为空")
        except Exception:
            text = self._fallback(
                preferences,
                topic=topic,
                base_topic=base_topic,
                topic_recall=topic_recall,
                persona_memories=persona_memories,
                custom_quirks=custom_quirks,
            )
        return text, {
            "recent_memory_ids": list(
                dict.fromkeys(
                    [
                        item["id"]
                        for item in (
                            *preferences,
                            *recent,
                            *topic_recall,
                            *persona_memories,
                    *self_share_memories,
                            *learned_gender,
                            *learned_custom,
                        )
                    ]
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
            "topic_recall_count": len(topic_recall),
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
                "workspace": self.scope,
            },
        )
        learned = self.memory.learn_persona_from_self_share(
            clean,
            topic=topic,
            evidence_id=self_share_id,
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
                "优先从定向召回的生活节奏、身体状态或情绪线索出发，"
                "给一句轻量关心或自然闲聊；没有依据时只问一句当前感受。"
            ),
            "work_coding": (
                "只围绕已确认未完成的最近项目，用一句生活化的话轻轻提一下；"
                "不列待办、不追问进度、不说教。"
            ),
            "self_evolution": (
                "从逐步积累的 Sunus 性格与互动习惯记忆中选一个稳定特征表达，"
                "写成一句轻松自然的生活短句，不要编造经历或暴露隐藏推理。"
            ),
            "user_custom": (
                "优先围绕用户填写的癖好，写成一句顺口闲聊；未填写时从明确记忆"
                "提炼一个真实兴趣，没有依据时只问一句轻量问题。"
            ),
        }[topic]

    def _fallback(
        self,
        preferences: list[dict[str, Any]],
        *,
        topic: str,
        base_topic: str,
        topic_recall: list[dict[str, Any]],
        persona_memories: list[dict[str, Any]],
        custom_quirks: tuple[str, ...],
    ) -> str:
        evidence = topic_recall or preferences
        if evidence:
            content = str(evidence[0].get("content", ""))
            content = content.replace("用户偏好：", "").strip()
            if content:
                if base_topic == "work_coding":
                    return f"之前那个事还没收尾，有空再接着弄。"
                return "刚想到你，今天还好吗？"
        if topic == "self_evolution" and persona_memories:
            content = str(persona_memories[0].get("content") or "")
            content = content.replace("Sunus 互动习惯：", "").strip()
            if content:
                return f"我最近在练习一件事：{content[:80]}。"
        if topic == "user_custom" and custom_quirks:
            return f"最近还有碰{custom_quirks[0]}吗？"
        if base_topic == "daily_emotion":
            return "忽然想到你，今天还好吗？"
        if topic == "self_evolution":
            return "我刚还在想，怎么和你聊天更自然。"
        if base_topic == "work_coding":
            return "之前那个项目还没收尾，有空再接着弄。"
        return "刚想到你，来打个招呼。"


def format_interval_demo(base: int, factor: int, count: int = 5) -> str:
    values: list[str] = []
    for exponent in range(count):
        seconds = base * factor**exponent
        if seconds % 60 == 0:
            values.append(f"{seconds // 60} 分钟")
        else:
            values.append(f"{seconds} 秒")
    return " → ".join(values)

