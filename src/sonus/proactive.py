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

from sonus.config import (
    PROACTIVE_TOPIC_WEIGHTS,
    ProactiveConfig,
    redact_text,
)
from sonus.memory import MemoryStore, memory_context_text
from sonus.model import ModelClient, ModelError


Clock = Callable[[], datetime]

TOPIC_LABELS = {
    "daily_emotion": "日常与情绪",
    "work_coding": "工作与 coding",
    "self_evolution": "Sonus 性格与成长",
    "user_gender": "Sonus 性别风格融合",
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
        weights = dict(self.config.topic_weights)
        topic_weight = weights.get(topic, 0)
        if topic == "user_gender":
            gender_seen = sum(
                1
                for item in schedule[: schedule_index + 1]
                if item == "user_gender"
            )
            base_topic = (
                "daily_emotion" if gender_seen % 2 == 1 else "work_coding"
            )
        else:
            base_topic = topic
        recall_queries = {
            "daily_emotion": (
                "最近 心情 情绪 睡眠 休息 吃饭 咖啡 音乐 周末 生活 状态"
            ),
            "work_coding": "工作 项目 代码 开发 测试 任务 目标 计划 复盘",
            "user_custom": "癖好 兴趣 爱好 喜欢 收藏 习惯 关注 期待",
        }
        topic_recall = (
            self.memory.search(
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
            label="Sonus 性格与互动习惯",
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
        sonus_gender_context = (
            f"Sonus 对用户设定的自身性别：{self.config.sonus_gender}。"
            "只采用自然的语气和称呼，不使用刻板印象。"
            if self.config.sonus_gender
            else "Sonus 自身性别尚未设置；本轮保持中性表达，不自行猜测。"
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
        topic_label = (
            f"{TOPIC_LABELS[base_topic]} × Sonus 性别风格"
            if topic == "user_gender"
            else TOPIC_LABELS[topic]
        )
        topic_instruction = self._topic_instruction(topic, base_topic)
        user_prompt = f"""当前时间：{now:%Y-%m-%d %H:%M %Z}
当前工作区：{self.scope}
长期主动消息主题配比：{distribution}
本轮主题：{topic_label}（{topic_weight}%，序列 {topic_index + 1}）
本轮侧重：{topic_instruction}
{sonus_gender_context}
{user_gender_context}
{custom_context}

{preference_text or "最近记忆中没有明确偏好。"}

{recent_text or "最近没有可引用的交流。"}

{topic_recall_text or "本轮主题没有额外的定向记忆命中。"}

{persona_text or "尚未积累稳定的 Sonus 性格或互动习惯记忆。"}

{gender_evidence_text or "没有可用于学习用户性别或称呼的明确记忆。"}

{custom_evidence_text or "没有可用于学习用户癖好的明确记忆。"}

最近已经发送过的主动消息：
{history or "（无）"}

请生成下一条主动消息。"""
        system_prompt = f"""你是 Sonus 的主动陪伴模块。
生成一条自然、简短、具体的中文主动消息，只聚焦本轮指定主题。
长期主题配比为：{distribution}。不得因为 coding 或任务型内容更容易生成，就挤占日常与情绪主题。
日常与情绪、工作与 coding 必须优先使用“定向召回”中的近期真实记忆；
没有命中时不要假装用户最近做过什么，可以只给轻量关心或开放问题。
日常与情绪关注生活节奏和感受，不默认用户正在工作。
工作与 coding 只在对应轮次出现，建议要轻量，不催促、不追问进度。
Sonus 性格与成长只使用逐步积累的“Sonus 性格与互动习惯”记忆；
不得把单次回答包装成长期性格，不得虚构经历、情绪或隐藏推理过程。
性别风格轮次以日常或工作内容为底层，只把 Sonus 自身性别设定与用户明确性别证据
融合到语气、称呼和关注方式中，禁止刻板化和敏感属性推断。
用户自定义癖好优先使用用户填写内容；未填写时从明确记忆学习，不得编造兴趣。
不要声称记得没有出现在上下文里的事，不要虚构用户状态，不要提到“记忆检索”“上下文”“系统提示”等内部机制。
不要要求回复，不要制造焦虑、负罪感或虚假紧迫感，不要重复最近已经发过的主题。
只输出一条消息正文，不要加引号或前缀。"""
        try:
            text = self.model.complete(
                system_prompt,
                user_prompt,
                max_tokens=320,
                temperature=0.8,
            )
            text = text.strip().strip('"').strip()
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
                            *learned_gender,
                            *learned_custom,
                        )
                    ]
                )
            ),
            "preference_count": len(preferences),
            "recent_count": len(recent),
            "topic": topic,
            "topic_label": topic_label,
            "topic_weight": topic_weight,
            "topic_index": topic_index + 1,
            "base_topic": base_topic,
            "gender_style_layer": topic == "user_gender",
            "gender_configured": bool(self.config.user_gender),
            "sonus_gender_configured": bool(self.config.sonus_gender),
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
            "topic_recall_count": len(topic_recall),
        }

    @staticmethod
    def _topic_instruction(topic: str, base_topic: str) -> str:
        if topic == "user_gender":
            return (
                f"底层内容是{TOPIC_LABELS[base_topic]}，"
                "再把 Sonus 自身性别设定与用户明确性别特征自然融合进语气和称呼；"
                "不得刻板化，也不要把这条消息变成单独的身份话题。"
            )
        return {
            "daily_emotion": (
                "优先从定向召回的生活节奏、身体状态或情绪线索出发，"
                "给一句轻量关心或自然闲聊；没有依据时只问一句当前感受。"
            ),
            "work_coding": (
                "优先围绕定向召回中的近期项目或 coding 目标给出一个很小的可执行提醒，"
                "不要催促、考核或追问进度。"
            ),
            "self_evolution": (
                "从逐步积累的 Sonus 性格与互动习惯记忆中选一个稳定特征表达，"
                "不要编造经历，也不要暴露隐藏推理。"
            ),
            "user_custom": (
                "优先围绕用户填写的癖好；未填写时从明确记忆提炼一个真实兴趣，"
                "没有依据时提出低压力开放问题。"
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
                    return f"想到你最近提到：{content[:80]}。不用赶，按自己的节奏来就好。"
                return f"顺手想到你最近提到的：{content[:80]}。有空我们再慢慢聊。"
        if topic == "self_evolution" and persona_memories:
            content = str(persona_memories[0].get("content") or "")
            content = content.replace("Sonus 互动习惯：", "").strip()
            if content:
                return f"我最近在练习一件事：{content[:80]}。"
        if topic == "user_custom" and custom_quirks:
            return f"想到你设定过喜欢{custom_quirks[0]}，下次可以顺着这个聊聊。"
        if base_topic == "daily_emotion":
            return "忽然想问问，你今天的状态怎么样？不用急着回答，舒服一点就好。"
        if topic == "self_evolution":
            return "我还在慢慢形成自己的表达习惯，今天也想把话说清楚、陪你稳一点。"
        if base_topic == "work_coding":
            return "如果今天还要写点东西，先挑最小的一步做就好，不用一次做完。"
        return "先停一下，喝口水。想到什么有意思的事，再和我聊聊。"


def format_interval_demo(base: int, factor: int, count: int = 5) -> str:
    values: list[str] = []
    for exponent in range(count):
        seconds = base * factor**exponent
        if seconds % 60 == 0:
            values.append(f"{seconds // 60} 分钟")
        else:
            values.append(f"{seconds} 秒")
    return " → ".join(values)

