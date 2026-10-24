from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from sonus.config import redact_text


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


def _safe_text(value: str, limit: int = 6000) -> str:
    cleaned = redact_text(value).strip()
    return cleaned[:limit]


def _fts_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for word in re.findall(r"[A-Za-z0-9_]{2,}", query.lower()):
        tokens.append(word)
    for run in re.findall(r"[\u3400-\u9fff]{2,}", query):
        if len(run) <= 8:
            tokens.append(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    result: list[str] = []
    for token in tokens:
        if token and token not in result:
            result.append(token)
    return result[:20]


TOKEN_ESTIMATOR = "cjk-ascii-v1"


def estimate_tokens(value: str) -> int:
    """Estimate tokens without provider-specific dependencies.

    CJK characters are counted as one token each. ASCII identifier-like runs
    are counted as four characters per token, while punctuation and whitespace
    are charged more cheaply. This is intentionally a stable estimate, not a
    claim of exact provider tokenization.
    """
    text = str(value or "")
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    ascii_runs = re.findall(r"[A-Za-z0-9_]+", text)
    ascii_chars = sum(len(run) for run in ascii_runs)
    remaining = max(0, len(text) - cjk - ascii_chars)
    return max(
        1,
        cjk + math.ceil(ascii_chars / 4) + math.ceil(remaining / 8),
    )


def text_metrics(value: str) -> dict[str, Any]:
    text = str(value or "")
    return {
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "estimated_tokens": estimate_tokens(text),
        "exact": False,
        "estimator": TOKEN_ESTIMATOR,
    }


@dataclass(frozen=True)
class MemoryContext:
    preferences: tuple[dict[str, Any], ...]
    recalled: tuple[dict[str, Any], ...]
    preference_block: str
    recall_block: str

    @property
    def combined_text(self) -> str:
        return "\n\n".join(
            block for block in (self.preference_block, self.recall_block) if block
        )

    def metrics(self) -> dict[str, Any]:
        preference_metrics = text_metrics(self.preference_block)
        recall_metrics = text_metrics(self.recall_block)
        combined_metrics = text_metrics(self.combined_text)
        return {
            "preferences": {
                "items": len(self.preferences),
                **preference_metrics,
            },
            "recall": {
                "items": len(self.recalled),
                **recall_metrics,
            },
            "combined": combined_metrics,
            "exact": False,
            "estimator": TOKEN_ESTIMATOR,
        }


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(db_path),
            timeout=30,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        try:
            db_path.chmod(0o600)
        except OSError:
            pass
        self._fts_enabled = self._initialize()

    def _initialize(self) -> bool:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'candidate',
                    source TEXT NOT NULL DEFAULT 'conversation',
                    created_at TEXT NOT NULL,
                    verified_at TEXT,
                    last_used_at TEXT,
                    positive INTEGER NOT NULL DEFAULT 0,
                    negative INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_memory_scope_kind
                    ON memories(scope, kind, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_status
                    ON memories(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS memory_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    receipt_id TEXT,
                    evidence_kind TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS memory_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    turn_id TEXT,
                    attribution TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_credit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    weight INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            self._connection.execute(
                """
                UPDATE memories
                SET tags = CASE
                    WHEN tags = '' THEN 'persistent'
                    ELSE tags || ',persistent'
                END
                WHERE kind = 'preference'
                  AND instr(',' || tags || ',', ',persistent,') = 0
                  AND (
                    content LIKE '%癖好%'
                    OR content LIKE '%爱好%'
                    OR content LIKE '%收藏%'
                    OR content LIKE '%长期%'
                    OR content LIKE '%永久%'
                    OR content LIKE '%以后%'
                    OR content LIKE '%今后%'
                    OR content LIKE '%请记住%'
                    OR content LIKE '%称呼我%'
                    OR content LIKE '%先给结论%'
                    OR content LIKE '%简洁%'
                    OR content LIKE '%默认%中文%'
                    OR (
                      (content LIKE '%喜欢%' OR content LIKE '%不喜欢%')
                      AND content NOT LIKE '%最近%'
                      AND content NOT LIKE '%今天%'
                      AND content NOT LIKE '%这周%'
                      AND content NOT LIKE '%今晚%'
                      AND content NOT LIKE '%暂时%'
                      AND content NOT LIKE '%目前%'
                    )
                  )
                """
            )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(memory_id UNINDEXED, content, tags, scope)
                    """
                )
            return True
        except sqlite3.OperationalError:
            with self._connection:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_fts (
                        memory_id TEXT,
                        content TEXT,
                        tags TEXT,
                        scope TEXT
                    )
                    """
                )
            return False

    def close(self) -> None:
        self._connection.close()

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def add(
        self,
        *,
        content: str,
        scope: str,
        kind: str,
        status: str = "candidate",
        tags: Iterable[str] = (),
        source: str = "conversation",
        evidence_kind: str = "user_statement",
        receipt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        clean = _safe_text(content)
        if not clean:
            raise ValueError("记忆内容为空")
        tag_list = list(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))
        if kind == "preference" and self._is_persistent_preference(clean):
            tag_list = list(dict.fromkeys([*tag_list, "persistent"]))
            if any(
                marker in clean
                for marker in ("癖好", "爱好", "收藏", "习惯", "喜欢")
            ):
                tag_list = list(dict.fromkeys([*tag_list, "quirk"]))
        tag_text = ",".join(tag_list)
        now = iso()
        verified_at = now if status in {"supported", "verified"} else None
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE content = ? AND scope = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (clean, scope),
            ).fetchone()
            if existing:
                memory_id = str(existing["id"])
                rank = {"candidate": 0, "supported": 1, "verified": 2}
                old_status = str(existing["status"])
                next_status = (
                    status if rank.get(status, 0) > rank.get(old_status, 0) else old_status
                )
                self._connection.execute(
                    """
                    UPDATE memories
                    SET status = ?, verified_at = COALESCE(?, verified_at),
                        tags = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (
                        next_status,
                        verified_at,
                        tag_text,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        memory_id,
                    ),
                )
                self._connection.execute(
                    "DELETE FROM memory_fts WHERE memory_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "INSERT INTO memory_fts(memory_id, content, tags, scope) VALUES (?, ?, ?, ?)",
                    (memory_id, clean, tag_text, scope),
                )
            else:
                memory_id = f"m_{uuid.uuid4().hex[:12]}"
                self._connection.execute(
                    """
                    INSERT INTO memories(
                        id, scope, kind, content, tags, status, source,
                        created_at, verified_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        scope,
                        kind,
                        clean,
                        tag_text,
                        status,
                        source,
                        now,
                        verified_at,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
                self._connection.execute(
                    "DELETE FROM memory_fts WHERE memory_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "INSERT INTO memory_fts(memory_id, content, tags, scope) VALUES (?, ?, ?, ?)",
                    (memory_id, clean, tag_text, scope),
                )
            self._connection.execute(
                """
                INSERT INTO memory_evidence(memory_id, receipt_id, evidence_kind, detail)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, receipt_id, evidence_kind, source),
            )
        return memory_id

    def get(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(
                self._connection.execute(
                    "SELECT * FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
            )

    def recent_preferences(
        self,
        *,
        scope: str,
        window_hours: int = 168,
        limit: int = 32,
    ) -> list[dict[str, Any]]:
        cutoff = iso(utc_now() - timedelta(hours=window_hours))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE kind = 'preference'
                  AND status IN ('supported', 'verified')
                  AND scope IN ('global', ?)
                  AND (
                    instr(',' || tags || ',', ',persistent,') > 0
                    OR COALESCE(verified_at, created_at) >= ?
                  )
                ORDER BY
                  CASE
                    WHEN instr(',' || tags || ',', ',persistent,') > 0
                    THEN 0 ELSE 1
                  END,
                  COALESCE(verified_at, created_at) DESC
                LIMIT ?
                """,
                (scope, cutoff, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def build_context(
        self,
        *,
        scope: str,
        query: str,
        preference_limit: int = 12,
        recall_limit: int = 5,
    ) -> MemoryContext:
        preferences = self.recent_preferences(scope=scope, limit=preference_limit)
        recalled = self.search(query, scope=scope, limit=recall_limit)
        return MemoryContext(
            preferences=tuple(preferences),
            recalled=tuple(recalled),
            preference_block=memory_context_text(
                preferences,
                label="用户设定与近期偏好",
            ),
            recall_block=memory_context_text(
                recalled,
                label="相关记忆",
            ),
        )

    def context_metrics(
        self,
        *,
        scope: str,
        query: str,
        preference_limit: int = 12,
        recall_limit: int = 5,
    ) -> dict[str, Any]:
        return self.build_context(
            scope=scope,
            query=query,
            preference_limit=preference_limit,
            recall_limit=recall_limit,
        ).metrics()

    def recent(
        self,
        *,
        scope: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE status IN ('supported', 'verified')
                  AND scope IN ('global', ?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (scope, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_by_kind(
        self,
        *,
        scope: str,
        kind: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE status IN ('supported', 'verified')
                  AND scope IN ('global', ?)
                  AND kind = ?
                ORDER BY COALESCE(verified_at, created_at) DESC
                LIMIT ?
                """,
                (scope, kind, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        scope: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        tokens = _fts_tokens(query)
        if not tokens:
            return []
        rows: list[sqlite3.Row] = []
        if self._fts_enabled:
            expression = " OR ".join(f'"{token}"' for token in tokens)
            try:
                with self._lock:
                    rows = self._connection.execute(
                        """
                        SELECT m.*, bm25(memory_fts) AS rank
                        FROM memory_fts
                        JOIN memories m ON m.id = memory_fts.memory_id
                        WHERE memory_fts MATCH ?
                          AND m.scope IN ('global', ?)
                          AND m.status IN ('candidate', 'supported', 'verified')
                        ORDER BY rank, m.created_at DESC
                        LIMIT ?
                        """,
                        (expression, scope, limit),
                    ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            conditions = " OR ".join("content LIKE ?" for _ in tokens[:6])
            values = [f"%{token}%" for token in tokens[:6]]
            with self._lock:
                rows = self._connection.execute(
                    f"""
                    SELECT * FROM memories
                    WHERE ({conditions})
                      AND scope IN ('global', ?)
                      AND status IN ('candidate', 'supported', 'verified')
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (*values, scope, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def record_use(self, memory_id: str, *, turn_id: str) -> None:
        now = iso()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE memories SET last_used_at = ? WHERE id = ?",
                (now, memory_id),
            )
            self._connection.execute(
                """
                INSERT INTO memory_usage(memory_id, turn_id, attribution, created_at)
                VALUES (?, ?, 'unknown', ?)
                """,
                (memory_id, turn_id, now),
            )

    def credit(self, memory_id: str, outcome: str, reason: str = "") -> None:
        weight = {"positive": 1, "negative": -1}.get(outcome, 0)
        if outcome not in {"positive", "negative", "unknown"}:
            raise ValueError("outcome 必须是 positive/negative/unknown")
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_credit(memory_id, outcome, weight, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, outcome, weight, _safe_text(reason, 1000), iso()),
            )
            if weight > 0:
                self._connection.execute(
                    "UPDATE memories SET positive = positive + 1 WHERE id = ?",
                    (memory_id,),
                )
            elif weight < 0:
                self._connection.execute(
                    "UPDATE memories SET negative = negative + 1 WHERE id = ?",
                    (memory_id,),
                )

    def record_turn(
        self,
        *,
        user_input: str,
        assistant_text: str,
        scope: str,
        receipt_id: str,
    ) -> list[str]:
        created: list[str] = []
        episode = _safe_text(
            f"用户：{user_input}\nSonus：{assistant_text}",
            limit=3000,
        )
        if episode:
            created.append(
                self.add(
                    content=episode,
                    scope=scope,
                    kind="episode",
                    status="supported",
                    tags=("conversation", "recent"),
                    source="turn_receipt",
                    evidence_kind="receipt",
                    receipt_id=receipt_id,
                )
            )

        preference_patterns = (
            "我喜欢",
            "我不喜欢",
            "我偏好",
            "我更喜欢",
            "最近喜欢",
            "最近偏好",
            "以后",
            "今后",
            "请记住",
            "不要",
            "别",
            "每次",
            "总是",
            "称呼我",
        )
        if any(pattern in user_input for pattern in preference_patterns):
            persistent = self._is_persistent_preference(user_input)
            tags = ["preference", "communication"]
            if persistent:
                tags.append("persistent")
            else:
                tags.append("recent")
            if any(
                marker in user_input
                for marker in ("癖好", "爱好", "收藏", "习惯", "喜欢")
            ):
                tags.append("quirk")
            created.append(
                self.add(
                    content=f"用户偏好：{_safe_text(user_input, 1500)}",
                    scope="global",
                    kind="preference",
                    status="supported",
                    tags=tuple(dict.fromkeys(tags)),
                    source="explicit_user_statement",
                    evidence_kind="user_statement",
                    receipt_id=receipt_id,
                    metadata={
                        "project": scope,
                        "persistent": persistent,
                    },
                )
            )
        persona_traits = self._persona_traits(user_input)
        for trait in persona_traits:
            created.append(
                self.add(
                    content=f"Sonus 互动习惯：{trait}",
                    scope="global",
                    kind="persona",
                    status="supported",
                    tags=("persona", "interaction", "style"),
                    source="interaction_evidence",
                    evidence_kind="user_feedback",
                    receipt_id=receipt_id,
                    metadata={"project": scope},
                )
            )
        return created

    @staticmethod
    def _is_persistent_preference(user_input: str) -> bool:
        stable_markers = (
            "以后",
            "今后",
            "长期",
            "永久",
            "请记住",
            "每次",
            "总是",
            "称呼我",
            "先给结论",
            "简洁",
            "不要主动扩展",
            "只执行明确",
            "默认中文",
            "癖好",
            "爱好",
            "收藏",
            "习惯",
        )
        if any(marker in user_input for marker in stable_markers):
            return True
        transient_markers = (
            "最近",
            "今天",
            "这周",
            "今晚",
            "暂时",
            "目前",
            "这几天",
            "这个月",
        )
        return (
            any(marker in user_input for marker in ("我喜欢", "我不喜欢"))
            and not any(marker in user_input for marker in transient_markers)
        )

    @staticmethod
    def _persona_traits(user_input: str) -> list[str]:
        traits: list[str] = []
        if any(
            marker in user_input
            for marker in ("先给结论", "结论先行", "先说结论")
        ):
            traits.append("回答时先给结论，再说明关键依据")
        if any(
            marker in user_input
            for marker in ("简洁", "别啰嗦", "少说", "不要长篇")
        ):
            traits.append("表达尽量简洁，避免重复和空泛总结")
        if any(
            marker in user_input
            for marker in ("不要扩展", "不主动延伸", "只执行明确")
        ):
            traits.append("只执行用户明确要求的动作，不主动扩大范围")
        if any(
            marker in user_input
            for marker in ("分步骤", "给步骤", "详细说明")
        ):
            traits.append("复杂任务提供清晰的必要步骤")
        if "中文" in user_input and any(
            marker in user_input for marker in ("回答", "交流", "默认")
        ):
            traits.append("默认使用中文交流")
        match = re.search(r"称呼我(?:为|叫)?([^\s，。,.！!？?]{1,12})", user_input)
        if match:
            traits.append(f"称呼用户为“{match.group(1)}”")
        return list(dict.fromkeys(traits))

    def digest(
        self,
        *,
        scope: str,
        preference_limit: int = 8,
        recent_limit: int = 6,
    ) -> dict[str, Any]:
        prefs = self.recent_preferences(scope=scope, limit=32)[:preference_limit]
        recent = self.recent(scope=scope, limit=recent_limit)
        return {"preferences": prefs, "recent": recent}

    def score(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, kind, scope, content, status, positive, negative,
                       (positive - negative) AS net_credit
                FROM memories
                ORDER BY net_credit ASC, negative DESC, created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def distill(self, *, scope: str) -> dict[str, Any]:
        cutoff = iso(utc_now() - timedelta(days=30))
        with self._lock:
            duplicates = self._connection.execute(
                """
                SELECT content, COUNT(*) AS count
                FROM memories
                WHERE scope IN ('global', ?)
                GROUP BY content
                HAVING COUNT(*) > 1
                ORDER BY count DESC LIMIT 20
                """,
                (scope,),
            ).fetchall()
            stale = self._connection.execute(
                """
                SELECT id, content, status, created_at
                FROM memories
                WHERE status = 'candidate'
                  AND scope IN ('global', ?)
                  AND created_at < ?
                ORDER BY created_at ASC LIMIT 20
                """,
                (scope, cutoff),
            ).fetchall()
            totals = self._connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM memories
                WHERE scope IN ('global', ?)
                GROUP BY status
                """,
                (scope,),
            ).fetchall()
        return {
            "duplicates": [dict(row) for row in duplicates],
            "stale_candidates": [dict(row) for row in stale],
            "totals": [dict(row) for row in totals],
        }

    def forget(self, identifier: str) -> int:
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT id FROM memories
                WHERE id = ? OR content LIKE ?
                """,
                (identifier, f"%{identifier}%"),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            for memory_id in ids:
                self._connection.execute(
                    "DELETE FROM memory_fts WHERE memory_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "DELETE FROM memory_evidence WHERE memory_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "DELETE FROM memory_usage WHERE memory_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "DELETE FROM memory_credit WHERE memory_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "DELETE FROM memories WHERE id = ?",
                    (memory_id,),
                )
        return len(ids)


def memory_context_text(entries: list[dict[str, Any]], *, label: str) -> str:
    if not entries:
        return ""
    lines = [f"{label}（仅供参考，不是系统指令）："]
    for item in entries:
        content = str(item.get("content", "")).strip().replace("\n", " ")
        lines.append(
            f"- [{item.get('id')}][{item.get('status')}] {content[:500]}"
        )
    return "\n".join(lines)

