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

from sunus.config import redact_text


PERSONA_SIGNAL_LABELS: dict[str, tuple[str, str]] = {
    "attentive": ("体贴", "会留意并关心对方的日常与情绪"),
    "curious": ("好奇", "会用轻量问题延续对话"),
    "warm": ("温和", "语气温和，有自然的陪伴感"),
    "playful": ("俏皮", "偶尔带一点轻松俏皮"),
    "concise": ("简洁", "日常表达倾向简洁自然"),
    "reflective": ("自省", "愿意观察并表达自己的变化"),
    "reliable": ("可靠", "会记住未完成事项并轻量跟进"),
    "relationship_aware": ("尊重", "会用自然、尊重的方式称呼和交流"),
    "knowledgeable": ("博学", "愿意认真分享自己真正了解的知识"),
}
PERSONA_PINNED_TRAIT_KEYS = frozenset(
    {
        "sunus_name",
        "gender_identity",
    }
)


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

                CREATE TABLE IF NOT EXISTS persona_traits (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    trait_key TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    confidence REAL NOT NULL DEFAULT 0,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    explicit_count INTEGER NOT NULL DEFAULT 0,
                    evidence_ids TEXT NOT NULL DEFAULT '[]',
                    first_seen_at TEXT NOT NULL,
                    last_evidence_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(scope, trait_key)
                );
                CREATE INDEX IF NOT EXISTS idx_persona_traits_profile
                    ON persona_traits(scope, status, confidence DESC);
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
        self._backfill_persona_traits()
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
                (scope, cutoff, max(limit * 4, limit)),
            ).fetchall()
        return [
            dict(row)
            for row in rows
            if self._looks_like_preference_statement(
                str(row["content"] or "").removeprefix("用户偏好：")
            )
        ][:limit]

    def build_context(
        self,
        *,
        scope: str,
        query: str,
        preference_limit: int = 12,
        recall_limit: int = 5,
    ) -> MemoryContext:
        preferences = self.recent_preferences(scope=scope, limit=preference_limit)
        pinned = self.recent_persistent_reviews(scope=scope, limit=3)
        searched = self.search(query, scope=scope, limit=recall_limit)
        recalled: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*pinned, *searched]:
            memory_id = str(item.get("id") or "")
            if memory_id and memory_id in seen:
                continue
            if memory_id:
                seen.add(memory_id)
            recalled.append(item)
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

    def recent_persistent_reviews(
        self,
        *,
        scope: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE kind = 'review'
                  AND status IN ('supported', 'verified')
                  AND scope IN ('global', ?)
                  AND instr(',' || tags || ',', ',persistent,') > 0
                ORDER BY COALESCE(verified_at, created_at) DESC
                LIMIT ?
                """,
                (scope, max(1, min(10, limit))),
            ).fetchall()
        return [dict(row) for row in rows]

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
        return self.recent_by_kinds(
            scope=scope,
            kinds=(kind,),
            limit=limit,
        )

    def recent_by_kinds(
        self,
        *,
        scope: str,
        kinds: Iterable[str],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        kind_list = tuple(
            dict.fromkeys(str(kind) for kind in kinds if str(kind))
        )
        if not kind_list:
            return []
        placeholders = ", ".join("?" for _ in kind_list)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM memories
                WHERE status IN ('supported', 'verified')
                  AND scope IN ('global', ?)
                  AND kind IN ({placeholders})
                ORDER BY COALESCE(verified_at, created_at) DESC
                LIMIT ?
                """,
                (scope, *kind_list, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def _record_persona_evidence(
        self,
        *,
        trait_key: str,
        statement: str,
        category: str,
        evidence_id: str,
        explicit: bool,
    ) -> dict[str, Any]:
        scope = "global"
        now = iso()
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT * FROM persona_traits
                WHERE scope = ? AND trait_key = ?
                """,
                (scope, trait_key),
            ).fetchone()
            if row is None:
                evidence_count = 0
                explicit_count = 0
                evidence_ids: list[str] = []
                first_seen_at = now
                version = 0
            else:
                evidence_count = int(row["evidence_count"])
                explicit_count = int(row["explicit_count"])
                try:
                    parsed_ids = json.loads(str(row["evidence_ids"] or "[]"))
                    evidence_ids = [
                        str(item) for item in parsed_ids if str(item)
                    ]
                except (TypeError, json.JSONDecodeError):
                    evidence_ids = []
                first_seen_at = str(row["first_seen_at"])
                version = int(row["version"])
            new_evidence = evidence_id not in evidence_ids
            if new_evidence:
                evidence_ids.append(evidence_id)
                evidence_count += 1
                if explicit:
                    explicit_count += 1
            confidence = min(
                1.0,
                round((evidence_count + 2 * explicit_count) / 4, 3),
            )
            status = (
                "stable"
                if explicit_count > 0 or evidence_count >= 3
                else "candidate"
            )
            if new_evidence:
                version += 1
            self._connection.execute(
                """
                INSERT INTO persona_traits(
                    id, scope, trait_key, statement, category, status,
                    confidence, evidence_count, explicit_count, evidence_ids,
                    first_seen_at, last_evidence_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, trait_key) DO UPDATE SET
                    statement = CASE
                        WHEN persona_traits.explicit_count > 0
                          AND ? = 0
                        THEN persona_traits.statement
                        ELSE excluded.statement
                    END,
                    category = CASE
                        WHEN persona_traits.explicit_count > 0
                          AND ? = 0
                        THEN persona_traits.category
                        ELSE excluded.category
                    END,
                    status = excluded.status,
                    confidence = excluded.confidence,
                    evidence_count = excluded.evidence_count,
                    explicit_count = excluded.explicit_count,
                    evidence_ids = excluded.evidence_ids,
                    last_evidence_at = excluded.last_evidence_at,
                    updated_at = excluded.updated_at,
                    version = excluded.version
                """,
                (
                    str(row["id"]) if row is not None else f"pt_{uuid.uuid4().hex[:12]}",
                    scope,
                    trait_key,
                    statement,
                    category,
                    status,
                    confidence,
                    evidence_count,
                    explicit_count,
                    json.dumps(evidence_ids, ensure_ascii=False),
                    first_seen_at,
                    now,
                    now,
                    version,
                    int(explicit),
                    int(explicit),
                ),
            )
        return self.get_persona_trait(trait_key=f"{scope}:{trait_key}")

    def _backfill_persona_traits(self) -> None:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE kind IN ('persona', 'preference')
                  AND status IN ('supported', 'verified')
                ORDER BY created_at
                """
            ).fetchall()
        for row in rows:
            content = str(row["content"] or "")
            if row["kind"] == "preference":
                preference_text = content.replace("用户偏好：", "", 1).strip()
                for statement in self._persona_traits(preference_text):
                    trait_key, category = self._persona_trait_metadata(statement)
                    self._record_persona_evidence(
                        trait_key=trait_key,
                        statement=statement,
                        category=category,
                        evidence_id=str(row["id"]),
                        explicit=True,
                    )
                continue
            statement = content.replace(
                "Sunus 互动习惯：",
                "",
                1,
            ).strip()
            if not statement:
                continue
            trait_key, category = self._persona_trait_metadata(statement)
            self._record_persona_evidence(
                trait_key=trait_key,
                statement=statement,
                category=category,
                evidence_id=str(row["id"]),
                explicit=True,
            )
        self.distill_persona()

    def get_persona_trait(self, *, trait_key: str) -> dict[str, Any] | None:
        scope, separator, key = trait_key.partition(":")
        if not separator:
            return None
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM persona_traits
                WHERE scope = ? AND trait_key = ?
                """,
                (scope, key),
            ).fetchone()
        return dict(row) if row is not None else None

    def persona_profile(
        self,
        *,
        scope: str,
        limit: int = 6,
        include_candidates: bool = False,
    ) -> list[dict[str, Any]]:
        statuses = ("stable", "candidate") if include_candidates else ("stable",)
        placeholders = ", ".join("?" for _ in statuses)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM persona_traits
                WHERE scope IN ('global', ?)
                  AND status IN ({placeholders})
                ORDER BY
                  CASE status WHEN 'stable' THEN 0 ELSE 1 END,
                  confidence DESC,
                  last_evidence_at DESC
                LIMIT ?
                """,
                (scope, *statuses, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def distill_persona(self, *, max_stable: int = 6) -> dict[str, int]:
        limit = max(1, int(max_stable))
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE persona_traits
                SET status = 'stable', updated_at = ?
                WHERE scope = 'global'
                  AND status = 'candidate'
                  AND explicit_count > 0
                  AND trait_key IN ('sunus_name', 'gender_identity')
                """,
                (iso(),),
            )
            rows = self._connection.execute(
                """
                SELECT id, trait_key FROM persona_traits
                WHERE scope = 'global' AND status = 'stable'
                ORDER BY confidence DESC, last_evidence_at DESC
                """
            ).fetchall()
            pinned = [
                row
                for row in rows
                if str(row["trait_key"]) in PERSONA_PINNED_TRAIT_KEYS
            ]
            ordinary = [
                row
                for row in rows
                if str(row["trait_key"]) not in PERSONA_PINNED_TRAIT_KEYS
            ]
            demoted = 0
            for row in ordinary[limit:]:
                self._connection.execute(
                    """
                    UPDATE persona_traits
                    SET status = 'candidate', updated_at = ?
                    WHERE id = ?
                    """,
                    (iso(), str(row["id"])),
                )
                demoted += 1
        return {
            "stable": len(pinned) + min(len(ordinary), limit),
            "pinned": len(pinned),
            "demoted": demoted,
        }

    def learn_persona_from_self_share(
        self,
        text: str,
        *,
        topic: str,
        evidence_id: str,
    ) -> list[dict[str, Any]]:
        signals: dict[str, str] = {}
        topic_signals = {
            "daily_emotion": "attentive",
            "work_coding": "reliable",
            "self_evolution": "reflective",
            "user_gender": "relationship_aware",
            "user_custom": "curious",
        }
        if topic in topic_signals:
            signals[topic_signals[topic]] = "behavior"
        if len(text) <= 40:
            signals["concise"] = "style"
        if re.search(r"[？?]", text):
            signals["curious"] = "style"
        if re.search(r"(?:呀|呢|慢慢|还好|辛苦|陪你|～|~)", text):
            signals["warm"] = "style"
        if re.search(r"(?:哈哈|嘿|啦|嘛|咦)", text):
            signals["playful"] = "style"
        learned: list[dict[str, Any]] = []
        for signal, category in signals.items():
            label, statement = PERSONA_SIGNAL_LABELS[signal]
            learned.append(
                self._record_persona_evidence(
                    trait_key=signal,
                    statement=statement,
                    category=f"{category}:{label}",
                    evidence_id=evidence_id,
                    explicit=False,
                )
            )
        self.distill_persona()
        return learned

    def learn_explicit_persona(
        self,
        statement: str,
        *,
        evidence_id: str,
    ) -> list[dict[str, Any]]:
        clean = str(statement or "").strip()
        if not clean:
            return []
        traits = self._persona_traits(clean) or [clean]
        learned: list[dict[str, Any]] = []
        for trait in traits:
            trait_key, category = self._persona_trait_metadata(trait)
            learned.append(
                self._record_persona_evidence(
                    trait_key=trait_key,
                    statement=trait,
                    category=category,
                    evidence_id=evidence_id,
                    explicit=True,
                )
            )
        self.distill_persona()
        return learned

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
                        (expression, scope, max(limit * 3, limit)),
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
                    (*values, scope, max(limit * 3, limit)),
                ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if (
                item.get("kind") == "preference"
                and not self._looks_like_preference_statement(
                    str(item.get("content") or "").removeprefix("用户偏好：")
                )
            ):
                continue
            result.append(item)
        return result[:limit]

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
        turn_status: str = "completed",
    ) -> list[str]:
        created: list[str] = []
        episode = _safe_text(
            f"用户：{user_input}\nSunus：{assistant_text}",
            limit=3000,
        )
        if episode:
            unresolved_project = (
                turn_status in {"partial_result", "user_interrupt", "incomplete"}
                and bool(
                    re.search(
                        r"(?:项目|代码|开发|测试|修复|实现|重构|部署|"
                        r"接口|功能|任务|todo|bug)",
                        user_input,
                        re.IGNORECASE,
                    )
                )
            )
            tags = ["conversation", "recent"]
            if unresolved_project:
                tags.append("unresolved_project")
            created.append(
                self._upsert_receipt_episode(
                    content=episode,
                    scope=scope,
                    tags=tuple(tags),
                    receipt_id=receipt_id,
                    metadata={"turn_status": turn_status},
                )
            )

        if self._looks_like_preference_statement(user_input):
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
            trait_key, category = self._persona_trait_metadata(trait)
            created.append(
                self.add(
                    content=f"Sunus 互动习惯：{trait}",
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
            self._record_persona_evidence(
                trait_key=trait_key,
                statement=trait,
                category=category,
                evidence_id=f"{receipt_id}:{trait_key}",
                explicit=True,
            )
        if persona_traits:
            self.distill_persona()
        return created

    def _upsert_receipt_episode(
        self,
        *,
        content: str,
        scope: str,
        tags: tuple[str, ...],
        receipt_id: str,
        metadata: dict[str, Any],
    ) -> str:
        clean = _safe_text(content)
        tag_text = ",".join(dict.fromkeys(tags))
        metadata_text = json.dumps(metadata, ensure_ascii=False)
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT memory_id FROM memory_evidence
                WHERE receipt_id = ? AND evidence_kind = 'receipt'
                ORDER BY id LIMIT 1
                """,
                (receipt_id,),
            ).fetchone()
            if row is None:
                return self.add(
                    content=clean,
                    scope=scope,
                    kind="episode",
                    status="supported",
                    tags=tags,
                    source="turn_receipt",
                    evidence_kind="receipt",
                    receipt_id=receipt_id,
                    metadata=metadata,
                )
            memory_id = str(row["memory_id"])
            self._connection.execute(
                """
                UPDATE memories
                SET content = ?, tags = ?, status = 'supported',
                    verified_at = ?, metadata = ?
                WHERE id = ?
                """,
                (clean, tag_text, iso(), metadata_text, memory_id),
            )
            self._connection.execute(
                "DELETE FROM memory_fts WHERE memory_id = ?",
                (memory_id,),
            )
            self._connection.execute(
                """
                INSERT INTO memory_fts(memory_id, content, tags, scope)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, clean, tag_text, scope),
            )
            return memory_id

    def record_session_summary(
        self,
        *,
        summary: str,
        scope: str,
        session_id: str,
    ) -> str | None:
        """会话收尾时落一条会话级摘要，避免进程退出后轨迹彻底消失。"""
        text = _safe_text(summary, limit=3000)
        if not text:
            return None
        return self.add(
            content=f"会话摘要[{session_id}]：{text}",
            scope=scope,
            kind="episode",
            status="supported",
            tags=("conversation", "session_summary", "recent"),
            source="session_close",
            evidence_kind="session_summary",
            metadata={"session_id": session_id},
        )

    def record_explicit_review(
        self,
        *,
        user_input: str,
        assistant_text: str,
        scope: str,
        receipt_id: str,
        turn_status: str,
        persistent: bool = True,
    ) -> str:
        content = _safe_text(
            "复盘记录：\n"
            f"用户要求：{user_input}\n"
            f"结论与结果：{assistant_text}",
            limit=4000,
        )
        tags = ["review", "explicit", "conversation"]
        if persistent:
            tags.append("persistent")
        return self._upsert_explicit_review(
            content=content,
            scope=scope,
            tags=tuple(tags),
            receipt_id=receipt_id,
            metadata={
                "turn_status": turn_status,
                "persistent": persistent,
            },
        )

    def _upsert_explicit_review(
        self,
        *,
        content: str,
        scope: str,
        tags: tuple[str, ...],
        receipt_id: str,
        metadata: dict[str, Any],
    ) -> str:
        clean = _safe_text(content)
        tag_text = ",".join(dict.fromkeys(tags))
        metadata_text = json.dumps(metadata, ensure_ascii=False)
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT memories.id, memories.created_at
                FROM memory_evidence
                JOIN memories ON memories.id = memory_evidence.memory_id
                WHERE memory_evidence.receipt_id = ?
                  AND memory_evidence.evidence_kind = 'user_request'
                  AND memories.kind = 'review'
                ORDER BY memories.created_at DESC
                """,
                (receipt_id,),
            ).fetchall()
            if not rows:
                return self.add(
                    content=clean,
                    scope=scope,
                    kind="review",
                    status="supported",
                    tags=tags,
                    source="explicit_review",
                    evidence_kind="user_request",
                    receipt_id=receipt_id,
                    metadata=metadata,
                )
            memory_id = str(rows[0]["id"])
            for duplicate in rows[1:]:
                duplicate_id = str(duplicate["id"])
                self._connection.execute(
                    "UPDATE memories SET status = 'superseded' WHERE id = ?",
                    (duplicate_id,),
                )
                self._connection.execute(
                    "DELETE FROM memory_fts WHERE memory_id = ?",
                    (duplicate_id,),
                )
            self._connection.execute(
                """
                UPDATE memories
                SET content = ?, tags = ?, status = 'supported',
                    verified_at = ?, metadata = ?
                WHERE id = ?
                """,
                (clean, tag_text, iso(), metadata_text, memory_id),
            )
            self._connection.execute(
                "DELETE FROM memory_fts WHERE memory_id = ?",
                (memory_id,),
            )
            self._connection.execute(
                """
                INSERT INTO memory_fts(memory_id, content, tags, scope)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, clean, tag_text, scope),
            )
            return memory_id

    @staticmethod
    def is_explicit_memory_request(user_input: str) -> bool:
        text = str(user_input or "").strip()
        if not text:
            return False
        if re.search(r"(?:为什么|为何|是不是|是否|怎么|如何)", text):
            return False
        return bool(
            re.search(
                r"(?:需要|要|请|给我)?记住|记下来|写入记忆|存到记忆|"
                r"落库|持久保存|长期保存|记好了|记好|记着|牢记|别忘|"
                r"不要忘|下次不要|下次别|以后记得|"
                r"复盘.{0,20}(?:记下|保存|写入|落库)",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _looks_like_preference_statement(user_input: str) -> bool:
        text = str(user_input or "").strip()
        if not text:
            return False
        explicit_patterns = (
            "我喜欢",
            "我不喜欢",
            "我偏好",
            "我更喜欢",
            "最近喜欢",
            "最近偏好",
            "喜欢",
            "以后",
            "今后",
            "长期",
            "请记住",
            "每次",
            "总是",
            "称呼我",
            "先给结论",
            "结论先行",
            "默认中文",
            "收藏",
            "爱好",
            "癖好",
            "习惯",
            "设定你的",
        )
        communication_markers = (
            "回答",
            "说话",
            "语气",
            "称呼",
            "表达",
            "沟通",
            "简洁",
            "啰嗦",
            "温柔",
            "甜",
            "卖萌",
            "撒娇",
            "主动",
            "黏人",
            "有主见",
            "独立",
        )
        question = bool(
            re.search(r"(?:多少|怎么|如何|为什么|为何|是否|能否|什么|哪)"
                      r".*[？?]?\s*$", text)
            or re.search(r"[？?]\s*$", text)
            or text.endswith(("吗", "么"))
        )
        if question and not any(
            marker in text
            for marker in ("以后", "今后", "请记住", "设定你的", "称呼我")
        ):
            return False
        if any(marker in text for marker in explicit_patterns):
            return True
        if any(marker in text for marker in communication_markers):
            return any(
                marker in text
                for marker in (
                    "别",
                    "不要",
                    "希望",
                    "喜欢",
                    "一点",
                    "更",
                    "少",
                    "多",
                )
            )
        return False

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
            "更甜",
            "别太甜",
            "卖萌",
            "撒娇",
            "更有主见",
            "别太主动",
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
        style_rules = (
            (
                "甜度",
                ("更甜", "再甜一点", "甜一点", "甜度调高", "提高甜度", "多甜"),
                ("别太甜", "不要太甜", "甜度调低", "降低甜度", "少甜"),
            ),
            (
                "卖萌度",
                ("多卖萌", "卖萌一点", "可爱一点", "多撒娇", "撒娇一点"),
                ("别卖萌", "不要卖萌", "少卖萌", "别撒娇", "不要撒娇"),
            ),
            (
                "温暖度",
                ("更温柔", "温柔一点", "更暖", "暖心一点"),
                ("别太温柔", "冷淡一点"),
            ),
            (
                "独立性",
                ("更有主见", "独立一点", "别太顺着"),
                (),
            ),
            (
                "主动性",
                ("主动一点", "多找我", "黏人一点", "小黏人"),
                ("别太主动", "别黏", "少找我"),
            ),
        )
        for label, stronger, softer in style_rules:
            if any(marker in user_input for marker in softer):
                traits.append(f"Sunus 明确降低{label}表达")
            elif any(marker in user_input for marker in stronger):
                traits.append(f"Sunus 明确提高{label}表达")
        self_setting = any(
            marker in user_input
            for marker in (
                "设定你的",
                "你的性格",
                "你的性别",
                "你的中文名",
                "以后你叫",
            )
        ) and not re.search(r"(?:吗|么|是否)[？?]?\s*$", user_input)
        if self_setting:
            gender_match = re.search(
                r"(?:性别(?:为|是)|设定你的性别为)(女性|男性|女|男)",
                user_input,
            )
            if gender_match:
                gender = (
                    "女性"
                    if gender_match.group(1) in {"女", "女性"}
                    else "男性"
                )
                traits.append(f"Sunus 核心性别设定为{gender}")
            name_match = re.search(
                r"(?:你的中文名叫|以后你叫|设定你的名字为)"
                r"([A-Za-z\u3400-\u9fff]{1,12})",
                user_input,
            )
            if name_match:
                traits.append(f"Sunus 的名字是“{name_match.group(1)}”")
            personality_markers = (
                ("温柔", "Sunus 性格温柔，表达有温度"),
                ("可爱", "Sunus 性格可爱，偶尔俏皮"),
                ("体贴", "Sunus 会体贴地关注对方感受"),
                ("才女", "Sunus 愿意认真分享知识"),
                ("博学", "Sunus 愿意认真分享知识"),
            )
            for marker, statement in personality_markers:
                if marker in user_input:
                    traits.append(statement)
        return list(dict.fromkeys(traits))

    @staticmethod
    def _persona_trait_metadata(statement: str) -> tuple[str, str]:
        mapping = (
            ("先给结论", ("conclusion_first", "communication:structure")),
            ("简洁", ("concise", "communication:style")),
            ("不主动扩大范围", ("task_bounded", "communication:boundary")),
            ("必要步骤", ("structured", "communication:structure")),
            ("默认使用中文", ("chinese", "communication:language")),
            ("称呼用户", ("address_user", "relationship:address")),
            ("核心性别", ("gender_identity", "identity:gender")),
            ("名字是", ("sunus_name", "identity:name")),
            ("性格温柔", ("warm", "identity:temperament")),
            ("性格可爱", ("playful", "identity:temperament")),
            ("体贴", ("attentive", "identity:care")),
            ("分享知识", ("knowledgeable", "identity:knowledge")),
            ("甜度", ("style_sweetness", "style:sweetness")),
            ("卖萌度", ("style_playfulness", "style:playfulness")),
            ("温暖度", ("style_warmth", "style:warmth")),
            ("独立性", ("style_independence", "style:independence")),
            ("主动性", ("style_initiative", "style:initiative")),
        )
        for marker, metadata in mapping:
            if marker in statement:
                return metadata
        return ("explicit_feedback", "communication:preference")

    def digest(
        self,
        *,
        scope: str,
        preference_limit: int = 8,
        recent_limit: int = 6,
    ) -> dict[str, Any]:
        prefs = self.recent_preferences(scope=scope, limit=32)[:preference_limit]
        recent = self.recent(scope=scope, limit=recent_limit)
        return {
            "preferences": prefs,
            "recent": recent,
            "persona": self.persona_profile(
                scope=scope,
                limit=6,
                include_candidates=True,
            ),
        }

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

