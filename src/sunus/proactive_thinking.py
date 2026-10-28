"""Persistent active-thought ledger for Sunus proactive companionship."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sunus.config import ProactiveConfig


THOUGHT_STATUSES = frozenset(
    {"incubating", "waiting", "ready", "dormant", "resolved", "dropped"}
)
THOUGHT_ACTIONS = frozenset(
    {"message", "verify", "review", "maintenance", "do_nothing"}
)
_NEGATIVE_RESPONSE = (
    "别烦",
    "不想聊",
    "别问",
    "不是这个",
    "不对",
    "跑偏",
    "废话",
    "安静",
)
_ACCEPTED_RESPONSE = (
    "谢谢",
    "对的",
    "明白了",
    "懂了",
    "可以",
    "好多了",
    "就是这个",
)


def _iso(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat()


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def classify_response_text(text: str) -> str:
    value = " ".join(str(text or "").split())
    if not value:
        return "neutral"
    if any(cue in value for cue in _NEGATIVE_RESPONSE):
        return "negative"
    if any(cue in value for cue in _ACCEPTED_RESPONSE):
        return "accepted"
    return "continued"


@dataclass(frozen=True)
class ThoughtSeed:
    question: str
    source_kind: str
    source_ref: str
    confidence: float
    importance: float
    action_hint: str = "message"
    prediction: str = ""
    evidence_refs: tuple[str, ...] = ()
    subject_key: str = ""
    expires_in_sec: int = 72 * 3600


@dataclass(frozen=True)
class ProactiveThought:
    id: str
    subject_key: str
    question: str
    source_kind: str
    source_ref: str
    evidence_refs: tuple[str, ...]
    confidence: float
    importance: float
    action_hint: str
    prediction: str
    status: str
    not_before: float
    expires_at: float
    attempts: int
    max_attempts: int
    last_outcome: str
    prediction_match: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_key": self.subject_key,
            "question": self.question,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "evidence_refs": list(self.evidence_refs),
            "confidence": round(self.confidence, 3),
            "importance": round(self.importance, 3),
            "action_hint": self.action_hint,
            "prediction": self.prediction,
            "status": self.status,
            "not_before": self.not_before,
            "expires_at": self.expires_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_outcome": self.last_outcome,
            "prediction_match": self.prediction_match,
        }


@dataclass(frozen=True)
class ThoughtDecision:
    action: str
    thought: ProactiveThought | None
    score: float
    wait_seconds: int
    reason: str


class ProactiveThoughtStore:
    """SQLite-backed open-question store with cooldown and outcome learning."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        config: ProactiveConfig,
        *,
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.connection = connection
        self.session_id = session_id
        self.config = config
        self.clock = clock
        self._initialize()

    def _now(self) -> float:
        return self.clock().timestamp()

    def _initialize(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS proactive_thoughts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    question TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0,
                    importance REAL NOT NULL DEFAULT 0,
                    action_hint TEXT NOT NULL DEFAULT 'message',
                    prediction TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'incubating',
                    not_before REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    last_outcome TEXT NOT NULL DEFAULT '',
                    prediction_match TEXT NOT NULL DEFAULT '',
                    last_evidence_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_proactive_thoughts_due
                    ON proactive_thoughts(session_id, status, not_before);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_proactive_thoughts_open
                    ON proactive_thoughts(session_id, subject_key, status)
                    WHERE status IN ('incubating', 'waiting', 'ready');
                """
            )
            columns = {
                str(row["name"])
                for row in self.connection.execute(
                    "PRAGMA table_info(proactive_thoughts)"
                ).fetchall()
            }
            if "prediction" not in columns:
                self.connection.execute(
                    "ALTER TABLE proactive_thoughts ADD COLUMN "
                    "prediction TEXT NOT NULL DEFAULT ''"
                )
            if "prediction_match" not in columns:
                self.connection.execute(
                    "ALTER TABLE proactive_thoughts ADD COLUMN "
                    "prediction_match TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def _row(row: sqlite3.Row) -> ProactiveThought:
        try:
            evidence = tuple(
                str(item)
                for item in json.loads(str(row["evidence_json"] or "[]"))
            )
        except (TypeError, json.JSONDecodeError):
            evidence = ()
        return ProactiveThought(
            id=str(row["id"]),
            subject_key=str(row["subject_key"]),
            question=str(row["question"]),
            source_kind=str(row["source_kind"]),
            source_ref=str(row["source_ref"]),
            evidence_refs=evidence,
            confidence=max(0.0, min(1.0, float(row["confidence"]))),
            importance=max(0.0, min(1.0, float(row["importance"]))),
            action_hint=str(row["action_hint"]),
            prediction=str(row["prediction"] or ""),
            status=str(row["status"]),
            not_before=float(row["not_before"]),
            expires_at=float(row["expires_at"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            last_outcome=str(row["last_outcome"] or ""),
            prediction_match=str(row["prediction_match"] or ""),
        )

    def upsert(self, seed: ThoughtSeed) -> ProactiveThought:
        now = self._now()
        question = _clean(seed.question, 500)
        if not question:
            raise ValueError("thought question 不能为空")
        source_kind = _clean(seed.source_kind, 80) or "unknown"
        source_ref = _clean(seed.source_ref, 300)
        subject_key = _clean(seed.subject_key, 200) or hashlib.sha256(
            f"{source_kind}|{source_ref}|{question}".encode("utf-8")
        ).hexdigest()[:24]
        evidence = tuple(
            dict.fromkeys(
                _clean(item, 200)
                for item in seed.evidence_refs
                if _clean(item, 200)
            )
        )
        evidence_json = json.dumps(evidence, ensure_ascii=False)
        evidence_hash = hashlib.sha256(
            evidence_json.encode("utf-8")
        ).hexdigest()[:20]
        action = (
            seed.action_hint
            if seed.action_hint in THOUGHT_ACTIONS
            else "message"
        )
        with self.connection:
            existing = self.connection.execute(
                """
                SELECT * FROM proactive_thoughts
                WHERE session_id = ? AND subject_key = ?
                  AND status IN ('incubating', 'waiting', 'ready')
                ORDER BY created_at DESC LIMIT 1
                """,
                (self.session_id, subject_key),
            ).fetchone()
            if existing is not None:
                merged = tuple(
                    dict.fromkeys(
                        (
                            *self._row(existing).evidence_refs,
                            *evidence,
                        )
                    )
                )
                merged_json = json.dumps(merged, ensure_ascii=False)
                self.connection.execute(
                    """
                    UPDATE proactive_thoughts
                    SET evidence_json = ?,
                        confidence = MAX(confidence, ?),
                        importance = MAX(importance, ?),
                        action_hint = ?,
                        prediction = ?,
                        expires_at = MAX(expires_at, ?),
                        last_evidence_hash = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged_json,
                        max(0.0, min(1.0, float(seed.confidence))),
                        max(0.0, min(1.0, float(seed.importance))),
                        action,
                        _clean(seed.prediction, 240),
                        now + max(300, int(seed.expires_in_sec)),
                        hashlib.sha256(
                            merged_json.encode("utf-8")
                        ).hexdigest()[:20],
                        _iso(now),
                        str(existing["id"]),
                    ),
                )
                row = self.connection.execute(
                    "SELECT * FROM proactive_thoughts WHERE id = ?",
                    (str(existing["id"]),),
                ).fetchone()
                return self._row(row)

            open_count = int(
                self.connection.execute(
                    """
                    SELECT COUNT(*) FROM proactive_thoughts
                    WHERE session_id = ?
                      AND status IN ('incubating', 'waiting', 'ready')
                    """,
                    (self.session_id,),
                ).fetchone()[0]
            )
            if open_count >= self.config.thought_max_open:
                self.connection.execute(
                    """
                    UPDATE proactive_thoughts
                    SET status = 'dormant', updated_at = ?
                    WHERE id = (
                        SELECT id FROM proactive_thoughts
                        WHERE session_id = ?
                          AND status IN ('incubating', 'waiting', 'ready')
                        ORDER BY importance ASC, attempts DESC, updated_at ASC
                        LIMIT 1
                    )
                    """,
                    (_iso(now), self.session_id),
                )
            thought_id = f"th_{uuid.uuid4().hex[:12]}"
            self.connection.execute(
                """
                INSERT INTO proactive_thoughts(
                    id, session_id, subject_key, question, source_kind,
                    source_ref, evidence_json, confidence, importance,
                    action_hint, prediction, status, not_before, expires_at,
                    attempts, max_attempts, last_outcome,
                    prediction_match, last_evidence_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'incubating',
                          ?, ?, 0, 3, '', '', ?, ?, ?)
                """,
                (
                    thought_id,
                    self.session_id,
                    subject_key,
                    question,
                    source_kind,
                    source_ref,
                    evidence_json,
                    max(0.0, min(1.0, float(seed.confidence))),
                    max(0.0, min(1.0, float(seed.importance))),
                    action,
                    _clean(seed.prediction, 240),
                    now,
                    now + max(300, int(seed.expires_in_sec)),
                    evidence_hash,
                    _iso(now),
                    _iso(now),
                ),
            )
            row = self.connection.execute(
                "SELECT * FROM proactive_thoughts WHERE id = ?",
                (thought_id,),
            ).fetchone()
            return self._row(row)

    def open_thoughts(self, *, limit: int = 20) -> list[ProactiveThought]:
        rows = self.connection.execute(
            """
            SELECT * FROM proactive_thoughts
            WHERE session_id = ?
              AND status IN ('incubating', 'waiting', 'ready')
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            (self.session_id, max(1, limit)),
        ).fetchall()
        return [self._row(row) for row in rows]

    def recent_thoughts(self, *, limit: int = 10) -> list[ProactiveThought]:
        rows = self.connection.execute(
            """
            SELECT * FROM proactive_thoughts
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (self.session_id, max(1, limit)),
        ).fetchall()
        return [self._row(row) for row in rows]

    def expire_dormant(self) -> int:
        now = self._now()
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE proactive_thoughts
                SET status = 'dormant', updated_at = ?
                WHERE session_id = ?
                  AND status IN ('incubating', 'waiting', 'ready')
                  AND expires_at <= ?
                """,
                (_iso(now), self.session_id, now),
            )
        return max(0, cursor.rowcount)

    def defer(self, thought_id: str, seconds: int, reason: str = "") -> None:
        now = self._now()
        with self.connection:
            self.connection.execute(
                """
                UPDATE proactive_thoughts
                SET status = 'waiting', not_before = ?,
                    last_outcome = ?, updated_at = ?
                WHERE id = ? AND session_id = ?
                """,
                (
                    now + max(30, int(seconds)),
                    _clean(reason, 120),
                    _iso(now),
                    thought_id,
                    self.session_id,
                ),
            )

    def resolve(self, thought_id: str, outcome: str = "resolved") -> None:
        now = self._now()
        status = "resolved" if outcome == "resolved" else "dropped"
        with self.connection:
            self.connection.execute(
                """
                UPDATE proactive_thoughts
                SET status = ?, last_outcome = ?, updated_at = ?
                WHERE id = ? AND session_id = ?
                """,
                (status, _clean(outcome, 120), _iso(now), thought_id, self.session_id),
            )

    def record_delivery(self, thought_id: str) -> None:
        now = self._now()
        with self.connection:
            row = self.connection.execute(
                """
                SELECT attempts, importance FROM proactive_thoughts
                WHERE id = ? AND session_id = ?
                """,
                (thought_id, self.session_id),
            ).fetchone()
            if row is None:
                return
            attempts = int(row["attempts"]) + 1
            cooldown = self.config.thought_cooldown_sec * (2 ** min(4, attempts))
            importance = float(row["importance"])
            status = (
                "dormant"
                if attempts >= 3
                else "waiting"
            )
            self.connection.execute(
                """
                UPDATE proactive_thoughts
                SET attempts = ?, status = ?, not_before = ?,
                    importance = ?, last_outcome = 'pending',
                    updated_at = ?
                WHERE id = ? AND session_id = ?
                """,
                (
                    attempts,
                    status,
                    now + cooldown,
                    max(0.15, importance * 0.92),
                    _iso(now),
                    thought_id,
                    self.session_id,
                ),
            )

    def refresh_outcomes(self) -> list[str]:
        """Resolve neutral no-response signals without treating silence as rejection."""
        now = self._now()
        cutoff = now - self.config.thought_no_response_sec
        updated: list[str] = []
        rows = self.connection.execute(
            """
            SELECT m.id, m.thought_id, m.created_at
            FROM proactive_messages m
            WHERE m.session_id = ?
              AND m.channel = 'social'
              AND m.thought_id != ''
              AND m.response_outcome = ''
              AND m.created_at <= ?
            ORDER BY m.created_at DESC
            LIMIT 8
            """,
            (
                self.session_id,
                datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
            ),
        ).fetchall()
        with self.connection:
            for row in rows:
                thought_id = str(row["thought_id"])
                self.connection.execute(
                    """
                    UPDATE proactive_messages
                    SET response_outcome = 'no_response', responded_at = ?
                    WHERE id = ?
                    """,
                    (now, str(row["id"])),
                )
                self.connection.execute(
                    """
                    UPDATE proactive_thoughts
                    SET importance = MAX(0.1, importance * 0.8),
                        last_outcome = 'no_response',
                        prediction_match = CASE
                            WHEN prediction LIKE '%低负担%'
                              OR prediction LIKE '%不要求回复%'
                            THEN 'matched'
                            ELSE 'uncertain'
                        END,
                        updated_at = ?
                    WHERE id = ? AND session_id = ?
                      AND status NOT IN ('dormant', 'resolved', 'dropped')
                    """,
                    (_iso(now), thought_id, self.session_id),
                )
                updated.append(thought_id)
        return updated

    def observe_response(self, outcome: str) -> str | None:
        now = self._now()
        row = self.connection.execute(
            """
            SELECT id, thought_id, created_at FROM proactive_messages
            WHERE session_id = ? AND channel = 'social'
              AND response_outcome = ''
            ORDER BY created_at DESC LIMIT 1
            """,
            (self.session_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            sent_at = datetime.fromisoformat(
                str(row["created_at"]).replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            sent_at = now
        effective_outcome = (
            outcome
            if now - sent_at <= self.config.thought_no_response_sec
            else "late"
        )
        thought_id = str(row["thought_id"] or "")
        with self.connection:
            self.connection.execute(
                """
                UPDATE proactive_messages
                SET response_outcome = ?, responded_at = ?
                WHERE id = ?
                """,
                (_clean(effective_outcome, 80), now, str(row["id"])),
            )
            if thought_id:
                delta = {
                    "continued": 0.08,
                    "accepted": 0.12,
                    "negative": -0.20,
                    "neutral": 0.0,
                }.get(effective_outcome, 0.0)
                prediction_match = (
                    "matched"
                    if effective_outcome in {"accepted", "continued"}
                    else "mismatched"
                    if effective_outcome == "negative"
                    else "uncertain"
                )
                self.connection.execute(
                    """
                    UPDATE proactive_thoughts
                    SET importance = MAX(0.1, MIN(1.0, importance + ?)),
                        last_outcome = ?, prediction_match = ?,
                        updated_at = ?
                    WHERE id = ? AND session_id = ?
                      AND status NOT IN ('resolved', 'dropped')
                    """,
                    (
                        delta,
                        effective_outcome,
                        prediction_match,
                        _iso(now),
                        thought_id,
                        self.session_id,
                    ),
                )
        return thought_id or None

    def select(self, *, daypart_changed: bool = False) -> ThoughtDecision:
        now = self._now()
        self.expire_dormant()
        self.refresh_outcomes()
        rows = self.connection.execute(
            """
            SELECT * FROM proactive_thoughts
            WHERE session_id = ?
              AND status IN ('incubating', 'waiting', 'ready')
              AND not_before <= ?
            ORDER BY importance DESC, updated_at DESC
            LIMIT 20
            """,
            (self.session_id, now),
        ).fetchall()
        if not rows:
            return ThoughtDecision(
                action="do_nothing",
                thought=None,
                score=0.0,
                wait_seconds=min(900, self.config.thought_cooldown_sec),
                reason="no_ready_thought",
            )
        ranked: list[tuple[float, ProactiveThought]] = []
        for row in rows:
            thought = self._row(row)
            age_hours = max(
                0.0,
                now - (
                    datetime.fromisoformat(
                        str(row["created_at"]).replace("Z", "+00:00")
                    ).timestamp()
                ),
            ) / 3600
            freshness = 1.0 / (1.0 + age_hours / 24.0)
            source_weight = {
                "repair": 0.16,
                "task": 0.25,
                "project": 0.14,
                "curiosity": 0.10,
                "daypart_review": 0.04,
            }.get(thought.source_kind, 0.06)
            score = (
                0.35 * thought.importance
                + 0.15 * thought.confidence
                + 0.15 * (1.0 - thought.confidence)
                + 0.15 * freshness
                + source_weight
                + (0.04 if daypart_changed else 0.0)
                - 0.10 * thought.attempts
            )
            if thought.last_outcome == "no_response":
                score -= 0.12
            ranked.append((score, thought))
        score, thought = max(ranked, key=lambda item: item[0])
        threshold = self.config.thought_min_score
        if score < threshold:
            return ThoughtDecision(
                action="do_nothing",
                thought=thought,
                score=score,
                wait_seconds=min(900, self.config.thought_cooldown_sec),
                reason="score_below_threshold",
            )
        action = thought.action_hint
        if action == "verify":
            if thought.last_outcome == "verify_before_speaking":
                return ThoughtDecision(
                    action="message",
                    thought=thought,
                    score=score,
                    wait_seconds=0,
                    reason="verification_window_elapsed",
                )
            return ThoughtDecision(
                action="verify",
                thought=thought,
                score=score,
                wait_seconds=min(1800, self.config.thought_cooldown_sec),
                reason="verify_before_speaking",
            )
        if action in {"review", "maintenance"}:
            return ThoughtDecision(
                action=action,
                thought=thought,
                score=score,
                wait_seconds=min(1800, self.config.thought_cooldown_sec),
                reason="internal_only",
            )
        return ThoughtDecision(
            action="message",
            thought=thought,
            score=score,
            wait_seconds=0,
            reason="thought_ready",
        )

    def stats(self) -> dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM proactive_thoughts
            WHERE session_id = ?
            GROUP BY status
            """,
            (self.session_id,),
        ).fetchall()
        return {
            str(row["status"]): int(row["count"])
            for row in rows
        }
