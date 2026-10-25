from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAILURE_TYPES = {
    "internal_error",
    "model_error",
    "circuit_open",
    "proactive_message_failed",
}
OK_REASONS = {"completed", "partial_result", "blocked", "user_interrupt"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def _clip(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fingerprint(event: dict[str, Any]) -> str:
    text = " ".join(
        str(event.get(key) or "") for key in ("reason", "type", "summary")
    ).strip()
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b", "<n>", text)
    text = re.sub(r"\s+", " ", text)
    return _clip(text.lower(), 180) or "unknown"


def _local_time(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "?"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def recent_sessions(
    data_dir: Path,
    *,
    limit: int,
    exclude_session: str = "",
) -> list[dict[str, Any]]:
    sessions_dir = data_dir.expanduser() / "sessions"
    paths = sorted(
        sessions_dir.glob("s_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.stem == exclude_session:
            continue
        records = _read_jsonl(path)
        users = [
            str(record.get("text") or "")
            for record in records
            if record.get("type") == "user"
        ]
        receipts = [
            record for record in records if record.get("type") == "receipt"
        ]
        last_user = users[-1] if users else ""
        last_receipt = receipts[-1] if receipts else {}
        rows.append(
            {
                "session_id": path.stem,
                "modified_at": _local_time(
                    datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=timezone.utc,
                    ).isoformat()
                ),
                "turns": len(users),
                "last_user": _clip(last_user),
                "last_turn_id": str(last_receipt.get("turn_id") or ""),
                "last_reason": str(last_receipt.get("reason") or ""),
                "last_final": _clip(last_receipt.get("final_text")),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def recent_failures(
    data_dir: Path,
    *,
    days: int,
    limit: int,
) -> list[dict[str, Any]]:
    events_dir = data_dir.expanduser() / "events"
    paths = sorted(events_dir.glob("*.jsonl"))[-max(1, min(7, days)) :]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, record in enumerate(_read_jsonl(path), 1):
            kind = str(record.get("type") or "")
            reason = str(record.get("reason") or "")
            failed = kind in FAILURE_TYPES or (
                kind == "turn_end" and reason and reason not in OK_REASONS
            )
            if not failed:
                continue
            rows.append(
                {
                    "local_time": _local_time(record.get("ts")),
                    "turn_id": str(record.get("turn_id") or ""),
                    "type": kind,
                    "reason": reason,
                    "summary": _clip(
                        record.get("summary") or record.get("preview")
                    ),
                    "source": f"{path.name}:{line_number}",
                }
            )
    return rows[-limit:]


def latest_failure(data_dir: Path) -> list[dict[str, Any]]:
    events_dir = data_dir.expanduser() / "events"
    for path in sorted(events_dir.glob("*.jsonl"), reverse=True):
        records = _read_jsonl(path)
        for line_number in range(len(records), 0, -1):
            record = records[line_number - 1]
            kind = str(record.get("type") or "")
            reason = str(record.get("reason") or "")
            if kind not in FAILURE_TYPES and not (
                kind == "turn_end" and reason and reason not in OK_REASONS
            ):
                continue
            turn_id = str(record.get("turn_id") or "")
            selected = [record]
            if turn_id:
                root_errors = [
                    item
                    for item in records
                    if str(item.get("turn_id") or "") == turn_id
                    and str(item.get("type") or "") in FAILURE_TYPES
                ]
                if root_errors:
                    selected = (
                        root_errors
                        if kind in FAILURE_TYPES
                        else [*root_errors, record]
                    )
            return [
                {
                    "local_time": _local_time(item.get("ts")),
                    "turn_id": str(item.get("turn_id") or ""),
                    "type": str(item.get("type") or ""),
                    "reason": str(item.get("reason") or ""),
                    "summary": _clip(
                        item.get("summary") or item.get("preview")
                    ),
                }
                for item in selected
            ]
    return []


def failure_trend(
    data_dir: Path,
    *,
    days: int,
    limit: int,
) -> dict[str, Any]:
    events_dir = data_dir.expanduser() / "events"
    paths = sorted(events_dir.glob("*.jsonl"))[-max(1, min(365, days)) :]
    total = 0
    by_fingerprint: Counter[str] = Counter()
    by_day: Counter[str] = Counter()
    latest: dict[str, dict[str, Any]] = {}
    seen_turns: set[str] = set()
    for path in paths:
        for record in _read_jsonl(path):
            kind = str(record.get("type") or "")
            reason = str(record.get("reason") or "")
            if kind not in FAILURE_TYPES and not (
                kind == "turn_end" and reason and reason not in OK_REASONS
            ):
                continue
            turn_id = str(record.get("turn_id") or "")
            if turn_id and turn_id in seen_turns:
                continue
            if turn_id:
                seen_turns.add(turn_id)
            total += 1
            fingerprint = _fingerprint(record)
            by_fingerprint[fingerprint] += 1
            by_day[path.stem] += 1
            latest[fingerprint] = {
                "local_time": _local_time(record.get("ts")),
                "turn_id": str(record.get("turn_id") or ""),
                "summary": _clip(record.get("summary") or record.get("preview")),
            }
    return {
        "ok": bool(total),
        "mode": "trend",
        "files": len(paths),
        "total_failures": total,
        "by_day": [
            {"date": day, "count": count}
            for day, count in sorted(by_day.items())[-14:]
        ],
        "top_errors": [
            {
                "fingerprint": fingerprint,
                "count": count,
                "latest": latest.get(fingerprint, {}),
            }
            for fingerprint, count in by_fingerprint.most_common(
                max(1, min(20, limit))
            )
        ],
    }


def turn_timeline(
    data_dir: Path,
    *,
    turn_id: str,
    days: int,
    limit: int,
) -> list[dict[str, Any]]:
    events_dir = data_dir.expanduser() / "events"
    paths = sorted(events_dir.glob("*.jsonl"))[-max(1, min(7, days)) :]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, record in enumerate(_read_jsonl(path), 1):
            if str(record.get("turn_id") or "") != turn_id:
                continue
            row = {
                "local_time": _local_time(record.get("ts")),
                "type": str(record.get("type") or ""),
                "summary": _clip(
                    record.get("summary")
                    or record.get("preview")
                    or record.get("name")
                ),
                "source": f"{path.name}:{line_number}",
            }
            if record.get("reason"):
                row["reason"] = str(record["reason"])
            if record.get("traceback"):
                row["traceback"] = str(record["traceback"])[:4000]
            rows.append(row)
    return rows[-limit:]


def inspect_runtime_history(
    data_dir: Path,
    *,
    mode: str,
    limit: int = 10,
    days: int = 2,
    turn_id: str = "",
    exclude_session: str = "",
) -> dict[str, Any]:
    bounded_limit = max(1, min(50, int(limit)))
    if mode == "recent":
        rows = recent_sessions(
            data_dir,
            limit=bounded_limit,
            exclude_session=exclude_session,
        )
    elif mode == "failures":
        rows = recent_failures(data_dir, days=days, limit=bounded_limit)
    elif mode == "latest_failure":
        rows = latest_failure(data_dir)
    elif mode == "trend":
        return failure_trend(
            data_dir,
            days=days,
            limit=bounded_limit,
        )
    elif mode == "turn":
        if not turn_id:
            return {"ok": False, "error": "mode=turn 时必须提供 turn_id"}
        rows = turn_timeline(
            data_dir,
            turn_id=turn_id,
            days=days,
            limit=bounded_limit,
        )
    else:
        return {"ok": False, "error": f"未知 mode: {mode}"}
    return {
        "ok": bool(rows),
        "mode": mode,
        "returned": len(rows),
        "rows": rows,
    }
