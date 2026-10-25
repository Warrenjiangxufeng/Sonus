from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sunus.memory import MemoryStore


def _records(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    values.append(record)
    except OSError:
        return []
    return values


def import_session_memory(
    memory: MemoryStore,
    *,
    data_dir: Path,
    date: str,
) -> dict[str, Any]:
    sessions_dir = data_dir.expanduser() / "sessions"
    paths = sorted(sessions_dir.glob(f"s_{date.replace('-', '')}-*.jsonl"))
    processed = 0
    memory_ids: list[str] = []
    review_ids: list[str] = []
    sessions = 0
    for path in paths:
        records = _records(path)
        if not records:
            continue
        sessions += 1
        meta = next(
            (record for record in records if record.get("type") == "meta"),
            {},
        )
        scope = str(meta.get("workspace") or "")
        if not scope:
            continue
        users = {
            str(record.get("turn_id") or ""): str(record.get("text") or "")
            for record in records
            if record.get("type") == "user"
        }
        assistant_messages: dict[str, list[str]] = {}
        for record in records:
            if record.get("type") != "message":
                continue
            message = record.get("message") or {}
            content = str(message.get("content") or "").strip()
            turn_id = str(record.get("turn_id") or "")
            if content and turn_id:
                assistant_messages.setdefault(turn_id, []).append(content)
        for receipt in (
            record for record in records if record.get("type") == "receipt"
        ):
            turn_id = str(receipt.get("turn_id") or "")
            user_input = str(
                receipt.get("user_input") or users.get(turn_id) or ""
            ).strip()
            if not user_input:
                continue
            final_text = str(receipt.get("final_text") or "").strip()
            if not final_text:
                final_text = "\n\n".join(
                    assistant_messages.get(turn_id, [])
                ).strip()
            receipt_id = str(receipt.get("receipt_id") or "")
            reason = str(receipt.get("reason") or "completed")
            for memory_id in memory.record_turn(
                user_input=user_input,
                assistant_text=final_text,
                scope=scope,
                receipt_id=receipt_id,
                turn_status=reason,
            ):
                if memory_id not in memory_ids:
                    memory_ids.append(memory_id)
            if memory.is_explicit_memory_request(user_input):
                review_id = memory.record_explicit_review(
                    user_input=user_input,
                    assistant_text=final_text,
                    scope=scope,
                    receipt_id=receipt_id,
                    turn_status=reason,
                    persistent=True,
                )
                if review_id not in review_ids:
                    review_ids.append(review_id)
            processed += 1
    return {
        "date": date,
        "sessions": sessions,
        "turns": processed,
        "memory_ids": memory_ids,
        "review_ids": review_ids,
    }
