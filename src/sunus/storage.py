from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sunus.config import redact_text, redact_value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SessionTranscript:
    session_id: str
    path: Path
    messages: list[dict[str, Any]]
    summary: str
    turns: int


def _session_roots(workspace: Path, data_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for root in (workspace / ".sunus" / "sessions", data_dir / "sessions"):
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def list_sessions(
    workspace: Path,
    data_dir: Path,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    found: dict[str, Path] = {}
    for root in _session_roots(workspace, data_dir):
        if not root.is_dir():
            continue
        for path in root.glob("s_*.jsonl"):
            found.setdefault(path.stem, path)
    rows: list[dict[str, Any]] = []
    for session_id, path in found.items():
        last_user = ""
        last_reason = ""
        turns = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("type") == "user":
                    last_user = str(record.get("text") or "")
                    turns += 1
                elif record.get("type") == "message":
                    message = record.get("message") or {}
                    if isinstance(message, dict) and message.get("role") == "user":
                        last_user = str(message.get("content") or "")
                        turns += 1
                elif record.get("type") == "receipt":
                    last_reason = str(record.get("reason") or "")
        except (OSError, ValueError):
            continue
        stat = path.stat()
        rows.append(
            {
                "session_id": session_id,
                "path": str(path),
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "turns": turns,
                "last_user": last_user[:300],
                "last_reason": last_reason,
            }
        )
    rows.sort(key=lambda item: item["modified_at"], reverse=True)
    return rows[: max(1, limit)]


def resolve_session_path(
    workspace: Path,
    data_dir: Path,
    session_id: str | None,
    *,
    last: bool = False,
) -> tuple[str, Path]:
    if session_id and not last:
        for root in _session_roots(workspace, data_dir):
            path = root / f"{session_id}.jsonl"
            if path.is_file():
                return session_id, path
        raise FileNotFoundError(f"找不到会话: {session_id}")
    rows = list_sessions(workspace, data_dir, limit=1)
    if not rows:
        raise FileNotFoundError("当前工作区没有可恢复的会话")
    return str(rows[0]["session_id"]), Path(str(rows[0]["path"]))


def load_session(path: Path) -> SessionTranscript:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(record)
    has_messages = any(
        record.get("type") in {"message", "context_checkpoint"}
        for record in records
    )
    messages: list[dict[str, Any]] = []
    summary = ""
    turns = 0
    for record in records:
        record_type = record.get("type")
        if record_type == "user":
            turns += 1
            text = str(record.get("text") or "").strip()
            if text:
                messages.append({"role": "user", "content": text})
        elif record_type == "message":
            message = record.get("message")
            if isinstance(message, dict) and message.get("role") != "system":
                messages.append(dict(message))
        elif record_type == "context_summary":
            summary = str(record.get("summary") or "")
        elif record_type == "context_checkpoint":
            summary = str(record.get("summary") or "")
            checkpoint = record.get("messages")
            if isinstance(checkpoint, list):
                messages = [
                    dict(message)
                    for message in checkpoint
                    if isinstance(message, dict)
                    and message.get("role") != "system"
                ]
        elif record_type == "receipt" and not has_messages:
            final_text = str(record.get("final_text") or "").strip()
            if final_text:
                messages.append({"role": "assistant", "content": final_text})
    return SessionTranscript(
        session_id=path.stem,
        path=path,
        messages=messages,
        summary=summary,
        turns=turns,
    )


class ArtifactStore:
    def __init__(self, workspace: Path, data_dir: Path, secret: str = "") -> None:
        self._secret = secret
        preferred = workspace / ".sunus" / "artifacts"
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            preferred.chmod(0o700)
            self.root = preferred
        except OSError:
            self.root = data_dir / "artifacts"
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                self.root.chmod(0o700)
            except OSError:
                pass

    def save(self, content: str, *, prefix: str = "tool") -> Path:
        name = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{prefix}-{uuid.uuid4().hex[:8]}.txt"
        path = self.root / name
        path.write_text(redact_text(content, self._secret), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path


class SessionLogger:
    def __init__(
        self,
        session_id: str,
        workspace: Path,
        data_dir: Path,
        secret: str = "",
        *,
        path: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self._secret = secret
        if path is not None:
            self.path = path
            self.root = path.parent
            return
        preferred = workspace / ".sunus" / "sessions"
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            preferred.chmod(0o700)
            self.root = preferred
        except OSError:
            self.root = data_dir / "sessions"
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                self.root.chmod(0o700)
            except OSError:
                pass
        self.path = self.root / f"{session_id}.jsonl"

    def append(self, record_type: str, payload: Any) -> None:
        record = {
            "type": record_type,
            "ts": utc_now(),
            **redact_value(payload, self._secret),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def receipt(
        self,
        *,
        turn_id: str,
        user_input: str,
        final_text: str,
        tool_traces: list[dict[str, Any]],
        changed_files: list[str],
        reason: str,
        metrics: dict[str, Any] | None = None,
        structured_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt = {
            "receipt_id": f"r_{uuid.uuid4().hex[:12]}",
            "turn_id": turn_id,
            "session_id": self.session_id,
            "reason": reason,
            "user_input": redact_text(user_input, self._secret)[:2000],
            "final_text": redact_text(final_text, self._secret)[:6000],
            "tools": redact_value(tool_traces, self._secret),
            "changed_files": changed_files,
            "metrics": redact_value(metrics or {}, self._secret),
            "result": redact_value(structured_result or {}, self._secret),
            "created_at": utc_now(),
        }
        self.append("receipt", receipt)
        return receipt

