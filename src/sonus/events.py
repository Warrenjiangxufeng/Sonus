from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sonus.config import redact_value


class EventBus:
    def __init__(
        self,
        data_dir: Path,
        *,
        secret: str = "",
        verbose: bool = False,
        echo_json: bool = False,
        renderer: Callable[[dict[str, Any]], None] | None = None,
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._dir = data_dir / "events"
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            self._dir.chmod(0o700)
        except OSError:
            pass
        self._secret = secret
        self._verbose = verbose
        self._echo_json = echo_json
        self._renderer = renderer
        self._observer = observer
        self._lock = threading.Lock()
        self._path = self._dir / (
            datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl"
        )

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event_type: str, **data: Any) -> dict[str, Any]:
        event = {
            "type": event_type,
            "ts": datetime.now(timezone.utc).isoformat(),
            **redact_value(data, self._secret),
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            try:
                self._path.chmod(0o600)
            except OSError:
                pass
            if self._observer is not None:
                self._observer(event)
            if self._echo_json:
                print(line, file=sys.stderr, flush=True)
            elif self._renderer is not None:
                self._renderer(event)
            elif self._verbose:
                summary = event.get("summary") or event.get("reason") or ""
                print(
                    f"[sonus:{event_type}] {str(summary)[:160]}",
                    file=sys.stderr,
                    flush=True,
                )
        return event

