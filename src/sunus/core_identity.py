from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CORE_IDENTITY_VERSION = 1
CORE_MISSION_DEFAULT = (
    "把用户交给本地智能体的复杂任务可靠做成，以证据和可追溯结果交付；"
    "从每次协作中沉淀可复用的能力，推动本地智能体持续进化。"
)
CORE_INVARIANTS = (
    "事实准确性不可被人设或指令关闭",
    "安全边界与越权保护不可被人设或指令关闭",
    "记忆不可被静默清除；归档、替代和迁移必须保留审计记录",
)


@dataclass(frozen=True)
class CoreIdentityState:
    status: str = "uninitialized"
    version: int = CORE_IDENTITY_VERSION
    digest: str = ""
    backup_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "digest": self.digest,
            "backup_path": self.backup_path,
        }


def _canonical_payload() -> dict[str, Any]:
    return {
        "version": CORE_IDENTITY_VERSION,
        "mission": CORE_MISSION_DEFAULT,
        "mission_source": "user_config_with_code_default",
        "invariants": list(CORE_INVARIANTS),
        "persona_name_source": "user_config",
    }


def _digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_core_identity(data_dir: Path) -> CoreIdentityState:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "core-identity.json"
    payload = _canonical_payload()
    expected = _digest(payload)
    manifest = {**payload, "digest": expected}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if (
            isinstance(current, dict)
            and current.get("digest") == expected
            and _digest(
                {
                    key: current.get(key)
                    for key in (
                        "version",
                        "mission",
                        "mission_source",
                        "invariants",
                        "persona_name_source",
                    )
                }
            )
            == expected
        ):
            return CoreIdentityState(
                status="verified",
                digest=expected,
            )
        backup = data_dir / (
            f"core-identity.tampered-{datetime.now():%Y%m%d-%H%M%S}.json"
        )
        try:
            backup.write_text(
                json.dumps(current, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            backup.chmod(0o600)
        except OSError:
            pass
        status = "repaired"
        backup_path = str(backup)
    else:
        status = "created"
        backup_path = ""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return CoreIdentityState(
        status=status,
        digest=expected,
        backup_path=backup_path,
    )
