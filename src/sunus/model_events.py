from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelEvent:
    type: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
