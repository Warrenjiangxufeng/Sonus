"""Bounded host telemetry used by Sunus's digital body model.

These signals describe the local runtime, not biological facts. They are kept
small and deterministic so tests and offline runs remain reproducible.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HostTelemetry:
    sampled_at: float
    local_hour: float
    load_ratio: float
    disk_used_ratio: float
    observed: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampled_at": self.sampled_at,
            "local_hour": round(self.local_hour, 2),
            "load_ratio": round(self.load_ratio, 4),
            "disk_used_ratio": round(self.disk_used_ratio, 4),
            "observed": list(self.observed),
        }


def sample_host_telemetry(*, now: float | None = None) -> HostTelemetry:
    current = time.time() if now is None else float(now)
    observed: list[str] = []
    try:
        load_ratio = float(os.getloadavg()[0]) / max(1, os.cpu_count() or 1)
    except (AttributeError, OSError):
        load_ratio = 0.0
    try:
        usage = shutil.disk_usage(os.path.expanduser("~"))
        disk_used_ratio = usage.used / max(1, usage.total)
    except OSError:
        disk_used_ratio = 0.0
    local_hour = (
        datetime.fromtimestamp(current).astimezone().hour
        + datetime.fromtimestamp(current).astimezone().minute / 60.0
    )
    if load_ratio >= 0.8:
        observed.append("host_load_high")
    if disk_used_ratio >= 0.9:
        observed.append("disk_pressure")
    if local_hour < 6 or local_hour >= 23:
        observed.append("late_hours")
    return HostTelemetry(
        sampled_at=current,
        local_hour=local_hour,
        load_ratio=load_ratio,
        disk_used_ratio=disk_used_ratio,
        observed=tuple(observed),
    )


def body_effects(
    telemetry: HostTelemetry,
    *,
    elapsed_hours: float,
    interactions_last_hour: int,
) -> dict[str, float]:
    """Translate runtime facts into small deltas for interoceptive variables."""
    effects = {
        "energy": 0.0,
        "fatigue": 0.0,
        "tension": 0.0,
        "comfort": 0.0,
        "safety": 0.0,
        "arousal": 0.0,
    }
    if elapsed_hours >= 2.0:
        recovery = min(0.08, 0.018 * elapsed_hours)
        effects["energy"] += recovery
        effects["fatigue"] -= recovery
        effects["tension"] -= min(0.05, recovery * 0.6)
    if "late_hours" in telemetry.observed:
        effects["fatigue"] += 0.025
        effects["energy"] -= 0.015
    if "host_load_high" in telemetry.observed:
        effects["tension"] += 0.04
        effects["arousal"] += 0.02
        effects["comfort"] -= 0.02
    if "disk_pressure" in telemetry.observed:
        effects["tension"] += 0.02
        effects["safety"] -= 0.02
    if interactions_last_hour >= 12:
        effects["fatigue"] += 0.03
        effects["tension"] += 0.02
    return effects
