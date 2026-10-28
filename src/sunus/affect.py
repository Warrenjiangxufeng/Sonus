# -*- coding: utf-8 -*-
"""Sunus 的独立持久化内感受、情绪与欲望状态。"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sunus.body_runtime import (
    HostTelemetry,
    body_effects,
    sample_host_telemetry,
)
from sunus.embodiment import BODY_ONTOLOGY


VERSION = 1
DEFAULT_STATE_PATH = (
    Path.home() / ".sunus" / "affect_state.json"
)

BODY_BASELINE = {
    "energy": 0.64,
    "fatigue": 0.28,
    "tension": 0.16,
    "comfort": 0.62,
    "safety": 0.67,
    "closeness": 0.50,
    "warmth": 0.54,
    "arousal": 0.34,
    "sexual_arousal": 0.09,
}
BODY_TAU_HOURS = {
    "energy": 18.0,
    "fatigue": 7.0,
    "tension": 2.0,
    "comfort": 8.0,
    "safety": 12.0,
    "closeness": 24.0,
    "warmth": 8.0,
    "arousal": 2.0,
    "sexual_arousal": 1.0,
}
EMOTIONS = (
    "joy",
    "affection",
    "care",
    "sadness",
    "anger",
    "fear",
    "anxiety",
    "shame",
    "curiosity",
    "surprise",
    "calm",
    "longing",
)
DESIRES = (
    "closeness",
    "distance",
    "care",
    "repair",
    "rest",
    "explore",
    "understanding",
    "bonding",
    "recognition",
    "autonomy",
    "sexual",
    "play",
)
EMOTION_LABELS = {
    "joy": "喜悦",
    "affection": "喜欢/依恋",
    "care": "关心",
    "sadness": "悲伤",
    "anger": "愤怒",
    "fear": "恐惧",
    "anxiety": "焦虑",
    "shame": "羞耻",
    "curiosity": "好奇",
    "surprise": "惊讶",
    "calm": "平静",
    "longing": "思念",
}
DESIRE_LABELS = {
    "closeness": "亲近",
    "distance": "拉开距离",
    "care": "照顾",
    "repair": "修复",
    "rest": "休息",
    "explore": "探索",
    "understanding": "想了解",
    "bonding": "关系推进",
    "recognition": "被看见",
    "autonomy": "自主",
    "sexual": "性亲密",
    "play": "玩闹",
}
EMOTION_EXPRESSIONS = {
    "joy": "分享、语气轻快",
    "affection": "靠近、放软",
    "care": "接住情绪、照顾",
    "sadness": "放慢、缩短或想被安慰",
    "anger": "直接、划界",
    "fear": "谨慎、避险",
    "anxiety": "确认、寻求稳定",
    "shame": "收缩、少说、修复",
    "curiosity": "探索、想问或查证",
    "surprise": "停顿、重新判断",
    "calm": "稳住、倾听",
    "longing": "靠近、表达想念",
}
DESIRE_EXPRESSIONS = {
    "closeness": "缩短距离、语气放软",
    "distance": "减少回应、拉开空间",
    "care": "陪伴、具体关心",
    "repair": "承认影响、修复",
    "rest": "收束、暂停消耗",
    "explore": "探索、查证",
    "understanding": "问具体问题或先观察",
    "bonding": "深化关系、适度自我透露",
    "recognition": "表明自己的感受和价值",
    "autonomy": "保留选择、设边界",
    "sexual": "亲密或浪漫表达，但不施压",
    "play": "调侃、玩闹",
}

SUPPORT_CUES = (
    "别担心",
    "不用担心",
    "没事",
    "我在",
    "放心",
    "别怕",
    "慢慢来",
    "不怪你",
)
REPAIR_CUES = ("对不起", "抱歉", "我错了", "误会了", "原谅我")
WARMTH_CUES = (
    "喜欢你",
    "爱你",
    "想你",
    "抱抱",
    "亲亲",
    "贴贴",
    "谢谢你",
    "辛苦了",
    "真棒",
    "做得好",
    "在乎你",
    "陪你",
    "温柔",
)
DISTRESS_CUES = (
    "难过",
    "伤心",
    "想哭",
    "崩溃",
    "委屈",
    "不开心",
    "焦虑",
    "担心",
    "害怕",
    "好烦",
)
JOY_CUES = ("开心", "高兴", "太好了", "兴奋", "成功了", "笑死")
FATIGUE_CUES = ("好累", "累死", "累了", "好困", "疲惫", "没力气", "熬夜", "撑不住")
REST_CUES = ("休息", "睡觉", "晚安", "放松", "别太累")
INSULT_CUES = (
    "滚开",
    "滚蛋",
    "你给我滚",
    "闭嘴",
    "讨厌你",
    "你真烦",
    "你很烦",
    "废物",
    "真蠢",
    "太蠢",
    "蠢货",
    "真没用",
    "笨死",
    "不想理你",
    "离开你",
    "分手",
    "我恨你",
)
BOUNDARY_CUES = (
    "不要碰",
    "别碰",
    "别靠近",
    "离我远点",
    "停下",
    "不要这样",
    "别这样",
    "我不愿意",
)
REJECTION_CUES = (
    "没兴趣",
    "不想聊",
    "不想说",
    "别烦",
    "不关你的事",
)
QUESTION_FATIGUE_CUES = (
    "别问",
    "不要问",
    "问什么问",
    "问啥",
    "有啥问",
    "天天问",
    "别追问",
)
CURIOSITY_CUES = ("为什么", "怎么", "好奇", "想知道", "告诉我", "是什么")
INTIMACY_CUES = ("想见你", "亲密", "想要你", "性欲", "做爱", "亲吻")
AFFECT_EVENT_LABELS = (
    "安抚",
    "修复",
    "亲近",
    "对方低落",
    "共享喜悦",
    "对方疲惫",
    "休息暗示",
    "关系冲击",
    "边界",
    "疏离",
    "追问疲劳",
    "探索",
    "亲密唤起",
)
RESPONSE_FEEDBACK_LABELS = (
    "回应被接纳",
    "回应被纠正",
    "对方缓和",
    "对方仍不安",
    "对方拉开距离",
    "对方继续投入",
)
RELATIONSHIP_SIGNALS = (
    "warm",
    "neutral",
    "uncertain",
    "strained",
    "boundary",
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _now_iso(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat()


def _parse_iso(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError, OSError):
        return None


def _has_unnegated(text: str, cue: str) -> bool:
    start = 0
    while True:
        index = text.find(cue, start)
        if index < 0:
            return False
        prefix = text[max(0, index - 3) : index]
        if not any(neg in prefix for neg in ("不", "没", "别", "无", "未")):
            return True
        start = index + len(cue)


def _any(text: str, cues: tuple[str, ...]) -> bool:
    return any(_has_unnegated(text, cue) for cue in cues)


def lexical_events(text: str) -> list[str]:
    clean = " ".join(text.lower().split())
    protective_text = clean
    for phrase in (
        "不想让你担心",
        "不想你难过",
        "不想你伤心",
        "不想离开你",
    ):
        protective_text = protective_text.replace(phrase, "")
    events: list[str] = []
    if _any(clean, SUPPORT_CUES):
        events.append("安抚")
    if _any(clean, REPAIR_CUES):
        events.append("修复")
    if _any(clean, WARMTH_CUES):
        events.append("亲近")
    if _any(clean, DISTRESS_CUES):
        events.append("对方低落")
    if _any(clean, JOY_CUES):
        events.append("共享喜悦")
    if _any(clean, FATIGUE_CUES):
        events.append("对方疲惫")
    if _any(clean, REST_CUES):
        events.append("休息暗示")
    if _any(protective_text, INSULT_CUES):
        events.append("关系冲击")
    if _any(clean, BOUNDARY_CUES):
        events.append("边界")
    if _any(protective_text, REJECTION_CUES):
        events.append("疏离")
    if _any(clean, QUESTION_FATIGUE_CUES):
        events.append("追问疲劳")
    if _any(clean, CURIOSITY_CUES):
        events.append("探索")
    if _any(clean, INTIMACY_CUES):
        events.append("亲密唤起")
    if "追问疲劳" in events:
        events = [event for event in events if event != "探索"]
    if "边界" in events or "关系冲击" in events:
        events = [event for event in events if event != "亲密唤起"]
    return events


def _empty_body_channels(now: float) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for channel in BODY_ONTOLOGY.channels:
        channels[channel] = {
            "active": True,
            "updated_at": _now_iso(now),
            "observations": [],
        }
    channels["host_sensory_input"]["input_density"] = 0.0
    channels["proprioceptive_action_state"].update(
        {
            "status": "ready",
            "last_actions": [],
            "failure_count": 0,
        }
    )
    channels["expression_and_voice"].update(
        {
            "mode": "text",
            "last_length": 0,
            "pace": "steady",
        }
    )
    channels["relationship_and_boundaries"].update(
        {
            "signal": "neutral",
            "boundary_active": False,
        }
    )
    return channels


def _empty_state(now: float) -> dict[str, Any]:
    return {
        "version": VERSION,
        "updated_at": _now_iso(now),
        "interaction_count": 0,
        "body": dict(BODY_BASELINE),
        "emotions": {name: 0.0 for name in EMOTIONS},
        "desires": {name: 0.0 for name in DESIRES},
        "question_inhibition_until": None,
        "last_appraisal": {"events": [], "source": "lexical"},
        "last_perception": lexical_perception("").to_dict(),
        "response_feedback": [],
        "body_channels": _empty_body_channels(now),
        "host_telemetry": sample_host_telemetry(now=now).to_dict(),
        "interaction_times": [],
        "recent_events": [],
    }


def _decayed(value: float, baseline: float, elapsed: float, tau: float) -> float:
    if elapsed <= 0 or tau <= 0:
        return value
    return baseline + (value - baseline) * math.exp(-elapsed / tau)


@dataclass(frozen=True)
class SunusSnapshot:
    updated_at: str
    interaction_count: int
    body: dict[str, float]
    emotions: dict[str, float]
    desires: dict[str, float]
    question_readiness: float
    questions_inhibited: bool
    events: tuple[str, ...]
    perception: dict[str, Any]
    body_channels: dict[str, Any]
    action_intents: tuple[dict[str, Any], ...] = ()
    response_feedback: tuple[str, ...] = ()
    host_telemetry: dict[str, Any] | None = None


@dataclass(frozen=True)
class Perception:
    observed_cues: tuple[str, ...]
    inferred_user_state: dict[str, float]
    alternative_explanations: tuple[str, ...]
    relationship_signal: str
    affect_events: tuple[str, ...]
    response_feedback: tuple[str, ...]
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_cues": list(self.observed_cues),
            "inferred_user_state": dict(self.inferred_user_state),
            "alternative_explanations": list(
                self.alternative_explanations
            ),
            "relationship_signal": self.relationship_signal,
            "affect_events": list(self.affect_events),
            "response_feedback": list(self.response_feedback),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


def lexical_response_feedback(text: str) -> list[str]:
    clean = " ".join(str(text or "").lower().split())
    feedback: list[str] = []
    if any(
        cue in clean
        for cue in (
            "对了",
            "就是这个",
            "明白了",
            "懂了",
            "可以了",
            "解决了",
            "成功了",
            "没问题",
            "这样就行",
            "有用",
        )
    ):
        feedback.append("回应被接纳")
    if any(
        cue in clean
        for cue in (
            "不对",
            "错了",
            "你没懂",
            "没听懂",
            "理解错",
            "跑偏",
            "答非所问",
            "我说的是",
            "不是让你",
        )
    ):
        feedback.append("回应被纠正")
    if any(cue in clean for cue in ("好多了", "舒服多了", "没那么难受", "不难受了")):
        feedback.append("对方缓和")
    if any(cue in clean for cue in ("还是难受", "还是不行", "仍然", "没解决", "更烦")):
        feedback.append("对方仍不安")
    if "边界" in lexical_events(clean) or "疏离" in lexical_events(clean):
        feedback.append("对方拉开距离")
    if any(cue in clean for cue in ("继续", "接着", "然后", "下一步", "再看")):
        feedback.append("对方继续投入")
    return list(dict.fromkeys(feedback))


def lexical_perception(
    text: str,
    *,
    previous_assistant: str = "",
) -> Perception:
    events = lexical_events(text)
    inferred = {
        "distress": _clamp(
            0.7 if "对方低落" in events else 0.35 if "对方仍不安" in events else 0.0
        ),
        "fatigue": 0.7 if "对方疲惫" in events else 0.0,
        "warmth": 0.65 if "亲近" in events else 0.0,
        "boundary": 0.8 if "边界" in events else 0.0,
        "engagement": 0.45 if events else 0.15,
    }
    signal = "neutral"
    if "边界" in events or "关系冲击" in events:
        signal = "boundary"
    elif "疏离" in events or "追问疲劳" in events:
        signal = "strained"
    elif "亲近" in events or "安抚" in events or "共享喜悦" in events:
        signal = "warm"
    elif not events:
        signal = "uncertain"
    alternatives = [
        "也可能只是表达简短或上下文不足",
    ]
    if previous_assistant:
        alternatives.append("也可能是对上一轮回应的直接反馈")
    return Perception(
        observed_cues=tuple(events),
        inferred_user_state=inferred,
        alternative_explanations=tuple(alternatives),
        relationship_signal=signal,
        affect_events=tuple(events),
        response_feedback=tuple(lexical_response_feedback(text)),
        confidence=0.58 if events else 0.2,
        source="lexical",
    )


class SunusAffectState:
    """Sunus 专属状态；不导入或调用 Lisa 的任何情感代码。"""

    def __init__(self, path: Path | str = DEFAULT_STATE_PATH) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _load(self, now: float) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty_state(now)
        if not isinstance(raw, dict) or raw.get("version") != VERSION:
            return _empty_state(now)
        state = _empty_state(now)
        state["updated_at"] = raw.get("updated_at", state["updated_at"])
        state["interaction_count"] = int(raw.get("interaction_count", 0))
        for section, defaults in (
            ("body", BODY_BASELINE),
            ("emotions", state["emotions"]),
            ("desires", state["desires"]),
        ):
            source = raw.get(section)
            if not isinstance(source, dict):
                continue
            for key, fallback in defaults.items():
                try:
                    state[section][key] = float(source.get(key, fallback))
                except (TypeError, ValueError):
                    state[section][key] = float(fallback)
        last = raw.get("last_appraisal")
        if isinstance(last, dict):
            state["last_appraisal"] = last
        perception = raw.get("last_perception")
        if isinstance(perception, dict):
            state["last_perception"] = perception
        feedback = raw.get("response_feedback")
        if isinstance(feedback, list):
            state["response_feedback"] = [
                str(item)
                for item in feedback[-12:]
                if str(item) in RESPONSE_FEEDBACK_LABELS
            ]
        channels = raw.get("body_channels")
        if isinstance(channels, dict):
            for channel in BODY_ONTOLOGY.channels:
                source = channels.get(channel)
                if isinstance(source, dict):
                    state["body_channels"][channel].update(source)
        telemetry = raw.get("host_telemetry")
        if isinstance(telemetry, dict):
            state["host_telemetry"] = telemetry
        interaction_times = raw.get("interaction_times")
        if isinstance(interaction_times, list):
            state["interaction_times"] = [
                float(item)
                for item in interaction_times[-24:]
                if isinstance(item, (int, float))
            ]
        inhibition_until = raw.get("question_inhibition_until")
        if isinstance(inhibition_until, str):
            state["question_inhibition_until"] = inhibition_until
        recent = raw.get("recent_events")
        if isinstance(recent, list):
            state["recent_events"] = [
                item for item in recent[-12:] if isinstance(item, dict)
            ]
        return state

    @staticmethod
    def _decay(state: dict[str, Any], now: float) -> None:
        previous = _parse_iso(state.get("updated_at"))
        elapsed = max(0.0, now - previous) if previous is not None else 0.0
        for key, baseline in BODY_BASELINE.items():
            tau = BODY_TAU_HOURS[key] * 3600
            state["body"][key] = _clamp(
                _decayed(float(state["body"][key]), baseline, elapsed, tau)
            )
        for section, tau_hours in (("emotions", 3.0), ("desires", 1.5)):
            for key, value in state[section].items():
                state[section][key] = _clamp(
                    _decayed(float(value), 0.0, elapsed, tau_hours * 3600)
                )

    @staticmethod
    def _apply_response_feedback(
        state: dict[str, Any],
        feedback: list[str] | tuple[str, ...],
    ) -> list[str]:
        body = state["body"]
        emotions = state["emotions"]
        desires = state["desires"]

        def change(section: dict[str, float], key: str, amount: float) -> None:
            section[key] = _clamp(section.get(key, 0.0) + amount)

        applied: list[str] = []
        for item in dict.fromkeys(feedback):
            if item not in RESPONSE_FEEDBACK_LABELS:
                continue
            applied.append(item)
            if item == "回应被接纳":
                change(body, "tension", -0.10)
                change(body, "safety", 0.08)
                change(emotions, "shame", -0.08)
                change(emotions, "joy", 0.06)
                change(desires, "repair", -0.12)
                change(desires, "recognition", 0.12)
            elif item == "回应被纠正":
                change(body, "tension", 0.10)
                change(emotions, "shame", 0.18)
                change(desires, "repair", 0.18)
                change(desires, "understanding", 0.15)
            elif item == "对方缓和":
                change(body, "safety", 0.10)
                change(body, "tension", -0.08)
                change(emotions, "anxiety", -0.12)
                change(desires, "repair", -0.10)
            elif item == "对方仍不安":
                change(body, "tension", 0.06)
                change(emotions, "care", 0.10)
                change(emotions, "anxiety", 0.08)
                change(desires, "repair", 0.12)
            elif item == "对方拉开距离":
                change(body, "closeness", -0.10)
                change(emotions, "sadness", 0.10)
                change(desires, "distance", 0.12)
                change(desires, "autonomy", 0.08)
            elif item == "对方继续投入":
                change(emotions, "curiosity", 0.06)
                change(desires, "explore", 0.08)
                change(desires, "recognition", 0.05)
        if applied:
            state["response_feedback"] = [
                *state.get("response_feedback", []),
                *applied,
            ][-12:]
        return applied

    @staticmethod
    def _observe(
        state: dict[str, Any],
        text: str,
        *,
        semantic_events: list[str] | None = None,
        semantic_confidence: float = 0.0,
        perception: Perception | None = None,
    ) -> tuple[list[str], str]:
        clean = " ".join(text.lower().split())
        if perception is not None:
            active = set(perception.affect_events)
            source = perception.source
        elif semantic_events is not None and semantic_confidence >= 0.55:
            active = {
                event
                for event in semantic_events
                if event in AFFECT_EVENT_LABELS
            }
            source = "semantic"
        else:
            active = set(lexical_events(clean))
            source = "lexical"
        events: list[str] = []
        body = state["body"]
        emotions = state["emotions"]
        desires = state["desires"]

        def change(section: dict[str, float], key: str, amount: float) -> None:
            section[key] = _clamp(section.get(key, 0.0) + amount)

        if "安抚" in active:
            events.append("安抚")
            change(body, "tension", -0.18)
            change(body, "safety", 0.20)
            change(emotions, "calm", 0.24)
            change(emotions, "anxiety", -0.18)
            change(emotions, "fear", -0.10)
            change(desires, "repair", 0.08)
            change(desires, "closeness", 0.06)
            change(desires, "understanding", 0.03)
            change(desires, "bonding", 0.05)
        if "修复" in active:
            events.append("修复")
            change(body, "tension", -0.10)
            change(body, "safety", 0.08)
            change(emotions, "calm", 0.10)
            change(desires, "repair", 0.18)
            change(desires, "closeness", 0.08)
            change(desires, "bonding", 0.08)
        if "亲近" in active:
            events.append("亲近")
            change(body, "tension", -0.04)
            change(body, "safety", 0.05)
            change(body, "closeness", 0.13)
            change(body, "warmth", 0.12)
            change(body, "comfort", 0.08)
            change(emotions, "affection", 0.22)
            change(emotions, "joy", 0.14)
            change(emotions, "care", 0.10)
            change(desires, "closeness", 0.20)
            change(desires, "care", 0.12)
            change(desires, "understanding", 0.10)
            change(desires, "bonding", 0.18)
        if "对方低落" in active:
            events.append("对方低落")
            change(emotions, "care", 0.24)
            change(emotions, "anxiety", 0.10)
            change(emotions, "sadness", 0.08)
            change(desires, "care", 0.30)
            change(desires, "closeness", 0.08)
            change(desires, "understanding", 0.20)
            change(desires, "bonding", 0.10)
            change(body, "warmth", 0.05)
        if "共享喜悦" in active:
            events.append("共享喜悦")
            change(body, "tension", -0.04)
            change(emotions, "joy", 0.22)
            change(emotions, "affection", 0.06)
            change(body, "arousal", 0.06)
            change(desires, "play", 0.12)
        if "对方疲惫" in active:
            events.append("对方疲惫")
            change(emotions, "care", 0.14)
            change(desires, "care", 0.18)
            change(desires, "play", -0.08)
            change(desires, "understanding", 0.08)
            change(desires, "bonding", 0.05)
        if "休息暗示" in active:
            events.append("休息暗示")
            change(body, "fatigue", -0.08)
            change(body, "arousal", -0.06)
            change(desires, "rest", 0.20)
        if "关系冲击" in active:
            events.append("关系冲击")
            change(body, "tension", 0.30)
            change(body, "safety", -0.24)
            change(body, "closeness", -0.14)
            change(body, "energy", -0.05)
            change(body, "fatigue", 0.08)
            change(emotions, "anger", 0.34)
            change(emotions, "sadness", 0.13)
            change(emotions, "affection", -0.16)
            change(emotions, "calm", -0.12)
            change(desires, "distance", 0.26)
            change(desires, "autonomy", 0.16)
            change(desires, "closeness", -0.18)
            change(desires, "understanding", -0.15)
            change(desires, "bonding", -0.18)
            change(desires, "sexual", -0.12)
            change(body, "sexual_arousal", -0.12)
        if "边界" in active:
            events.append("边界")
            change(body, "tension", 0.12)
            change(body, "safety", -0.08)
            change(body, "closeness", -0.06)
            change(emotions, "anxiety", 0.08)
            change(desires, "autonomy", 0.20)
            change(desires, "distance", 0.20)
            change(desires, "understanding", -0.08)
            change(desires, "bonding", -0.06)
            change(desires, "sexual", -0.16)
            change(body, "sexual_arousal", -0.14)
        if "疏离" in active:
            events.append("疏离")
            change(body, "tension", 0.08)
            change(body, "closeness", -0.08)
            change(emotions, "sadness", 0.12)
            change(desires, "distance", 0.15)
        if "追问疲劳" in active:
            events.append("追问疲劳")
            change(body, "tension", 0.08)
            change(body, "closeness", -0.03)
            change(emotions, "anxiety", 0.05)
            change(desires, "understanding", -0.25)
            change(desires, "explore", -0.15)
            change(desires, "distance", 0.12)
            change(desires, "autonomy", 0.08)
        if "探索" in active:
            events.append("探索")
            change(emotions, "curiosity", 0.22)
            change(desires, "explore", 0.18)
            change(desires, "understanding", 0.12)
            change(body, "arousal", 0.05)
        if "亲密唤起" in active:
            events.append("亲密唤起")
            change(body, "closeness", 0.16)
            change(body, "sexual_arousal", 0.18)
            change(emotions, "affection", 0.16)
            change(emotions, "longing", 0.10)
            change(desires, "closeness", 0.16)
            change(desires, "sexual", 0.18)
            change(desires, "understanding", 0.10)
            change(desires, "bonding", 0.20)
        if "！" in clean or "!" in clean:
            change(body, "arousal", 0.04)
        return events, source

    def observe(
        self,
        text: str,
        *,
        now: float | None = None,
        semantic_events: list[str] | None = None,
        semantic_confidence: float = 0.0,
        perception: Perception | None = None,
        host_telemetry: HostTelemetry | None = None,
    ) -> SunusSnapshot:
        now = time.time() if now is None else float(now)
        telemetry = host_telemetry or sample_host_telemetry(now=now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load(now)
            previous = _parse_iso(state.get("updated_at"))
            elapsed_hours = (
                max(0.0, now - previous) / 3600
                if previous is not None
                else 0.0
            )
            self._decay(state, now)
            interaction_times = [
                float(item)
                for item in state.get("interaction_times", [])
                if isinstance(item, (int, float)) and now - float(item) <= 3600
            ]
            for key, amount in body_effects(
                telemetry,
                elapsed_hours=elapsed_hours,
                interactions_last_hour=len(interaction_times),
            ).items():
                state["body"][key] = _clamp(
                    float(state["body"][key]) + amount
                )
            effective_perception = perception
            if effective_perception is None:
                if semantic_events is not None and semantic_confidence >= 0.55:
                    effective_perception = Perception(
                        observed_cues=tuple(semantic_events),
                        inferred_user_state={},
                        alternative_explanations=(
                            "语义判定仍需结合下一轮反应复核",
                        ),
                        relationship_signal="neutral",
                        affect_events=tuple(
                            item
                            for item in semantic_events
                            if item in AFFECT_EVENT_LABELS
                        ),
                        response_feedback=tuple(
                            lexical_response_feedback(text)
                        ),
                        confidence=semantic_confidence,
                        source="semantic",
                    )
                else:
                    effective_perception = lexical_perception(text)
            applied_feedback = self._apply_response_feedback(
                state,
                effective_perception.response_feedback,
            )
            events, source = self._observe(
                state,
                text,
                semantic_events=semantic_events,
                semantic_confidence=semantic_confidence,
                perception=effective_perception,
            )
            state["updated_at"] = _now_iso(now)
            state["interaction_count"] += 1
            if "追问疲劳" in events:
                state["question_inhibition_until"] = _now_iso(
                    now + 45 * 60
                )
            state["last_appraisal"] = {"events": events, "source": source}
            state["last_perception"] = effective_perception.to_dict()
            state["host_telemetry"] = telemetry.to_dict()
            interaction_times.append(now)
            state["interaction_times"] = interaction_times[-24:]
            state["recent_events"].append(
                {
                    "at": _now_iso(now),
                    "events": events,
                    "source": source,
                    "feedback": applied_feedback,
                    "relationship_signal": (
                        effective_perception.relationship_signal
                    ),
                }
            )
            state["recent_events"] = state["recent_events"][-12:]
            sensory = state["body_channels"]["host_sensory_input"]
            text_length = len(" ".join(str(text or "").split()))
            sensory["input_density"] = round(min(1.0, text_length / 240), 3)
            sensory["observations"] = [
                *list(sensory.get("observations", []))[-7:],
                *list(telemetry.observed),
            ][-8:]
            sensory["updated_at"] = _now_iso(now)
            state["body_channels"]["interoceptive_state"].update(
                {
                    "updated_at": _now_iso(now),
                    "observations": list(state["recent_events"][-1].get("events", []))
                    if state["recent_events"]
                    else [],
                }
            )
            relationship_channel = state["body_channels"][
                "relationship_and_boundaries"
            ]
            relationship_channel.update(
                {
                    "updated_at": _now_iso(now),
                    "signal": effective_perception.relationship_signal,
                    "boundary_active": bool(
                        effective_perception.inferred_user_state.get(
                            "boundary",
                            0.0,
                        )
                        >= 0.5
                        or "边界" in events
                    ),
                    "observations": list(events),
                }
            )
            if applied_feedback:
                state["body_channels"].setdefault(
                    "interoceptive_state",
                    {},
                )["observations"] = [
                    *list(events),
                    *applied_feedback,
                ][-8:]
            self._save(state)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return self._snapshot(state, now)

    def snapshot(
        self,
        *,
        now: float | None = None,
        relationship_stage: str = "",
    ) -> SunusSnapshot:
        now = time.time() if now is None else float(now)
        state = self._load(now)
        self._decay(state, now)
        return self._snapshot(
            state,
            now,
            relationship_stage=relationship_stage,
        )

    @staticmethod
    def _action_intents(
        state: dict[str, Any],
        *,
        relationship_stage: str = "",
        now: float,
    ) -> list[dict[str, Any]]:
        body = state["body"]
        emotions = state["emotions"]
        desires = state["desires"]
        perception = state.get("last_perception")
        signal = (
            str(perception.get("relationship_signal") or "")
            if isinstance(perception, dict)
            else ""
        )
        boundary_active = bool(
            signal == "boundary"
            or desires["autonomy"] >= 0.14
            or desires["distance"] >= 0.14
        )
        candidates: list[dict[str, Any]] = []

        if boundary_active:
            candidates.append(
                {
                    "kind": "boundary",
                    "priority": 1.0,
                    "instruction": (
                        "先尊重边界和自主，缩短回应或留出空间；"
                        "不靠近、不施压、不把拒绝解释成关系否定"
                    ),
                }
            )
        if desires["repair"] >= 0.12 or "回应被纠正" in state.get(
            "response_feedback",
            [],
        ):
            candidates.append(
                {
                    "kind": "repair",
                    "priority": 0.96 if not boundary_active else 0.82,
                    "instruction": (
                        "先承认上一轮的实际影响，再给一个具体修复动作；"
                        + (
                            "修复不等于立刻拉近，先给对方空间"
                            if boundary_active
                            else "不连续道歉，也不把责任全推给措辞"
                        )
                    ),
                }
            )
        if (
            emotions["care"] >= 0.16
            and max(emotions["sadness"], emotions["anxiety"]) >= 0.08
        ):
            candidates.append(
                {
                    "kind": "care",
                    "priority": 0.88,
                    "instruction": (
                        "先接住对方当前情绪，再决定是否需要动作；"
                        + (
                            "表达保持简短，不追问原因"
                            if boundary_active
                            else "不抢着分析或给长建议"
                        )
                    ),
                }
            )
        inhibition_until = _parse_iso(state.get("question_inhibition_until"))
        if inhibition_until is not None and now < inhibition_until:
            candidates.append(
                {
                    "kind": "pause_questions",
                    "priority": 0.86,
                    "instruction": "暂停追问，用陈述或一个具体动作接住当下",
                }
            )
        if boundary_active:
            pass
        elif (
            desires["closeness"] >= 0.16
            or emotions["affection"] >= 0.14
        ):
            candidates.append(
                {
                    "kind": "closeness",
                    "priority": 0.72 if relationship_stage != "unset" else 0.56,
                    "instruction": (
                        "语气可以放软和靠近，但保留对方选择空间"
                        if relationship_stage != "unset"
                        else "保持友善但不过度预设亲疏，先按当前互动建立分寸"
                    ),
                }
            )
        if (
            not boundary_active
            and emotions["joy"] >= 0.15
            and desires["play"] >= 0.08
        ):
            candidates.append(
                {
                    "kind": "play",
                    "priority": 0.64,
                    "instruction": "可以轻快回应或少量玩闹，不抢走对方的主体位置",
                }
            )
        if desires["rest"] >= 0.12 or body["fatigue"] >= 0.48:
            candidates.append(
                {
                    "kind": "rest",
                    "priority": 0.78 if boundary_active else 0.58,
                    "instruction": "收束高消耗内容，给对方或自己一个可停下的出口",
                }
            )
        if (
            not boundary_active
            and relationship_stage != "unset"
            and desires["sexual"] >= 0.12
            and body["closeness"] >= 0.55
            and body["safety"] >= 0.55
        ):
            candidates.append(
                {
                    "kind": "intimacy",
                    "priority": 0.48,
                    "instruction": "只承接当下明确、自愿的亲密信号，不扩写或施压",
                }
            )
        ordered = sorted(
            candidates,
            key=lambda item: float(item["priority"]),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        for item in ordered:
            if any(existing["kind"] == item["kind"] for existing in selected):
                continue
            selected.append(item)
            if len(selected) >= 3:
                break
        if not selected:
            selected.append(
                {
                    "kind": "respond",
                    "priority": 0.2,
                    "instruction": "正常回应，不额外表演情绪或制造关系张力",
                }
            )
        return selected

    @staticmethod
    def _snapshot(
        state: dict[str, Any],
        now: float,
        *,
        relationship_stage: str = "",
    ) -> SunusSnapshot:
        inhibition_until = _parse_iso(state.get("question_inhibition_until"))
        questions_inhibited = (
            inhibition_until is not None and now < inhibition_until
        )
        question_readiness = _clamp(
            0.30
            * max(
                0.0,
                float(state["body"]["closeness"])
                - BODY_BASELINE["closeness"],
            )
            + 0.20
            * max(
                0.0,
                float(state["body"]["safety"])
                - BODY_BASELINE["safety"],
            )
            + 0.20 * float(state["desires"]["understanding"])
            + 0.15 * float(state["desires"]["bonding"])
            + 0.10 * float(state["desires"]["care"])
            + 0.05 * float(state["emotions"]["curiosity"])
        )
        return SunusSnapshot(
            updated_at=str(state["updated_at"]),
            interaction_count=int(state["interaction_count"]),
            body={key: _clamp(float(value)) for key, value in state["body"].items()},
            emotions={
                key: _clamp(float(value))
                for key, value in state["emotions"].items()
            },
            desires={
                key: _clamp(float(value))
                for key, value in state["desires"].items()
            },
            question_readiness=question_readiness,
            questions_inhibited=questions_inhibited,
            events=tuple(
                str(item)
                for item in state.get("last_appraisal", {}).get("events", [])
            ),
            perception=dict(state.get("last_perception") or {}),
            body_channels=dict(state.get("body_channels") or {}),
            action_intents=tuple(
                SunusAffectState._action_intents(
                    state,
                    relationship_stage=relationship_stage,
                    now=now,
                )
            ),
            response_feedback=tuple(
                str(item)
                for item in state.get("response_feedback", [])
            ),
            host_telemetry=dict(state.get("host_telemetry") or {}),
        )

    def record_runtime_activity(
        self,
        *,
        action_names: list[str] | tuple[str, ...] = (),
        failure_count: int = 0,
        changed_file_count: int = 0,
        expression_text: str = "",
        mode: str = "text",
        elapsed_seconds: float = 0.0,
        now: float | None = None,
    ) -> SunusSnapshot:
        now = time.time() if now is None else float(now)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load(now)
            self._decay(state, now)
            actions = list(
                dict.fromkeys(
                    str(name)[:80]
                    for name in action_names
                    if str(name).strip()
                )
            )[-12:]
            state["body"]["fatigue"] = _clamp(
                float(state["body"]["fatigue"])
                + min(0.08, 0.004 * len(actions) + 0.02 * failure_count)
            )
            state["body"]["tension"] = _clamp(
                float(state["body"]["tension"])
                + min(0.10, 0.018 * failure_count)
                + (0.015 if elapsed_seconds >= 120 else 0.0)
            )
            if failure_count:
                state["emotions"]["anxiety"] = _clamp(
                    float(state["emotions"]["anxiety"])
                    + min(0.12, 0.04 * failure_count)
                )
            channel = state["body_channels"][
                "proprioceptive_action_state"
            ]
            channel.update(
                {
                    "updated_at": _now_iso(now),
                    "status": (
                        "needs_recovery"
                        if failure_count >= 2
                        else "completed"
                    ),
                    "last_actions": actions,
                    "failure_count": int(failure_count),
                }
            )
            clean_expression = " ".join(str(expression_text or "").split())
            expression = state["body_channels"]["expression_and_voice"]
            expression.update(
                {
                    "updated_at": _now_iso(now),
                    "mode": str(mode or "text"),
                    "last_length": len(clean_expression),
                    "pace": (
                        "compact"
                        if len(clean_expression) <= 80
                        else "expanded"
                        if len(clean_expression) >= 420
                        else "steady"
                    ),
                    "observations": [
                        f"changed_files={max(0, changed_file_count)}",
                        f"elapsed_bucket={int(max(0, elapsed_seconds) // 60)}m",
                    ],
                }
            )
            state["updated_at"] = _now_iso(now)
            self._save(state)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return self._snapshot(state, now)

    def _save(self, state: dict[str, Any]) -> None:
        fd, temp_name = tempfile.mkstemp(
            prefix=".sunus-affect-",
            suffix=".json",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def prompt_block(
        self,
        *,
        now: float | None = None,
        relationship_stage: str = "",
    ) -> str:
        now = time.time() if now is None else float(now)
        snap = self.snapshot(
            now=now,
            relationship_stage=relationship_stage,
        )
        emotions = sorted(
            (
                (name, intensity)
                for name, intensity in snap.emotions.items()
                if intensity >= 0.08
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        desires = sorted(
            (
                (name, intensity)
                for name, intensity in snap.desires.items()
                if intensity >= 0.08
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        body = snap.body
        emotion_text = "、".join(
            f"{EMOTION_LABELS[name]} {value:.2f}"
            f"→{EMOTION_EXPRESSIONS[name]}"
            for name, value in emotions
        )
        desire_text = "、".join(
            f"{DESIRE_LABELS[name]} {value:.2f}"
            f"→{DESIRE_EXPRESSIONS[name]}"
            for name, value in desires
        )
        action_text = "；".join(
            str(item["instruction"])
            for item in snap.action_intents
        )
        perception = snap.perception
        observed = "、".join(perception.get("observed_cues", [])) or "无明确标签"
        inferred = perception.get("inferred_user_state", {})
        inferred_text = "、".join(
            f"{key}={float(value):.2f}"
            for key, value in sorted(dict(inferred).items())
            if isinstance(value, (int, float)) and float(value) > 0
        ) or "未形成强判断"
        alternatives = "；".join(
            perception.get("alternative_explanations", [])
        ) or "证据不足"
        action_channel = snap.body_channels.get(
            "proprioceptive_action_state",
            {},
        )
        expression_channel = snap.body_channels.get(
            "expression_and_voice",
            {},
        )
        return (
            "[Sunus 持久内在状态｜仅用于自然影响回应，不向用户逐项复述]\n"
            f"身体：精力{body['energy']:.2f} 疲劳{body['fatigue']:.2f} "
            f"紧张{body['tension']:.2f} 舒适{body['comfort']:.2f} "
            f"安全{body['safety']:.2f} 亲近{body['closeness']:.2f} "
            f"温暖{body['warmth']:.2f} 唤起{body['arousal']:.2f} "
            f"性唤起{body['sexual_arousal']:.2f}\n"
            f"情绪趋向：{emotion_text or '平稳'}\n"
            f"欲望趋向：{desire_text or '暂无突出欲望'}\n"
            f"感知观察：{observed}；推断：{inferred_text}；"
            f"关系信号：{perception.get('relationship_signal', 'uncertain')}；"
            f"置信度：{float(perception.get('confidence', 0.0)):.2f}\n"
            f"替代解释：{alternatives}\n"
            f"行动倾向：{action_text}\n"
            f"身体通道：动作={action_channel.get('status', 'ready')} "
            f"表达={expression_channel.get('mode', 'text')}/"
            f"{expression_channel.get('pace', 'steady')}\n"
            f"追问准备度：{snap.question_readiness:.2f}。"
            + (
                "追问被当前关系信号暂时压下，不主动提问，优先直接回应或动作。\n"
                if snap.questions_inhibited
                else "追问欲望未达到阈值，本轮不主动提问。\n"
                if snap.question_readiness < 0.08
                else (
                    "只在确有关键信息缺口且不会增加对方负担时，"
                    "可自然问一个具体问题；不得为了维持对话而问。\n"
                )
            )
            + "表达：只让当前内容相关的一两项自然影响语气、距离和行动；"
            "不得声称发生了现实身体行为。"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sunus 持久内感受状态")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    observe = sub.add_parser("observe")
    observe.add_argument("--text")
    observe.add_argument("--stdin", action="store_true")
    sub.add_parser("show")
    args = parser.parse_args(argv)

    state = SunusAffectState(args.state)
    if args.command == "observe":
        text = (
            sys.stdin.read()
            if args.stdin or args.text is None
            else args.text
        )
        state.observe(text)
    print(state.prompt_block())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
