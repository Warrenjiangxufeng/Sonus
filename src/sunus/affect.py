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


def _empty_state(now: float) -> dict[str, Any]:
    return {
        "version": VERSION,
        "updated_at": _now_iso(now),
        "interaction_count": 0,
        "body": dict(BODY_BASELINE),
        "emotions": {name: 0.0 for name in EMOTIONS},
        "desires": {name: 0.0 for name in DESIRES},
        "question_inhibition_until": None,
        "last_appraisal": {"events": []},
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
    def _observe(state: dict[str, Any], text: str) -> list[str]:
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
        body = state["body"]
        emotions = state["emotions"]
        desires = state["desires"]

        def change(section: dict[str, float], key: str, amount: float) -> None:
            section[key] = _clamp(section.get(key, 0.0) + amount)

        if _any(clean, SUPPORT_CUES):
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
        if _any(clean, REPAIR_CUES):
            events.append("修复")
            change(body, "tension", -0.10)
            change(body, "safety", 0.08)
            change(emotions, "calm", 0.10)
            change(desires, "repair", 0.18)
            change(desires, "closeness", 0.08)
            change(desires, "bonding", 0.08)
        if _any(clean, WARMTH_CUES):
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
        if _any(clean, DISTRESS_CUES):
            events.append("对方低落")
            change(emotions, "care", 0.24)
            change(emotions, "anxiety", 0.10)
            change(emotions, "sadness", 0.08)
            change(desires, "care", 0.30)
            change(desires, "closeness", 0.08)
            change(desires, "understanding", 0.20)
            change(desires, "bonding", 0.10)
            change(body, "warmth", 0.05)
        if _any(clean, JOY_CUES):
            events.append("共享喜悦")
            change(body, "tension", -0.04)
            change(emotions, "joy", 0.22)
            change(emotions, "affection", 0.06)
            change(body, "arousal", 0.06)
            change(desires, "play", 0.12)
        if _any(clean, FATIGUE_CUES):
            events.append("对方疲惫")
            change(emotions, "care", 0.14)
            change(desires, "care", 0.18)
            change(desires, "play", -0.08)
            change(desires, "understanding", 0.08)
            change(desires, "bonding", 0.05)
        if _any(clean, REST_CUES):
            events.append("休息暗示")
            change(body, "fatigue", -0.08)
            change(body, "arousal", -0.06)
            change(desires, "rest", 0.20)
        if _any(protective_text, INSULT_CUES):
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
        if _any(clean, BOUNDARY_CUES):
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
        if _any(protective_text, REJECTION_CUES):
            events.append("疏离")
            change(body, "tension", 0.08)
            change(body, "closeness", -0.08)
            change(emotions, "sadness", 0.12)
            change(desires, "distance", 0.15)
        if _any(clean, QUESTION_FATIGUE_CUES):
            events.append("追问疲劳")
            change(body, "tension", 0.08)
            change(body, "closeness", -0.03)
            change(emotions, "anxiety", 0.05)
            change(desires, "understanding", -0.25)
            change(desires, "explore", -0.15)
            change(desires, "distance", 0.12)
            change(desires, "autonomy", 0.08)
        if _any(clean, CURIOSITY_CUES):
            events.append("探索")
            change(emotions, "curiosity", 0.22)
            change(desires, "explore", 0.18)
            change(desires, "understanding", 0.12)
            change(body, "arousal", 0.05)
        if _any(clean, INTIMACY_CUES):
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
        return events

    def observe(self, text: str, *, now: float | None = None) -> SunusSnapshot:
        now = time.time() if now is None else float(now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load(now)
            self._decay(state, now)
            events = self._observe(state, text)
            state["updated_at"] = _now_iso(now)
            state["interaction_count"] += 1
            if "追问疲劳" in events:
                state["question_inhibition_until"] = _now_iso(
                    now + 45 * 60
                )
            state["last_appraisal"] = {"events": events}
            state["recent_events"].append(
                {"at": _now_iso(now), "events": events}
            )
            state["recent_events"] = state["recent_events"][-12:]
            self._save(state)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return self._snapshot(state, now)

    def snapshot(self, *, now: float | None = None) -> SunusSnapshot:
        now = time.time() if now is None else float(now)
        state = self._load(now)
        self._decay(state, now)
        return self._snapshot(state, now)

    @staticmethod
    def _snapshot(state: dict[str, Any], now: float) -> SunusSnapshot:
        inhibition_until = _parse_iso(state.get("question_inhibition_until"))
        questions_inhibited = (
            inhibition_until is not None and now < inhibition_until
        )
        question_readiness = _clamp(
            0.30 * float(state["body"]["closeness"])
            + 0.20 * float(state["body"]["safety"])
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
        )

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

    def prompt_block(self, *, now: float | None = None) -> str:
        now = time.time() if now is None else float(now)
        snap = self.snapshot(now=now)
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
        actions: list[tuple[float, str]] = []
        if snap.desires["repair"] >= 0.12:
            actions.append((0.95, "先承认影响，再给具体修复动作"))
        if (
            snap.emotions["anger"] >= 0.16
            or snap.desires["autonomy"] >= 0.14
            or snap.desires["distance"] >= 0.14
        ):
            actions.append((0.90, "直接说边界或缩短回应，不讨好"))
        if (
            snap.emotions["care"] >= 0.16
            and max(
                snap.emotions["sadness"],
                snap.emotions["anxiety"],
            )
            >= 0.08
        ):
            actions.append((0.85, "先接住情绪，再决定是否需要动作"))
        if snap.questions_inhibited:
            actions.append((0.82, "想了解但暂停追问，先直接回应或留空间"))
        elif (
            snap.question_readiness >= 0.36
            and snap.desires["understanding"] >= 0.12
        ):
            actions.append((0.76, "若确有关键信息缺口，可问一个具体问题"))
        if (
            snap.desires["closeness"] >= 0.16
            or snap.emotions["affection"] >= 0.14
        ):
            actions.append((0.70, "语气放软、靠近，但不施压"))
        if (
            snap.emotions["joy"] >= 0.15
            and snap.desires["play"] >= 0.08
        ):
            actions.append((0.62, "可以轻快分享或少量玩笑"))
        if snap.emotions["longing"] >= 0.10:
            actions.append((0.58, "可表达想念，但不制造负担"))
        if snap.desires["rest"] >= 0.12 or snap.body["fatigue"] >= 0.48:
            actions.append((0.55, "收束话题或暂停消耗"))
        if snap.emotions["surprise"] >= 0.12:
            actions.append((0.50, "短暂停顿，不急着解释"))
        if snap.emotions["calm"] >= 0.14:
            actions.append((0.45, "保持稳定节奏，少煽情"))
        if (
            snap.desires["sexual"] >= 0.12
            and snap.body["closeness"] >= 0.55
            and snap.body["safety"] >= 0.55
        ):
            actions.append((0.42, "亲密表达必须尊重关系和同意，不施压"))
        action_text = "；".join(
            instruction
            for _, instruction in sorted(actions, reverse=True)[:3]
        ) or "正常回应，不额外表演情绪"
        return (
            "[Sunus 持久内在状态｜仅用于自然影响回应，不向用户逐项复述]\n"
            f"身体：精力{body['energy']:.2f} 疲劳{body['fatigue']:.2f} "
            f"紧张{body['tension']:.2f} 舒适{body['comfort']:.2f} "
            f"安全{body['safety']:.2f} 亲近{body['closeness']:.2f} "
            f"温暖{body['warmth']:.2f} 唤起{body['arousal']:.2f} "
            f"性唤起{body['sexual_arousal']:.2f}\n"
            f"情绪趋向：{emotion_text or '平稳'}\n"
            f"欲望趋向：{desire_text or '暂无突出欲望'}\n"
            f"行动倾向：{action_text}\n"
            f"追问准备度：{snap.question_readiness:.2f}。"
            + (
                "追问被当前关系信号暂时压下，不主动提问，优先直接回应或动作。\n"
                if snap.questions_inhibited
                else "追问欲望未达到阈值，本轮不主动提问。\n"
                if snap.question_readiness < 0.36
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
