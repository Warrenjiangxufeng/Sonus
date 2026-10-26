from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RelationshipConfig:
    label: str = ""
    sunus_role: str = ""
    user_role: str = ""
    address: str = ""

    @property
    def configured(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_ROLE = (
    r"(?:女朋友|男朋友|恋人|伴侣|老婆|老公|妻子|丈夫|"
    r"姐姐|妹妹|哥哥|弟弟|女儿|儿子|妈妈|妈妈桑|父亲|母亲|"
    r"朋友|好友|挚友|搭档|伙伴|队友|同事|同学|同桌|学生|老师|"
    r"导师|助手|助理|主人|管家|下属|老板|照顾者|被照顾者|"
    r"[A-Za-z\u3400-\u9fff]{1,8})"
)
_ROLE_CAPTURE = f"(?P<role>{_ROLE})"
_QUESTION_OR_HYPOTHETICAL = re.compile(
    r"(?:吗|么|是不是|是否|像不像|如果|假如|要是|好像|像不像|"
    r"扮演|假装|演一下|演戏|游戏里|梦里|开玩笑|会怎样|怎么样|"
    r"如何|好不好)[？?]?",
    re.IGNORECASE,
)
_CLEAR_PATTERN = re.compile(
    r"(?:取消|解除|清空|删除|删掉|忘掉).{0,10}(?:关系|设定)|"
    r"(?:以后|今后|从现在开始|从今天起).{0,10}"
    r"(?:你不再是|你不是|我们不再是|我们不是).{0,12}"
    r"(?:朋友|恋人|伴侣|女朋友|男朋友|老婆|老公|关系)",
    re.IGNORECASE,
)
_SUNUS_ROLE_PATTERN = re.compile(
    rf"(?:以后|今后|从现在开始|从今天起|默认|就)?\s*"
    rf"你(?:就)?(?:是|当|做|作为)我(?:的)?{_ROLE_CAPTURE}",
    re.IGNORECASE,
)
_USER_ROLE_PATTERN = re.compile(
    rf"(?:以后|今后|从现在开始|从今天起|默认|就)?\s*"
    rf"我(?:就)?(?:是|当|做|作为)你(?:的)?{_ROLE_CAPTURE}",
    re.IGNORECASE,
)
_LABEL_PATTERN = re.compile(
    r"(?:以后|今后|从现在开始|从今天起|默认)?\s*"
    r"(?:我们(?:就)?(?:是|成为|升级为|升级成)|"
    r"设定(?:我们|你和我)(?:的)?关系(?:是|为)|"
    r"(?:你和我|我们)(?:的)?关系(?:是|为))"
    r"(?P<label>[^，。,.！!？?\n]{1,12})",
    re.IGNORECASE,
)
_MARRIAGE_PATTERN = re.compile(
    r"(?:以后|今后|从现在开始|从今天起)?\s*"
    r"(?:我们)?(?:结婚|升级(?:为|成)夫妻|成为夫妻)"
)
_ADDRESS_PATTERN = re.compile(
    r"(?:称呼我|叫我)(?:为|叫)?(?P<address>[^\s，。,.！!？?]{1,12})"
)
_INVALID_ROLES = frozenset(
    {
        "谁",
        "什么",
        "哪种",
        "怎样",
        "唯一",
        "全世界",
        "小可爱",
    }
)
_ROMANTIC_ROLES = frozenset(
    {"女朋友", "男朋友", "恋人", "伴侣", "老婆", "老公", "妻子", "丈夫"}
)
_SPOUSE_ROLES = frozenset({"老婆", "老公", "妻子", "丈夫"})
_FAMILY_ROLES = frozenset(
    {
        "姐姐",
        "妹妹",
        "哥哥",
        "弟弟",
        "女儿",
        "儿子",
        "妈妈",
        "父亲",
        "母亲",
    }
)


def _clean_value(value: Any, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _clean_role(role: str) -> str:
    value = _clean_value(role, limit=20)
    value = re.sub(r"(?:关系|吧|呀|啊|哦|啦|呢)+$", "", value).strip()
    return value


def relationship_from_mapping(raw: Any) -> RelationshipConfig:
    if raw in (None, ""):
        return RelationshipConfig()
    if isinstance(raw, str):
        payload: dict[str, Any] = {"label": raw}
    elif isinstance(raw, dict):
        payload = raw
    else:
        raise ValueError("persona.relationship 必须是 JSON object 或字符串")
    return RelationshipConfig(
        label=_clean_value(payload.get("label"), limit=40),
        sunus_role=_clean_value(payload.get("sunus_role"), limit=40),
        user_role=_clean_value(payload.get("user_role"), limit=40),
        address=_clean_value(payload.get("address"), limit=40),
    )


def _infer_label(
    *,
    label: str,
    sunus_role: str,
    user_role: str,
) -> str:
    cleaned = _clean_role(label) if label else ""
    roles = {role for role in (sunus_role, user_role) if role}
    if any(marker in cleaned for marker in ("夫妻", "已婚", "婚姻")):
        return "夫妻"
    if roles & _SPOUSE_ROLES:
        return "夫妻"
    if roles & _ROMANTIC_ROLES:
        return "恋人" if roles <= _ROMANTIC_ROLES else "伴侣"
    if roles & _FAMILY_ROLES:
        return "家人"
    if roles and roles <= {"朋友", "好友", "挚友"}:
        return "朋友"
    if roles and roles <= {"搭档", "伙伴", "队友"}:
        return "搭档"
    if cleaned:
        return cleaned
    return sunus_role or user_role


def extract_relationship_update(text: str) -> dict[str, str] | None:
    value = str(text or "").strip()
    if not value:
        return None
    if _CLEAR_PATTERN.search(value) and not re.search(r"[？?]\s*$|吗\s*$", value):
        return {}
    if _QUESTION_OR_HYPOTHETICAL.search(value):
        return None

    sunus_match = _SUNUS_ROLE_PATTERN.search(value)
    user_match = _USER_ROLE_PATTERN.search(value)
    label_match = _LABEL_PATTERN.search(value)
    address_match = _ADDRESS_PATTERN.search(value)
    sunus_role = _clean_role(sunus_match.group("role")) if sunus_match else ""
    user_role = _clean_role(user_match.group("role")) if user_match else ""
    if sunus_role in _INVALID_ROLES or user_role in _INVALID_ROLES:
        sunus_role = ""
        user_role = ""
    label = _clean_role(label_match.group("label")) if label_match else ""
    if not label and _MARRIAGE_PATTERN.search(value):
        label = "夫妻"
    address = _clean_value(
        address_match.group("address") if address_match else "",
        limit=40,
    )
    if not any((sunus_role, user_role, label, address)):
        return None
    label = _infer_label(
        label=label,
        sunus_role=sunus_role,
        user_role=user_role,
    )
    return {
        key: item
        for key, item in {
            "label": label,
            "sunus_role": sunus_role,
            "user_role": user_role,
            "address": address,
        }.items()
        if item
    }


def relationship_trait(relationship: RelationshipConfig | None) -> str:
    if relationship is None or not relationship.configured:
        return ""
    parts = [
        f"Sunus 与用户的人物关系设定为“{relationship.label}”"
        if relationship.label
        else "Sunus 与用户设有人物关系"
    ]
    if relationship.sunus_role:
        parts.append(f"Sunus 的角色是“{relationship.sunus_role}”")
    if relationship.user_role:
        parts.append(f"用户的角色是“{relationship.user_role}”")
    return "；".join(parts)


def relationship_stage(relationship: RelationshipConfig | None) -> str:
    if relationship is None or not relationship.configured:
        return "unset"
    label = relationship.label
    roles = {relationship.sunus_role, relationship.user_role}
    if roles & _SPOUSE_ROLES or any(
        marker in label for marker in ("夫妻", "已婚", "婚姻")
    ):
        return "spouse"
    if roles & _ROMANTIC_ROLES or any(
        marker in label for marker in ("情侣", "恋人", "伴侣")
    ):
        return "couple"
    return "locked"


@dataclass(frozen=True)
class RelationshipTransition:
    allowed: bool
    relationship: RelationshipConfig
    reason: str


def transition_relationship(
    current: RelationshipConfig | None,
    update: dict[str, str],
) -> RelationshipTransition:
    current = current or RelationshipConfig()
    stage = relationship_stage(current)
    if stage == "unset":
        if not update:
            return RelationshipTransition(True, RelationshipConfig(), "尚未设置关系")
        return RelationshipTransition(
            True,
            relationship_from_mapping(update),
            "初始关系已设定",
        )
    if stage == "couple":
        if not update:
            return RelationshipTransition(
                True,
                RelationshipConfig(),
                "情侣关系已解除",
            )
        merged = current.to_dict()
        merged.update(update)
        proposed = relationship_from_mapping(merged)
        if relationship_stage(proposed) == "spouse":
            return RelationshipTransition(
                True,
                proposed,
                "情侣关系已升级为夫妻",
            )
        if relationship_stage(proposed) == "couple":
            return RelationshipTransition(
                True,
                proposed,
                "情侣关系已更新",
            )
        return RelationshipTransition(
            False,
            current,
            "情侣关系只能调整为情侣、夫妻或解除",
        )
    if stage == "spouse":
        return RelationshipTransition(
            False,
            current,
            "夫妻关系已锁定，不能再修改或解除",
        )
    return RelationshipTransition(
        False,
        current,
        "人物关系已锁定，不能再修改或解除",
    )


def relationship_prompt(relationship: RelationshipConfig | None) -> str:
    if relationship is None or not relationship.configured:
        return ""
    lines = ["\n人物关系（用户设定，持续生效）："]
    if relationship.label:
        lines.append(f"- 双方关系：{relationship.label}")
    if relationship.sunus_role:
        lines.append(f"- Sunus 的角色：{relationship.sunus_role}")
    if relationship.user_role:
        lines.append(f"- 用户的角色：{relationship.user_role}")
    if relationship.address:
        lines.append(f"- Sunus 对用户的称呼：{relationship.address}")
    lines.extend(
        [
            "- 默认按这套关系的亲疏、称呼和边界组织对话；不要每轮重复声明关系。",
            "- 关系设定后默认锁定；只有情侣关系可调整、解除或升级为夫妻，夫妻及其他关系不可再改。",
            "- 关系语境不能改变事实准确性、安全边界、工具权限或用户明确要求。",
            "- 没有设定的角色、称呼或共同经历不得自行补全；用户改口时以新设定为准。",
        ]
    )
    return "\n".join(lines) + "\n"
