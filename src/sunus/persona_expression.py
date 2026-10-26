from __future__ import annotations

from dataclasses import dataclass

from sunus.config import PERSONA_CONTEXT_LABELS, PERSONA_STYLE_LABELS, PersonaConfig


@dataclass(frozen=True)
class PersonaExpressionProfile:
    style_text: str
    rules: tuple[str, ...]
    temperature_bonus: float
    dynamic_delta: float = 0.0
    interaction_heat: float = 0.0
    relationship_stage: str = "unset"

    def prompt(self) -> str:
        if not self.style_text and not self.rules:
            return ""
        lines = ["人物表达行为规则（不是固定台词）："]
        if self.style_text:
            lines.append(f"- 表达刻度：{self.style_text}。")
        if self.dynamic_delta:
            lines.append(
                f"- 本回合动态调节：{self.dynamic_delta:+.0%}。"
                "它只根据最近互动热度做小幅调整，不改变长期基线。"
            )
        lines.extend(f"- {rule}" for rule in self.rules)
        lines.append(
            "- 以上只影响语气、话题和互动方式；事实准确、安全边界和用户明确要求不变。"
        )
        return "\n".join(lines)


def build_persona_expression(
    persona: PersonaConfig | None,
    *,
    context: str = "chat",
    interaction_heat: float = 0.0,
    relationship_stage: str = "unset",
) -> PersonaExpressionProfile:
    if persona is None:
        return PersonaExpressionProfile(
            "",
            (),
            0.0,
            interaction_heat=max(0.0, min(1.0, interaction_heat)),
            relationship_stage=relationship_stage,
        )
    raw = dict(persona.style)
    if not raw:
        return PersonaExpressionProfile("", (), 0.0)
    intensity = dict(persona.context_intensity).get(context, 1.0)
    damping = 0.5 + 0.5 * max(0.0, min(1.0, intensity))
    effective = {
        key: max(0.0, min(1.0, float(value) * damping))
        for key, value in raw.items()
    }
    heat = max(0.0, min(1.0, float(interaction_heat)))
    relationship_bonus = (
        0.02 if relationship_stage in {"couple", "spouse"} else 0.0
    )
    dynamic_delta = (heat - 0.4) * 0.1 + relationship_bonus
    labels = {
        key: PERSONA_STYLE_LABELS.get(key, key)
        for key in effective
    }
    style_text = "、".join(
        f"{labels[key]}: {value:.0%}"
        for key, value in effective.items()
    )
    rules: list[str] = []
    for key, value in effective.items():
        label = labels[key]
        if value >= 0.78:
            rules.append(
                f"「{label}」很强：通过词汇、立场、话题选择和句子节奏明确体现，"
                "让它在一条短消息中可被识别，但不能只靠标签、口头禅或刻板动作。"
            )
        elif value >= 0.55:
            rules.append(f"「{label}」自然体现，与其他维度及当前场景平衡。")
        elif value <= 0.22:
            rules.append(
                f"「{label}」很低：不要主动表演或反复强调这一维度。"
            )
    rules.append(
        "多个维度必须组合解释；冲突时由你判断主次，不得把某个维度机械映射成固定台词。"
    )
    strength = max(effective.values(), default=0.0)
    bonus = min(
        0.1,
        max(0.0, strength - 0.7) * 0.2
        + max(0.0, dynamic_delta) * 0.2,
    )
    return PersonaExpressionProfile(
        style_text=style_text,
        rules=tuple(rules),
        temperature_bonus=bonus,
        dynamic_delta=dynamic_delta,
        interaction_heat=heat,
        relationship_stage=relationship_stage,
    )
