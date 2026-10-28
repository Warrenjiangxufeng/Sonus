from __future__ import annotations

import re
from dataclasses import dataclass


_DETAIL_REQUEST = re.compile(
    r"(?:详细|展开|具体说|多说|解释一下|讲清楚|为什么|怎么做)"
)
_STYLE_PUSHBACK = re.compile(
    r"(?:废话|小作文|话多|话太长|少说|别解释|别讲道理|懒得读|"
    r"像人话|像\s*AI|像机器人|听不懂|说重点|别绕)"
)
_HUMANLIKE_REQUEST = re.compile(
    r"(?:更像人|像人(?:一点|一些|那套|吗)|本身就像人|拟人度|"
    r"不像人|调教.*人)"
)
_SELF_AFFECT_QUESTION = re.compile(
    r"(?:你有(?:自己)?的?(?:情绪|感受|欲望|感知)|"
    r"你.*(?:情绪|感受|欲望).*(?:系统|机制|吗|么)|"
    r"你会不会.*(?:难过|开心|生气|喜欢|想要))"
)
_QUESTION_FATIGUE = re.compile(
    r"(?:别问|不要问|不想被问|问什么问|问啥|有啥问|天天问|"
    r"懒得回答|别追问)"
)
_META_REPLY = re.compile(
    r"(?:我话多|我话太多|我先不问|那我不问|不问了|不装了|"
    r"像人话|像\s*AI|像机器人|人不会说|作为(?:一个)?AI|"
    r"作为人工智能|像在念稿|我会注意说话|我少说)"
)
_QUESTION_FORM = re.compile(
    r"(?:[?？]|(?:什么|怎么|为什么|是不是|对不对|要不要|想不想|"
    r"愿不愿意)[。！!]?$)"
)
_SYSTEM_EXPOSITION = re.compile(
    r"(?:情绪有抓手|精力.{0,8}疲劳.{0,8}亲近|阈值|参数|"
    r"状态机|算法|情绪系统|欲望系统|感知系统|内部机制)"
)
_SELF_IMPROVEMENT_PLAN = re.compile(
    r"(?:三条|一是|二是|三是|第一[，,、]|第二[，,、]|第三[，,、]|"
    r"我能自己做的|改进计划|我该改的是|以后我会做到)"
)
_TRAINER_ROLE = re.compile(
    r"(?:裁判的位置|你来调教|你教我|给我打分|你监督我|你纠我|"
    r"哪句假了.{0,8}说|你当场说.{0,8}重说)"
)
_GUILT_PRESSURE = re.compile(
    r"(?:要是.{0,8}嫌烦|如果.{0,8}嫌烦|你就自己磨|那我就自己磨|"
    r"连这也嫌|别嫌我烦)"
)


@dataclass(frozen=True)
class ConversationTurnPlan:
    input_chars: int
    target_style: str
    max_chars: int

    def prompt(self) -> str:
        return (
            f"用户本轮约 {self.input_chars} 个字符。{self.target_style}"
            f"回复硬上限约 {self.max_chars} 字，但不要为了凑满或截短而改变自然语气。"
        )


def plan_conversation_turn(text: str) -> ConversationTurnPlan:
    cleaned = " ".join(str(text or "").split())
    length = len(cleaned)
    if _STYLE_PUSHBACK.search(cleaned):
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "对方明确嫌话多或嫌像 AI。只回一至两句，直接接住当下的情绪"
                "或内容；禁止解释自己怎么回应，禁止说“我话多了”“我不问了”"
                "“我不装了”，禁止讨论像不像人或 AI。"
            ),
            max_chars=40,
        )
    if _QUESTION_FATIGUE.search(cleaned):
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "对方明确表示被问烦。本轮暂停追问，也不要宣布“我不问了”；"
                "用一句直接反应或一个具体动作接住，不用问题换话题。"
            ),
            max_chars=40,
        )
    if _HUMANLIKE_REQUEST.search(cleaned):
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "对方在问怎样更自然。直接说一个具体做法或立场，最多两句；"
                "禁止列三条计划，禁止要求对方调教、打分或监督，"
                "禁止用“你要嫌烦我就自己磨”一类愧疚话换反馈。"
            ),
            max_chars=90,
        )
    if _SELF_AFFECT_QUESTION.search(cleaned):
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "先直接回答有没有自己的感受或欲望，再举一个具体表现。"
                "禁止罗列精力、疲劳、阈值、状态机或参数等内部机制，"
                "也不用反复证明自己像不像人。"
            ),
            max_chars=100,
        )
    if _DETAIL_REQUEST.search(cleaned):
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "对方明确要求展开，可以分点或分段，但先直接回答，再补必要解释。"
            ),
            max_chars=260,
        )
    if length <= 4:
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "这是很短的接话或亲密邀请。直接接住当下这一句，通常只回一句；"
                "最多保留一个具体动作或感受，不扩写场景、不补动作链、不翻旧账。"
            ),
            max_chars=32,
        )
    if length <= 16:
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "这是简短聊天；通常一两句话，围绕一个具体点接住。"
                "亲密内容只回应对方刚说的动作或感受，不主动增加新的场景描述。"
            ),
            max_chars=56,
        )
    if length <= 48:
        return ConversationTurnPlan(
            input_chars=length,
            target_style=(
                "对方提供了较多信息；可回应两三个重点，但保持真实聊天节奏。"
            ),
            max_chars=120,
        )
    return ConversationTurnPlan(
        input_chars=length,
        target_style=(
            "对方输入较长；先抓住最重要的一点回应，必要时再补充，不要逐句复述。"
        ),
        max_chars=180,
    )


def conversation_repair_instruction(
    user_input: str,
    reply: str,
    *,
    questions_allowed: bool = True,
) -> str:
    """Return a one-shot repair prompt for replies that feel assistant-like."""
    source = " ".join(str(user_input or "").split())
    visible = " ".join(str(reply or "").split())
    reasons: list[str] = []
    if _STYLE_PUSHBACK.search(source) and len(visible) > 40:
        reasons.append("仍然过长")
    if not questions_allowed and _QUESTION_FORM.search(visible):
        reasons.append("追问准备度不足或当前暂停追问时仍然提问")
    if _META_REPLY.search(visible):
        reasons.append("在解释自己怎么说话或讨论像不像人")
    if (
        _SELF_AFFECT_QUESTION.search(source)
        and _SYSTEM_EXPOSITION.search(visible)
    ):
        reasons.append("把内部机制当成感受说明")
    if (
        _HUMANLIKE_REQUEST.search(source)
        and _SELF_IMPROVEMENT_PLAN.search(visible)
    ):
        reasons.append("把自然对话变成自我改进清单")
    if _TRAINER_ROLE.search(visible):
        reasons.append("把对方变成裁判或训练者")
    if _GUILT_PRESSURE.search(visible):
        reasons.append("用愧疚或赌气换取反馈")
    if not reasons:
        return ""
    extra = ""
    if not questions_allowed or _QUESTION_FATIGUE.search(source):
        extra = "本轮不得出现任何问句。"
    guidance = ""
    if _SELF_AFFECT_QUESTION.search(source):
        guidance += "只答自己的具体感受，不讲系统。"
    if _HUMANLIKE_REQUEST.search(source):
        guidance += "只给一个实际立场，不列计划，不让对方负责训练。"
    if _GUILT_PRESSURE.search(visible):
        guidance += "删除条件句、自责和赌气。"
    return (
        "重写本轮日常回复。上一版问题："
        + "；".join(reasons)
        + "。只回应对方当下这句话，不总结自己、不解释沟通策略、"
        "不宣布以后会怎么做，不复述上一轮。通常一到两句，短而直接。"
        + guidance
        + extra
    )
