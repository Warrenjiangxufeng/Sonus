from __future__ import annotations

import re
from dataclasses import dataclass, replace


_QUESTION_PREFIX = re.compile(
    r"^\s*(?:如何|怎么|怎样|为什么|为何|是否|能否|可否|可不可以|"
    r"你觉得|你认为|有哪些|有什么|"
    r"可以.{0,16}吗[？?]?)"
)
_REQUEST_MARKER = re.compile(r"(?:^|\s)(?:请|帮我|帮忙|麻烦|执行|开始|做吧|继续)")
_CONFIRM_ONLY = re.compile(
    r"^\s*(?:(?:可以|好|行)[，,\s]*)?"
    r"(?:继续|做吧|开始|执行|按这个来|按方案做)"
    r"[。.!！\s]*$|^\s*(?:可以|好|行)[。.!！\s]*$"
)
_NEGATION_BEFORE = re.compile(
    r"(?:不要|别|无需|不用|不需要|不能|不可|不许|不准|禁止|避免).{0,8}$"
)
_NEGATED_FOLLOWUP = re.compile(
    r"^\s*(?:不要|别|无需|不用|不需要|不能|不可|不许|不准|禁止|先别)"
)
_MUTATION_PATTERN = re.compile(
    r"(?:修改|修复|创建|新建|写入|写到|添加|新增|增加|删除|移除|"
    r"重命名|移动|复制|实现|重构|重写|更新|替换|补上|补全|"
    r"生成|保存|落盘|改掉|改成|改一下|改下|直接改|给我改|动手改|"
    r"继续改|按刚才.{0,8}改|照刚才.{0,8}改|编写|写文档|整理|补充|"
    r"清理|格式化|优化|输出到|输出文档|输出文件|写一|写个|写份|"
    r"write|create|edit|modify|fix|implement|refactor|generate|save|"
    r"delete|rename|update)",
    re.IGNORECASE,
)
_COMMAND_PATTERN = re.compile(
    r"(?:运行|执行|测试|构建|编译|启动|安装|检查|扫描|查找|搜索|"
    r"跑一下|跑一遍|验证|查看网页|读取网页|浏览器|标签页|"
    r"网页.{0,8}(?:展示|内容|文本)|"
    r"git|pytest|run|test|build|install|scan|check|"
    r"(?:^|[，,。.!！？?\s]|就|快|给我)装(?:一下|上|好|个|吧|啊)?"
    r"(?=$|[，,。.!！？?\s]))",
    re.IGNORECASE,
)
_NETWORK_PATTERN = re.compile(
    r"(?:访问|下载|请求|联网|网络|搜索|搜一下|搜搜|查一下|"
    r"(?<!已)打开网页|访问网页|抓取网页|网页地址|"
    r"download|fetch|request|http)",
    re.IGNORECASE,
)
_SUGGESTION_PATTERN = re.compile(
    r"(?:建议|方案|优化|怎么|如何|是否|能否|可选|下一步|设计)",
    re.IGNORECASE,
)
_TECHNICAL_PATTERN = re.compile(
    r"(?:代码|项目|文件|目录|仓库|命令|终端|脚本|测试|用例|接口|"
    r"读取|查看|打开|列出|搜索|查找|"
    r"数据库|配置|部署|安装|运行|构建|编译|调试|修复|修改|实现|"
    r"重构|架构|模型|提示词|上下文|记忆|进程|线程|窗口|"
    r"\b(?:api|cli|git|python|shell|pytest|codex|agent|sunus|token|"
    r"prompt|http|url|github|bug)\b)",
    re.IGNORECASE,
)
_CASUAL_PATTERN = re.compile(
    r"(?:哈喽|你好|嗨|早安|午安|晚安|在吗|在干嘛|干嘛呢|"
    r"聊天|聊聊|陪我|无聊|心情|开心|难过|烦恼|好累|压力|"
    r"喜欢|想念|相亲|对象|感情|冷淡|女孩|男孩|"
    r"谢谢|多谢|哈哈|嘿嘿|抱抱|再见|拜拜|"
    r"\b(?:hi|hello|thanks|good\s+night)\b)",
    re.IGNORECASE,
)
_RELATIONAL_PATTERN = re.compile(
    r"(?:陪你|陪我|不想聊工作|太较真|绷得|别教育我|像客服|"
    r"正常朋友|聊得像人|像机器人|你自己哪里|判断一个人|"
    r"真的靠谱|嘴上会说|说实话|你觉得我|我们聊)",
    re.IGNORECASE,
)
_PATH_PATTERN = re.compile(r"(?:^|[\s`'\"])[~./][\w./-]+")
_REFERENTIAL_FOLLOWUP = re.compile(
    r"^\s*(?:就|那)?(?:试一下|试试|继续|接着来|按你说的|照这个做|"
    r"按刚才(?:的)?(?:来|做|继续|改|执行)?|"
    r"照刚才(?:的)?(?:来|做|继续|改|执行)?|就这么办|来吧|开始吧)"
    r"(?:[，,。.!！\s]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RequestScope:
    text: str
    allows_mutation: bool
    allows_commands: bool
    allows_network: bool
    allows_suggestions: bool
    required_capabilities: frozenset[str] = frozenset()


def _positive_action(pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 16) : match.start()]
        if not _NEGATION_BEFORE.search(prefix):
            return True
    return False


def classify_request(text: str) -> RequestScope:
    stripped = text.strip()
    confirm_only = bool(_CONFIRM_ONLY.fullmatch(stripped))
    question = bool(_QUESTION_PREFIX.search(stripped))
    request = bool(_REQUEST_MARKER.search(stripped))
    side_effect_allowed = confirm_only or not (question and not request)
    allows_mutation = (
        side_effect_allowed and _positive_action(_MUTATION_PATTERN, stripped)
    ) or confirm_only
    allows_commands = (
        side_effect_allowed and _positive_action(_COMMAND_PATTERN, stripped)
    ) or confirm_only
    allows_network = (
        side_effect_allowed and _positive_action(_NETWORK_PATTERN, stripped)
    ) or confirm_only
    required: set[str] = set()
    if allows_mutation:
        required.add("write")
    if allows_commands:
        required.add("command")
    if allows_network:
        required.add("network")
    if allows_mutation and re.search(r"(?:测试|验证|运行|pytest|unittest)", stripped):
        required.add("command")
    return RequestScope(
        text=stripped,
        allows_mutation=allows_mutation,
        # 代码修改必须能够运行对应测试，否则 Agent 只能盲改。
        allows_commands=allows_commands or allows_mutation,
        allows_network=allows_network,
        allows_suggestions=question
        or bool(_SUGGESTION_PATTERN.search(stripped)),
        required_capabilities=frozenset(required),
    )


def classify_request_with_context(
    text: str,
    *,
    previous_assistant: str = "",
    previous_scope: RequestScope | None = None,
) -> RequestScope:
    scope = classify_request(text)
    if (
        (previous_scope is None and not previous_assistant.strip())
        or not _REFERENTIAL_FOLLOWUP.search(text)
        or _QUESTION_PREFIX.search(text)
        or _NEGATED_FOLLOWUP.search(text)
    ):
        return scope
    previous = previous_scope or classify_request(previous_assistant)
    if not (
        previous.allows_mutation
        or previous.allows_commands
        or previous.allows_network
    ):
        return scope
    return replace(
        scope,
        allows_mutation=scope.allows_mutation or previous.allows_mutation,
        allows_commands=scope.allows_commands or previous.allows_commands,
        allows_network=scope.allows_network or previous.allows_network,
        required_capabilities=scope.required_capabilities
        | previous.required_capabilities,
    )


def is_referential_followup(text: str) -> bool:
    return bool(_REFERENTIAL_FOLLOWUP.search(str(text or "")))


def is_casual_conversation(
    text: str,
    *,
    previous_assistant: str = "",
) -> bool:
    """Return whether a turn may be social chat rather than project work."""
    return conversation_hint(
        text,
        previous_assistant=previous_assistant,
    ) in {"casual", "ambiguous"}


def conversation_hint(
    text: str,
    *,
    previous_assistant: str = "",
) -> str:
    """Classify unambiguous turns and mark short follow-ups for model review."""
    stripped = text.strip()
    if not stripped or len(stripped) > 160:
        return "task"
    if _TECHNICAL_PATTERN.search(stripped) or _PATH_PATTERN.search(stripped):
        return "task"
    scope = classify_request(stripped)
    if previous_assistant.strip() and _REFERENTIAL_FOLLOWUP.search(stripped):
        return "task"
    if scope.allows_mutation or scope.allows_commands or scope.allows_network:
        return "task"
    if _RELATIONAL_PATTERN.search(stripped):
        return "casual"
    if _CASUAL_PATTERN.search(stripped):
        return "casual"
    if len(stripped) <= 24 and not _MUTATION_PATTERN.search(stripped):
        return "ambiguous"
    return "task"
