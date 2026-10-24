from __future__ import annotations

import re
from dataclasses import dataclass


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
_MUTATION_PATTERN = re.compile(
    r"(?:修改|修复|创建|新建|写入|写到|添加|新增|增加|删除|移除|"
    r"重命名|移动|复制|实现|重构|重写|更新|替换|补上|补全|"
    r"生成|保存|落盘|改掉|改成|改一下|编写|写文档|整理|补充|"
    r"清理|格式化|优化|输出到|输出文档|输出文件|写一|写个|写份|"
    r"write|create|edit|modify|fix|implement|refactor|generate|save|"
    r"delete|rename|update)",
    re.IGNORECASE,
)
_COMMAND_PATTERN = re.compile(
    r"(?:运行|执行|测试|构建|编译|启动|安装|检查|扫描|查找|搜索|"
    r"跑一下|跑一遍|验证|git|pytest|run|test|build|install|scan|check)",
    re.IGNORECASE,
)
_NETWORK_PATTERN = re.compile(
    r"(?:访问|下载|请求|联网|打开网址|抓取|"
    r"download|fetch|request|http)",
    re.IGNORECASE,
)
_SUGGESTION_PATTERN = re.compile(
    r"(?:建议|方案|优化|怎么|如何|是否|能否|可选|下一步|设计)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RequestScope:
    text: str
    allows_mutation: bool
    allows_commands: bool
    allows_network: bool
    allows_suggestions: bool


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
    return RequestScope(
        text=stripped,
        allows_mutation=(
            side_effect_allowed and _positive_action(_MUTATION_PATTERN, stripped)
        )
        or confirm_only,
        allows_commands=(
            side_effect_allowed and _positive_action(_COMMAND_PATTERN, stripped)
        )
        or confirm_only,
        allows_network=(
            side_effect_allowed and _positive_action(_NETWORK_PATTERN, stripped)
        )
        or confirm_only,
        allows_suggestions=question
        or bool(_SUGGESTION_PATTERN.search(stripped)),
    )
