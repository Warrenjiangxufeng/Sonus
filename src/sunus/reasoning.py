from __future__ import annotations

import re
from dataclasses import dataclass

from sunus.scope import RequestScope


_DIAGNOSTIC_PATTERN = re.compile(
    r"(?:为什么|为何|原因|根因|排查|诊断|怎么回事|"
    r"检查.{0,12}(?:问题|异常|失败)|"
    r"哪里.{0,8}(?:错|出问题)|是否.{0,8}(?:问题|异常)|失败|报错|异常|"
    r"找不到|没有.{0,8}(?:工具|数据|记录|文件)|无法|不能|用不了|装不了)",
    re.IGNORECASE,
)

_STATUS_QUERY_PATTERN = re.compile(
    r"(?:git\s+status|ahead|behind|"
    r"(?:有没有|是否|看看|检查|查看|确认).{0,30}"
    r"(?:提交|推送|同步|状态|干净|改动)|"
    r"(?:提交|推送|同步).{0,16}(?:没有|没|吗|状态|完成))",
    re.IGNORECASE,
)
_COMPLEX_PATTERN = re.compile(
    r"(?:为什么|根因|排查|诊断|崩溃|熔断|异常|失败|修复|实现|重构|"
    r"迁移|架构|多步骤|全量测试|浏览器|跨轮|排查机制)",
    re.IGNORECASE,
)
_LOOKUP_PATTERN = re.compile(
    r"(?:只读任务|只给.*(?:结论|结果|文件名)|只看|读取|查看|检查|"
    r"告诉我.*(?:内容|结果|文件|页面|标题|URL))",
    re.IGNORECASE,
)
_PLAYWRIGHT_PATTERN = re.compile(
    r"(?:playwright-cli|playwright\s+cli)",
    re.IGNORECASE,
)
_BROWSER_PATTERN = re.compile(r"(?:浏览器|标签页|已打开网页|网页展示)", re.I)
_TEMPORAL_PATTERN = re.compile(r"(?:上次|刚才|最近)", re.I)


@dataclass(frozen=True)
class ReasoningPlan:
    mode: str
    complexity: str
    capabilities: tuple[str, ...]
    preferred_tools: tuple[str, ...]
    max_exploration_calls: int
    stop_condition: str
    answer_style: str
    drop_details: bool = False
    strategy: str = ""

    def prompt(self) -> str:
        lines = [
            "代码生成的单层推理计划（只执行这一套，不叠加其他模式提示）：",
            f"- 模式: {self.mode}",
            f"- 复杂度: {self.complexity}",
            f"- 必需能力: {', '.join(self.capabilities) or '无副作用'}",
            f"- 首选路径: {', '.join(self.preferred_tools) or '直接回答'}",
            f"- 探索上限: {self.max_exploration_calls} 个动作",
            f"- 停止条件: {self.stop_condition}",
            f"- 表达: {self.answer_style}",
        ]
        if self.strategy:
            lines.append(f"- 专项策略: {self.strategy}")
        return "\n".join(lines)


def build_reasoning_plan(
    text: str,
    scope: RequestScope,
    *,
    conversation: bool = False,
) -> ReasoningPlan:
    value = str(text or "")
    capabilities = tuple(sorted(scope.required_capabilities))
    if conversation:
        return ReasoningPlan(
            mode="关系性对话",
            complexity="低",
            capabilities=(),
            preferred_tools=(),
            max_exploration_calls=0,
            stop_condition="回应当下主要需求后立即停止，不展开成分析报告。",
            answer_style=(
                "像正常熟人说话，通常 1-3 句；不使用固定昵称、括号动作或强行撒娇；"
                "可以保留一点犹豫，不默认列点或做咨询式总结。"
            ),
            strategy=(
                "先判断对方要陪伴、理解、评价、建议还是纠正；只选一个回应动作；"
                "只依据真实上下文，不过度心理分析。"
            ),
        )
    if is_status_query_request(value):
        return ReasoningPlan(
            mode="简单状态查询",
            complexity="低",
            capabilities=capabilities,
            preferred_tools=("最小只读检查", "直接回答"),
            max_exploration_calls=3,
            stop_condition="拿到状态和关键标识即停止，不扩展未点名对象。",
            answer_style="首句给结论，1-3 句；details 留空，不复述工具日志。",
            drop_details=True,
        )
    if is_playwright_cli_request(value):
        return ReasoningPlan(
            mode="Playwright CLI 工具链",
            complexity="中",
            capabilities=capabilities or ("command",),
            preferred_tools=("inspect_environment", "playwright-cli", "inspect_browser"),
            max_exploration_calls=5,
            stop_condition=(
                "版本不兼容、登录页、验证码或现有 Chrome 状态无法复用时立即停止并说明下一步。"
            ),
            answer_style="报告实际状态、阻塞点和唯一下一步；不展示整段排障过程。",
            strategy=(
                "最多一次环境检查和一次包元数据；禁止伪造 process.versions、"
                "临时 Node shim 或修改 node_modules。现有 Chrome 与新 Playwright profile 分离。"
            ),
        )
    if is_diagnostic_request(value):
        tools = (
            ("inspect_runtime_history", "turn")
            if _TEMPORAL_PATTERN.search(value)
            else ("search_text", "read_file", "run_command")
        )
        return ReasoningPlan(
            mode="根因诊断",
            complexity="高",
            capabilities=capabilities,
            preferred_tools=tools,
            max_exploration_calls=6,
            stop_condition=(
                "证据足以区分假设后停止；两个以上候选无法区分时询问一个时间或对象锚点。"
            ),
            answer_style="结论、最多 3 条依据、1 条修复方向；不展示完整代码和工具日志。",
            strategy=(
                "至少形成两个竞争假设，并主动寻找能区分它们的证据；"
                "单个失败入口不能证明整体缺失；失败先归因为权限、路径、"
                "超时、模型或验证，再换路径。"
            ),
        )
    if _BROWSER_PATTERN.search(value):
        return ReasoningPlan(
            mode="浏览器只读查证",
            complexity="低",
            capabilities=capabilities or ("command",),
            preferred_tools=("inspect_browser", "submit_result"),
            max_exploration_calls=2,
            stop_condition=(
                "拿到页面标题、URL或正文即停止；权限关闭时说明一个恢复入口，不猜内容。"
            ),
            answer_style="直接给页面事实，1-3 句；details 留空。",
            drop_details=True,
            strategy="禁止剪贴板、模拟按键和多轮 osascript 试探。",
        )
    if is_simple_lookup_request(value):
        return ReasoningPlan(
            mode="只读查证",
            complexity="低",
            capabilities=capabilities,
            preferred_tools=("list_files", "search_text", "read_file"),
            max_exploration_calls=4,
            stop_condition="拿到点名对象的结果即停止，不扩大目录。",
            answer_style="直接结果，1-3 句；details 留空。",
            drop_details=True,
        )
    if scope.allows_mutation:
        return ReasoningPlan(
            mode="修改交付",
            complexity="中",
            capabilities=capabilities or ("write", "command"),
            preferred_tools=("read_file", "patch_file", "run_command", "submit_result"),
            max_exploration_calls=8,
            stop_condition="改动验证完成或出现明确阻塞即停止。",
            answer_style="简要说明改动、验证结果和剩余风险。",
            strategy="先读后改；修改后运行最相关验证。",
        )
    return ReasoningPlan(
        mode="一般任务",
        complexity="低",
        capabilities=capabilities,
        preferred_tools=("按需使用最小只读工具",),
        max_exploration_calls=4,
        stop_condition="足以回答时立即停止。",
        answer_style="结论先行，最长 3 个短段。",
    )


def is_diagnostic_request(text: str) -> bool:
    return bool(_DIAGNOSTIC_PATTERN.search(str(text or "")))


def is_status_query_request(text: str) -> bool:
    value = str(text or "")
    return bool(_STATUS_QUERY_PATTERN.search(value)) and not is_diagnostic_request(
        value
    )


def is_complex_request(text: str) -> bool:
    value = str(text or "")
    return bool(_COMPLEX_PATTERN.search(value)) and not is_status_query_request(
        value
    )


def is_simple_lookup_request(text: str) -> bool:
    value = str(text or "")
    return bool(_LOOKUP_PATTERN.search(value)) and not is_diagnostic_request(
        value
    )


def is_playwright_cli_request(text: str) -> bool:
    return bool(_PLAYWRIGHT_PATTERN.search(str(text or "")))


def playwright_cli_prompt() -> str:
    return """本轮明确涉及 playwright-cli，按固定检查链执行：
1. 先用一次 inspect_environment 确认 CLI、Node 版本和可用解释器；需要版本兼容信息时最多再看一次包元数据。
2. 如果 CLI 与当前 Node 不兼容，直接报告所需版本并请用户决定安装或改用其他路径；禁止伪造 process.versions、注入临时 shim、修改 node_modules 来假装兼容。
3. 明确区分“已打开的 Chrome”和“Playwright 新 Chromium profile”。Playwright 新 profile 不能复用现有 Chrome 登录态；用户要读现有页面时优先 inspect_browser 或已配置的 CDP，不要擅自打开登录页。
4. 打开、关闭、帮助查询和内容读取属于不同步骤；不要从浏览器安装目录、Electron 应用或全局包中碰运气。
5. 出现登录页、验证码或用户凭据要求时立即停止，向用户说明当前窗口和所需下一步，不尝试自行登录。"""


def lookup_query_prompt() -> str:
    return """本轮是只读查证任务：
1. 只检查用户点名的对象，用最少工具拿到结果。
2. 首句直接回答结果；能力失败时只说明阻塞点和下一步。
3. 不列工具日志、不截断粘贴整段内容、不重复调用已经给出答案的步骤。
4. submit_result.details 留空，通常用 1-3 句完成。"""


def complex_task_prompt() -> str:
    return """本轮属于复杂任务。在第一次工具调用前先完成内部规划：
1. 明确目标、约束、所需能力、成功判据和停止条件。
2. 按信息增益排序候选路径；优先用一个检查区分多个假设。
3. 工具失败后先归因（权限、路径、环境、模型、验证失败），再决定换路径；不要重复同一假设。
4. 已有证据足以回答，或两个以上候选无法区分时立即收束；如果缺少时间、对象或权限，直接询问一次。"""


def conversation_thinking_prompt() -> str:
    return """本轮是关系性对话，先做一次内部判断再开口：
1. 识别对方此刻主要需要什么：陪伴、被理解、一个真实评价、建议，还是对上一句的纠正。不要把所有需求都当成分析任务。
2. 只选择一个主要回应动作；先回应当下，不必一次解决全部问题。
3. 判断必须基于当前对话里真实出现的信息，不虚构共同经历，也不过度心理分析。
4. 开口前检查：像正常熟人会说的话吗？是否模板化、咨询师化、客服化，或为了亲昵而强行撒娇？如果像，重写得更直接自然。"""


def status_query_prompt() -> str:
    return """本轮是简单状态查询，不按根因诊断展开：
1. 只检查用户明确点名的仓库、目录或对象；同级的备份目录、worktree 和副本不主动审计，除非用户点名，或其状态会直接改变答案。
2. 使用最少只读检查，直接回答是否完成、当前状态和关键标识（如 commit 哈希）。
3. 首句给出明确结论，通常 1-3 句；不列文件清单，不复述命令输出，不重复证据。
4. submit_result.details 留空，把必要标识直接写进 summary。"""


def diagnostic_protocol_prompt() -> str:
    return """本轮是诊断或根因分析，按证据门控执行（内部遵守，不要输出隐藏推理）：
1. 先区分“观察到的现象”“期望结果”“实际差异”，不要把推测写成事实。
2. 从当前配置、实时环境和工具输出获取权威事实；历史记忆只作线索，不作最终证据。
3. 至少形成两个竞争假设，并主动寻找能区分它们的证据。
4. 优先尝试证伪第一个直觉解释；单个失败入口不能证明工具或数据整体不存在。
5. 对“没有、找不到、无法、未安装、配置错误”等强结论，必须核对对应权威入口：
   - 数据问题：核对当前运行时数据目录及数据库/事件/会话。
   - 本机能力问题：调用 inspect_environment，或实际查询 PATH、版本和解释器。
   - 执行失败问题：保留命令、退出码和原始错误，再判断是权限、环境还是代码问题。
6. 只有现有证据足以排除替代解释时才下结论；否则明确写“未确认”和还缺哪项证据。
7. 回答中给出结论、关键证据、已排除的替代解释和剩余不确定性，不输出思维链。
8. 对外回答保持简洁：先结论，最多 3 条关键依据和 1 条修复方向；除非用户明确索要，不展示完整代码、diff、函数实现或大段日志。"""
