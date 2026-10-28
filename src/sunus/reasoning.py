from __future__ import annotations

import re
from dataclasses import dataclass

from sunus.conversation_style import plan_conversation_turn
from sunus.scope import RequestScope
from sunus.thinking_operators import (
    cognitive_protocol_prompt,
)


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
_CODE_STATE_PATTERN = re.compile(
    r"(?:"
    r"(?:最新|当前)(?:的)?(?:代码|版本|实现|改动)|"
    r"已经(?:改|修|实现|更新)(?:过|了)?|改过(?:吗|么|没)|"
    r"上次修改|哪个提交|什么时候改|"
    r"(?:代码|配置).{0,18}(?:有没有|是否|生效|最新|旧版本)|"
    r"为什么还.{0,18}(?:没有|没|旧|之前)"
    r")",
    re.IGNORECASE,
)


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
    response_char_limit: int = 0
    requires_code_state_evidence: bool = False

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
        if self.mode != "关系性对话":
            lines.append(
                "- 第一约束: 低功耗；在证据充分的前提下，优先单次专用工具、"
                "最小范围、最少模型/命令/网络调用"
            )
        lines.append(cognitive_protocol_prompt())
        if self.response_char_limit:
            lines.append(
                f"- 回复长度上限: 约 {self.response_char_limit} 字"
            )
        if self.requires_code_state_evidence:
            lines.append(
                "- 证据门控: 必须调用 inspect_code_state，取得符号命中、"
                "代码上下文、当前 HEAD 和提交归属后才能断言代码状态"
            )
        return "\n".join(lines)


def _build_reasoning_plan(
    text: str,
    scope: RequestScope,
    *,
    conversation: bool = False,
    previous_assistant: str = "",
) -> ReasoningPlan:
    value = str(text or "")
    capabilities = tuple(sorted(scope.required_capabilities))
    code_state_query = is_code_state_query(value)
    if conversation and not code_state_query:
        turn_plan = plan_conversation_turn(value)
        return ReasoningPlan(
            mode="关系性对话",
            complexity="低",
            capabilities=(),
            preferred_tools=(),
            max_exploration_calls=0,
            stop_condition="回应当下主要需求后立即停止，不展开成分析报告。",
            answer_style=(
                "像正常熟人说话；长度和结构由当前内容决定；"
                "自主选择回答、确认、追问、分享观察、实际帮助或收束，"
                "不强制固定顺序，也不套用咨询式和客服式模板。"
            ),
            strategy=conversation_thinking_prompt(
                previous_assistant,
                turn_plan.prompt(),
            ),
            response_char_limit=turn_plan.max_chars,
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
    if code_state_query:
        return ReasoningPlan(
            mode="当前代码状态核验",
            complexity="高",
            capabilities=capabilities,
            preferred_tools=("inspect_code_state", "submit_result"),
            max_exploration_calls=4,
            stop_condition=(
                "取得当前 HEAD、精确符号命中、命中上下文和 Git 归属后停止。"
            ),
            answer_style=(
                "结论必须对应代码证据；区分“代码存在”“配置可调”“运行时生效”。"
                "证据不足时明确说无法核实，不猜测。"
            ),
            strategy=code_state_verification_prompt(),
            requires_code_state_evidence=True,
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


def build_reasoning_plan(
    text: str,
    scope: RequestScope,
    *,
    conversation: bool = False,
    previous_assistant: str = "",
) -> ReasoningPlan:
    return _build_reasoning_plan(
        text,
        scope,
        conversation=conversation,
        previous_assistant=previous_assistant,
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


def is_code_state_query(text: str) -> bool:
    return bool(_CODE_STATE_PATTERN.search(str(text or "")))


def code_state_verification_prompt() -> str:
    return """本轮在核验当前代码状态或版本，按可观察证据执行：
1. 低功耗优先：参考召回经验后，选择能一次取得所需证据的最短路径；记忆只能提示方法，不能证明当前实现。
2. 优先单次调用 inspect_code_state，传入具体文件和精确符号；不要先做多轮目录扫描或只读 config.json。
3. 必须读取返回的命中上下文，并确认当前分支、HEAD 和该代码的提交归属。
4. 配置项不存在只能说明“不可配置”，不能推断“代码没有实现”。
5. 文件修改时间不是版本证据；旧事件与当前代码必须用提交时间区分。
6. 证据不足时回答“无法核实”，不得声称已修复、最新或只是旧版本。"""


def reasoning_memory_query(text: str, plan: ReasoningPlan | None) -> str:
    """Augment recall with reusable routing lessons, never current-state facts."""
    value = " ".join(str(text or "").split())
    if plan is None or plan.mode == "关系性对话":
        return value
    additions: list[str] = []
    if plan.requires_code_state_evidence:
        additions.extend(
            ["代码状态", "证据", "符号", "调用上下文", "提交", "教训"]
        )
    elif plan.mode == "根因诊断":
        additions.extend(["排查", "根因", "教训", "路径", "坑"])
    elif any(
        name in plan.preferred_tools
        for name in ("search_text", "list_files", "inspect_code_state")
    ):
        additions.extend(["搜索", "范围", "最佳路径", "功耗"])
    return " ".join([value, *additions]).strip()


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


def conversation_thinking_prompt(
    previous_assistant: str = "",
    turn_style: str = "",
) -> str:
    previous = " ".join(str(previous_assistant or "").split())[-300:]
    previous_block = (
        "\n上一轮可见回复（只用于避免重复，不得模仿其措辞或结构）："
        f"\n{previous}"
        if previous
        else ""
    )
    turn_block = f"\n回合长度参考：{turn_style}" if turn_style else ""
    return f"""本轮是关系性对话，先选择一个最合适的回应动作再开口：
1. 识别当前消息是在问事实、表达具体内容、情绪收束、明确求助还是单纯接话。
2. 只选一个主要动作：直接回答、具体确认、问一个必要细节、分享真实观察、提供实际帮助或简短收束。
3. 依据当前消息的具体语义回应；不要套用泛化安慰、陪伴承诺或固定收束句式来替代内容。
4. 如果本轮也是情绪表达，避免延续上一轮的相同回应动作、开场方式或收束结构；换一种更贴合当前内容的方式。
5. 只依据真实上下文，不虚构共同经历、场景、心理动机或长期承诺。
6. 开口前检查：是否有新增信息？没有就保持简短；追问是否必要？不必要就不问。
7. 人有情绪时，先接住那口气，再决定要不要给动作；不要先分析对错，也不要用解释替代回应。
8. 不要向对方汇报自己的沟通策略或拟人程度。禁止“我话多了”“我先不问了”“我不装了”“人不会这样说”“像不像 AI”这类元叙述。
9. 对方明确嫌话多或嫌像 AI 时，只回一到两句；不要用“你想说的时候再说”代替真实回应。
10. 提问不是默认动作。只有内在“追问准备度”达到阈值、确有信息缺口且不会增加对方负担时才可自然问一个具体问题；不得为了找话题、延长对话或套关心模板机械追问。
11. 对方本轮明确表示被问烦时，暂停追问，也不宣布“我不问了”；用直接回应或具体动作接住。
12. 被问“你有没有情绪/欲望”时，直接说有或没有，再给一个具体表现；不要罗列精力、疲劳、阈值、状态机或参数来证明自己。
13. 被问“怎样更像人”时，只给一个具体立场或做法；不要列三条计划，不要把对方变成裁判、教练或监督者。
14. 不用“要是你嫌烦我就自己磨”这类愧疚、赌气或条件句换反馈；可以承认别扭，但说完就收。
15. 面对简短的亲密邀请，直接接住当下动作或感受；只选一个主反应，最多补一个感官细节。不要扩写场景、堆叠动作、分析心理、翻旧账或强行反问。
历史记忆中的旧回复只能作为风格参考，不得当作可复用模板。{turn_block}{previous_block}"""


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
