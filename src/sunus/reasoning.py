from __future__ import annotations

import re


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


def is_diagnostic_request(text: str) -> bool:
    return bool(_DIAGNOSTIC_PATTERN.search(str(text or "")))


def is_status_query_request(text: str) -> bool:
    value = str(text or "")
    return bool(_STATUS_QUERY_PATTERN.search(value)) and not is_diagnostic_request(
        value
    )


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
