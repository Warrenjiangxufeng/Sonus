from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_CURRENT_CODE_CLAIM = re.compile(
    r"(?:"
    r"当前(?:代码|实现|版本)|最新(?:代码|版本|实现|改动)|"
    r"代码(?:里|中|已经|还没)|配置项|可配置|不可配置|"
    r"已经(?:修复|改|实现|更新)|没有(?:实现|独立预算|配置项)|"
    r"旧版本|哪个提交|HEAD|提交时间|运行时生效|实际生效"
    r")",
    re.IGNORECASE,
)


def claims_current_code_state(text: str) -> bool:
    return bool(_CURRENT_CODE_CLAIM.search(str(text or "")))


@dataclass
class CodeStateEvidenceGate:
    """Require observable current-code evidence before a version claim."""

    enabled: bool = False
    prompt_required: bool = False
    attempted: bool = False
    verified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def record_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        content: str,
    ) -> None:
        if not self.enabled or name != "inspect_code_state":
            return
        self.attempted = True
        if not ok:
            return
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self.evidence = payload
        self.verified = bool(
            payload.get("head")
            and payload.get("target_exists")
            and payload.get("version_verified")
        )

    def missing_instruction(self, reply: str = "") -> str:
        if (
            not self.enabled
            or self.verified
            or not claims_current_code_state(reply)
        ):
            return ""
        if not self.attempted:
            return (
                "回答中出现了当前代码状态或版本断言，回答前必须调用只读工具 "
                "inspect_code_state，传入具体 path 和 symbol。"
                "不得只查 config.json 或凭记忆推断代码是否存在、是否最新。"
            )
        return (
            "inspect_code_state 尚未取得完整证据。必须确认符号命中、"
            "当前 HEAD 和该行的 Git 归属；若工具无法取得版本，"
            "明确回答无法核实版本，不得声称已修复、最新或旧版本。"
        )

    def prompt_block(self) -> str:
        if not self.enabled or not self.prompt_required:
            return ""
        return (
            "当前代码状态证据门控：必须先调用 inspect_code_state(path, symbol)。"
            "只有同时拿到符号命中、代码上下文、当前 HEAD 和提交归属后，"
            "才能回答代码是否存在、是否可配置、是否最新。"
        )

    def unverified_answer(self) -> str:
        if self.attempted and self.evidence:
            return (
                "当前无法完整核实代码状态：只取得了部分代码证据，"
                "尚未确认当前离线代码与提交归属。请允许读取目标调用点并运行只读 Git 检查后重试。"
            )
        return (
            "当前无法核实代码状态：本轮没有取得目标符号、调用上下文和 Git 版本证据。"
            "不能用配置文件缺失或旧记忆推断当前实现。"
        )
