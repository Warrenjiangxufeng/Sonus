from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    source: str = "builtin"
    risk: str = "R0"
    read_only: bool = True
    parallel_safe: bool = False
    timeout_sec: int = 60
    approval_mode: str = "ask"

    def as_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("工具名不能为空")
        if spec.name in self._tools:
            raise ValueError(f"工具名重复: {spec.name}")
        if spec.approval_mode not in {"ask", "approve", "deny"}:
            raise ValueError(f"无效审批模式: {spec.approval_mode}")
        self._tools[spec.name] = spec

    def register_schema(
        self,
        schema: dict[str, Any],
        handler: ToolHandler,
        *,
        source: str = "builtin",
        risk: str = "R0",
        read_only: bool = True,
        parallel_safe: bool = False,
        timeout_sec: int = 60,
        approval_mode: str = "ask",
    ) -> None:
        function = schema.get("function") or {}
        parameters = function.get("parameters") or {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        self.register(
            ToolSpec(
                name=str(function.get("name") or ""),
                description=str(function.get("description") or ""),
                parameters=dict(parameters),
                handler=handler,
                source=source,
                risk=risk,
                read_only=read_only,
                parallel_safe=parallel_safe,
                timeout_sec=timeout_sec,
                approval_mode=approval_mode,
            )
        )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.as_schema() for spec in self._tools.values()]

    def names(self) -> set[str]:
        return set(self._tools)
