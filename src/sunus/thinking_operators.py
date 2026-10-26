from __future__ import annotations

from dataclasses import dataclass


MAX_OPERATORS = 4

OPERATOR_LIBRARY = (
    "演绎",
    "归纳",
    "溯因",
    "类比",
    "因果",
    "概率",
    "发散",
    "组合创造",
    "反事实",
    "心智模拟",
    "批判",
    "系统",
    "决策优化",
    "元认知",
    "叙事时间",
    "心理理论",
    "伦理实践智慧",
    "具身直觉",
)

MODE_ROUTES: dict[str, tuple[str, ...]] = {
    "关系性对话": ("叙事时间", "心理理论"),
    "简单状态查询": ("演绎", "批判"),
    "Playwright CLI 工具链": ("溯因", "因果", "反事实", "元认知"),
    "根因诊断": ("溯因", "因果", "反事实", "批判"),
    "浏览器只读查证": ("演绎", "批判"),
    "只读查证": ("演绎", "批判"),
    "修改交付": ("发散", "反事实", "决策优化", "元认知"),
    "一般任务": ("归纳", "类比", "概率", "元认知"),
}

CONTRAST_CHECKS: dict[str, str] = {
    "演绎": "检查前提和边界是否成立",
    "归纳": "寻找反例或独立样本",
    "溯因": "比较可解释现象的替代原因",
    "类比": "检查关键差异是否破坏迁移",
    "因果": "检查混杂、共同原因和反向因果",
    "概率": "检查基础率、样本偏差和证据独立性",
    "发散": "用约束、成本和可验证性筛掉候选",
    "组合创造": "检查组合后的新冲突和风险",
    "反事实": "确认改变条件后的推演与证据一致",
    "心智模拟": "寻找现实约束和异常路径",
    "批判": "检查最强反方解释和隐含假设",
    "系统": "用局部检查验证系统解释",
    "决策优化": "检查代价、可逆性和尾部风险",
    "元认知": "确认策略是否增加证据并设置停止条件",
    "叙事时间": "检查时间、指代和缺失环节",
    "心理理论": "寻找不依赖意图或情绪的替代解释",
    "伦理实践智慧": "检查受影响方、权利边界和情境差异",
    "具身直觉": "用可验证证据检查直觉是否失真",
}


@dataclass(frozen=True)
class OperatorSelection:
    operators: tuple[str, ...]
    contrast_checks: tuple[str, ...]

    def prompt(self) -> str:
        return (
            f"- 当前首选思维算子: {', '.join(self.operators)}\n"
            f"- 反向复核: {'；'.join(self.contrast_checks)}"
        )


def select_operators(
    *,
    mode: str,
    complexity: str,
) -> OperatorSelection:
    """Use only the already-computed plan mode; no model, tool, or I/O cost."""
    operators = list(MODE_ROUTES.get(mode, MODE_ROUTES["一般任务"]))
    if complexity == "低":
        operators = operators[:2]
    elif "元认知" not in operators and len(operators) < MAX_OPERATORS:
        operators.append("元认知")
    selected = tuple(operators[:MAX_OPERATORS])
    contrasts = tuple(
        CONTRAST_CHECKS[operator]
        for operator in selected
        if operator in CONTRAST_CHECKS
    )[:2]
    return OperatorSelection(
        operators=selected,
        contrast_checks=contrasts or (CONTRAST_CHECKS["批判"],),
    )
