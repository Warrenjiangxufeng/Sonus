from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any


DOCUMENT_PROFILES: dict[str, tuple[str, ...]] = {
    "readme": ("概述", "快速开始", "架构", "配置", "测试", "边界"),
    "design": ("目标", "架构", "数据流", "安全边界", "验证"),
    "report": ("结论", "证据", "风险", "下一步"),
    "changelog": ("变更", "验证", "兼容性"),
    "analysis": ("问题", "证据", "根因", "方案", "验证"),
}


def _section_key(value: str) -> str:
    return re.sub(r"[\s#:：_\-]+", "", value).lower()


def render_document(
    *,
    profile: str,
    title: str,
    summary: str,
    sections: list[dict[str, Any]],
    sources: list[str] | None = None,
    generated_at: str | None = None,
) -> str:
    profile = profile.strip().lower()
    if profile not in DOCUMENT_PROFILES:
        raise ValueError(
            f"未知文档模板: {profile}；可选: "
            f"{', '.join(sorted(DOCUMENT_PROFILES))}"
        )
    title = title.strip()
    if not title:
        raise ValueError("文档 title 不能为空")
    if not isinstance(sections, list) or not sections:
        raise ValueError("文档 sections 不能为空")

    normalized_sections: list[tuple[str, str]] = []
    for item in sections:
        if not isinstance(item, dict):
            raise ValueError("section 必须是 object")
        heading = str(item.get("heading") or "").strip()
        body = str(item.get("body") or "").strip()
        if not heading or not body:
            raise ValueError("section 需要非空 heading 和 body")
        normalized_sections.append((heading, body))

    available = {_section_key(heading) for heading, _ in normalized_sections}
    required = DOCUMENT_PROFILES[profile]
    missing = [heading for heading in required if _section_key(heading) not in available]
    if missing:
        raise ValueError(
            f"模板 {profile} 缺少章节: {', '.join(missing)}"
        )

    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    source_values = [str(item).strip() for item in (sources or []) if str(item).strip()]
    lines = [
        "---",
        f"sonus_document: {profile}",
        f"sonus_generated_at: {json.dumps(stamp, ensure_ascii=False)}",
    ]
    if source_values:
        lines.append("sonus_sources:")
        lines.extend(
            f"  - {json.dumps(item, ensure_ascii=False)}"
            for item in source_values
        )
    lines.extend(["---", "", f"# {title}"])
    if summary.strip():
        lines.extend(["", summary.strip()])
    for heading, body in normalized_sections:
        lines.extend(["", f"## {heading}", "", body])
    return "\n".join(lines).rstrip() + "\n"
