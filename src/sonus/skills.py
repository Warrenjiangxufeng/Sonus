"""本地技能库：把可复用的操作流程做成 markdown，按需载入，而不是写进代码。

一个技能就是一个目录：<workspace>/skills/<name>/SKILL.md

    ---
    name: pdf-split
    description: 把一个 PDF 按页拆成多个文件
    when_to_use: 用户要求拆分 PDF 或提取指定页时
    ---
    1. 先确认源文件路径……
    2. 用 run_command 调用……
    3. 校验输出的页数与源文件一致后再回报。

设计要点：
- 技能是数据不是代码，新增技能不需要改 Sonus 源码，也不需要重新安装。
- 只在系统提示里暴露 name/description（省 token），完整流程通过 read_skill 读取。
- 技能内容属于参考方法，不覆盖安全原则与授权模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SKILL_FILE = "SKILL.md"
SKILLS_DIRNAME = "skills"
MAX_SKILL_BYTES = 256 * 1024
MAX_CATALOG_ITEMS = 40
MAX_DESCRIPTION_CHARS = 180

_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff._-]{0,63}$")
_FRONTMATTER_KEYS = ("name", "description", "when_to_use", "version")
_HEADING_RE = re.compile(r"^\s*#")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    when_to_use: str
    relative_path: str
    body: str

    def catalog_line(self) -> str:
        line = f"- {self.name}: {self.description}"
        if self.when_to_use:
            line += f"（适用：{self.when_to_use}）"
        return line


def _first_summary(body: str) -> str:
    for raw in body.splitlines():
        line = raw.strip()
        if not line or _HEADING_RE.match(line) or line.startswith("```"):
            continue
        if len(line) > MAX_DESCRIPTION_CHARS:
            return line[:MAX_DESCRIPTION_CHARS] + "…"
        return line
    return "(无描述)"


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析可选的 `---` 头部元数据，返回 (meta, body)。"""
    if not text.startswith("---"):
        return {}, text.strip()
    lines = text.splitlines()
    end: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = index
            break
    if end is None:
        return {}, text.strip()
    meta: dict[str, str] = {}
    for raw in lines[1:end]:
        key, sep, value = raw.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        if key in _FRONTMATTER_KEYS:
            meta[key] = value.strip().strip("'\"")
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


class SkillLibrary:
    """按工作区扫描技能目录。每次 discover 都重新读盘，方便边用边改技能。"""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / SKILLS_DIRNAME

    def discover(self) -> list[Skill]:
        if not self.root.is_dir():
            return []
        skills: list[Skill] = []
        try:
            entries = sorted(self.root.iterdir(), key=lambda item: item.name)
        except OSError:
            return []
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            target = entry / SKILL_FILE
            if not target.is_file():
                continue
            try:
                skill = self._load(entry.name, target)
            except (OSError, ValueError):
                continue
            skills.append(skill)
        return skills

    def _load(self, folder_name: str, path: Path) -> Skill:
        if path.stat().st_size > MAX_SKILL_BYTES:
            raise ValueError(f"技能文件过大: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(text)
        name = meta.get("name") or folder_name
        if not _SKILL_NAME_RE.match(name):
            raise ValueError(f"技能名非法: {name}")
        return Skill(
            name=name,
            description=meta.get("description") or _first_summary(body),
            when_to_use=meta.get("when_to_use", ""),
            relative_path=str(path.relative_to(self.workspace)),
            body=body,
        )

    def get(self, name: str) -> Skill | None:
        wanted = name.strip()
        for skill in self.discover():
            if skill.name == wanted:
                return skill
        lowered = wanted.lower()
        for skill in self.discover():
            if skill.name.lower() == lowered:
                return skill
        return None

    def catalog_text(self, limit: int = MAX_CATALOG_ITEMS) -> str:
        skills = self.discover()
        if not skills:
            return ""
        lines = [f"可用技能（{len(skills)} 个，位于 {SKILLS_DIRNAME}/<name>/{SKILL_FILE}）："]
        lines.extend(skill.catalog_line() for skill in skills[:limit])
        if len(skills) > limit:
            lines.append(f"- …另有 {len(skills) - limit} 个未列出")
        lines.append(
            "用到某个技能时，先 read_skill(name) 读完整步骤，再按其中的流程调用工具。"
            "技能只提供方法，不改变安全原则与授权模型。"
        )
        return "\n".join(lines)
