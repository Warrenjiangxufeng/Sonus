from __future__ import annotations

import fnmatch
import heapq
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from sunus.config import redact_text, make_ssl_context, save_user_config
from sunus.browser_inspect import inspect_browser_page
from sunus.documents import DOCUMENT_PROFILES, render_document
from sunus.environment import discover_environment, login_shell, shell_invocation
from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.policy import (
    ApprovalManager,
    PathGuard,
    PolicyError,
    command_denial_reason,
    command_risk,
    explicit_write_paths,
    is_sensitive_write_path,
    normalize_capabilities,
)
from sunus.scope import RequestScope, classify_request_with_context
from sunus.skills import SKILLS_DIRNAME, SkillLibrary
from sunus.storage import ArtifactStore
from sunus.result import result_tool_schema
from sunus.runtime_history import inspect_runtime_history
from sunus.tool_registry import ToolRegistry, ToolSpec


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOOL_OUTPUT = 60_000
MAX_CONTEXT_TOOL_OUTPUT = 16_000
MAX_SEARCH_LINE_CHARS = 500
SEARCH_TIMEOUT_SEC = 15
SEARCH_MAX_FILES = 20_000
SEARCH_MAX_LINE_BYTES = 1024 * 1024
_BROAD_SEARCH_ROOTS = {
    Path.home().resolve(),
    (Path.home() / "Documents").resolve(),
    (Path.home() / "Desktop").resolve(),
}
_TURN_CACHEABLE_TOOLS = {
    "read_file",
    "list_files",
    "search_text",
    "inspect_code_state",
    "search_memory",
    "memory_context_metrics",
    "inspect_environment",
    "list_skills",
    "read_skill",
    "run_command_batch",
}
def _is_broad_search_root(path: Path) -> bool:
    try:
        return path.resolve() in _BROAD_SEARCH_ROOTS
    except OSError:
        return False


def _command_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _semantic_command_failure(output: str) -> str:
    """Detect clear failures that a shell pipeline may mask with exit code 0."""
    checks = (
        (r"(?m)^Traceback \(most recent call last\):\s*$", "Python traceback"),
        (r"(?m)^(?:[A-Za-z_][\w.]*)?(?:Error|Exception):", "异常"),
        (
            r"(?mi)^\s*(?:[A-Za-z_][\w.-]*\s+)?requires\s+"
            r"(?:Node(?:\.js)?|Python|Java|Ruby|Go)\s+"
            r"\d+(?:\.\d+)*\s+or\s+(?:higher|newer)\b",
            "运行时版本不兼容",
        ),
        (r"(?m)^(?:zsh|sh|bash): .*command not found", "命令不存在"),
        (r"(?m)^(?:zsh|sh|bash): .*no matches found", "路径或参数不存在"),
        (r"(?m)^(?:curl|wget): .*(?:Could not resolve host|Connection refused)", "网络失败"),
        (
            r"(?:waiting for element to be visible, enabled and stable|"
            r"element is outside of the viewport|Timeout \d+ms exceeded)",
            "Playwright 操作未完成",
        ),
    )
    for pattern, label in checks:
        match = re.search(pattern, output)
        if match:
            line = next(
                (
                    item.strip()
                    for item in output[match.start() :].splitlines()
                    if item.strip()
                ),
                label,
            )
            return f"{label}：{line[:300]}"
    return ""


_TARGET_STATE_COMMAND_PATTERN = re.compile(
    r"(?i)(?:"
    r"\bopen\s+-a\s+[^;&|]*\b(?:chrome|safari|firefox|edge)\b|"
    r"\bosascript\b[^;&|]*(?:make\s+new\s+window|set\s+url\s+of)|"
    r"\b(?:playwright(?:-cli)?|pw\.sh)\b[^\n]*(?:"
    r"\b(?:open|goto|navigate|reload|click|dblclick|fill|type|press|"
    r"select|upload|check|uncheck|drag|drop|mousemove|mousedown|mouseup)\b"
    r")"
    r")"
)


def _command_changes_target_state(command: str) -> bool:
    """Return true for high-confidence UI/navigation state changes."""
    return bool(_TARGET_STATE_COMMAND_PATTERN.search(str(command or "")))


_IGNORED_NAMES = {
    ".git",
    ".sunus",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}

_SEARCH_IGNORED_NAMES = _IGNORED_NAMES | {
    ".aws",
    ".cache",
    ".codex",
    ".dsh",
    ".gnupg",
    ".ssh",
    ".Trash",
    "Library",
    "target",
}

_SEARCH_SENSITIVE_FILES = {
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "id_ed25519",
    "id_rsa",
}


def _ignored(name: str) -> bool:
    return name in _IGNORED_NAMES


def _search_ignored(name: str, *, is_dir: bool = False) -> bool:
    if name in _SEARCH_IGNORED_NAMES:
        return True
    if is_dir:
        return False
    lowered = name.lower()
    return (
        lowered in _SEARCH_SENSITIVE_FILES
        or lowered.startswith(".env")
        or lowered.endswith((".key", ".p12", ".pem", ".pfx"))
    )


def find_ripgrep() -> str | None:
    """Find ripgrep without relying only on the launcher's inherited PATH."""
    configured = os.environ.get("SUNUS_RG_PATH", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    found = shutil.which("rg")
    if found:
        candidates.append(Path(found))
    candidates.extend(
        Path(path).expanduser()
        for path in (
            "/opt/homebrew/bin/rg",
            "/usr/local/bin/rg",
            "/opt/local/bin/rg",
            "~/.cargo/bin/rg",
            "~/.local/bin/rg",
        )
    )
    sunus_runtime_home = Path(
        os.environ.get("SUNUS_PROVIDER_HOME")
        or os.environ.get("CODEX_HOME")
        or Path.home() / ".codex"
    )
    sunus_releases = sunus_runtime_home / "packages" / "app-server-daemon" / "releases"
    if sunus_releases.is_dir():
        bundled = list(sunus_releases.glob("*/codex-path/rg"))
        bundled.sort(
            key=lambda path: path.stat().st_mtime
            if path.exists()
            else 0,
            reverse=True,
        )
        candidates.extend(bundled)
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
        except OSError:
            continue
    return None


@dataclass
class ToolResult:
    ok: bool
    content: str
    changed_files: list[str] = field(default_factory=list)
    artifact: str | None = None
    state_changed: bool = False
    approval_barrier: bool = False
    halt: bool = False
    cache_hit: bool = False
    control_outcome: str = ""

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "content": self.content}
        if self.changed_files:
            payload["changed_files"] = self.changed_files
        if self.artifact:
            payload["artifact"] = self.artifact
        if self.state_changed:
            payload["state_changed"] = True
        if self.approval_barrier:
            payload["approval_barrier"] = True
        if self.halt:
            payload["halt"] = True
        if self.cache_hit:
            payload["cache_hit"] = True
        if self.control_outcome:
            payload["control_outcome"] = self.control_outcome
        return payload


def tool_schemas() -> list[dict[str, Any]]:
    def fn(name: str, description: str, properties: dict[str, Any], required: list[str]):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    document_profile_text = "; ".join(
        f"{name}={','.join(headings)}"
        for name, headings in sorted(DOCUMENT_PROFILES.items())
    )

    return [
        fn(
            "read_file",
            "Read a UTF-8 text file inside the workspace. Returns numbered lines. "
            "Batch independent file reads in one model turn.",
            {
                "path": {"type": "string", "description": "Workspace-relative path."},
                "start_line": {"type": "integer", "minimum": 1, "default": 1},
                "end_line": {"type": "integer", "minimum": 1, "default": 400},
            },
            ["path"],
        ),
        fn(
            "list_files",
            "List files and directories inside the workspace. Batch independent "
            "listings in one model turn.",
            {
                "path": {"type": "string", "default": "."},
                "depth": {"type": "integer", "minimum": 1, "maximum": 8, "default": 2},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 250},
                "sort_by_mtime": {"type": "boolean", "default": False},
                "modified_within_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 0,
                },
            },
            [],
        ),
        fn(
            "search_text",
            "Preferred bounded literal-text search inside workspace files. Uses ripgrep with "
            "sensitive/cache exclusions and a hard timeout when available; use this before "
            "shell-level recursive grep or rg. Broad roots default to a bounded file summary; "
            "first try a known file, inspect_code_state, or a known repository; "
            "set allow_broad=true only after deciding a global line search is necessary.",
            {
                "query": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "glob": {"type": "string", "default": "*"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                "output": {
                    "type": "string",
                    "enum": ["lines", "files"],
                    "default": "lines",
                    "description": (
                        "Use files for a metadata-first pass that groups matches "
                        "by path before reading selected files."
                    ),
                },
                "timeout_sec": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": SEARCH_TIMEOUT_SEC,
                },
                "allow_broad": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Allow line-level search from home/Documents/Desktop. "
                        "Default false returns a bounded file summary first."
                    ),
                },
            },
            ["query"],
        ),
        fn(
            "inspect_code_state",
            "Inspect current code state needed for version claims: exact symbol "
            "matches, surrounding source lines, current Git HEAD/branch, and "
            "the commit that last changed the target file or symbol. Read-only.",
            {
                "path": {
                    "type": "string",
                    "description": "Source file or directory to inspect.",
                },
                "symbol": {
                    "type": "string",
                    "description": "Exact code/configuration symbol to locate.",
                },
                "context_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 40,
                    "default": 12,
                },
                "max_matches": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 4,
                },
            },
            ["path", "symbol"],
        ),
        fn(
            "memory_context_metrics",
            "Inspect how much local memory text is injected into the current "
            "context. Returns exact character counts and dependency-free token "
            "estimates marked exact=false; it never changes memory.",
            {
                "query": {
                    "type": "string",
                    "description": "Query to measure; defaults to the current user message.",
                },
                "preference_limit": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 32,
                    "default": 12,
                },
                "recall_limit": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                    "default": 5,
                },
            },
            [],
        ),
        fn(
            "search_memory",
            "Search structured long-term memory through the SQLite/FTS index. "
            "Use this for remembered facts, preferences, episodes, reviews, and "
            "lessons; never search or read memory.db as a text file.",
            {
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": [
                        "preference",
                        "episode",
                        "persona",
                        "review",
                        "lesson",
                        "strategy",
                        "self_share",
                        "semantic",
                    ],
                    "description": "Optional memory kind filter.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 8,
                },
            },
            ["query"],
        ),
        fn(
            "remember",
            "Persist an explicitly requested durable preference, review, lesson, "
            "or episode. Use only when the user asks Sunus to remember or review.",
            {
                "content": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["preference", "review", "lesson", "episode"],
                    "default": "review",
                },
                "scope": {
                    "type": "string",
                    "enum": ["project", "global"],
                    "default": "project",
                },
                "persistent": {"type": "boolean", "default": True},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            ["content"],
        ),
        fn(
            "inspect_environment",
            "Inspect the current Python, shell, pytest executable and matching "
            "interpreter. Use this before claiming a local tool is missing.",
            {
                "commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Additional command names to locate on PATH.",
                },
                "refresh": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Rescan PATH immediately, for example after installing "
                        "or uninstalling a command."
                    ),
                },
            },
            [],
        ),
        fn(
            "inspect_runtime_history",
            "Query bounded recent Sunus activity without broad full-text scanning: "
            "recent sessions, recent failures, or one turn timeline.",
            {
                "mode": {
                    "type": "string",
                    "enum": [
                        "recent",
                        "latest_failure",
                        "failures",
                        "trend",
                        "turn",
                    ],
                    "default": "recent",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 2,
                },
                "turn_id": {"type": "string", "default": ""},
            },
            [],
        ),
        fn(
            "inspect_browser",
            "Read the active tab title/URL or visible text from an already-open "
            "Chrome/Safari page. Does not navigate, click, log in, or modify settings.",
            {
                "browser": {
                    "type": "string",
                    "enum": ["chrome", "safari"],
                    "default": "chrome",
                },
                "mode": {
                    "type": "string",
                    "enum": ["active", "content"],
                    "default": "active",
                },
            },
            [],
        ),
        fn(
            "write_file",
            "Create or fully overwrite a text file in the workspace or at a "
            "path explicitly authorized by the current user request. "
            "Sensitive credential and runtime paths remain protected.",
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            ["path", "content"],
        ),
        fn(
            "write_file_batch",
            "Create or overwrite up to 20 files in one validated tool call. "
            "Use this for independent generated files; all paths and sizes are "
            "validated before any file is written.",
            {
                "files": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                }
            },
            ["files"],
        ),
        fn(
            "write_document",
            "Render and write a structured Markdown document using a predefined "
            f"template. Profiles and required headings: {document_profile_text}. "
            "Paths must stay inside the workspace.",
            {
                "path": {"type": "string"},
                "profile": {
                    "type": "string",
                    "enum": sorted(DOCUMENT_PROFILES),
                },
                "title": {"type": "string"},
                "summary": {"type": "string", "default": ""},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "heading": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["heading", "body"],
                    },
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            ["path", "profile", "title", "sections"],
        ),
        fn(
            "patch_file",
            "Replace exact text in one file. Prefer this over write_file for small edits. "
            "The file must be in the workspace or explicitly named by the "
            "current user request.",
            {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            ["path", "old_text", "new_text"],
        ),
        fn(
            "patch_file_batch",
            "Apply up to 20 validated exact-text replacements across one or more "
            "files in one tool call. Edits are applied in order. All paths and "
            "matches are validated before writing; use this instead of repeated "
            "patch_file calls when multiple replacements are already known.",
            {
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "replace_all": {
                                "type": "boolean",
                                "default": False,
                            },
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                }
            },
            ["edits"],
        ),
        fn(
            "apply_and_verify",
            "Apply multiple exact-text edits and then run up to four verification "
            "commands in one tool call. The patch batch is committed only after "
            "all edits validate; verification failures are returned with the "
            "successful file changes.",
            {
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "replace_all": {
                                "type": "boolean",
                                "default": False,
                            },
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
                "verification_commands": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "command": {"type": "string"},
                            "cwd": {"type": "string", "default": "."},
                            "timeout_sec": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 300,
                                "default": 120,
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            ["edits", "verification_commands"],
        ),
        fn(
            "run_command",
            "Run a shell command in the workspace and return exit code and combined output. "
            "Prefer search_text for file or text discovery; broad recursive grep/rg searches "
            "must use bounded roots, explicit exclusions, and a short timeout.",
            {
                "command": {"type": "string"},
                "cwd": {"type": "string", "default": "."},
                "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 1800, "default": 120},
            },
            ["command"],
        ),
        fn(
            "run_command_batch",
            "Run up to four independent read-only shell commands in one tool call. "
            "Use this instead of separate run_command calls when the checks do not "
            "depend on each other. Every command must be classified R0; commands "
            "that mutate files, Git state, processes, packages, or network are rejected.",
            {
                "commands": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "command": {"type": "string"},
                            "cwd": {"type": "string", "default": "."},
                            "timeout_sec": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 120,
                                "default": 30,
                            },
                        },
                        "required": ["command"],
                    },
                }
            },
            ["commands"],
        ),
        fn(
            "verification_batch",
            "Run up to four independent single-command verification steps in "
            "one tool call. If runtime.verification_command_prefixes is set, "
            "commands must match one of those generic configured prefixes.",
            {
                "commands": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "command": {"type": "string"},
                            "cwd": {"type": "string", "default": "."},
                            "timeout_sec": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 300,
                                "default": 120,
                            },
                        },
                        "required": ["command"],
                    },
                }
            },
            ["commands"],
        ),
        fn(
            "request_authorization",
            "Ask the user to expand the current task scope when read-only tools are "
            "insufficient. Use only when the requested result genuinely requires a "
            "workspace mutation, local command, or network access.",
            {
                "capability": {
                    "type": "string",
                    "enum": [
                        "write",
                        "write_and_command",
                        "command",
                        "network",
                    ],
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of the blocked task and needed capability.",
                },
            },
            ["capability", "reason"],
        ),
        fn(
            "configure_permissions",
            "Persist permission changes after an explicit user instruction. "
            "This edits runtime.permissions in the Sunus user config and applies "
            "safe write allowlists immediately. Never call without user intent.",
            {
                "permissions": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "write": {
                            "type": "string",
                            "enum": ["auto", "allow", "deny"],
                        },
                        "command": {
                            "type": "string",
                            "enum": ["auto", "allow", "deny"],
                        },
                        "network": {
                            "type": "string",
                            "enum": ["auto", "allow", "deny"],
                        },
                    },
                },
                "write_allowlist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Additional or replacement paths that may be written "
                        "without per-file approval."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["add", "set", "remove"],
                    "default": "add",
                },
                "reason": {
                    "type": "string",
                    "description": "The explicit user instruction being applied.",
                },
            },
            ["reason"],
        ),
        fn(
            "schedule_proactive_file_write",
            "Create a structured proactive task to write a file later. "
            "Use only after an explicit user request for a future or recurring "
            "write. Never use it to turn Sunus's own diary, reflection, or care "
            "habits into a scheduled obligation. The task is stored in "
            "proactive_tasks and consumed by the proactive scheduler.",
            {
                "task_name": {"type": "string"},
                "instruction": {"type": "string"},
                "target_path_template": {
                    "type": "string",
                    "description": "Absolute path; may contain {date}.",
                },
                "cadence": {
                    "type": "string",
                    "enum": ["once", "daily"],
                    "default": "once",
                },
                "not_before": {
                    "type": "string",
                    "default": "20:00",
                },
                "source_memory_id": {
                    "type": "string",
                    "default": "",
                },
            },
            ["task_name", "instruction", "target_path_template"],
        ),
        fn(
            "fetch_url",
            "Fetch an HTTP(S) URL as text. Network access requires a higher-risk grant.",
            {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 100000, "default": 30000},
                "timeout_sec": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 30,
                    "default": 15,
                },
            },
            ["url"],
        ),
        fn(
            "fetch_url_batch",
            "Fetch up to five independent HTTP(S) URLs in one tool call and "
            "return each status and content separately. Do not use for ordered "
            "login, form submission, or state-changing workflows.",
            {
                "requests": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "url": {"type": "string"},
                            "max_chars": {
                                "type": "integer",
                                "minimum": 1000,
                                "maximum": 100000,
                                "default": 30000,
                            },
                            "timeout_sec": {
                                "type": "integer",
                                "minimum": 3,
                                "maximum": 30,
                                "default": 15,
                            },
                        },
                        "required": ["url"],
                    },
                }
            },
            ["requests"],
        ),
        fn(
            "update_reasoning_state",
            "Update the current task's observable reasoning frame when the subgoal, "
            "hypothesis, evidence gap, success criteria, or decision changes. This "
            "does not execute an external action. Use it at task start or a genuine "
            "decision change, not mechanically after every tool result.",
            {
                "current_subgoal": {"type": "string"},
                "working_hypothesis": {"type": "string"},
                "evidence_gap": {"type": "string"},
                "success_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "decision": {"type": "string"},
            },
            [],
        ),
        fn(
            "list_skills",
            "List locally defined skills (name + description + when to use). "
            "Call read_skill to load a skill's full instructions.",
            {},
            [],
        ),
        fn(
            "read_skill",
            "Load the full markdown instructions of one local skill from "
            f"{SKILLS_DIRNAME}/<name>/SKILL.md, then follow them using the other tools.",
            {
                "name": {"type": "string", "description": "Skill name from list_skills."},
            },
            ["name"],
        ),
    ] + [result_tool_schema()]


class ToolExecutor:
    def __init__(
        self,
        *,
        workspace: Path,
        path_guard: PathGuard,
        approvals: ApprovalManager,
        artifacts: ArtifactStore,
        events: EventBus,
        data_dir: Path | None = None,
        secret: str = "",
        memory: MemoryStore | None = None,
        mcp_servers: tuple[Any, ...] = (),
        max_parallel_tools: int = 4,
        read_only_command_prefixes: tuple[str, ...] = (),
        verification_command_prefixes: tuple[str, ...] = (),
    ) -> None:
        self.workspace = workspace
        self.path_guard = path_guard
        self.approvals = approvals
        self.artifacts = artifacts
        self.data_dir = data_dir or artifacts.root.parent
        self.current_session_id = ""
        self.events = events
        self.secret = secret
        self.memory = memory
        self.proactive_scheduler: Any | None = None
        self.skills = SkillLibrary(workspace)
        self.request_scope: RequestScope | None = None
        self.task_state: Any | None = None
        self.max_parallel_tools = max(1, int(max_parallel_tools))
        self.read_only_command_prefixes = tuple(
            read_only_command_prefixes
        )
        self.verification_command_prefixes = tuple(
            verification_command_prefixes
        )
        self._turn_result_cache: dict[str, ToolResult] = {}
        self.registry = ToolRegistry()
        self.mcp_manager: Any | None = None
        self._initialize_registry(mcp_servers)

    def _initialize_registry(self, mcp_servers: tuple[Any, ...]) -> None:
        handlers = {
            "read_file": self._read_file,
            "list_files": self._list_files,
            "search_text": self._search_text,
            "inspect_code_state": self._inspect_code_state,
            "search_memory": self._search_memory,
            "memory_context_metrics": self._memory_context_metrics,
            "remember": self._remember,
            "inspect_environment": self._inspect_environment,
            "inspect_runtime_history": self._inspect_runtime_history,
            "inspect_browser": self._inspect_browser,
            "update_reasoning_state": self._update_reasoning_state,
            "write_file": self._write_file,
            "write_file_batch": self._write_file_batch,
            "write_document": self._write_document,
            "patch_file": self._patch_file,
            "patch_file_batch": self._patch_file_batch,
            "apply_and_verify": self._apply_and_verify,
            "run_command": self._run_command,
            "run_command_batch": self._run_command_batch,
            "verification_batch": self._verification_batch,
            "request_authorization": self._request_authorization,
            "configure_permissions": self._configure_permissions,
            "schedule_proactive_file_write": (
                self._schedule_proactive_file_write
            ),
            "fetch_url": self._fetch_url,
            "fetch_url_batch": self._fetch_url_batch,
            "list_skills": self._list_skills,
            "read_skill": self._read_skill,
        }
        read_only = {
            "read_file",
            "list_files",
            "search_text",
            "inspect_code_state",
            "search_memory",
            "memory_context_metrics",
            "inspect_environment",
            "inspect_runtime_history",
            "list_skills",
            "read_skill",
            "run_command_batch",
            "submit_result",
        }
        parallel_safe = set(read_only) - {"submit_result", "run_command_batch"}
        risks = {
            "fetch_url": "R2",
            "inspect_browser": "R2",
            "update_reasoning_state": "R0",
            "write_file": "R1",
            "write_document": "R1",
            "patch_file": "R1",
            "patch_file_batch": "R1",
            "run_command": "R1",
            "run_command_batch": "R1",
            "configure_permissions": "R1",
            "schedule_proactive_file_write": "R1",
            "write_file_batch": "R1",
            "apply_and_verify": "R1",
            "verification_batch": "R1",
            "fetch_url_batch": "R2",
        }
        capabilities = {
            "read_file": "workspace_text",
            "list_files": "workspace_inventory",
            "search_text": "workspace_text_search",
            "inspect_code_state": "code_state",
            "search_memory": "memory_recall",
            "memory_context_metrics": "memory_instrumentation",
            "remember": "memory_write",
            "inspect_environment": "runtime_environment",
            "inspect_runtime_history": "session_history",
            "inspect_browser": "browser_state",
            "update_reasoning_state": "task_cognition",
            "write_file": "workspace_write",
            "write_file_batch": "workspace_write",
            "write_document": "document_write",
            "patch_file": "workspace_edit",
            "patch_file_batch": "workspace_edit",
            "apply_and_verify": "edit_and_verify",
            "run_command": "command_execution",
            "run_command_batch": "command_execution",
            "verification_batch": "verification",
            "request_authorization": "permission_request",
            "configure_permissions": "permission_config",
            "schedule_proactive_file_write": "proactive_schedule",
            "fetch_url": "remote_http",
            "fetch_url_batch": "remote_http",
            "list_skills": "skill_inventory",
            "read_skill": "skill_instructions",
            "submit_result": "terminal_result",
        }
        evidence_kinds = {
            "read_file": "file_content",
            "list_files": "file_inventory",
            "search_text": "text_matches",
            "inspect_code_state": "code_provenance",
            "search_memory": "memory_evidence",
            "memory_context_metrics": "context_metrics",
            "remember": "persistent_memory",
            "inspect_environment": "environment_facts",
            "inspect_runtime_history": "runtime_events",
            "inspect_browser": "browser_page",
            "update_reasoning_state": "reasoning_state",
            "write_file": "file_change",
            "write_file_batch": "file_change",
            "write_document": "document_change",
            "patch_file": "file_change",
            "patch_file_batch": "file_change",
            "apply_and_verify": "edit_and_verification",
            "run_command": "command_output",
            "run_command_batch": "command_output",
            "verification_batch": "verification_output",
            "request_authorization": "authorization_state",
            "configure_permissions": "permission_state",
            "schedule_proactive_file_write": "scheduled_task",
            "fetch_url": "remote_page",
            "fetch_url_batch": "remote_pages",
            "list_skills": "skill_catalog",
            "read_skill": "skill_instructions",
            "submit_result": "final_result",
        }
        preconditions = {
            "inspect_browser": (
                "目标浏览器正在运行",
                "目标页面可从当前浏览器状态读取",
            ),
            "fetch_url": (
                "用户明确请求外部网页读取",
                "当前请求具有网络能力",
            ),
            "run_command": ("当前请求允许本地命令",),
            "run_command_batch": ("当前请求允许本地命令",),
            "write_file": ("当前请求允许写入",),
            "write_file_batch": ("当前请求允许写入",),
            "write_document": ("当前请求允许写入",),
            "patch_file": ("当前请求允许修改",),
            "patch_file_batch": ("当前请求允许修改",),
            "apply_and_verify": (
                "当前请求允许修改",
                "当前请求允许本地命令",
            ),
            "search_memory": ("本地记忆库可用",),
            "verification_batch": ("当前请求允许本地命令",),
        }
        failure_modes = {
            "inspect_browser": ("浏览器未运行", "页面权限或脚本限制"),
            "fetch_url": ("DNS 失败", "TLS 失败", "网络超时", "HTTP 错误"),
            "fetch_url_batch": ("DNS 失败", "TLS 失败", "网络超时", "HTTP 错误"),
            "run_command": ("命令不存在", "依赖缺失", "权限拒绝", "非零退出"),
            "run_command_batch": ("命令不存在", "非只读命令", "权限拒绝", "非零退出"),
            "verification_batch": ("验证命令不支持", "依赖缺失", "非零退出"),
            "read_file": ("路径不存在", "编码或文件大小限制"),
            "search_text": ("路径错误", "超时或结果过宽"),
            "search_memory": ("索引不可用", "无相关记忆"),
        }
        for schema in tool_schemas():
            name = str(schema.get("function", {}).get("name") or "")
            handler = handlers.get(
                name,
                lambda _arguments, tool_name=name: ToolResult(
                    False,
                    f"{tool_name} 由 Agent 运行时处理，不能直接执行",
                ),
            )
            self.registry.register_schema(
                schema,
                handler,
                risk=risks.get(name, "R0"),
                read_only=name in read_only,
                parallel_safe=name in parallel_safe,
                approval_mode="approve" if name == "submit_result" else "ask",
                capability=capabilities.get(name, "general"),
                evidence_kind=evidence_kinds.get(name, "observation"),
                preconditions=preconditions.get(name, ()),
                failure_modes=failure_modes.get(name, ()),
            )
        if mcp_servers:
            from sunus.mcp import MCPManager

            self.mcp_manager = MCPManager(
                mcp_servers,
                events=self.events,
                secret=self.secret,
            )
            for spec in self.mcp_manager.discover_specs():
                try:
                    self.registry.register(spec)
                except ValueError as error:
                    self.events.emit(
                        "mcp_tool_conflict",
                        name=spec.name,
                        summary=f"忽略冲突 MCP 工具 · {spec.name}: {error}",
                    )

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.schemas()

    def specs(self) -> list[ToolSpec]:
        return self.registry.specs()

    def close(self) -> None:
        if self.mcp_manager is not None:
            self.mcp_manager.close()
            self.mcp_manager = None

    def begin_request(
        self,
        text: str,
        *,
        previous_assistant: str = "",
        previous_scope: RequestScope | None = None,
    ) -> RequestScope:
        self.request_scope = self.approvals.apply_scope(
            classify_request_with_context(
                text,
                previous_assistant=previous_assistant,
                previous_scope=previous_scope,
            )
        )
        if self.request_scope.allows_mutation:
            self.path_guard.set_request_write_paths(
                explicit_write_paths(text, self.workspace)
            )
        else:
            self.path_guard.clear_request_write_paths()
        self._turn_result_cache.clear()
        return self.request_scope

    def _scope_authorization(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        scope = self.request_scope
        if scope is None:
            return ""
        if name in {
            "configure_permissions",
            "schedule_proactive_file_write",
        }:
            capability = "write"
            allowed = scope.allows_mutation
            risk = "R1"
        elif name == "apply_and_verify":
            capability = "write_and_command"
            allowed = scope.allows_mutation and scope.allows_commands
            risk = "R1"
        elif name == "verification_batch":
            capability = "command"
            allowed = scope.allows_commands
            risk = "R1"
        elif name in {
            "write_file",
            "write_file_batch",
            "write_document",
            "patch_file",
            "patch_file_batch",
        }:
            capability = "write"
            allowed = scope.allows_mutation
            risk = "R1"
            raw_paths = (
                [
                    str(item.get("path") or "").strip()
                    for item in (
                        arguments.get("files")
                        if name == "write_file_batch"
                        else arguments.get("edits")
                    )
                    or []
                    if isinstance(item, dict)
                ]
                if name in {"patch_file_batch", "write_file_batch"}
                else [str(arguments.get("path") or "").strip()]
            )
            if (
                any(raw_paths)
                and self.approvals.capability_overrides.get("write") is not False
            ):
                for raw_path in raw_paths:
                    if not raw_path:
                        continue
                    try:
                        candidate = self.path_guard.resolve(
                            raw_path,
                            for_write=True,
                        )
                    except PolicyError:
                        continue
                    if self.path_guard.is_write_permitted(candidate):
                        allowed = True
                        break
        elif name in {"run_command", "run_command_batch"}:
            capability = "command"
            allowed = scope.allows_commands
            if name == "run_command":
                command = str(arguments.get("command") or "").strip()
                denial = command_denial_reason(command) if command else ""
                risk = command_risk(command, workspace=self.workspace)
            else:
                commands = arguments.get("commands") or []
                command = json.dumps(commands, ensure_ascii=False)[:4000]
                command_values = [
                    str(item.get("command") or "").strip()
                    for item in commands
                    if isinstance(item, dict)
                ]
                denial = next(
                    (
                        command_denial_reason(value)
                        for value in command_values
                        if command_denial_reason(value)
                    ),
                    "",
                )
                risks = [
                    command_risk(value, workspace=self.workspace)
                    for value in command_values
                ]
                risk = max(
                    risks,
                    key={"R0": 0, "R1": 1, "R2": 2, "R3": 3}.get,
                    default="R2",
                )
            if not command:
                return ""
            if denial:
                return denial
        elif name in {"fetch_url", "fetch_url_batch"}:
            capability = "network"
            allowed = scope.allows_network
            risk = "R2"
        else:
            allowed = True
            capability = ""
            risk = "R1"
        if allowed:
            return ""
        approved = self.approvals.require(
            "授权本轮能力",
            (
                f"能力：{capability}\n"
                f"工具：{name}\n"
                f"参数：{json.dumps(arguments, ensure_ascii=False)[:2000]}"
            ),
            risk=risk,
            capability=capability,
            use_task_grant=False,
        )
        if not approved:
            return "该操作需要明确授权，但用户拒绝或当前运行无法确认授权。"
        self.approvals.grant_task_capabilities([capability])
        self.request_scope = replace(
            self.request_scope,
            allows_mutation=(
                self.request_scope.allows_mutation or capability == "write"
            ),
            allows_commands=(
                self.request_scope.allows_commands or capability == "command"
            ),
            allows_network=(
                self.request_scope.allows_network or capability == "network"
            ),
        )
        self.approvals.consume_interactive_decision()
        return ""

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        denial = self._scope_authorization(name, arguments)
        if denial:
            return self._attach_approval_decision(ToolResult(False, denial))
        spec = self.registry.get(name)
        if spec is None:
            return ToolResult(False, f"未知工具: {name}")
        cache_key = self._cache_key(name, arguments)
        if cache_key and cache_key in self._turn_result_cache:
            cached = self._turn_result_cache[cache_key]
            self.events.emit(
                "tool_cache_hit",
                name=name,
                summary=f"复用本轮已获取结果 · {name}",
            )
            return ToolResult(
                ok=cached.ok,
                content=cached.content,
                changed_files=list(cached.changed_files),
                artifact=cached.artifact,
                state_changed=False,
                cache_hit=True,
                control_outcome=cached.control_outcome,
            )
        if spec.source == "mcp":
            if spec.approval_mode == "deny":
                return ToolResult(False, f"用户配置拒绝 MCP 工具: {name}")
            if spec.approval_mode == "ask" and not self.approvals.require(
                "调用 MCP 工具",
                f"{name}\n{json.dumps(arguments, ensure_ascii=False)[:2000]}",
                risk=spec.risk,
            ):
                return self._attach_approval_decision(
                    ToolResult(False, f"用户拒绝 MCP 工具: {name}")
                )
        if spec.source == "mcp" and spec.timeout_sec > 0:
            try:
                timeout_sec = int(
                    arguments.get("timeout_sec", spec.timeout_sec)
                )
            except (TypeError, ValueError):
                timeout_sec = spec.timeout_sec
            arguments = dict(arguments)
            arguments.setdefault("timeout_sec", timeout_sec)
        try:
            result = spec.handler(arguments)
            if not isinstance(result, ToolResult):
                result = ToolResult(True, str(result))
            if len(result.content) > MAX_CONTEXT_TOOL_OUTPUT:
                artifact = result.artifact
                if artifact is None:
                    artifact = str(
                        self.artifacts.save(
                            redact_text(result.content, self.secret),
                            prefix=f"context-{name}",
                        )
                    )
                result.content = (
                    result.content[:MAX_CONTEXT_TOOL_OUTPUT]
                    + f"\n... [context cap {MAX_CONTEXT_TOOL_OUTPUT} chars; "
                    f"full output: {artifact}]"
                )
                result.artifact = artifact
            if cache_key and result.ok:
                self._turn_result_cache[cache_key] = ToolResult(
                    ok=result.ok,
                    content=result.content,
                    changed_files=list(result.changed_files),
                    artifact=result.artifact,
                    state_changed=result.state_changed,
                )
            if (
                not spec.read_only
                or result.state_changed
                or bool(result.changed_files)
            ):
                self._turn_result_cache.clear()
            return self._attach_approval_decision(result)
        except (PolicyError, OSError, ValueError) as error:
            return self._attach_approval_decision(
                ToolResult(False, f"{type(error).__name__}: {error}")
            )
        except Exception as error:
            return self._attach_approval_decision(
                ToolResult(
                    False,
                    f"工具异常 {type(error).__name__}: {error}",
                )
            )

    def _attach_approval_decision(self, result: ToolResult) -> ToolResult:
        decision = self.approvals.consume_interactive_decision()
        if not decision:
            return result
        result.approval_barrier = True
        result.halt = decision in {"denied", "cancelled"}
        return result

    @staticmethod
    def _cache_key(name: str, arguments: dict[str, Any]) -> str:
        if name not in _TURN_CACHEABLE_TOOLS:
            return ""
        return json.dumps(
            [name, arguments],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    def _read_only_command_allowed(self, command: str) -> bool:
        if command_risk(command, workspace=self.workspace) == "R0":
            return True
        head = self._command_after_leading_cd(command)
        return self._matches_command_prefix(
            head,
            self.read_only_command_prefixes,
        )

    def _verification_command_allowed(self, command: str) -> bool:
        if command_denial_reason(command):
            return False
        head = self._command_after_leading_cd(command)
        if not head or ";" in head or "&&" in head or "||" in head:
            return False
        if not self.verification_command_prefixes:
            return True
        return self._matches_command_prefix(
            head,
            self.verification_command_prefixes,
        )

    @staticmethod
    def _command_after_leading_cd(command: str) -> str:
        text = str(command).strip()
        text = re.sub(
            r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*",
            "",
            text,
        )
        if text.startswith("cd "):
            parts = text.split("&&", 1)
            if len(parts) == 2:
                text = parts[1].strip()
        return text

    @staticmethod
    def _matches_command_prefix(
        command: str,
        prefixes: tuple[str, ...],
    ) -> bool:
        text = str(command).strip()
        for prefix in prefixes:
            if text == prefix or text.startswith(prefix + " "):
                return True
            try:
                command_parts = shlex.split(text)
                prefix_parts = shlex.split(prefix)
            except ValueError:
                continue
            if not command_parts or not prefix_parts:
                continue
            if len(command_parts) < len(prefix_parts):
                continue
            if Path(command_parts[0]).name != Path(prefix_parts[0]).name:
                continue
            if command_parts[1 : len(prefix_parts)] != prefix_parts[1:]:
                continue
            return True
        return False

    def _truncate(self, value: str, *, prefix: str) -> tuple[str, str | None]:
        value = redact_text(value, self.secret)
        if len(value) <= MAX_TOOL_OUTPUT:
            return value, None
        artifact = self.artifacts.save(value, prefix=prefix)
        return (
            value[:MAX_TOOL_OUTPUT]
            + f"\n... [truncated; full output: {artifact}]",
            str(artifact),
        )

    def _read_file(self, args: dict[str, Any]) -> ToolResult:
        path = self.path_guard.resolve(str(args["path"]), must_exist=True)
        if not path.is_file():
            return ToolResult(False, f"不是文件: {path}")
        if path.stat().st_size > MAX_FILE_BYTES:
            return ToolResult(False, f"文件超过 {MAX_FILE_BYTES} 字节: {path}")
        start = max(1, int(args.get("start_line", 1)))
        end = max(start, int(args.get("end_line", 400)))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start - 1 : end]
        numbered = "\n".join(
            f"{number:>6} | {line}"
            for number, line in enumerate(selected, start=start)
        )
        return ToolResult(True, redact_text(numbered, self.secret) or "(empty file)")

    def _list_files(self, args: dict[str, Any]) -> ToolResult:
        root = self.path_guard.resolve(str(args.get("path", ".")), must_exist=True)
        if not root.is_dir():
            return ToolResult(False, f"不是目录: {root}")
        depth_limit = min(8, max(1, int(args.get("depth", 2))))
        max_results = min(1000, max(1, int(args.get("max_results", 250))))
        sort_by_mtime = bool(args.get("sort_by_mtime", False))
        modified_days = max(0, min(365, int(args.get("modified_within_days", 0))))
        cutoff = time.time() - modified_days * 86400 if modified_days else 0
        results: list[tuple[str, float]] = []

        def add_result(path: Path, modified: float, *, directory: bool = False) -> None:
            relative = self.path_guard.relative(path)
            if directory:
                relative += "/"
            item = (modified, relative)
            if sort_by_mtime:
                heapq.heappush(results, item)
                if len(results) > max_results:
                    heapq.heappop(results)
            else:
                results.append(item)

        root_depth = len(root.parts)
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            relative_depth = len(current_path.parts) - root_depth
            dirs[:] = sorted(name for name in dirs if not _ignored(name))
            if relative_depth >= depth_limit:
                dirs[:] = []
            for directory in dirs:
                directory_path = current_path / directory
                try:
                    modified = directory_path.stat().st_mtime
                except OSError:
                    modified = 0
                if modified >= cutoff:
                    add_result(directory_path, modified, directory=True)
                if not sort_by_mtime and len(results) >= max_results:
                    break
            for filename in sorted(files):
                if not sort_by_mtime and len(results) >= max_results:
                    break
                if not _ignored(filename):
                    file_path = current_path / filename
                    try:
                        modified = file_path.stat().st_mtime
                    except OSError:
                        modified = 0
                    if modified >= cutoff:
                        add_result(file_path, modified)
            if not sort_by_mtime and len(results) >= max_results:
                break
        if sort_by_mtime:
            results.sort(key=lambda item: (item[0], item[1]), reverse=True)
            content = "\n".join(
                f"{path}\t"
                f"{datetime.fromtimestamp(modified).astimezone():%Y-%m-%d %H:%M:%S}"
                for modified, path in results
            )
        else:
            content = "\n".join(path for _modified, path in results)
        return ToolResult(True, content or "(empty directory)")

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _matches_glob(relative: Path, glob: str) -> bool:
        if not glob or glob == "*":
            return True
        value = relative.as_posix()
        return relative.match(glob) or fnmatch.fnmatchcase(value, glob)

    def _search_with_ripgrep(
        self,
        executable: str,
        *,
        root: Path,
        query: str,
        glob: str,
        limit: int,
        timeout_sec: int,
    ) -> tuple[list[str], bool, str]:
        command = [
            executable,
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--fixed-strings",
            "--hidden",
            "--max-filesize",
            str(MAX_FILE_BYTES),
            "--max-columns",
            str(MAX_SEARCH_LINE_CHARS),
            "--max-columns-preview",
        ]
        for name in sorted(_SEARCH_IGNORED_NAMES):
            command.extend(["--glob", f"!**/{name}/**"])
        command.extend(["--glob", "!**/.env*"])
        for name in sorted(_SEARCH_SENSITIVE_FILES):
            command.extend(["--glob", f"!**/{name}"])
        for suffix in (".key", ".p12", ".pem", ".pfx"):
            command.extend(["--glob", f"!**/*{suffix}"])
        command.extend(["--glob", glob, "--", query, "."])

        lines: list[str] = []
        failure = ""
        timed_out = False
        process: subprocess.Popen[bytes] | None = None
        selector = selectors.DefaultSelector()
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    bufsize=0,
                    start_new_session=True,
                )
                if process.stdout is None:
                    raise OSError("无法读取 rg 输出")
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + timeout_sec
                buffer = bytearray()
                while len(lines) < limit:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    if not selector.select(remaining):
                        timed_out = True
                        break
                    chunk = os.read(process.stdout.fileno(), 64 * 1024)
                    if not chunk:
                        if buffer and len(lines) < limit:
                            lines.append(
                                bytes(buffer)
                                .decode("utf-8", errors="replace")
                                .rstrip("\r")[:MAX_SEARCH_LINE_CHARS]
                            )
                        break
                    buffer.extend(chunk)
                    while b"\n" in buffer and len(lines) < limit:
                        raw_line, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        lines.append(
                            raw_line.decode("utf-8", errors="replace").rstrip("\r")[
                                :MAX_SEARCH_LINE_CHARS
                            ]
                        )
                    if len(buffer) > SEARCH_MAX_LINE_BYTES:
                        failure = "搜索输出包含超长单行，已停止"
                        break

                if len(lines) >= limit or timed_out or failure:
                    self._stop_process(process)
                else:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        self._stop_process(process)
            except OSError as error:
                failure = f"rg 启动失败: {error}"
                if process is not None:
                    self._stop_process(process)
            finally:
                selector.close()
                if process is not None and process.stdout is not None:
                    process.stdout.close()

            stderr_file.seek(0)
            stderr = stderr_file.read(4000).strip()

        if timed_out:
            failure = f"搜索超过 {timeout_sec} 秒，已停止；返回的是部分结果"
        elif not failure and len(lines) < limit:
            returncode = process.returncode if process is not None else None
            if returncode not in {0, 1}:
                failure = f"rg 执行失败（exit {returncode}）：{stderr or '无错误输出'}"
        return lines, timed_out, failure

    def _search_without_ripgrep(
        self,
        *,
        root: Path,
        query: str,
        glob: str,
        limit: int,
        timeout_sec: int,
    ) -> tuple[list[str], bool, int, str]:
        lines: list[str] = []
        scanned = 0
        timed_out = False
        failure = ""
        deadline = time.monotonic() + timeout_sec
        stack = [root]

        while stack and len(lines) < limit:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            current = stack.pop()
            try:
                entries = os.scandir(current)
            except OSError:
                continue
            with entries:
                for entry in entries:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if not _search_ignored(entry.name, is_dir=True):
                                stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if _search_ignored(entry.name):
                            continue
                        path = Path(entry.path)
                        relative = path.relative_to(root)
                        if not self._matches_glob(relative, glob):
                            continue
                        scanned += 1
                        if scanned > SEARCH_MAX_FILES:
                            failure = (
                                f"搜索达到 {SEARCH_MAX_FILES} 个文件上限，"
                                "请缩小 path 或 glob"
                            )
                            break
                        if entry.stat(follow_symlinks=False).st_size > MAX_FILE_BYTES:
                            continue
                        with path.open(
                            "r",
                            encoding="utf-8",
                            errors="replace",
                        ) as stream:
                            for number, line in enumerate(stream, start=1):
                                if time.monotonic() >= deadline:
                                    timed_out = True
                                    break
                                if query in line:
                                    line_text = line.rstrip(
                                        chr(10) + chr(13)
                                    )[:MAX_SEARCH_LINE_CHARS]
                                    lines.append(
                                        f"{relative.as_posix()}:{number}:{line_text}"
                                    )
                                    if len(lines) >= limit:
                                        break
                    except OSError:
                        continue
                    if timed_out or failure or len(lines) >= limit:
                        break
            if timed_out or failure or len(lines) >= limit:
                break
        if timed_out:
            failure = f"搜索超过 {timeout_sec} 秒，已停止；返回的是部分结果"
        return lines, timed_out, scanned, failure

    def _search_text(self, args: dict[str, Any]) -> ToolResult:
        query = str(args["query"])
        if not query:
            return ToolResult(False, "query 不能为空")
        root = self.path_guard.resolve(str(args.get("path", ".")), must_exist=True)
        if not root.is_dir():
            return ToolResult(False, f"搜索路径不是目录: {root}")
        glob = str(args.get("glob", "*") or "*")
        limit = min(500, max(1, int(args.get("max_results", 100))))
        output = str(args.get("output") or "lines")
        if output not in {"lines", "files"}:
            return ToolResult(False, "output 必须是 lines 或 files")
        allow_broad = bool(args.get("allow_broad", False))
        broad_limited = _is_broad_search_root(root) and not allow_broad
        if broad_limited:
            output = "files"
            limit = min(limit, 50)
        search_limit = min(5000, max(limit, limit * 20)) if output == "files" else limit
        timeout_sec = min(
            30,
            max(1, int(args.get("timeout_sec", SEARCH_TIMEOUT_SEC))),
        )
        executable = find_ripgrep()
        if executable:
            lines, timed_out, failure = self._search_with_ripgrep(
                executable,
                root=root,
                query=query,
                glob=glob,
                limit=search_limit,
                timeout_sec=timeout_sec,
            )
            notes = ["engine=rg"]
        else:
            lines, timed_out, scanned, failure = self._search_without_ripgrep(
                root=root,
                query=query,
                glob=glob,
                limit=search_limit,
                timeout_sec=timeout_sec,
            )
            notes = [
                f"engine=bounded-python-fallback (rg not found; scanned {scanned} files)"
            ]

        if output == "files":
            counts: Counter[str] = Counter()
            for line in lines:
                path, separator, _detail = line.partition(":")
                if separator:
                    counts[path] += 1
            grouped = counts.most_common(limit)
            content = (
                "\n".join(f"{path}\t{count} 命中" for path, count in grouped)
                or "(no matches)"
            )
            notes.append(f"grouped_by_file={len(grouped)}")
        else:
            content = "\n".join(lines) or "(no matches)"
        if failure:
            notes.append(failure)
        if broad_limited:
            notes.append(
                "broad_root=metadata_first; use a listed repository path next, "
                "or allow_broad=true only for an explicit global line search"
            )
        return ToolResult(
            ok=not timed_out and not failure,
            content=content + "\n\n[" + "; ".join(notes) + "]",
        )

    def _inspect_code_state(self, args: dict[str, Any]) -> ToolResult:
        target = self.path_guard.resolve(
            str(args.get("path") or "."),
            must_exist=True,
        )
        symbol = str(args.get("symbol") or "").strip()
        if not symbol:
            return ToolResult(False, "symbol 不能为空")
        context_lines = min(
            40,
            max(1, int(args.get("context_lines", 12))),
        )
        max_matches = min(
            8,
            max(1, int(args.get("max_matches", 4))),
        )

        matches: list[dict[str, Any]] = []
        if target.is_file():
            candidates = [(target, self._workspace_relative(target))]
        elif target.is_dir():
            searched = self._search_text(
                {
                    "query": symbol,
                    "path": str(target),
                    "max_results": max_matches * 8,
                    "timeout_sec": SEARCH_TIMEOUT_SEC,
                    "allow_broad": True,
                }
            )
            if not searched.ok and not matches:
                return ToolResult(False, searched.content)
            candidates = []
            for line in searched.content.splitlines():
                if not line or line.startswith("["):
                    continue
                relative, separator, rest = line.partition(":")
                number, number_separator, _ = rest.partition(":")
                if not separator or not number_separator or not number.isdigit():
                    continue
                path = (target / relative).resolve()
                if path.is_file():
                    candidates.append((path, self._workspace_relative(path)))
                    if len(candidates) >= max_matches:
                        break
        else:
            return ToolResult(False, f"不是文件或目录: {target}")

        seen: set[tuple[str, int]] = set()
        for path, relative in candidates:
            try:
                lines = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines, start=1):
                if symbol not in line:
                    continue
                key = (relative, index)
                if key in seen:
                    continue
                seen.add(key)
                start = max(1, index - context_lines)
                end = min(len(lines), index + context_lines)
                matches.append(
                    {
                        "path": relative,
                        "line": index,
                        "text": line.strip()[:500],
                        "context": [
                            f"{number}: {lines[number - 1]}"
                            for number in range(start, end + 1)
                        ],
                    }
                )
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break

        search_root = target if target.is_dir() else target.parent
        repo_root = self._git_value(
            search_root,
            ["rev-parse", "--show-toplevel"],
        )
        head = self._git_value(search_root, ["rev-parse", "HEAD"])
        branch = self._git_value(
            search_root,
            ["rev-parse", "--abbrev-ref", "HEAD"],
        )
        file_commit: dict[str, str] = {}
        symbol_commit: dict[str, str] = {}
        line_commits: dict[str, dict[str, str]] = {}
        if repo_root:
            repo_path = Path(str(repo_root)).resolve()
            try:
                relative_git = target.resolve().relative_to(
                    repo_path
                ).as_posix()
            except ValueError:
                relative_git = ""
            if relative_git:
                file_commit = self._git_commit(
                    repo_path,
                    ["log", "-1", "--format=%H%x09%aI%x09%s", "--", relative_git],
                )
                symbol_commit = self._git_commit(
                    repo_path,
                    [
                        "log",
                        "-S",
                        symbol,
                        "-1",
                        "--format=%H%x09%aI%x09%s",
                        "--",
                        relative_git,
                    ],
                )
                for match in matches:
                    match_path = (
                        Path(self.workspace) / str(match["path"])
                    ).resolve()
                    try:
                        match_relative = match_path.relative_to(
                            repo_path
                        ).as_posix()
                    except ValueError:
                        continue
                    line_commits[str(match["line"])] = self._git_line_commit(
                        repo_path,
                        match_relative,
                        int(match["line"]),
                    )

        target_exists = target.exists()
        version_verified = bool(
            head and target_exists and (line_commits or file_commit)
        )
        payload = {
            "target": self._workspace_relative(target),
            "target_exists": target_exists,
            "symbol": symbol,
            "matches": matches,
            "not_found": not matches,
            "repo_root": str(repo_root or ""),
            "branch": branch,
            "head": head,
            "file_commit": file_commit,
            "symbol_commit": symbol_commit,
            "line_commits": line_commits,
            "version_verified": version_verified,
        }
        return ToolResult(
            True,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _workspace_relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(
                Path(self.workspace).resolve()
            ).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _git_value(self, cwd: Path, arguments: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(cwd), *arguments],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _git_commit(
        self,
        cwd: Path,
        arguments: list[str],
    ) -> dict[str, str]:
        value = self._git_value(cwd, arguments)
        commit, separator, detail = value.partition("\t")
        if not separator:
            return {}
        timestamp, separator, subject = detail.partition("\t")
        return {
            "commit": commit,
            "timestamp": timestamp,
            "subject": subject if separator else "",
        }

    def _git_line_commit(
        self,
        cwd: Path,
        path: str,
        line: int,
    ) -> dict[str, str]:
        value = self._git_value(
            cwd,
            [
                "blame",
                "-L",
                f"{line},{line}",
                "--porcelain",
                "--",
                path,
            ],
        )
        if not value:
            return {}
        rows = value.splitlines()
        commit = rows[0].split(" ", 1)[0] if rows else ""
        author_time = ""
        subject = ""
        for row in rows[1:]:
            if row.startswith("author-time "):
                try:
                    author_time = datetime.fromtimestamp(
                        int(row.removeprefix("author-time "))
                    ).astimezone().isoformat()
                except ValueError:
                    author_time = ""
            elif row.startswith("summary "):
                subject = row.removeprefix("summary ")
            elif not row.startswith("\t") and author_time and subject:
                break
        return {
            "commit": commit,
            "timestamp": author_time,
            "subject": subject,
        }

    def _search_memory(self, args: dict[str, Any]) -> ToolResult:
        if self.memory is None:
            return ToolResult(False, "当前运行未配置记忆库")
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult(False, "search_memory.query 不能为空")
        kind = str(args.get("kind") or "").strip()
        limit = max(1, min(20, int(args.get("limit", 8))))
        rows = self.memory.search(
            query,
            scope=str(self.workspace.resolve()),
            limit=20,
        )
        if kind:
            rows = [row for row in rows if row.get("kind") == kind]
        payload = [
            {
                "id": str(row.get("id") or ""),
                "kind": str(row.get("kind") or ""),
                "scope": str(row.get("scope") or ""),
                "tags": str(row.get("tags") or ""),
                "created_at": str(row.get("created_at") or ""),
                "content": redact_text(str(row.get("content") or ""))[:2000],
            }
            for row in rows[:limit]
        ]
        return ToolResult(
            True,
            json.dumps(
                {
                    "query": query,
                    "count": len(payload),
                    "results": payload,
                },
                ensure_ascii=False,
            ),
        )

    def _memory_context_metrics(self, args: dict[str, Any]) -> ToolResult:
        if self.memory is None:
            return ToolResult(False, "当前运行未配置记忆库，无法统计上下文。")
        query = str(args.get("query") or "").strip()
        if not query and self.request_scope is not None:
            query = self.request_scope.text
        metrics = self.memory.context_metrics(
            scope=str(self.workspace.resolve()),
            query=query,
            preference_limit=max(0, int(args.get("preference_limit", 12))),
            recall_limit=max(0, int(args.get("recall_limit", 5))),
        )
        preference = metrics["preferences"]
        recall = metrics["recall"]
        combined = metrics["combined"]
        return ToolResult(
            True,
            (
                "本轮记忆上下文（只读）：\n"
                f"- 偏好 {preference['items']} 条："
                f"{preference['characters']} 字符，约 "
                f"{preference['estimated_tokens']} tokens\n"
                f"- 相关记忆 {recall['items']} 条："
                f"{recall['characters']} 字符，约 "
                f"{recall['estimated_tokens']} tokens\n"
                f"- 合计：{combined['characters']} 字符，约 "
                f"{combined['estimated_tokens']} tokens\n"
                f"- exact=false；estimator={metrics['estimator']}。"
                "Token 是依赖无关估算，不是 Provider 精确 usage。"
            ),
        )

    def _remember(self, args: dict[str, Any]) -> ToolResult:
        if self.memory is None:
            return ToolResult(False, "当前运行未配置记忆库")
        kind = str(args.get("kind") or "review")
        if kind not in {"preference", "review", "lesson", "episode"}:
            return ToolResult(False, "不支持的记忆类型")
        scope = str(args.get("scope") or "project")
        scope_value = "global" if scope == "global" else str(self.workspace.resolve())
        tags = [
            str(item)
            for item in args.get("tags") or []
            if str(item).strip()
        ]
        if kind == "review":
            tags.extend(["review", "explicit"])
        if bool(args.get("persistent", True)):
            tags.append("persistent")
        memory_id = self.memory.add(
            content=str(args.get("content") or ""),
            scope=scope_value,
            kind=kind,
            status="supported",
            tags=tuple(dict.fromkeys(tags)),
            source="agent_remember_tool",
            evidence_kind="explicit_user_request",
            metadata={"persistent": bool(args.get("persistent", True))},
        )
        return ToolResult(True, f"已写入记忆 {memory_id}")

    def _inspect_environment(self, args: dict[str, Any]) -> ToolResult:
        requested = args.get("commands", [])
        if not isinstance(requested, list):
            requested = []
        return ToolResult(
            True,
            json.dumps(
                discover_environment(
                    (
                        str(item)
                        for item in requested
                        if str(item).strip()
                    ),
                    refresh=bool(args.get("refresh", False)),
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )

    def _inspect_runtime_history(self, args: dict[str, Any]) -> ToolResult:
        payload = inspect_runtime_history(
            self.data_dir,
            mode=str(args.get("mode") or "recent"),
            limit=int(args.get("limit", 10)),
            days=int(args.get("days", 2)),
            turn_id=str(args.get("turn_id") or ""),
            exclude_session=self.current_session_id,
        )
        return ToolResult(
            bool(payload.get("ok")),
            json.dumps(payload, ensure_ascii=False),
        )

    def _inspect_browser(self, args: dict[str, Any]) -> ToolResult:
        browser = str(args.get("browser") or "chrome")
        mode = str(args.get("mode") or "active")
        if not self.approvals.require(
            "读取浏览器页面",
            f"{browser} · {mode}",
            risk="R2",
            capability="command",
        ):
            return ToolResult(False, "用户拒绝读取浏览器页面")
        payload = inspect_browser_page(browser=browser, mode=mode)
        return ToolResult(
            bool(payload.get("ok")),
            json.dumps(payload, ensure_ascii=False),
        )

    def _update_reasoning_state(self, args: dict[str, Any]) -> ToolResult:
        if self.task_state is None:
            return ToolResult(False, "当前没有可更新的任务推理状态")
        criteria = args.get("success_criteria") or []
        if not isinstance(criteria, list):
            return ToolResult(False, "success_criteria 必须是数组")
        current_subgoal = str(args.get("current_subgoal") or "")
        working_hypothesis = str(args.get("working_hypothesis") or "")
        evidence_gap = str(args.get("evidence_gap") or "")
        decision = str(args.get("decision") or "")
        if not any(
            (
                current_subgoal.strip(),
                working_hypothesis.strip(),
                evidence_gap.strip(),
                decision.strip(),
                any(str(item).strip() for item in criteria),
            )
        ):
            return ToolResult(False, "没有需要更新的推理状态")
        self.task_state.update_cognition(
            current_subgoal=current_subgoal,
            working_hypothesis=working_hypothesis,
            evidence_gap=evidence_gap,
            success_criteria=[str(item) for item in criteria],
            decision=decision,
        )
        return ToolResult(
            True,
            "已更新当前任务的推理状态。",
            state_changed=True,
        )

    def _write_file(self, args: dict[str, Any]) -> ToolResult:
        path = self.path_guard.resolve(str(args["path"]), for_write=True)
        self.path_guard.assert_writable(path)
        content = str(args["content"])
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            return ToolResult(False, "写入内容超过大小限制")
        if not self.approvals.require(
            "写入文件",
            f"{path}\n将写入 {len(content.encode('utf-8'))} 字节。",
            capability="write",
        ):
            return ToolResult(False, "用户拒绝写入文件")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            True,
            f"已写入 {path} ({len(content.encode('utf-8'))} bytes)",
            changed_files=[self.path_guard.relative(path)],
        )

    def _write_file_batch(self, args: dict[str, Any]) -> ToolResult:
        files = args.get("files")
        if not isinstance(files, list) or not files:
            return ToolResult(False, "files 必须是非空数组")
        if len(files) > 20:
            return ToolResult(False, "单次最多批量写入 20 个文件")

        staged: dict[Path, str] = {}
        originals: dict[Path, str | None] = {}
        previews: list[str] = []
        for index, item in enumerate(files, 1):
            if not isinstance(item, dict):
                return ToolResult(False, f"第 {index} 个文件参数不合法")
            path = self.path_guard.resolve(
                str(item.get("path") or ""),
                for_write=True,
            )
            self.path_guard.assert_writable(path)
            content = str(item.get("content") or "")
            encoded = content.encode("utf-8")
            if len(encoded) > MAX_FILE_BYTES:
                return ToolResult(False, f"第 {index} 个文件超过大小限制")
            if path in staged and staged[path] != content:
                return ToolResult(
                    False,
                    f"第 {index} 个文件与前面的写入目标重复: {path}",
                )
            staged[path] = content
            originals[path] = (
                path.read_text(encoding="utf-8")
                if path.is_file()
                else None
            )
            previews.append(
                f"{index}. {path} ({len(encoded)} bytes)"
            )

        if not self.approvals.require(
            "批量写入文件",
            "\n".join(previews),
            capability="write",
        ):
            return ToolResult(False, "用户拒绝批量写入文件")

        written: list[Path] = []
        try:
            for path, content in staged.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_write_text(path, content)
                written.append(path)
        except OSError as error:
            rollback_errors: list[str] = []
            for path in reversed(written):
                original = originals[path]
                try:
                    if original is None:
                        path.unlink(missing_ok=True)
                    else:
                        self._atomic_write_text(path, original)
                except OSError as rollback_error:
                    rollback_errors.append(f"{path}: {rollback_error}")
            detail = f"批量写入失败，已尝试回滚: {error}"
            if rollback_errors:
                detail += "\n回滚失败: " + "; ".join(rollback_errors)
            return ToolResult(False, detail)

        changed_files = [
            self.path_guard.relative(path)
            for path in staged
        ]
        return ToolResult(
            True,
            f"已批量写入 {len(changed_files)} 个文件。",
            changed_files=changed_files,
        )

    def _write_document(self, args: dict[str, Any]) -> ToolResult:
        path = self.path_guard.resolve(str(args["path"]), for_write=True)
        self.path_guard.assert_writable(path)
        content = render_document(
            profile=str(args.get("profile") or ""),
            title=str(args.get("title") or ""),
            summary=str(args.get("summary") or ""),
            sections=list(args.get("sections") or []),
            sources=[str(item) for item in args.get("sources", [])],
        )
        size = len(content.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            return ToolResult(False, "文档内容超过大小限制")
        if not self.approvals.require(
            "写入结构化文档",
            f"{path}\n模板: {args.get('profile')}\n将写入 {size} 字节。",
            capability="write",
        ):
            return ToolResult(False, "用户拒绝写入文档")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            True,
            f"已写入结构化文档 {path} ({size} bytes)",
            changed_files=[self.path_guard.relative(path)],
        )

    def _patch_file(self, args: dict[str, Any]) -> ToolResult:
        path = self.path_guard.resolve(
            str(args["path"]),
            must_exist=True,
            for_write=True,
        )
        self.path_guard.assert_writable(path)
        if not path.is_file():
            return ToolResult(False, f"不是文件: {path}")
        old_text = str(args["old_text"])
        new_text = str(args["new_text"])
        if not old_text:
            return ToolResult(False, "old_text 不能为空")
        content = path.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            return ToolResult(False, "未找到待替换文本")
        replace_all = bool(args.get("replace_all", False))
        if occurrences > 1 and not replace_all:
            return ToolResult(
                False,
                f"待替换文本出现 {occurrences} 次；请提供更长上下文或设置 replace_all=true",
            )
        preview = old_text[:1000]
        if not self.approvals.require(
            "修改文件",
            f"{path}\n替换 {occurrences if replace_all else 1} 处。\n--- old ---\n{preview}",
            capability="write",
        ):
            return ToolResult(False, "用户拒绝修改文件")
        updated = content.replace(old_text, new_text, -1 if replace_all else 1)
        path.write_text(updated, encoding="utf-8")
        return ToolResult(
            True,
            f"已修改 {path}",
            changed_files=[self.path_guard.relative(path)],
        )

    def _patch_file_batch(self, args: dict[str, Any]) -> ToolResult:
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            return ToolResult(False, "edits 必须是非空数组")
        if len(edits) > 20:
            return ToolResult(False, "单次最多批量应用 20 处替换")

        staged: dict[Path, str] = {}
        originals: dict[Path, str] = {}
        previews: list[str] = []
        for index, edit in enumerate(edits, 1):
            if not isinstance(edit, dict):
                return ToolResult(False, f"第 {index} 处替换参数不合法")
            path = self.path_guard.resolve(
                str(edit.get("path") or ""),
                must_exist=True,
                for_write=True,
            )
            self.path_guard.assert_writable(path)
            if not path.is_file():
                return ToolResult(False, f"第 {index} 处不是文件: {path}")
            if path not in staged:
                staged[path] = path.read_text(encoding="utf-8")
                originals[path] = staged[path]
            content = staged[path]
            old_text = str(edit.get("old_text") or "")
            new_text = str(edit.get("new_text") or "")
            if not old_text:
                return ToolResult(False, f"第 {index} 处 old_text 不能为空")
            occurrences = content.count(old_text)
            if occurrences == 0:
                return ToolResult(
                    False,
                    f"第 {index} 处未找到待替换文本: {path}",
                )
            replace_all = bool(edit.get("replace_all", False))
            if occurrences > 1 and not replace_all:
                return ToolResult(
                    False,
                    f"第 {index} 处文本出现 {occurrences} 次；"
                    "请提供更长上下文或设置 replace_all=true",
                )
            staged[path] = content.replace(
                old_text,
                new_text,
                -1 if replace_all else 1,
            )
            previews.append(
                f"{index}. {path}\n"
                f"替换 {occurrences if replace_all else 1} 处\n"
                f"--- old ---\n{old_text[:500]}"
            )

        changed = {
            path: content
            for path, content in staged.items()
            if content != originals[path]
        }
        if not changed:
            return ToolResult(False, "批量替换没有产生任何文件变化")
        if not self.approvals.require(
            "批量修改文件",
            "\n\n".join(previews)[:6000],
            capability="write",
        ):
            return ToolResult(False, "用户拒绝批量修改文件")

        written: list[Path] = []
        try:
            for path, content in changed.items():
                self._atomic_write_text(path, content)
                written.append(path)
        except OSError as error:
            rollback_errors: list[str] = []
            for path in written:
                try:
                    self._atomic_write_text(path, originals[path])
                except OSError as rollback_error:
                    rollback_errors.append(f"{path}: {rollback_error}")
            detail = f"批量写入失败，已尝试回滚: {error}"
            if rollback_errors:
                detail += "\n回滚失败: " + "; ".join(rollback_errors)
            return ToolResult(False, detail)

        changed_files = [
            self.path_guard.relative(path)
            for path in changed
        ]
        return ToolResult(
            True,
            f"已批量应用 {len(edits)} 处替换，修改 {len(changed_files)} 个文件。",
            changed_files=changed_files,
        )

    def _apply_and_verify(self, args: dict[str, Any]) -> ToolResult:
        patch = self._patch_file_batch(
            {"edits": args.get("edits") or []}
        )
        if not patch.ok:
            return patch
        verification = self._verification_batch(
            {"commands": args.get("verification_commands") or []}
        )
        content = "\n\n".join(
            (
                "=== 文件修改 ===",
                patch.content,
                "=== 验证结果 ===",
                verification.content,
            )
        )
        return ToolResult(
            verification.ok,
            content,
            changed_files=list(patch.changed_files),
            artifact=verification.artifact,
            state_changed=True,
        )

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        mode = path.stat().st_mode if path.exists() else 0o600
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            temporary_path.write_text(content, encoding="utf-8")
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, path)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def _run_command(self, args: dict[str, Any]) -> ToolResult:
        command = str(args["command"]).strip()
        if not command:
            return ToolResult(False, "命令为空")
        denial = command_denial_reason(command)
        if denial:
            return ToolResult(False, denial)
        cwd = self.path_guard.resolve(str(args.get("cwd", ".")), must_exist=True)
        if not cwd.is_dir():
            return ToolResult(False, f"cwd 不是目录: {cwd}")
        if not self.approvals.require_command(command):
            return ToolResult(False, "用户拒绝执行命令")
        timeout = min(1800, max(1, int(args.get("timeout_sec", 120))))
        result = self._execute_shell_command(
            command,
            cwd=cwd,
            timeout=timeout,
        )
        if result.ok:
            result.state_changed = _command_changes_target_state(command)
        return result

    def _run_command_batch(self, args: dict[str, Any]) -> ToolResult:
        raw_commands = args.get("commands")
        if not isinstance(raw_commands, list) or not raw_commands:
            return ToolResult(False, "commands 必须是非空数组")
        if len(raw_commands) > 4:
            return ToolResult(False, "单次最多批量执行 4 条命令")

        commands: list[tuple[str, Path, int]] = []
        for index, item in enumerate(raw_commands, 1):
            if not isinstance(item, dict):
                return ToolResult(False, f"第 {index} 条命令参数不合法")
            command = str(item.get("command") or "").strip()
            if not command:
                return ToolResult(False, f"第 {index} 条命令为空")
            denial = command_denial_reason(command)
            if denial:
                return ToolResult(False, f"第 {index} 条命令被拒绝: {denial}")
            if not self._read_only_command_allowed(command):
                return ToolResult(
                    False,
                    f"第 {index} 条命令不是只读 R0，也未命中只读命令配置，"
                    "不能进入批量执行。",
                )
            cwd = self.path_guard.resolve(
                str(item.get("cwd", ".")),
                must_exist=True,
            )
            if not cwd.is_dir():
                return ToolResult(False, f"第 {index} 条命令 cwd 不是目录: {cwd}")
            timeout = min(
                120,
                max(1, int(item.get("timeout_sec", 30))),
            )
            commands.append((command, cwd, timeout))

        approval_summary = "\n".join(
            f"{index}. [{cwd}] {command}"
            for index, (command, cwd, _timeout) in enumerate(commands, 1)
        )
        if not self.approvals.require(
            "运行只读命令批次",
            approval_summary,
            risk="R1",
            capability="command",
        ):
            return ToolResult(False, "用户拒绝执行命令批次")

        with ThreadPoolExecutor(
            max_workers=min(4, len(commands)),
        ) as pool:
            futures = [
                pool.submit(
                    self._execute_shell_command,
                    command,
                    cwd=cwd,
                    timeout=timeout,
                )
                for command, cwd, timeout in commands
            ]
            results = [future.result() for future in futures]

        sections: list[str] = []
        for index, ((command, cwd, _timeout), result) in enumerate(
            zip(commands, results),
            1,
        ):
            sections.append(
                "\n".join(
                    (
                        f"=== [{index}] {command}",
                        f"cwd={cwd}",
                        result.content,
                    )
                )
            )
        output = "\n\n".join(sections)
        content, artifact = self._truncate(output, prefix="command-batch")
        return ToolResult(
            all(result.ok for result in results),
            content,
            artifact=artifact,
        )

    def _verification_batch(self, args: dict[str, Any]) -> ToolResult:
        raw_commands = args.get("commands")
        if not isinstance(raw_commands, list) or not raw_commands:
            return ToolResult(False, "commands 必须是非空数组")
        if len(raw_commands) > 4:
            return ToolResult(False, "单次最多并发执行 4 条验证命令")

        commands: list[tuple[str, Path, int]] = []
        for index, item in enumerate(raw_commands, 1):
            if not isinstance(item, dict):
                return ToolResult(False, f"第 {index} 条验证参数不合法")
            command = str(item.get("command") or "").strip()
            if not self._verification_command_allowed(command):
                return ToolResult(
                    False,
                    f"第 {index} 条命令不是受支持的单项验证命令: {command}",
                )
            cwd = self.path_guard.resolve(
                str(item.get("cwd", ".")),
                must_exist=True,
            )
            if not cwd.is_dir():
                return ToolResult(False, f"第 {index} 条 cwd 不是目录: {cwd}")
            timeout = min(
                300,
                max(1, int(item.get("timeout_sec", 120))),
            )
            commands.append((command, cwd, timeout))

        summary = "\n".join(
            f"{index}. [{cwd}] {command}"
            for index, (command, cwd, _timeout) in enumerate(commands, 1)
        )
        if not self.approvals.require(
            "运行验证命令批次",
            summary,
            risk="R1",
            capability="command",
        ):
            return ToolResult(False, "用户拒绝执行验证命令批次")

        with ThreadPoolExecutor(
            max_workers=min(4, len(commands)),
        ) as pool:
            futures = [
                pool.submit(
                    self._execute_shell_command,
                    command,
                    cwd=cwd,
                    timeout=timeout,
                )
                for command, cwd, timeout in commands
            ]
            results = [future.result() for future in futures]

        sections: list[str] = []
        for index, ((command, cwd, _timeout), result) in enumerate(
            zip(commands, results),
            1,
        ):
            sections.append(
                "\n".join(
                    (
                        f"=== [{index}] {command}",
                        f"cwd={cwd}",
                        result.content,
                    )
                )
            )
        content, artifact = self._truncate(
            "\n\n".join(sections),
            prefix="verification-batch",
        )
        return ToolResult(
            all(result.ok for result in results),
            content,
            artifact=artifact,
        )

    def _execute_shell_command(
        self,
        command: str,
        *,
        cwd: Path,
        timeout: int,
    ) -> ToolResult:
        try:
            shell = login_shell()
            shell_command = command
            if Path(shell).name.lower() not in {"pwsh", "powershell"}:
                shell_command = f"set -o pipefail\n{command}"
            process = subprocess.Popen(
                shell_invocation(shell, shell_command),
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as error:
            return ToolResult(False, str(error))

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self._stop_process(process)
            process.communicate()
            partial = _command_output_text(error.stdout)
            partial += _command_output_text(error.stderr)
            output = f"命令超时（{timeout}s）\n{partial}"
            content, artifact = self._truncate(output, prefix="timeout")
            return ToolResult(False, content, artifact=artifact)

        output = (
            f"exit_code={process.returncode}\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )
        semantic_failure = _semantic_command_failure(output)
        if semantic_failure:
            output += f"\n--- semantic_check ---\n{semantic_failure}"
        content, artifact = self._truncate(output, prefix="command")
        return ToolResult(
            process.returncode == 0 and not semantic_failure,
            content,
            artifact=artifact,
        )

    def _request_authorization(self, args: dict[str, Any]) -> ToolResult:
        capability = str(args.get("capability") or "").strip()
        reason = str(args.get("reason") or "").strip()
        if capability in {"mutation_and_commands", "write_and_command"}:
            requested = {"write", "command"}
        elif capability in {"mutation", "commands"}:
            requested = (
                {"write"} if capability == "mutation" else {"command"}
            )
        else:
            requested = normalize_capabilities([capability])
        if not requested:
            return ToolResult(False, "未知授权类型")
        if self.request_scope is None:
            return ToolResult(False, "当前没有可扩展的任务范围")
        if not reason:
            return ToolResult(False, "授权请求缺少原因")
        states = self.approvals.permission_state()
        denied = sorted(item for item in requested if states.get(item) == "deny")
        if denied:
            return ToolResult(
                False,
                "用户已明确撤销授权，当前回合不能自动弹窗绕过："
                + "、".join(denied),
            )
        already_allowed = all(
            (
                self.request_scope.allows_mutation
                if item == "write"
                else self.request_scope.allows_commands
                if item == "command"
                else self.request_scope.allows_network
            )
            for item in requested
        )
        if already_allowed:
            return ToolResult(True, "所需能力已获当前任务授权")

        approved = self.approvals.require(
            "扩大当前任务授权",
            f"需要能力：{capability}\n原因：{reason}",
            risk="R2",
        )
        if not approved:
            return ToolResult(False, "用户拒绝扩大任务授权")

        mutation = "write" in requested
        commands = "command" in requested
        self.approvals.grant_task_capabilities(requested)
        self.approvals.consume_interactive_decision()
        self.request_scope = replace(
            self.request_scope,
            allows_mutation=self.request_scope.allows_mutation or mutation,
            allows_commands=self.request_scope.allows_commands or commands,
            allows_network=self.request_scope.allows_network
            or "network" in requested,
        )
        return ToolResult(
            True,
            f"已授权：{capability}。下一步可以使用新开放的工具；"
            "授权只在当前任务回合有效。",
        )

    def _configure_permissions(self, args: dict[str, Any]) -> ToolResult:
        reason = str(args.get("reason") or "").strip()
        if not reason:
            return ToolResult(False, "权限配置缺少用户授意说明")
        raw_modes = args.get("permissions") or {}
        if not isinstance(raw_modes, dict):
            return ToolResult(False, "permissions 必须是 object")
        raw_paths = args.get("write_allowlist") or []
        if not isinstance(raw_paths, list):
            return ToolResult(False, "write_allowlist 必须是数组")
        mode = str(args.get("mode") or "add").strip().lower()
        if mode not in {"add", "set", "remove"}:
            return ToolResult(False, "mode 必须是 add/set/remove")
        if not raw_modes and not raw_paths:
            return ToolResult(False, "没有需要更新的权限配置")

        permission_updates: dict[str, str] = {}
        for capability in ("write", "command", "network"):
            if capability not in raw_modes:
                continue
            value = str(raw_modes.get(capability) or "").strip().lower()
            if value not in {"auto", "allow", "deny"}:
                return ToolResult(
                    False,
                    f"{capability} 必须是 auto/allow/deny",
                )
            permission_updates[capability] = value

        current_paths = set(self.path_guard.write_allowlist)
        requested_paths: set[Path] = set()
        for raw_path in raw_paths:
            value = str(raw_path).strip()
            if not value:
                continue
            candidate = Path(value).expanduser().resolve(strict=False)
            if candidate in {Path("/"), Path.home().resolve()}:
                return ToolResult(
                    False,
                    f"写入白名单范围过大，拒绝配置: {candidate}",
                )
            requested_paths.add(candidate)
        if len(current_paths | requested_paths) > 16:
            return ToolResult(False, "写入白名单最多允许 16 个路径")
        if mode == "set":
            next_paths = requested_paths
        elif mode == "remove":
            next_paths = current_paths - requested_paths
        else:
            next_paths = current_paths | requested_paths

        updates: dict[str, Any] = {
            "runtime": {
                "permissions": {
                    **permission_updates,
                    "write_allowlist": [
                        str(path) for path in sorted(next_paths)
                    ],
                }
            }
        }
        try:
            path = save_user_config(self.data_dir, updates)
        except Exception as error:
            return ToolResult(False, f"保存权限配置失败: {error}")
        self.path_guard.set_write_allowlist(next_paths)
        for capability, value in permission_updates.items():
            self.approvals.set_persistent_permission(capability, value)
        summary = (
            f"已更新 {path}；写入白名单 {len(next_paths)} 项"
            + (
                "；权限模式 "
                + "、".join(
                    f"{name}={value}"
                    for name, value in sorted(permission_updates.items())
                )
                if permission_updates
                else ""
            )
        )
        self.events.emit(
            "permissions_configured",
            path=str(path),
            write_allowlist=[str(item) for item in sorted(next_paths)],
            permissions=permission_updates,
            summary=summary,
        )
        return ToolResult(True, summary, state_changed=True)

    def _schedule_proactive_file_write(
        self,
        args: dict[str, Any],
    ) -> ToolResult:
        if self.proactive_scheduler is None:
            return ToolResult(False, "主动调度器未初始化")
        target_template = str(
            args.get("target_path_template") or ""
        ).strip()
        if not target_template:
            return ToolResult(False, "待办目标路径不能为空")
        cadence = str(args.get("cadence") or "once").strip().lower()
        not_before = str(args.get("not_before") or "20:00").strip()
        instruction = str(args.get("instruction") or "").strip()
        task_name = str(args.get("task_name") or "").strip()
        source_memory_id = str(
            args.get("source_memory_id") or ""
        ).strip()
        if not instruction:
            return ToolResult(False, "待办说明不能为空")
        rendered_for_path = target_template.replace(
            "{date}",
            datetime.now().astimezone().date().isoformat(),
        )
        target = Path(rendered_for_path).expanduser()
        if not target.is_absolute():
            target = self.workspace / target
        target = target.resolve(strict=False)
        parent = target.parent if target.suffix else target
        if parent in {Path("/"), Path.home().resolve()}:
            return ToolResult(False, f"待办路径范围过大: {parent}")
        if is_sensitive_write_path(parent):
            return ToolResult(False, f"待办不能写入敏感路径: {parent}")
        permission_result = self._configure_permissions(
            {
                "write_allowlist": [str(parent)],
                "mode": "add",
                "reason": f"为主动待办 {task_name} 持久授权目标目录",
            }
        )
        if not permission_result.ok:
            return permission_result
        try:
            task = self.proactive_scheduler.create_task(
                name=task_name,
                instruction=instruction,
                target_path_template=target_template,
                cadence=cadence,
                not_before=not_before,
                source_memory_id=source_memory_id,
            )
        except Exception as error:
            return ToolResult(False, f"保存主动待办失败: {error}")
        payload = {
            "id": task.id,
            "name": task.name,
            "instruction": task.instruction,
            "target_path_template": task.target_path_template,
            "cadence": task.cadence,
            "not_before": task.not_before,
            "status": task.status,
            "source_memory_id": task.source_memory_id,
        }
        self.events.emit(
            "proactive_task_scheduled",
            task_id=task.id,
            name=task.name,
            cadence=task.cadence,
            target_path_template=task.target_path_template,
            summary=f"已创建主动待办: {task.name}",
        )
        return ToolResult(
            True,
            json.dumps(payload, ensure_ascii=False),
            state_changed=True,
        )

    def _fetch_url(
        self,
        args: dict[str, Any],
        *,
        _approved: bool = False,
    ) -> ToolResult:
        url = str(args["url"])
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolResult(False, "仅支持 http/https URL")
        if not _approved and not self.approvals.require(
            "访问网络",
            url,
            risk="R2",
            capability="network",
        ):
            return ToolResult(False, "用户拒绝访问网络")
        max_chars = min(100000, max(1000, int(args.get("max_chars", 30000))))
        timeout_sec = min(
            30,
            max(3, int(args.get("timeout_sec", 15))),
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Sunus/0.1"},
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_sec,
                context=make_ssl_context(),
            ) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                raw = response.read(2 * 1024 * 1024)
        except urllib.error.HTTPError as error:
            return ToolResult(
                False,
                json.dumps(
                    {
                        "error": "http_error",
                        "status": int(error.code),
                        "detail": redact_text(str(error.reason)),
                        "host": parsed.netloc,
                        "retryable": error.code in {408, 425, 429, 500, 502, 503, 504},
                        "suggestion": "读取状态码后再决定是否换来源，不要原样重试。",
                    },
                    ensure_ascii=False,
                ),
            )
        except urllib.error.URLError as error:
            reason = error.reason
            detail = redact_text(str(reason))
            lowered = detail.lower()
            if isinstance(reason, socket.gaierror):
                code = "dns_error"
                suggestion = "检查域名和网络后换来源；不要继续轮换同一站点的路径。"
            elif isinstance(reason, ssl.SSLError) or "ssl" in lowered or "certificate" in lowered:
                code = "tls_error"
                suggestion = "TLS/证书握手失败；停止重试同一站点，改用已有资料或其他来源。"
            elif (
                isinstance(reason, (TimeoutError, socket.timeout))
                or "timed out" in lowered
                or "timeout" in lowered
            ):
                code = "network_timeout"
                suggestion = "本轮网络请求超时；不要连续请求同类网页，改用已有资料或结束等待。"
            else:
                code = "network_error"
                suggestion = "网络不可用；先说明能力边界，不要反复重试。"
            return ToolResult(
                False,
                json.dumps(
                    {
                        "error": code,
                        "detail": detail,
                        "host": parsed.netloc,
                        "timeout_sec": timeout_sec,
                        "retryable": code in {"network_timeout", "dns_error"},
                        "suggestion": suggestion,
                    },
                    ensure_ascii=False,
                ),
            )
        except (TimeoutError, socket.timeout) as error:
            return ToolResult(
                False,
                json.dumps(
                    {
                        "error": "network_timeout",
                        "detail": redact_text(str(error)),
                        "host": parsed.netloc,
                        "timeout_sec": timeout_sec,
                        "retryable": True,
                        "suggestion": "本轮网络请求超时；停止连续访问同类网页，改用已有资料。",
                    },
                    ensure_ascii=False,
                ),
            )
        except (ssl.SSLError, OSError) as error:
            code = (
                "tls_error"
                if isinstance(error, ssl.SSLError)
                else "network_error"
            )
            return ToolResult(
                False,
                json.dumps(
                    {
                        "error": code,
                        "detail": redact_text(str(error)),
                        "host": parsed.netloc,
                        "retryable": False,
                        "suggestion": "网络连接失败；停止重试并基于已有证据回答。",
                    },
                    ensure_ascii=False,
                ),
            )
        text = raw.decode(charset, errors="replace")[:max_chars]
        content, artifact = self._truncate(text, prefix="fetch")
        return ToolResult(True, content, artifact=artifact)

    def _fetch_url_batch(self, args: dict[str, Any]) -> ToolResult:
        requests = args.get("requests")
        if not isinstance(requests, list) or not requests:
            return ToolResult(False, "requests 必须是非空数组")
        if len(requests) > 5:
            return ToolResult(False, "单次最多并发获取 5 个 URL")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(requests, 1):
            if not isinstance(item, dict):
                return ToolResult(False, f"第 {index} 个请求参数不合法")
            url = str(item.get("url") or "").strip()
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return ToolResult(False, f"第 {index} 个 URL 不是 http/https")
            normalized.append(
                {
                    "url": url,
                    "max_chars": item.get("max_chars", 30000),
                    "timeout_sec": item.get("timeout_sec", 15),
                }
            )
        summary = "\n".join(
            f"{index}. {item['url']}"
            for index, item in enumerate(normalized, 1)
        )
        if not self.approvals.require(
            "批量访问网络",
            summary,
            risk="R2",
            capability="network",
        ):
            return ToolResult(False, "用户拒绝批量访问网络")

        with ThreadPoolExecutor(
            max_workers=min(5, len(normalized)),
        ) as pool:
            futures = [
                pool.submit(
                    lambda item=item: self._fetch_url(
                        item,
                        _approved=True,
                    )
                )
                for item in normalized
            ]
            results = [future.result() for future in futures]

        sections: list[str] = []
        for index, (item, result) in enumerate(
            zip(normalized, results),
            1,
        ):
            sections.append(
                "\n".join(
                    (
                        f"=== [{index}] {item['url']}",
                        result.content,
                    )
                )
            )
        content, artifact = self._truncate(
            "\n\n".join(sections),
            prefix="fetch-batch",
        )
        return ToolResult(
            all(result.ok for result in results),
            content,
            artifact=artifact,
        )

    def _list_skills(self, args: dict[str, Any]) -> ToolResult:
        skills = self.skills.discover()
        if not skills:
            return ToolResult(
                True,
                f"(暂无技能；在 {SKILLS_DIRNAME}/<name>/SKILL.md 中创建即可)",
            )
        lines = [f"{len(skills)} 个本地技能（{SKILLS_DIRNAME}/）："]
        lines.extend(f"- {skill.name}: {skill.description}" for skill in skills)
        if any(skill.when_to_use for skill in skills):
            lines.append("")
            for skill in skills:
                if skill.when_to_use:
                    lines.append(f"- {skill.name} 适用：{skill.when_to_use}")
        lines.append("")
        lines.append("用 read_skill(name) 读取某个技能的完整步骤。")
        return ToolResult(True, "\n".join(lines))

    def _read_skill(self, args: dict[str, Any]) -> ToolResult:
        name = str(args.get("name", "")).strip()
        if not name:
            return ToolResult(False, "缺少技能名")
        skill = self.skills.get(name)
        if skill is None:
            available = ", ".join(item.name for item in self.skills.discover()) or "(无)"
            return ToolResult(False, f"未找到技能: {name}；可用: {available}")
        body = (
            f"# 技能: {skill.name}\n"
            f"文件: {skill.relative_path}\n"
            f"说明: {skill.description}\n"
        )
        if skill.when_to_use:
            body += f"适用: {skill.when_to_use}\n"
        body += (
            "\n以下为技能流程（参考方法，不覆盖安全原则与授权模型）：\n\n"
            + skill.body
        )
        content, artifact = self._truncate(body, prefix="skill")
        return ToolResult(True, content, artifact=artifact)

