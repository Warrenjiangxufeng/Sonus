from __future__ import annotations

import os
import re
import select
import shlex
import shutil
import sys
import termios
import threading
import tty
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Literal

from sunus.scope import RequestScope


CAPABILITY_ALIASES = {
    "write": "write",
    "edit": "write",
    "files": "write",
    "file": "write",
    "写入": "write",
    "写文件": "write",
    "command": "command",
    "commands": "command",
    "shell": "command",
    "run": "command",
    "命令": "command",
    "network": "network",
    "web": "network",
    "网络": "network",
}
CAPABILITIES = frozenset(CAPABILITY_ALIASES.values())
_EXPLICIT_PATH = re.compile(
    r"(?<![\w])(?:~|\.{1,2})?/[^\s`'\"“”‘’，。！？；;：:（）()\[\]{}<>]+"
    r"|(?<![\w])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
)
_QUOTED_PATH = re.compile(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"|“([^”]+)”")
_SENSITIVE_WRITE_PARTS = {
    ".aws",
    ".codex",
    ".dsh",
    ".ssh",
    "Keychains",
}


def normalize_capabilities(values: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    for value in values:
        for item in re.split(r"[,，、\s]+", str(value).strip().lower()):
            if not item:
                continue
            if item in {"all", "全部", "所有"}:
                selected.update(CAPABILITIES)
                continue
            capability = CAPABILITY_ALIASES.get(item)
            if capability:
                selected.add(capability)
    return selected


class PolicyError(RuntimeError):
    pass


def explicit_write_paths(text: str, workspace: Path) -> tuple[Path, ...]:
    """Extract paths explicitly named in the current user request."""
    root = workspace.expanduser().resolve()
    candidates: list[str] = []
    for match in _QUOTED_PATH.finditer(str(text or "")):
        value = next((group for group in match.groups() if group), "")
        value = value.strip()
        if (
            value
            and "/" in value
            and not value.startswith(("http://", "https://"))
        ):
            candidates.append(value)
    candidates.extend(_EXPLICIT_PATH.findall(str(text or "")))

    resolved_paths: list[Path] = []
    for raw in candidates:
        value = raw.strip().rstrip(".,;:!?，。；：！？")
        if not value or value.startswith(("http://", "https://")):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved in {Path("/"), Path.home().resolve()}:
            continue
        if resolved not in resolved_paths:
            resolved_paths.append(resolved)
    return tuple(resolved_paths)


def is_sensitive_write_path(path: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    try:
        parts = resolved.relative_to(Path.home().resolve()).parts
    except ValueError:
        parts = ()
    return any(part in _SENSITIVE_WRITE_PARTS for part in parts)


class PathGuard:
    def __init__(
        self,
        workspace: Path,
        allow_outside: bool = False,
        write_allowlist: Iterable[Path] = (),
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.allow_outside = allow_outside
        self.write_allowlist = tuple(
            path.expanduser().resolve(strict=False)
            for path in write_allowlist
        )
        self.request_write_paths: tuple[Path, ...] = ()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def set_write_allowlist(self, paths: Iterable[Path]) -> None:
        self.write_allowlist = tuple(
            path.expanduser().resolve(strict=False)
            for path in paths
        )

    def is_write_allowlisted(self, path: Path) -> bool:
        resolved = path.expanduser().resolve(strict=False)
        return any(
            resolved == root or root in resolved.parents
            for root in self.write_allowlist
        )

    def set_request_write_paths(self, paths: Iterable[Path]) -> None:
        self.request_write_paths = tuple(
            path.expanduser().resolve(strict=False)
            for path in paths
        )

    def clear_request_write_paths(self) -> None:
        self.request_write_paths = ()

    def is_request_write_path(self, path: Path) -> bool:
        resolved = path.expanduser().resolve(strict=False)
        return any(
            resolved == root or root in resolved.parents
            for root in self.request_write_paths
        )

    def is_write_permitted(self, path: Path) -> bool:
        return (
            self.is_write_allowlisted(path)
            or self.is_request_write_path(path)
        )

    def resolve(
        self,
        value: str | Path,
        *,
        must_exist: bool = False,
        for_write: bool = False,
    ) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve(strict=False)
        if must_exist and not resolved.exists():
            raise PolicyError(f"路径不存在: {resolved}")
        if not self.allow_outside and not (
            resolved == self.workspace or self.workspace in resolved.parents
        ) and not (
            for_write and self.is_write_permitted(resolved)
        ):
            raise PolicyError(
                f"路径越界: {resolved}；工作区为 {self.workspace}。"
                "如确有需要，请在当前请求中明确指定目标路径。"
            )
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.workspace))
        except ValueError:
            return str(path)

    def assert_writable(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        if not self.allow_outside and not (
            resolved == self.workspace or self.workspace in resolved.parents
        ) and not self.is_write_permitted(resolved):
            raise PolicyError(
                f"路径越界: {resolved}；工作区为 {self.workspace}。"
                "写文件时需在当前请求中明确指定目标路径。"
            )
        if is_sensitive_write_path(resolved):
            raise PolicyError(f"敏感路径不可由文件工具修改: {resolved}")
        protected_parts = {".git", "__pycache__"}
        try:
            relative = resolved.relative_to(self.workspace)
        except ValueError:
            relative = None
        if relative and any(part in protected_parts for part in relative.parts):
            raise PolicyError(f"受保护路径不可修改: {resolved}")
        if (
            relative
            and ".sunus" in relative.parts
            and not self.is_write_allowlisted(resolved)
        ):
            raise PolicyError(f"受保护路径不可修改: {resolved}")


_DANGEROUS_PATTERNS = (
    re.compile(r"(?i)\bsudo\b"),
    re.compile(r"(?i)\b(shutdown|reboot|halt)\b"),
    re.compile(r"(?i)\bdiskutil\s+(erase|reformat)"),
    re.compile(r"(?i)\bmkfs(\.|\b)"),
    re.compile(r"(?i)\bdd\b[^\n]*\bof\s*=\s*/dev/"),
    re.compile(r"(?i)\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)\s+(?:/|~|\\$HOME)(?:\s|$)"),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"),
    re.compile(
        r"(?i)Object\.defineProperty\s*\(\s*process\.versions"
        r"|process\.versions\.node\s*="
    ),
    re.compile(
        r"(?i)\bNODE_OPTIONS\s*=.*--require\s+"
        r"(?:/tmp/|/private/tmp/).*(?:shim|node20|pw-node)"
    ),
)

_SENSITIVE_COMMAND_PATTERNS = (
    re.compile(r"(?i)(?:^|[/\\\s])\.codex[/\\]config\.toml\b"),
    re.compile(r"(?i)(?:^|[/\\\s])\.dsh[/\\]\.credentials(?:\.yaml)?\b"),
    re.compile(r"(?i)(?:^|[/\\\s])\.ssh(?:[/\\]|\b)"),
    re.compile(r"(?i)(?:^|[/\\\s])\.aws[/\\]credentials\b"),
    re.compile(r"(?i)\bsecurity\s+(?:find-generic-password|dump-keychain)\b"),
)

RiskLevel = Literal["R0", "R1", "R2", "R3"]
_RISK_ORDER: dict[RiskLevel, int] = {"R0": 0, "R1": 1, "R2": 2, "R3": 3}
_SHELL_CONTROL = re.compile(r"[;&|<>`\n]|\$\(|\$\{")
_READ_ONLY_COMMANDS = {
    "cat",
    "cd",
    "cmp",
    "date",
    "df",
    "diff",
    "du",
    "echo",
    "file",
    "find",
    "grep",
    "head",
    "ls",
    "printf",
    "pwd",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "true",
    "tree",
    "uname",
    "wc",
    "which",
    "whoami",
}
_LOCAL_COMMANDS = {
    "bash",
    "cargo",
    "cp",
    "go",
    "make",
    "mkdir",
    "mv",
    "node",
    "npm",
    "pnpm",
    "pytest",
    "python",
    "python3",
    "ruby",
    "sh",
    "swift",
    "touch",
    "unittest",
    "yarn",
    "zsh",
}
_NETWORK_OR_DESTRUCTIVE_COMMANDS = {
    "chmod",
    "chown",
    "curl",
    "defaults",
    "kill",
    "launchctl",
    "open",
    "osascript",
    "pkill",
    "rm",
    "rmdir",
    "rsync",
    "scp",
    "shred",
    "ssh",
    "unlink",
    "wget",
}


@dataclass
class TaskGrant:
    """One task-scoped grant. It expires when the turn ends."""

    risk_ceiling: RiskLevel = "R1"

    def allows(self, risk: RiskLevel) -> bool:
        return _RISK_ORDER[risk] <= _RISK_ORDER[self.risk_ceiling]


def _contains_outside_path(tokens: list[str], workspace: Path) -> bool:
    root = workspace.expanduser().resolve()
    for token in tokens:
        if token.startswith(("http://", "https://")):
            continue
        if not (
            token.startswith("/")
            or token.startswith("~")
            or token == ".."
            or token.startswith("../")
            or "/../" in token
        ):
            continue
        try:
            resolved = Path(token).expanduser().resolve(strict=False)
        except OSError:
            return True
        if resolved != root and root not in resolved.parents:
            return True
    return False


def _has_unquoted_newline(command: str) -> bool:
    quote = ""
    escaped = False
    for character in command:
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if character == quote:
                quote = ""
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "\n":
            return True
    return False


def _single_heredoc_head(command: str) -> str | None:
    lines = command.splitlines()
    if len(lines) < 2 or lines[0].count("<<") != 1:
        return None
    match = re.search(
        r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[ \t]*$",
        lines[0],
    )
    if not match:
        return None
    delimiter = match.group(2)
    closing = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == delimiter),
        None,
    )
    if closing is None:
        return None
    head = lines[0][: match.start()].rstrip()
    tail = " ".join(line.strip() for line in lines[closing + 1 :] if line.strip())
    return f"{head}; {tail}" if tail else head


def _simple_command_risk(tokens: list[str], *, workspace: Path) -> RiskLevel:
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return "R2"
    executable = Path(tokens[0]).name

    if executable in {"python", "python3", "pytest", "unittest"}:
        return "R1" if not _contains_outside_path(tokens[1:], workspace) else "R2"
    if executable in _LOCAL_COMMANDS:
        if executable in {"npm", "pnpm", "yarn"} and len(tokens) > 1:
            if tokens[1] in {"install", "i", "ci", "add", "publish", "update", "upgrade"}:
                return "R2"
        if executable == "go" and len(tokens) > 1 and tokens[1] in {"get", "install"}:
            return "R2"
        if executable == "cargo" and len(tokens) > 1 and tokens[1] in {"install", "add"}:
            return "R2"
        return "R1" if not _contains_outside_path(tokens[1:], workspace) else "R2"
    if executable == "git":
        subcommand = tokens[1] if len(tokens) > 1 else ""
        if subcommand in {"status", "diff", "log", "show", "rev-parse"}:
            if any(token in {"--output", "-o"} for token in tokens[2:]):
                return "R2"
            return "R0" if not _contains_outside_path(tokens[2:], workspace) else "R2"
        return "R2"
    if executable in {"sed", "sort"}:
        if any(
            token == "-i"
            or token.startswith("-i")
            or token == "-o"
            or token.startswith("--output")
            for token in tokens[1:]
        ):
            return "R2"
    if executable == "find" and any(
        token in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
        for token in tokens[1:]
    ):
        return "R2"
    if executable in _READ_ONLY_COMMANDS:
        return "R0" if not _contains_outside_path(tokens[1:], workspace) else "R2"
    if executable in _NETWORK_OR_DESTRUCTIVE_COMMANDS:
        return "R2"
    return "R1" if not _contains_outside_path(tokens[1:], workspace) else "R2"


def command_risk(command: str, *, workspace: Path | None = None) -> RiskLevel:
    if command_denial_reason(command):
        return "R3"
    cleaned = re.sub(r"\d?>&\d", "", command)
    heredoc_head = _single_heredoc_head(cleaned)
    if heredoc_head is not None:
        cleaned = heredoc_head
    elif _has_unquoted_newline(cleaned):
        return "R2"
    if re.search(r"\$\(|\$\{|`", cleaned):
        return "R2"
    try:
        lexer = shlex.shlex(cleaned, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return "R2"
    if not tokens:
        return "R2"

    root = (workspace or Path.cwd()).expanduser().resolve()
    highest = "R0"
    segment: list[str] = []
    redirect_risk: RiskLevel | None = None

    def apply(risk: RiskLevel) -> None:
        nonlocal highest
        if _RISK_ORDER[risk] > _RISK_ORDER[highest]:
            highest = risk

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"&&", "||", ";", "|"}:
            if not segment:
                return "R2"
            if redirect_risk is not None:
                apply(redirect_risk)
            apply(_simple_command_risk(segment, workspace=root))
            segment = []
            redirect_risk = None
        elif token == "&":
            return "R2"
        elif token in {"<", ">", ">>", "<>", ">|", "&>", "&>>"}:
            if index + 1 >= len(tokens):
                return "R2"
            target = tokens[index + 1]
            if target != "/dev/null" and _contains_outside_path([target], root):
                return "R2"
            risk: RiskLevel = "R0" if token == "<" else "R1"
            redirect_risk = (
                risk
                if redirect_risk is None or _RISK_ORDER[risk] > _RISK_ORDER[redirect_risk]
                else redirect_risk
            )
            index += 1
        else:
            segment.append(token)
        index += 1

    if segment:
        if not redirect_risk:
            apply(_simple_command_risk(segment, workspace=root))
        else:
            apply(redirect_risk)
            apply(_simple_command_risk(segment, workspace=root))
    elif redirect_risk:
        apply(redirect_risk)
    else:
        return "R2"
    return highest


def command_denial_reason(command: str) -> str:
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return f"命令命中高危规则: {pattern.pattern}"
    for pattern in _SENSITIVE_COMMAND_PATTERNS:
        if pattern.search(command):
            return f"命令尝试访问受保护凭据: {pattern.pattern}"
    return ""


_APPROVAL_OPTIONS = (
    "允许本次（↑/↓ 选择 · Enter / y）",
    "本会话全部允许（↑/↓ 选择 · 2 / a）",
    "拒绝（↑/↓ 选择 · 3 / x / Esc / Tab）",
)
_APPROVAL_SELECTED = "\x1b[1;7;36m"
_APPROVAL_DIM = "\x1b[2m"
_APPROVAL_RESET = "\x1b[0m"


def _read_menu_key(descriptor: int) -> str:
    def read_character() -> str:
        return os.read(descriptor, 1).decode("utf-8", errors="ignore")

    character = read_character()
    if character == "":
        return "EOF"
    if character == "\t":
        return "TAB"
    if character in {"\r", "\n"}:
        return "ENTER"
    if character == "\x1b":
        if not select.select([descriptor], [], [], 0.03)[0]:
            return "ESC"
        second = read_character()
        if second != "[":
            return "ESC"
        if not select.select([descriptor], [], [], 0.03)[0]:
            return "ESC"
        third = read_character()
        return {"A": "UP", "B": "DOWN"}.get(third, "ESC")
    return character.lower()


def _read_choice_key() -> str:
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if line == "":
            return "EOF"
        value = line.strip().lower()
        return "ENTER" if not value else value
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor, when=termios.TCSADRAIN)
        return _read_menu_key(descriptor)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _approval_color_enabled() -> bool:
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _display_width(value: str) -> int:
    width = 0
    for character in value:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def _approval_option_line(
    index: int,
    label: str,
    *,
    selected: bool,
    color_enabled: bool,
) -> str:
    plain = f"  {'❯' if selected else ' '} {index + 1}. {label}"
    if not color_enabled:
        return plain
    if not selected:
        return f"{_APPROVAL_DIM}{plain}{_APPROVAL_RESET}"
    columns = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns)
    padding = " " * max(0, columns - 1 - _display_width(plain))
    return f"{_APPROVAL_SELECTED}{plain}{padding}{_APPROVAL_RESET}"


def _render_approval_options(
    selected: int,
    *,
    first: bool,
    color_enabled: bool | None = None,
) -> None:
    enabled = (
        _approval_color_enabled()
        if color_enabled is None
        else color_enabled
    )
    if not first:
        sys.stdout.write(f"\x1b[{len(_APPROVAL_OPTIONS)}A")
    for index, label in enumerate(_APPROVAL_OPTIONS):
        line = _approval_option_line(
            index,
            label,
            selected=index == selected,
            color_enabled=enabled,
        )
        sys.stdout.write(f"\r\x1b[2K{line}\n")
    sys.stdout.flush()


def _clear_approval_options() -> None:
    count = len(_APPROVAL_OPTIONS)
    sys.stdout.write(f"\x1b[{count}A")
    for index in range(count):
        sys.stdout.write("\r\x1b[2K")
        if index < count - 1:
            sys.stdout.write("\x1b[1B")
    if count > 1:
        sys.stdout.write(f"\x1b[{count - 1}A")
    sys.stdout.write("\r")
    sys.stdout.flush()


def _choose_approval_option(*, key_reader=None) -> int:
    selected = 0
    first = True
    rendered = False
    read_key = key_reader or _read_choice_key
    while True:
        if key_reader is None and not sys.stdin.isatty():
            value = input("选择 [1/2/3，Enter=允许本次]：").strip().lower()
            key = "ENTER" if not value else value
        else:
            _render_approval_options(selected, first=first)
            first = False
            rendered = True
            try:
                key = read_key()
            except (EOFError, KeyboardInterrupt):
                _clear_approval_options()
                raise
        if key == "UP":
            selected = (selected - 1) % len(_APPROVAL_OPTIONS)
        elif key == "DOWN":
            selected = (selected + 1) % len(_APPROVAL_OPTIONS)
        elif key == "ENTER":
            if rendered:
                _clear_approval_options()
            return selected
        elif key in {"1", "y"}:
            if rendered:
                _clear_approval_options()
            return 0
        elif key in {"2", "a"}:
            if rendered:
                _clear_approval_options()
            return 1
        elif key in {"3", "x", "n", "ESC", "EOF", "TAB"}:
            if rendered:
                _clear_approval_options()
            return 2


class ApprovalManager:
    def __init__(
        self,
        *,
        auto_approve: bool = False,
        persistent_permissions: dict[str, str] | None = None,
        interactive: bool | None = None,
        workspace: Path | None = None,
        default_grant: TaskGrant | None = None,
        prompt_event: threading.Event | None = None,
        before_prompt: Callable[[], None] | None = None,
        after_prompt: Callable[[], None] | None = None,
    ) -> None:
        self.auto_approve = auto_approve
        self.workspace = (workspace or Path.cwd()).expanduser().resolve()
        self.default_grant = default_grant or TaskGrant()
        self.grant = self.default_grant
        self.interactive = (
            bool(getattr(sys.stdin, "isatty", lambda: False)())
            if interactive is None
            else interactive
        )
        self.session_allow = auto_approve
        self.prompt_event = prompt_event
        self.before_prompt = before_prompt
        self.after_prompt = after_prompt
        self._interactive_decision = ""
        self.capability_overrides: dict[str, bool] = {}
        for capability, mode in (persistent_permissions or {}).items():
            if capability not in CAPABILITIES:
                continue
            normalized = str(mode).strip().lower()
            if normalized == "allow":
                self.capability_overrides[capability] = True
            elif normalized == "deny":
                self.capability_overrides[capability] = False
        if auto_approve:
            self.capability_overrides = {
                capability: True for capability in CAPABILITIES
            }

    def set_persistent_permission(
        self,
        capability: str,
        mode: str,
    ) -> None:
        if capability not in CAPABILITIES:
            raise ValueError(f"未知能力: {capability}")
        normalized = str(mode).strip().lower()
        if normalized == "allow":
            self.capability_overrides[capability] = True
        elif normalized == "deny":
            self.capability_overrides[capability] = False
        elif normalized == "auto":
            self.capability_overrides.pop(capability, None)
        else:
            raise ValueError("权限模式必须是 auto/allow/deny")

    def begin_task(self, risk_ceiling: RiskLevel | None = None) -> TaskGrant:
        self.grant = (
            TaskGrant(risk_ceiling=risk_ceiling)
            if risk_ceiling is not None
            else self.default_grant
        )
        return self.grant

    def end_task(self) -> None:
        self.grant = TaskGrant(risk_ceiling="R0")

    def describe(self) -> dict[str, object]:
        return {
            "task_grant": self.grant.risk_ceiling,
            "session_allow": self.session_allow,
            "interactive": self.interactive,
            "capabilities": self.permission_state(),
        }

    def permission_state(self) -> dict[str, str]:
        return {
            capability: (
                "allow"
                if self.capability_overrides.get(capability) is True
                else "deny"
                if self.capability_overrides.get(capability) is False
                else "auto"
            )
            for capability in sorted(CAPABILITIES)
        }

    def grant_capabilities(self, capabilities: Iterable[str]) -> set[str]:
        selected = normalize_capabilities(capabilities)
        for capability in selected:
            self.capability_overrides[capability] = True
        return selected

    def revoke_capabilities(self, capabilities: Iterable[str]) -> set[str]:
        selected = normalize_capabilities(capabilities)
        for capability in selected:
            self.capability_overrides[capability] = False
        return selected

    def reset_capabilities(self, capabilities: Iterable[str]) -> set[str]:
        selected = normalize_capabilities(capabilities)
        if not selected:
            selected = set(CAPABILITIES)
        for capability in selected:
            self.capability_overrides.pop(capability, None)
        return selected

    def apply_scope(self, scope: RequestScope) -> RequestScope:
        allows_mutation = scope.allows_mutation
        allows_commands = scope.allows_commands
        allows_network = scope.allows_network
        for capability, allowed in self.capability_overrides.items():
            if capability == "write":
                allows_mutation = allowed
            elif capability == "command":
                allows_commands = allowed
            elif capability == "network":
                allows_network = allowed
        return replace(
            scope,
            allows_mutation=allows_mutation,
            allows_commands=allows_commands,
            allows_network=allows_network,
        )

    def require(
        self,
        action: str,
        detail: str,
        *,
        risk: RiskLevel = "R1",
        capability: str | None = None,
    ) -> bool:
        if capability:
            override = self.capability_overrides.get(capability)
            if override is False:
                return False
            if override is True:
                return True
        if self.session_allow or self.grant.allows(risk):
            return True
        if not self.interactive:
            print(
                f"Sunus 已拒绝 {action}：超出当前任务授权（{risk}）；"
                "请在交互终端确认，或由本次任务显式授权。",
                file=sys.stderr,
            )
            return False
        self._interactive_decision = ""
        try:
            if self.prompt_event is not None:
                self.prompt_event.set()
            if self.before_prompt is not None:
                self.before_prompt()
            print(f"\nSunus 请求执行：{action}（风险 {risk}）")
            print(detail.strip()[:3000])
            try:
                choice = _choose_approval_option()
            except (EOFError, KeyboardInterrupt):
                print()
                self._interactive_decision = "cancelled"
                return False
        finally:
            if self.after_prompt is not None:
                self.after_prompt()
            if self.prompt_event is not None:
                self.prompt_event.clear()
        if choice == 1:
            self.session_allow = True
            self._interactive_decision = "session_approved"
            return True
        self._interactive_decision = (
            "approved" if choice == 0 else "denied"
        )
        return choice == 0

    def consume_interactive_decision(self) -> str:
        decision = self._interactive_decision
        self._interactive_decision = ""
        return decision

    def require_command(self, command: str) -> bool:
        denial = command_denial_reason(command)
        if denial:
            print(f"Sunus 已阻止命令：{denial}", file=sys.stderr)
            return False
        risk = command_risk(command, workspace=self.workspace)
        try:
            quoted = shlex.join([command])
        except Exception:
            quoted = command
        return self.require(
            "运行命令",
            quoted,
            risk=risk,
            capability="command",
        )


def summarize_paths(paths: Iterable[Path], limit: int = 12) -> str:
    values = [str(path) for path in paths]
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit] + [f"... 共 {len(values)} 项"])

