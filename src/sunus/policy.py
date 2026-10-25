from __future__ import annotations

import os
import re
import select
import shlex
import sys
import termios
import threading
import tty
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


class PathGuard:
    def __init__(self, workspace: Path, allow_outside: bool = False) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.allow_outside = allow_outside
        self.workspace.mkdir(parents=True, exist_ok=True)

    def resolve(self, value: str | Path, *, must_exist: bool = False) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve(strict=False)
        if must_exist and not resolved.exists():
            raise PolicyError(f"路径不存在: {resolved}")
        if not self.allow_outside and not (
            resolved == self.workspace or self.workspace in resolved.parents
        ):
            raise PolicyError(
                f"路径越界: {resolved}；工作区为 {self.workspace}。"
                "如确需访问外部路径，请使用 --allow-outside。"
            )
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.workspace))
        except ValueError:
            return str(path)

    def assert_writable(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        protected_parts = {".git", ".sunus", "__pycache__"}
        try:
            relative = resolved.relative_to(self.workspace)
        except ValueError:
            relative = None
        if relative and any(part in protected_parts for part in relative.parts):
            raise PolicyError(f"受保护路径不可修改: {resolved}")


_DANGEROUS_PATTERNS = (
    re.compile(r"(?i)\bsudo\b"),
    re.compile(r"(?i)\b(shutdown|reboot|halt)\b"),
    re.compile(r"(?i)\bdiskutil\s+(erase|reformat)"),
    re.compile(r"(?i)\bmkfs(\.|\b)"),
    re.compile(r"(?i)\bdd\b[^\n]*\bof\s*=\s*/dev/"),
    re.compile(r"(?i)\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)\s+(?:/|~|\\$HOME)(?:\s|$)"),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"),
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
    "允许本次",
    "本会话全部允许",
    "拒绝",
)


def _read_menu_key(descriptor: int) -> str:
    def read_character() -> str:
        return os.read(descriptor, 1).decode("utf-8", errors="ignore")

    character = read_character()
    if character == "":
        return "EOF"
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


def _render_approval_options(selected: int, *, first: bool) -> None:
    if not first:
        sys.stdout.write(f"\x1b[{len(_APPROVAL_OPTIONS) + 1}A")
    for index, label in enumerate(_APPROVAL_OPTIONS):
        marker = "❯" if index == selected else " "
        sys.stdout.write(f"\r\x1b[2K  {marker} {index + 1}. {label}\n")
    sys.stdout.write("\r\x1b[2K  ↑/↓ 选择 · Enter 确认 · y 允许 · x 拒绝\n")
    sys.stdout.flush()


def _choose_approval_option(*, key_reader=None) -> int:
    selected = 0
    first = True
    read_key = key_reader or _read_choice_key
    while True:
        if key_reader is None and not sys.stdin.isatty():
            value = input("选择 [1/2/3，Enter=允许本次]：").strip().lower()
            key = "ENTER" if not value else value
        else:
            _render_approval_options(selected, first=first)
            first = False
            key = read_key()
        if key == "UP":
            selected = (selected - 1) % len(_APPROVAL_OPTIONS)
        elif key == "DOWN":
            selected = (selected + 1) % len(_APPROVAL_OPTIONS)
        elif key == "ENTER":
            return selected
        elif key in {"1", "y"}:
            return 0
        elif key in {"2", "a"}:
            return 1
        elif key in {"3", "x", "n", "ESC", "EOF"}:
            return 2


class ApprovalManager:
    def __init__(
        self,
        *,
        auto_approve: bool = False,
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
        self.capability_overrides: dict[str, bool] = {}
        if auto_approve:
            self.capability_overrides = {
                capability: True for capability in CAPABILITIES
            }

    def begin_task(self) -> TaskGrant:
        self.grant = self.default_grant
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
                return False
        finally:
            if self.after_prompt is not None:
                self.after_prompt()
            if self.prompt_event is not None:
                self.prompt_event.clear()
        if choice == 1:
            self.session_allow = True
            return True
        return choice == 0

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

