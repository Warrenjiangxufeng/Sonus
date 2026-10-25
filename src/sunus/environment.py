from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


_VERSION = re.compile(r"\d+\.\d+(?:\.\d+)?")
_PYTHON_NAME = re.compile(r"^(?:python(?:3(?:\.\d+)*)?|pypy3?)$")
_TEST_RUNNER_NAME = re.compile(r"^(?:pytest(?:-?\d+(?:\.\d+)*)?|py\.test)$")
_RUNTIME_NAMES = {
    "node": re.compile(r"^(?:node|nodejs)$"),
    "node_package_manager": re.compile(r"^(?:npm|pnpm|yarn|bun|deno)$"),
    "vcs": re.compile(r"^(?:git|hg|svn)$"),
    "search": re.compile(r"^(?:rg|fd|fdfind|grep)$"),
    "database": re.compile(r"^(?:sqlite3|psql|mysql)$"),
    "http": re.compile(r"^(?:curl|wget)$"),
}
_PROBE_PYTHON = """
import importlib.util
import json
import sys

print(json.dumps({
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "has_pytest": importlib.util.find_spec("pytest") is not None,
}))
"""
_CACHE_TTL_SEC = 30.0
_CACHE_AT = 0.0
_CACHE_REPORT: dict[str, Any] | None = None
_CACHE_INVENTORY: dict[str, list[str]] | None = None


def _run(command: list[str], *, timeout: float = 4.0) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or completed.stderr or "").strip()


def login_shell() -> str:
    candidate = os.environ.get("SHELL", "")
    if not candidate:
        try:
            candidate = pwd.getpwuid(os.getuid()).pw_shell
        except KeyError:
            candidate = ""
    name = Path(candidate).name if candidate else ""
    return shutil.which(name) or shutil.which("sh") or ""


def _shell_path_roots() -> tuple[list[str], list[str]]:
    process_roots = [
        os.path.abspath(os.path.expanduser(item))
        for item in os.environ.get("PATH", "").split(os.pathsep)
        if item
    ]
    login_roots: list[str] = []
    shell = login_shell()
    if shell:
        output = _run([shell, "-lc", 'printf "%s" "$PATH"'], timeout=3.0)
        login_roots = [
            os.path.abspath(os.path.expanduser(item))
            for item in output.split(os.pathsep)
            if item
        ]
    roots: list[str] = []
    seen_roots: set[str] = set()
    for item in [*login_roots, *process_roots]:
        resolved = os.path.realpath(item)
        if resolved not in seen_roots and os.path.isdir(resolved):
            seen_roots.add(resolved)
            roots.append(resolved)
    return roots, login_roots


def scan_path_executables() -> dict[str, Any]:
    roots, login_roots = _shell_path_roots()
    inventory: dict[str, list[str]] = {}
    errors: list[str] = []
    for root in roots:
        try:
            entries = os.scandir(root)
        except OSError as error:
            errors.append(f"{root}: {error}")
            continue
        with entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        if not entry.is_symlink():
                            continue
                    path = entry.path
                    if not os.access(path, os.X_OK):
                        continue
                except OSError:
                    continue
                inventory.setdefault(entry.name, []).append(path)
    return {
        "path_roots": roots,
        "login_shell_path_used": bool(login_roots),
        "executable_count": sum(len(paths) for paths in inventory.values()),
        "command_count": len(inventory),
        "inventory": inventory,
        "errors": errors,
    }


def find_executables(
    names: Iterable[str],
    *,
    inventory: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    paths_by_name = inventory or scan_path_executables()["inventory"]
    found: list[dict[str, str]] = []
    for name in dict.fromkeys(str(item) for item in names if item):
        for path in paths_by_name.get(name, []):
            found.append({"name": name, "command": path})
    return found


def _probe_python(executable: str) -> dict[str, Any]:
    output = _run([executable, "-c", _PROBE_PYTHON])
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    version = str(payload.get("version") or "")
    if not version:
        matches = _VERSION.findall(_run([executable, "--version"]))
        version = matches[0] if matches else ""
    return {
        "command": executable,
        "executable": str(payload.get("executable") or executable),
        "version": version,
        "has_pytest": bool(payload.get("has_pytest")),
    }


def _version(executable: str) -> str:
    for arguments in (("--version",), ("-version",), ("-V",)):
        output = _run([executable, *arguments], timeout=3.0)
        matches = _VERSION.findall(output)
        if matches:
            return matches[0]
    return ""


def _shebang_interpreter(executable: Path) -> str:
    try:
        first_line = executable.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[0]
    except (OSError, IndexError):
        return ""
    if not first_line.startswith("#!"):
        return ""
    tokens = first_line[2:].strip().split()
    if not tokens:
        return ""
    if Path(tokens[0]).name == "env" and len(tokens) > 1:
        return shutil.which(tokens[1]) or ""
    return tokens[0]


def _version_at_least(value: str, minimum: tuple[int, int]) -> bool:
    match = _VERSION.search(value)
    if not match:
        return False
    parts = tuple(int(part) for part in match.group(0).split(".")[:2])
    return parts >= minimum


def _matching(
    names: Iterable[str],
    pattern: re.Pattern[str],
) -> list[str]:
    return sorted(name for name in names if pattern.fullmatch(name))


def discover_environment(
    requested_commands: Iterable[str] = (),
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    global _CACHE_AT, _CACHE_INVENTORY, _CACHE_REPORT
    requested = list(dict.fromkeys(str(item) for item in requested_commands if item))
    if (
        not refresh
        and _CACHE_REPORT is not None
        and _CACHE_INVENTORY is not None
        and time.monotonic() - _CACHE_AT < _CACHE_TTL_SEC
    ):
        report = dict(_CACHE_REPORT)
        report["requested_commands"] = [
            {**item, "version": _version(item["command"])}
            for item in find_executables(
                requested,
                inventory=_CACHE_INVENTORY,
            )
        ]
        return report

    path_report = scan_path_executables()
    inventory = path_report["inventory"]
    all_names = set(inventory)

    python_names = _matching(all_names, _PYTHON_NAME)
    python_candidates = [
        _probe_python(item["command"])
        for name in python_names
        for item in find_executables((name,), inventory=inventory)
    ]

    runner_names = _matching(all_names, _TEST_RUNNER_NAME)
    pytest_entries: list[dict[str, Any]] = []
    for entry in find_executables(runner_names, inventory=inventory):
        interpreter = _shebang_interpreter(Path(entry["command"]))
        interpreter_version = (
            _probe_python(interpreter).get("version", "")
            if interpreter
            else ""
        )
        pytest_entries.append(
            {
                **entry,
                "version": _version(entry["command"]),
                "interpreter": interpreter,
                "interpreter_version": interpreter_version,
                "project_compatible": _version_at_least(
                    interpreter_version,
                    (3, 10),
                ),
            }
        )
    pytest_entries.sort(
        key=lambda item: (item["name"] != "pytest", item["command"])
    )

    module_candidates = [
        item
        for item in python_candidates
        if item["has_pytest"]
        and _version_at_least(str(item["version"]), (3, 10))
    ]
    compatible_runner = next(
        (
            item
            for item in pytest_entries
            if item["project_compatible"]
        ),
        None,
    )
    if module_candidates:
        recommended = f"{module_candidates[0]['command']} -m pytest"
    elif compatible_runner:
        recommended = compatible_runner["command"]
    elif pytest_entries:
        recommended = pytest_entries[0]["command"]
    else:
        recommended = ""

    categories: dict[str, list[dict[str, str]]] = {
        "test_runner": pytest_entries,
        "python": python_candidates,
    }
    for category, pattern in _RUNTIME_NAMES.items():
        categories[category] = find_executables(
            _matching(all_names, pattern),
            inventory=inventory,
        )
    shell = login_shell()
    categories["shell"] = (
        [{"name": Path(shell).name, "command": shell}] if shell else []
    )

    requested_report: list[dict[str, Any]] = []
    for item in find_executables(requested, inventory=inventory):
        requested_report.append({**item, "version": _version(item["command"])})

    public_path = {
        key: value
        for key, value in path_report.items()
        if key != "inventory"
    }
    report = {
        "path": public_path,
        "categories": categories,
        "pytest_candidates": pytest_entries,
        "recommended_test_command": recommended,
        "requested_commands": requested_report,
    }
    _CACHE_AT = time.monotonic()
    _CACHE_INVENTORY = inventory
    _CACHE_REPORT = {
        **report,
        "requested_commands": [],
    }
    return report


def shell_invocation(shell: str, command: str) -> list[str]:
    if not shell:
        raise FileNotFoundError("未找到可用 shell")
    if Path(shell).name.lower() in {"pwsh", "powershell"}:
        return [shell, "-NoProfile", "-Command", command]
    return [shell, "-lc", command]


def environment_prompt() -> str:
    report = discover_environment()
    path = report["path"]
    pytest_entries = report["pytest_candidates"]
    python_entries = report["categories"]["python"]
    lines = [
        "本地执行环境（每次读取时实时扫描登录 shell 与进程 PATH）：",
        f"- PATH 目录: {len(path['path_roots'])}；"
        f"可执行命令: {path['command_count']} 个",
        f"- 登录 shell: {login_shell() or 'not found'}",
    ]
    if python_entries:
        lines.append(
            "- Python 候选: "
            + ", ".join(
                f"{item['command']} ({item['version'] or 'unknown'}"
                + ("+pytest" if item["has_pytest"] else "")
                + ")"
                for item in python_entries
            )
        )
    if pytest_entries:
        lines.append(
            "- pytest 候选: "
            + ", ".join(
                f"{item['command']} ({item['version'] or 'unknown'}; "
                f"{item['interpreter'] or 'unknown interpreter'}"
                + (
                    "; project-compatible"
                    if item["project_compatible"]
                    else "; interpreter differs from project requirement"
                )
                + ")"
                for item in pytest_entries
            )
        )
    else:
        lines.append(
            "- pytest 候选: PATH 中未发现；不得仅凭某个 Python "
            "的 import 失败推断 pytest 未安装。"
        )
    if report["recommended_test_command"]:
        lines.append(
            f"- 当前推荐测试入口: {report['recommended_test_command']}"
        )
    lines.extend(
        (
            "- 需要检查本提示未列出的其他命令时，调用 inspect_environment，"
            "传入实际命令名；它会重新扫描 PATH 并返回所有命中位置与版本。",
            "- 工具是否安装以实际 PATH/可执行文件为准，不读取本代码中的固定路径。",
        )
    )
    return "\n".join(lines)
