from __future__ import annotations

import subprocess
from typing import Any


_SCRIPTS = {
    "chrome": {
        "active": """
tell application "Google Chrome"
    if not running then return "BROWSER_NOT_RUNNING"
    tell active tab of front window
        return ((title as text) & linefeed & (URL as text))
    end tell
end tell
""".strip(),
        "content": """
tell application "Google Chrome"
    if not running then return "BROWSER_NOT_RUNNING"
    tell active tab of front window
        return execute javascript "document.body.innerText"
    end tell
end tell
""".strip(),
    },
    "safari": {
        "active": """
tell application "Safari"
    if not (exists front document) then return "BROWSER_NOT_RUNNING"
    tell front document
        return ((name as text) & linefeed & (URL as text))
    end tell
end tell
""".strip(),
        "content": """
tell application "Safari"
    if not (exists front document) then return "BROWSER_NOT_RUNNING"
    tell front document
        return do JavaScript "document.body.innerText"
    end tell
end tell
""".strip(),
    },
}


def inspect_browser_page(
    *,
    browser: str = "chrome",
    mode: str = "active",
    timeout_sec: int = 15,
) -> dict[str, Any]:
    provider = str(browser or "chrome").lower()
    if provider not in _SCRIPTS:
        return {
            "ok": False,
            "error": "unsupported_browser",
            "next_action": "目前支持 chrome 和 safari。",
        }
    if mode not in {"active", "content"}:
        return {
            "ok": False,
            "error": "unsupported_mode",
            "next_action": "mode 使用 active 或 content。",
        }
    try:
        completed = subprocess.run(
            ["/usr/bin/osascript", "-e", _SCRIPTS[provider][mode]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(3, min(30, timeout_sec)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "ok": False,
            "browser": provider,
            "mode": mode,
            "error": type(error).__name__,
            "detail": str(error),
            "next_action": "检查浏览器自动化权限后重试。",
        }
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0 or not stdout:
        combined = " ".join([stderr, stdout]).strip()
        lowered = combined.lower()
        next_action = "检查 macOS 自动化权限后重试。"
        if "javascript" in lowered and (
            "关闭" in combined
            or "disabled" in lowered
            or "not allowed" in lowered
        ):
            next_action = (
                "Chrome 需手动开启“查看 > 开发者 > 允许 Apple 事件中的 JavaScript”；"
                "Safari 需在开发菜单允许 Apple Events 的 JavaScript。"
            )
        return {
            "ok": False,
            "browser": provider,
            "mode": mode,
            "error": "browser_access_failed",
            "detail": combined[:2000],
            "next_action": next_action,
        }
    if stdout == "BROWSER_NOT_RUNNING":
        return {
            "ok": False,
            "browser": provider,
            "mode": mode,
            "error": "browser_not_running",
            "next_action": "打开目标浏览器页面后重试。",
        }
    if mode == "active":
        first, _, second = stdout.partition("\n")
        return {
            "ok": True,
            "browser": provider,
            "mode": mode,
            "title": first.strip(),
            "url": second.strip(),
        }
    return {
        "ok": True,
        "browser": provider,
        "mode": mode,
        "text": stdout[:15000],
    }
