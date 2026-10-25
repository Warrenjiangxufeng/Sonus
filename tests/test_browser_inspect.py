from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sunus.browser_inspect import inspect_browser_page


class BrowserInspectTests(unittest.TestCase):
    def test_reads_active_tab_title_and_url(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="DeepSeek 开放平台\nhttps://platform.deepseek.com/usage\n",
            stderr="",
        )
        with patch("sunus.browser_inspect.subprocess.run", return_value=completed):
            result = inspect_browser_page(browser="chrome", mode="active")

        self.assertTrue(result["ok"])
        self.assertEqual(result["title"], "DeepSeek 开放平台")
        self.assertEqual(result["url"], "https://platform.deepseek.com/usage")

    def test_javascript_disabled_returns_single_recovery_action(self) -> None:
        completed = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "通过 AppleScript 执行 JavaScript 的功能已关闭。"
                "请开启允许 Apple 事件中的 JavaScript"
            ),
        )
        with patch("sunus.browser_inspect.subprocess.run", return_value=completed):
            result = inspect_browser_page(browser="chrome", mode="content")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "browser_access_failed")
        self.assertIn("允许 Apple 事件中的 JavaScript", result["next_action"])

    def test_rejects_unsupported_browser(self) -> None:
        result = inspect_browser_page(browser="firefox", mode="active")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "unsupported_browser")


if __name__ == "__main__":
    unittest.main()
