from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.policy import ApprovalManager, PathGuard
from sunus.storage import ArtifactStore
from sunus.tools import ToolExecutor, find_ripgrep, tool_schemas


class ToolTests(unittest.TestCase):
    def test_tool_schemas_include_structured_output_and_documents(self) -> None:
        names = {item["function"]["name"] for item in tool_schemas()}
        self.assertIn("submit_result", names)
        self.assertIn("write_document", names)
        self.assertIn("memory_context_metrics", names)
        self.assertIn("remember", names)
        self.assertIn("inspect_environment", names)
        self.assertIn("inspect_browser", names)
        self.assertIn("inspect_runtime_history", names)
        self.assertIn("request_authorization", names)

    def make_executor(
        self,
        root: Path,
        *,
        memory: MemoryStore | None = None,
        interactive: bool = False,
    ) -> ToolExecutor:
        return ToolExecutor(
            workspace=root,
            path_guard=PathGuard(root),
            approvals=ApprovalManager(
                auto_approve=False,
                interactive=interactive,
                workspace=root,
            ),
            artifacts=ArtifactStore(root, root / ".data"),
            events=EventBus(root / ".data"),
            data_dir=root / ".data",
            memory=memory,
        )

    def test_read_write_patch_search_and_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            write = executor.execute("write_file", {"path": "a.txt", "content": "hello\n"})
            self.assertTrue(write.ok)
            patch = executor.execute(
                "patch_file",
                {"path": "a.txt", "old_text": "hello", "new_text": "hello sunus"},
            )
            self.assertTrue(patch.ok)
            read = executor.execute("read_file", {"path": "a.txt"})
            self.assertIn("hello sunus", read.content)
            search = executor.execute("search_text", {"query": "sunus"})
            self.assertIn("a.txt", search.content)
            listing = executor.execute("list_files", {"depth": 2})
            self.assertIn("a.txt", listing.content)

    def test_search_can_aggregate_hits_by_file_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            executor.execute(
                "write_file",
                {"path": "a.txt", "content": "needle\nneedle\n"},
            )
            executor.execute(
                "write_file",
                {"path": "b.txt", "content": "needle\n"},
            )

            result = executor.execute(
                "search_text",
                {"query": "needle", "output": "files"},
            )

            self.assertTrue(result.ok)
            self.assertIn("a.txt\t2 命中", result.content)
            self.assertIn("b.txt\t1 命中", result.content)
            self.assertNotIn("needle\nneedle", result.content)

    def test_list_files_can_filter_and_sort_by_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            old = root / "old.txt"
            new = root / "new.txt"
            old.write_text("old", encoding="utf-8")
            new.write_text("new", encoding="utf-8")
            now = time.time()
            os.utime(old, (now - 10 * 86400, now - 10 * 86400))
            os.utime(new, (now - 60, now - 60))

            result = executor.execute(
                "list_files",
                {
                    "depth": 1,
                    "sort_by_mtime": True,
                    "modified_within_days": 2,
                },
            )

            self.assertTrue(result.ok)
            self.assertIn("new.txt", result.content)
            self.assertNotIn("old.txt", result.content)

    def test_command_output_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp))
            result = executor.execute("run_command", {"command": "printf 'ok\\n'"})
            self.assertTrue(result.ok)
            self.assertIn("exit_code=0", result.content)
            self.assertIn("ok", result.content)

    def test_write_file_creates_missing_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)

            result = executor.execute(
                "write_file",
                {
                    "path": "Desktop/not-yet-created/file.txt",
                    "content": "hello",
                },
            )

            self.assertTrue(result.ok)
            self.assertEqual(
                (root / "Desktop" / "not-yet-created" / "file.txt").read_text(
                    encoding="utf-8"
                ),
                "hello",
            )

    def test_authorization_request_can_expand_current_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root, interactive=True)
            executor.begin_request("解释这个文件")

            with (
                patch("sunus.policy._choose_approval_option", return_value=0),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                result = executor.execute(
                    "request_authorization",
                    {
                        "capability": "mutation_and_commands",
                        "reason": "需要修改代码并运行测试",
                    },
                )

            self.assertTrue(result.ok)
            self.assertIsNotNone(executor.request_scope)
            self.assertTrue(executor.request_scope.allows_mutation)
            self.assertTrue(executor.request_scope.allows_commands)

    def test_remember_writes_supported_persistent_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = MemoryStore(root / ".data" / "memory.db")
            executor = self.make_executor(root, memory=memory)

            result = executor.execute(
                "remember",
                {
                    "content": "复盘：授权能力名必须统一。",
                    "kind": "review",
                    "persistent": True,
                },
            )

            self.assertTrue(result.ok)
            rows = memory.recent_persistent_reviews(
                scope=str(root.resolve()),
                limit=3,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["kind"], "review")
            self.assertEqual(rows[0]["status"], "supported")
            self.assertIn("persistent", rows[0]["tags"].split(","))
            memory.close()

    def test_command_authorization_is_not_confused_with_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root, interactive=True)
            executor.begin_request("访问网页")
            self.assertTrue(executor.request_scope.allows_network)
            self.assertFalse(executor.request_scope.allows_commands)

            with (
                patch("sunus.policy._choose_approval_option", return_value=0),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                result = executor.execute(
                    "request_authorization",
                    {
                        "capability": "commands",
                        "reason": "需要运行浏览器命令",
                    },
                )

            self.assertTrue(result.ok)
            self.assertTrue(executor.request_scope.allows_commands)
            self.assertTrue(executor.request_scope.allows_network)

    def test_command_timeout_terminates_the_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp))
            started = time.monotonic()
            result = executor.execute(
                "run_command",
                {"command": "sh -c 'sleep 5'", "timeout_sec": 1},
            )
            elapsed = time.monotonic() - started

            self.assertFalse(result.ok)
            self.assertIn("命令超时（1s）", result.content)
            self.assertLess(elapsed, 3)

    def test_environment_tool_queries_path_without_inventory_dump(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp))
            result = executor.execute(
                "inspect_environment",
                {"commands": ["git"], "refresh": True},
            )
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertNotIn("inventory", payload["path"])
            self.assertIn("requested_commands", payload)

    def test_runtime_history_tool_returns_aggregate_not_raw_log_dump(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events_path = root / ".data" / "events" / "2026-10-25.jsonl"
            events_path.parent.mkdir(parents=True)
            events_path.write_text(
                json.dumps(
                    {
                        "type": "internal_error",
                        "ts": "2026-10-25T10:00:00+00:00",
                        "turn_id": "t_1",
                        "summary": "TypeError: first",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            executor = self.make_executor(root)

            result = executor.execute(
                "inspect_runtime_history",
                {"mode": "trend", "days": 365},
            )

            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["total_failures"], 1)
            self.assertNotIn("rows", payload)

    def test_browser_tool_returns_structured_active_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root, interactive=True)
            executor.approvals.grant_capabilities(["command"])
            with patch(
                "sunus.tools.inspect_browser_page",
                return_value={
                    "ok": True,
                    "browser": "chrome",
                    "mode": "active",
                    "title": "DeepSeek 开放平台",
                    "url": "https://platform.deepseek.com/usage",
                },
            ):
                result = executor.execute(
                    "inspect_browser",
                    {"browser": "chrome", "mode": "active"},
                )

            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["title"], "DeepSeek 开放平台")
            self.assertEqual(
                payload["url"],
                "https://platform.deepseek.com/usage",
            )

    def test_pipeline_masked_python_error_is_marked_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp))
            result = executor.execute(
                "run_command",
                {
                    "command": (
                        "python3 -c 'import sunus_missing_tokenizer' "
                        "2>&1 | head -2"
                    )
                },
            )
            self.assertFalse(result.ok)
            self.assertIn("exit_code=0", result.content)
            self.assertIn("semantic_check", result.content)
            self.assertIn("Python traceback", result.content)

    def test_memory_context_metrics_is_read_only_and_allowed_for_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = MemoryStore(root / ".data" / "memory.db")
            memory.add(
                content="用户偏好：先给结论",
                scope="global",
                kind="preference",
                status="supported",
            )
            executor = self.make_executor(root, memory=memory)
            executor.begin_request("每次送入上下文的 token 有多少？")

            result = executor.execute("memory_context_metrics", {})

            self.assertTrue(result.ok)
            self.assertIn("偏好", result.content)
            self.assertIn("exact=false", result.content)
            memory.close()

    def test_outside_path_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp) / "workspace")
            result = executor.execute("read_file", {"path": "/etc/hosts"})
            self.assertFalse(result.ok)
            self.assertIn("路径越界", result.content)

    def test_bounded_fallback_ignores_large_and_sensitive_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            (root / "src").mkdir()
            (root / "src" / "visible.txt").write_text("needle\n", encoding="utf-8")
            for directory in (".git", ".sunus", "node_modules", ".ssh", "Library"):
                path = root / directory
                path.mkdir(exist_ok=True)
                (path / "hidden.txt").write_text("needle\n", encoding="utf-8")
            (root / ".env").write_text("needle=secret\n", encoding="utf-8")

            with patch("sunus.tools.find_ripgrep", return_value=None):
                result = executor.execute(
                    "search_text",
                    {"query": "needle", "timeout_sec": 2},
                )

            self.assertTrue(result.ok)
            self.assertIn("src/visible.txt", result.content)
            self.assertNotIn(".git", result.content)
            self.assertNotIn(".sunus", result.content)
            self.assertNotIn("node_modules", result.content)
            self.assertNotIn(".ssh", result.content)
            self.assertNotIn(".env", result.content)
            self.assertIn("bounded-python-fallback", result.content)

    def test_large_search_result_is_capped_and_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            (root / "large.txt").write_text(
                ("needle " + ("x" * 450) + "\n") * 50,
                encoding="utf-8",
            )

            with patch("sunus.tools.find_ripgrep", return_value=None):
                result = executor.execute(
                    "search_text",
                    {"query": "needle", "timeout_sec": 2},
                )

            self.assertTrue(result.ok)
            self.assertLess(len(result.content), 17_000)
            self.assertIn("context cap", result.content)
            self.assertIsNotNone(result.artifact)
            self.assertTrue(Path(str(result.artifact)).is_file())

    def test_fallback_timeout_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            (root / "a.txt").write_text("needle\n", encoding="utf-8")

            with (
                patch("sunus.tools.find_ripgrep", return_value=None),
                patch("sunus.tools.time.monotonic", side_effect=[0.0, 2.0]),
            ):
                result = executor.execute(
                    "search_text",
                    {"query": "needle", "timeout_sec": 1},
                )

            self.assertFalse(result.ok)
            self.assertIn("超过 1 秒", result.content)
            self.assertIn("bounded-python-fallback", result.content)

    def test_find_ripgrep_honors_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "rg"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o700)
            with patch.dict(os.environ, {"SUNUS_RG_PATH": str(executable)}):
                self.assertEqual(find_ripgrep(), str(executable.resolve()))

    def test_write_document_uses_profile_and_rejects_missing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            valid = executor.execute(
                "write_document",
                {
                    "path": "docs/DESIGN.md",
                    "profile": "design",
                    "title": "设计",
                    "summary": "摘要",
                    "sections": [
                        {"heading": "目标", "body": "目标"},
                        {"heading": "架构", "body": "架构"},
                        {"heading": "数据流", "body": "数据流"},
                        {"heading": "安全边界", "body": "边界"},
                        {"heading": "验证", "body": "验证"},
                    ],
                    "sources": ["src/a.py"],
                },
            )
            self.assertTrue(valid.ok)
            content = (root / "docs" / "DESIGN.md").read_text(encoding="utf-8")
            self.assertIn("sunus_document: design", content)
            self.assertIn("## 架构", content)

            invalid = executor.execute(
                "write_document",
                {
                    "path": "docs/INVALID.md",
                    "profile": "report",
                    "title": "报告",
                    "sections": [{"heading": "结论", "body": "结论"}],
                },
            )
            self.assertFalse(invalid.ok)
            self.assertIn("缺少章节", invalid.content)
            self.assertFalse((root / "docs" / "INVALID.md").exists())

    def test_unrequested_side_effects_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "note.txt").write_text("safe", encoding="utf-8")
            executor = self.make_executor(root)
            executor.begin_request("这个文件为什么有问题？只分析。")
            read = executor.execute("read_file", {"path": "note.txt"})
            write = executor.execute(
                "write_file",
                {"path": "new.txt", "content": "unrequested"},
            )
            command = executor.execute("run_command", {"command": "ls"})
            fetch = executor.execute(
                "fetch_url",
                {"url": "https://example.com"},
            )
            self.assertTrue(read.ok)
            for result in (write, command, fetch):
                self.assertFalse(result.ok)
                self.assertIn("没有明确要求", result.content)
            self.assertFalse((root / "new.txt").exists())

    def test_explicit_capability_revoke_overrides_user_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            executor.approvals.revoke_capabilities(["write"])
            scope = executor.begin_request("修改文件并运行测试")

            self.assertFalse(scope.allows_mutation)
            self.assertTrue(scope.allows_commands)
            result = executor.execute(
                "write_file",
                {"path": "blocked.txt", "content": "blocked"},
            )
            self.assertFalse(result.ok)
            self.assertIn("没有明确要求", result.content)
            self.assertFalse((root / "blocked.txt").exists())


if __name__ == "__main__":
    unittest.main()

