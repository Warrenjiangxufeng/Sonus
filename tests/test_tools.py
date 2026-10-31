from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sunus.config import ProactiveConfig
from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.policy import ApprovalManager, PathGuard
from sunus.proactive import ProactiveScheduler
from sunus.storage import ArtifactStore
from sunus.task_state import TaskState
from sunus.tools import (
    ToolExecutor,
    ToolResult,
    _command_changes_target_state,
    _is_broad_search_root,
    find_ripgrep,
    tool_schemas,
)


class ToolTests(unittest.TestCase):
    def test_tool_schemas_include_structured_output_and_documents(self) -> None:
        names = {item["function"]["name"] for item in tool_schemas()}
        self.assertIn("submit_result", names)
        self.assertIn("write_document", names)
        self.assertIn("write_file_batch", names)
        self.assertIn("patch_file_batch", names)
        self.assertIn("apply_and_verify", names)
        self.assertIn("verification_batch", names)
        self.assertIn("fetch_url_batch", names)
        self.assertIn("memory_context_metrics", names)
        self.assertIn("search_memory", names)
        self.assertIn("inspect_code_state", names)
        self.assertIn("remember", names)
        self.assertIn("inspect_environment", names)
        self.assertIn("inspect_browser", names)
        self.assertIn("inspect_runtime_history", names)
        self.assertIn("request_authorization", names)
        self.assertIn("update_reasoning_state", names)
        self.assertIn("run_command_batch", names)

    def test_reasoning_state_tool_updates_only_task_cognition(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            state = TaskState(objective="定位并验证目标")
            executor.task_state = state

            result = executor.execute(
                "update_reasoning_state",
                {
                    "current_subgoal": "区分两个候选目标",
                    "working_hypothesis": "当前证据层无法访问目标内部",
                    "evidence_gap": "缺少跨证据层的可操作性检查",
                    "success_criteria": ["目标状态发生预期变化"],
                    "decision": "停止更换同类查询",
                },
            )

            self.assertTrue(result.ok)
            self.assertTrue(result.state_changed)
            self.assertEqual(state.current_subgoal, "区分两个候选目标")
            self.assertEqual(
                state.working_hypothesis,
                "当前证据层无法访问目标内部",
            )
            self.assertEqual(
                state.success_criteria,
                ["目标状态发生预期变化"],
            )
            self.assertEqual(state.stalled_steps, 0)
            self.assertIn("cognition: 停止更换同类查询", state.decision_log[-1])

    def test_broad_home_roots_are_rejected_before_recursive_search(self) -> None:
        self.assertTrue(_is_broad_search_root(Path.home()))
        self.assertTrue(_is_broad_search_root(Path.home() / "Documents"))
        self.assertTrue(_is_broad_search_root(Path.home() / "Desktop"))
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(_is_broad_search_root(Path(temp)))

    def make_executor(
        self,
        root: Path,
        *,
        memory: MemoryStore | None = None,
        interactive: bool = False,
        read_only_command_prefixes: tuple[str, ...] = (),
        verification_command_prefixes: tuple[str, ...] = (),
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
            read_only_command_prefixes=read_only_command_prefixes,
            verification_command_prefixes=verification_command_prefixes,
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

    def test_batch_patch_applies_multiple_hunks_and_preserves_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "script.py"
            path.write_text("alpha\nbeta\n", encoding="utf-8")
            path.chmod(0o755)
            executor = self.make_executor(root)

            result = executor.execute(
                "patch_file_batch",
                {
                    "edits": [
                        {
                            "path": "script.py",
                            "old_text": "alpha",
                            "new_text": "ALPHA",
                        },
                        {
                            "path": "script.py",
                            "old_text": "beta",
                            "new_text": "BETA",
                        },
                    ]
                },
            )

            self.assertTrue(result.ok, result.content)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "ALPHA\nBETA\n",
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o755)
            self.assertEqual(result.changed_files, ["script.py"])
            self.assertIn("2 处替换", result.content)

    def test_batch_patch_validates_all_edits_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "script.py"
            path.write_text("alpha\nbeta\n", encoding="utf-8")
            executor = self.make_executor(root)

            result = executor.execute(
                "patch_file_batch",
                {
                    "edits": [
                        {
                            "path": "script.py",
                            "old_text": "alpha",
                            "new_text": "ALPHA",
                        },
                        {
                            "path": "script.py",
                            "old_text": "missing",
                            "new_text": "MISSING",
                        },
                    ]
                },
            )

            self.assertFalse(result.ok)
            self.assertIn("第 2 处未找到", result.content)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "alpha\nbeta\n",
            )

    def test_batch_write_creates_multiple_files_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)

            result = executor.execute(
                "write_file_batch",
                {
                    "files": [
                        {
                            "path": "src/a.py",
                            "content": "A = 1\n",
                        },
                        {
                            "path": "src/b.py",
                            "content": "B = 2\n",
                        },
                    ]
                },
            )

            self.assertTrue(result.ok, result.content)
            self.assertEqual(
                (root / "src" / "a.py").read_text(encoding="utf-8"),
                "A = 1\n",
            )
            self.assertEqual(
                (root / "src" / "b.py").read_text(encoding="utf-8"),
                "B = 2\n",
            )
            self.assertEqual(
                result.changed_files,
                ["src/a.py", "src/b.py"],
            )

    def test_batch_write_validation_prevents_partial_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)

            result = executor.execute(
                "write_file_batch",
                {
                    "files": [
                        {"path": "good.txt", "content": "ok"},
                        {"path": "../outside.txt", "content": "bad"},
                    ]
                },
            )

            self.assertFalse(result.ok)
            self.assertFalse((root / "good.txt").exists())

    def test_apply_and_verify_returns_patch_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "note.txt"
            path.write_text("alpha\n", encoding="utf-8")
            executor = self.make_executor(root)

            result = executor.execute(
                "apply_and_verify",
                {
                    "edits": [
                        {
                            "path": "note.txt",
                            "old_text": "alpha",
                            "new_text": "beta",
                        }
                    ],
                    "verification_commands": [
                        {"command": "python3 -m unittest --help"},
                    ],
                },
            )

            self.assertTrue(result.ok, result.content)
            self.assertEqual(path.read_text(encoding="utf-8"), "beta\n")
            self.assertIn("=== 文件修改 ===", result.content)
            self.assertIn("=== 验证结果 ===", result.content)
            self.assertEqual(result.changed_files, ["note.txt"])
            self.assertTrue(result.state_changed)

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

    def test_broad_search_defaults_to_bounded_file_summary(self) -> None:
        home = Path.home()
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            executor = ToolExecutor(
                workspace=home,
                path_guard=PathGuard(home),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=home,
                ),
                artifacts=ArtifactStore(home, data),
                events=EventBus(data),
                data_dir=data,
            )
            with (
                patch("sunus.tools.find_ripgrep", return_value="/usr/bin/rg"),
                patch.object(
                    executor,
                    "_search_with_ripgrep",
                    return_value=(["repo/src/app.py:2:max_tokens"], False, ""),
                ) as search,
            ):
                result = executor.execute(
                    "search_text",
                    {"query": "max_tokens", "output": "lines"},
                )
        self.assertTrue(result.ok)
        self.assertIn("repo/src/app.py\t1 命中", result.content)
        self.assertIn("broad_root=metadata_first", result.content)
        self.assertEqual(search.call_args.kwargs["limit"], 1_000)

    def test_explicit_global_line_search_is_still_allowed(self) -> None:
        home = Path.home()
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            executor = ToolExecutor(
                workspace=home,
                path_guard=PathGuard(home),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=home,
                ),
                artifacts=ArtifactStore(home, data),
                events=EventBus(data),
                data_dir=data,
            )
            with (
                patch("sunus.tools.find_ripgrep", return_value="/usr/bin/rg"),
                patch.object(
                    executor,
                    "_search_with_ripgrep",
                    return_value=(["repo/src/app.py:2:max_tokens"], False, ""),
                ),
            ):
                result = executor.execute(
                    "search_text",
                    {
                        "query": "max_tokens",
                        "output": "lines",
                        "allow_broad": True,
                    },
                )
        self.assertTrue(result.ok)
        self.assertIn("repo/src/app.py:2:max_tokens", result.content)
        self.assertNotIn("broad_root=metadata_first", result.content)

    def test_inspect_code_state_returns_context_and_git_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "src" / "app.py"
            source.parent.mkdir()
            source.write_text("value = 1\nmax_tokens = 256\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "add token budget"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            executor = self.make_executor(root)

            result = executor.execute(
                "inspect_code_state",
                {
                    "path": "src/app.py",
                    "symbol": "max_tokens",
                    "context_lines": 2,
                },
            )

            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertTrue(payload["version_verified"])
            self.assertTrue(payload["head"])
            self.assertEqual(payload["matches"][0]["line"], 2)
            self.assertIn("max_tokens", payload["matches"][0]["text"])
            self.assertTrue(payload["file_commit"]["commit"])
            self.assertTrue(payload["line_commits"]["2"]["commit"])

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

    def test_read_only_command_batch_aggregates_labeled_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")
            executor = self.make_executor(root)
            original_execute = executor._execute_shell_command
            lock = threading.Lock()
            active = 0
            maximum = 0

            def tracked_execute(*args, **kwargs):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    time.sleep(0.05)
                    return original_execute(*args, **kwargs)
                finally:
                    with lock:
                        active -= 1

            with patch.object(
                executor,
                "_execute_shell_command",
                side_effect=tracked_execute,
            ):
                result = executor.execute(
                    "run_command_batch",
                    {
                        "commands": [
                            {"command": "cat a.txt"},
                            {"command": "cat b.txt"},
                        ]
                    },
                )

            self.assertTrue(result.ok, result.content)
            self.assertGreaterEqual(maximum, 2)
            self.assertIn("=== [1] cat a.txt", result.content)
            self.assertIn("=== [2] cat b.txt", result.content)
            self.assertEqual(result.content.count("exit_code=0"), 2)
            self.assertIn("alpha", result.content)
            self.assertIn("beta", result.content)

    def test_command_batch_rejects_non_read_only_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)

            result = executor.execute(
                "run_command_batch",
                {
                    "commands": [
                        {"command": "printf changed > state.txt"},
                    ]
                },
            )

            self.assertFalse(result.ok)
            self.assertIn("不是只读 R0", result.content)
            self.assertFalse((root / "state.txt").exists())

    def test_command_batch_accepts_configured_read_only_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper = root / "helper.sh"
            helper.write_text("#!/bin/sh\nprintf 'helper-ok\\n'\n", encoding="utf-8")
            helper.chmod(0o755)
            executor = self.make_executor(
                root,
                read_only_command_prefixes=("./helper.sh",),
            )

            result = executor.execute(
                "run_command_batch",
                {
                    "commands": [
                        {"command": "./helper.sh"},
                    ]
                },
            )

            self.assertTrue(result.ok, result.content)
            self.assertIn("helper-ok", result.content)

    def test_verification_batch_accepts_custom_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            verifier = root / "verify.sh"
            verifier.write_text(
                "#!/bin/sh\nprintf 'verify-ok\\n'\n",
                encoding="utf-8",
            )
            verifier.chmod(0o755)
            executor = self.make_executor(
                root,
                verification_command_prefixes=("./verify.sh",),
            )

            result = executor.execute(
                "verification_batch",
                {
                    "commands": [
                        {"command": "./verify.sh --fast"},
                    ]
                },
            )

            self.assertTrue(result.ok, result.content)
            self.assertIn("verify-ok", result.content)

    def test_verification_prefix_matches_absolute_executable_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(
                Path(temp),
                verification_command_prefixes=("check",),
            )

            self.assertTrue(
                executor._verification_command_allowed(
                    "/usr/local/bin/check --full"
                )
            )

    def test_read_only_tool_result_is_cached_until_state_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "note.txt"
            path.write_text("first", encoding="utf-8")
            executor = self.make_executor(root)
            executor.begin_request("读取 note.txt")

            first = executor.execute("read_file", {"path": "note.txt"})
            second = executor.execute("read_file", {"path": "note.txt"})
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)

            path.write_text("second", encoding="utf-8")
            executor.task_state = TaskState(objective="刷新读取缓存")
            executor.execute(
                "update_reasoning_state",
                {"decision": "文件状态可能变化，重新读取"},
            )
            third = executor.execute("read_file", {"path": "note.txt"})

            self.assertFalse(third.cache_hit)
            self.assertIn("second", third.content)

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

    def test_configure_permissions_persists_and_applies_write_allowlist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / ".sunus"
            memory = MemoryStore(data / "memory.db")
            executor = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=EventBus(data),
                data_dir=data,
                memory=memory,
            )
            scope = executor.begin_request(
                "我授权你以后直接写日记，修改这个权限配置"
            )
            self.assertTrue(scope.allows_mutation)
            diary = data / "diary"

            configured = executor.execute(
                "configure_permissions",
                {
                    "write_allowlist": [str(diary)],
                    "mode": "set",
                    "reason": "用户授权直接写日记",
                },
            )

            self.assertTrue(configured.ok)
            config = json.loads((data / "config.json").read_text(encoding="utf-8"))
            permissions = config["runtime"]["permissions"]
            self.assertEqual(permissions["write_allowlist"], [str(diary.resolve())])
            executor.begin_request("写今天的日记")
            diary_file = diary / "2026-10-26.md"
            written = executor.execute(
                "write_file",
                {"path": str(diary_file), "content": "今天的记录"},
            )
            self.assertTrue(written.ok)
            self.assertEqual(diary_file.read_text(encoding="utf-8"), "今天的记录")

            protected = executor.execute(
                "write_file",
                {"path": str(data / "other.txt"), "content": "blocked"},
            )
            self.assertFalse(protected.ok)
            self.assertFalse((data / "other.txt").exists())
            executor.close()
            memory.close()

    def test_schedule_proactive_file_write_persists_task_and_permission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / ".sunus"
            memory = MemoryStore(data / "memory.db")
            scheduler = ProactiveScheduler(
                data / "memory.db",
                ProactiveConfig(),
                "task-test",
            )
            executor = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=EventBus(data),
                data_dir=data,
                memory=memory,
            )
            executor.proactive_scheduler = scheduler
            executor.begin_request("下次主动消息写日记")
            target_template = str(data / "diary" / "{date}.txt")

            result = executor.execute(
                "schedule_proactive_file_write",
                {
                    "task_name": "daily_diary",
                    "instruction": "写今天的日记",
                    "target_path_template": target_template,
                    "cadence": "daily",
                    "not_before": "20:00",
                    "source_memory_id": "m_test",
                },
            )

            self.assertTrue(result.ok)
            tasks = scheduler.list_tasks(status="active")
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].name, "daily_diary")
            self.assertEqual(tasks[0].source_memory_id, "m_test")
            self.assertIn(
                (data / "diary").resolve(),
                executor.path_guard.write_allowlist,
            )
            config = json.loads((data / "config.json").read_text(encoding="utf-8"))
            self.assertIn(
                str((data / "diary").resolve()),
                config["runtime"]["permissions"]["write_allowlist"],
            )
            scheduler.close()
            executor.close()
            memory.close()

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

    def test_side_effect_approval_does_not_depend_on_request_words(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root, interactive=True)
            scope = executor.begin_request(
                "登陆了，你看看关掉后下次能不能带登陆态打开，"
                "也许可以用mcp或者cdp模式我记得"
            )
            self.assertFalse(scope.allows_commands)
            executor.approvals.begin_task()

            with (
                patch(
                    "sunus.policy._choose_approval_option",
                    return_value=0,
                ) as choose,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                first = executor.execute(
                    "run_command",
                    {"command": "printf first", "timeout_sec": 10},
                )
                second = executor.execute(
                    "run_command",
                    {"command": "printf second", "timeout_sec": 10},
                )

            self.assertTrue(first.ok, first.content)
            self.assertTrue(second.ok, second.content)
            self.assertIn("first", first.content)
            self.assertIn("second", second.content)
            self.assertEqual(choose.call_count, 1)
            self.assertTrue(executor.request_scope.allows_commands)
            executor.approvals.end_task()

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
            self.assertIn("exit_code=1", result.content)
            self.assertIn("semantic_check", result.content)
            self.assertIn("Python traceback", result.content)

    def test_runtime_version_requirement_is_marked_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp))
            result = executor.execute(
                "run_command",
                {
                    "command": (
                        "printf 'Playwright requires Node.js 20 or higher.\\n'"
                    )
                },
            )

            self.assertFalse(result.ok)
            self.assertIn("exit_code=0", result.content)
            self.assertIn("运行时版本不兼容", result.content)

    def test_pipeline_failure_and_playwright_timeout_are_not_marked_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp))
            failed_pipe = executor.execute(
                "run_command",
                {"command": "sh -c 'exit 7' | tail -1"},
            )
            playwright_timeout = executor.execute(
                "run_command",
                {
                    "command": (
                        "printf 'waiting for element to be visible, "
                        "enabled and stable\\n'"
                    )
                },
            )

            self.assertFalse(failed_pipe.ok)
            self.assertIn("exit_code=7", failed_pipe.content)
            self.assertFalse(playwright_timeout.ok)
            self.assertIn("Playwright 操作未完成", playwright_timeout.content)

    def test_browser_navigation_is_marked_as_state_change(self) -> None:
        self.assertTrue(
            _command_changes_target_state(
                "osascript -e 'tell application \"Google Chrome\" "
                "to make new window'"
            )
        )
        self.assertTrue(
            _command_changes_target_state(
                "playwright-cli open https://platform.deepseek.com/usage"
            )
        )
        self.assertTrue(
            _command_changes_target_state(
                "P=skills/playwright-cli-launcher/scripts/pw.sh && "
                "bash $P -s=xhs click e126"
            )
        )
        self.assertFalse(
            _command_changes_target_state(
                "pgrep -fl 'Google Chrome'; osascript -e "
                "'tell application \"Google Chrome\" to count windows'"
            )
        )

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

    def test_search_memory_uses_structured_index_instead_of_database_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = MemoryStore(root / ".data" / "memory.db")
            memory.add(
                content="用户喜欢循环的歌曲是《你瞒我瞒》",
                scope="global",
                kind="preference",
                status="supported",
            )
            executor = self.make_executor(root, memory=memory)
            executor.begin_request("帮我查记忆里喜欢的歌")

            result = executor.execute(
                "search_memory",
                {"query": "循环 歌曲", "kind": "preference", "limit": 5},
            )

            self.assertTrue(result.ok)
            self.assertIn("你瞒我瞒", result.content)
            self.assertNotIn("memory.db", result.content)
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
                self.assertIn("需要明确授权", result.content)
            self.assertFalse((root / "new.txt").exists())

    def test_fetch_url_timeout_returns_structured_network_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            executor.approvals.grant_capabilities(["network"])
            executor.begin_request(
                "读取 https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices"
            )
            with patch(
                "sunus.tools.urllib.request.urlopen",
                side_effect=urllib.error.URLError(
                    socket.timeout("handshake timed out")
                ),
            ) as urlopen:
                result = executor.execute(
                    "fetch_url",
                    {
                        "url": (
                            "https://en.wikipedia.org/wiki/"
                            "Raven%27s_Progressive_Matrices"
                        ),
                        "timeout_sec": 7,
                    },
                )

            self.assertFalse(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["error"], "network_timeout")
            self.assertEqual(payload["host"], "en.wikipedia.org")
            self.assertEqual(payload["timeout_sec"], 7)
            self.assertTrue(payload["retryable"])
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 7)

    def test_fetch_url_batch_runs_independent_requests_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = self.make_executor(root)
            executor.approvals.session_allow = True
            lock = threading.Lock()
            active = 0
            maximum = 0

            def fake_fetch(arguments, *, _approved=False):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    time.sleep(0.05)
                    return ToolResult(True, f"body:{arguments['url']}")
                finally:
                    with lock:
                        active -= 1

            with patch.object(
                executor,
                "_fetch_url",
                side_effect=fake_fetch,
            ):
                result = executor.execute(
                    "fetch_url_batch",
                    {
                        "requests": [
                            {"url": "https://example.com/a"},
                            {"url": "https://example.com/b"},
                        ]
                    },
                )

            self.assertTrue(result.ok)
            self.assertGreaterEqual(maximum, 2)
            self.assertIn("body:https://example.com/a", result.content)
            self.assertIn("body:https://example.com/b", result.content)

    def test_explicit_request_path_is_writable_for_that_turn_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            target = Path(temp) / "exports" / "result.md"
            executor = self.make_executor(root)

            executor.begin_request(f"把结果写到 {target}")
            written = executor.execute(
                "write_file",
                {"path": str(target), "content": "结果"},
            )

            self.assertTrue(written.ok)
            self.assertEqual(target.read_text(encoding="utf-8"), "结果")

            executor.begin_request("只分析，不要写文件")
            denied = executor.execute(
                "write_file",
                {"path": str(Path(temp) / "later.md"), "content": "blocked"},
            )
            self.assertFalse(denied.ok)
            self.assertFalse((Path(temp) / "later.md").exists())

    def test_sensitive_credential_path_stays_blocked_even_if_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            executor = self.make_executor(Path(temp))
            target = Path.home() / ".ssh" / "sunus-test"

            executor.begin_request(f"把文件写到 {target}")
            result = executor.execute(
                "write_file",
                {"path": str(target), "content": "blocked"},
            )

            self.assertFalse(result.ok)
            self.assertIn("敏感路径", result.content)
            self.assertFalse(target.exists())

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
            self.assertIn("需要明确授权", result.content)
            self.assertFalse((root / "blocked.txt").exists())


if __name__ == "__main__":
    unittest.main()

