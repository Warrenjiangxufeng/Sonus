from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sunus.events import EventBus
from sunus.policy import ApprovalManager, PathGuard
from sunus.storage import ArtifactStore
from sunus.tool_registry import ToolRegistry, ToolSpec
from sunus.tools import ToolExecutor, ToolResult


class ToolRegistryTests(unittest.TestCase):
    def make_executor(self, root: Path) -> ToolExecutor:
        data = root / ".data"
        events = EventBus(data)
        return ToolExecutor(
            workspace=root,
            path_guard=PathGuard(root),
            approvals=ApprovalManager(
                auto_approve=False,
                interactive=False,
                workspace=root,
            ),
            artifacts=ArtifactStore(root, data),
            events=events,
        )

    def test_builtin_tools_have_capability_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = self.make_executor(root)
            read = tools.registry.get("read_file")
            write = tools.registry.get("write_file")
            command = tools.registry.get("run_command")
            authorization = tools.registry.get("request_authorization")
            configure_permissions = tools.registry.get("configure_permissions")
            schedule_proactive = tools.registry.get(
                "schedule_proactive_file_write"
            )
            runtime_history = tools.registry.get("inspect_runtime_history")
            code_state = tools.registry.get("inspect_code_state")
            memory_search = tools.registry.get("search_memory")
            browser = tools.registry.get("inspect_browser")
            remember = tools.registry.get("remember")
            result = tools.registry.get("submit_result")

            self.assertIsNotNone(read)
            self.assertTrue(read.read_only)
            self.assertTrue(read.parallel_safe)
            self.assertEqual(read.risk, "R0")
            self.assertEqual(read.capability, "workspace_text")
            self.assertEqual(read.evidence_kind, "file_content")
            self.assertFalse(write.read_only)
            self.assertFalse(write.parallel_safe)
            self.assertEqual(write.risk, "R1")
            self.assertFalse(command.parallel_safe)
            self.assertFalse(authorization.read_only)
            self.assertFalse(authorization.parallel_safe)
            self.assertFalse(configure_permissions.read_only)
            self.assertFalse(configure_permissions.parallel_safe)
            self.assertEqual(configure_permissions.risk, "R1")
            self.assertFalse(schedule_proactive.read_only)
            self.assertFalse(schedule_proactive.parallel_safe)
            self.assertEqual(schedule_proactive.risk, "R1")
            self.assertTrue(runtime_history.read_only)
            self.assertTrue(runtime_history.parallel_safe)
            self.assertTrue(code_state.read_only)
            self.assertTrue(code_state.parallel_safe)
            self.assertTrue(memory_search.read_only)
            self.assertTrue(memory_search.parallel_safe)
            self.assertFalse(browser.read_only)
            self.assertFalse(browser.parallel_safe)
            self.assertEqual(browser.capability, "browser_state")
            self.assertEqual(browser.evidence_kind, "browser_page")
            self.assertTrue(browser.preconditions)
            self.assertTrue(browser.failure_modes)
            fetch = tools.registry.get("fetch_url")
            self.assertEqual(fetch.capability, "remote_http")
            self.assertEqual(fetch.evidence_kind, "remote_page")
            self.assertFalse(remember.read_only)
            self.assertFalse(remember.parallel_safe)
            self.assertFalse(result.parallel_safe)
            self.assertEqual(
                {item.name for item in tools.specs()},
                {
                    "read_file",
                    "list_files",
                    "search_text",
                    "inspect_code_state",
                    "search_memory",
                    "memory_context_metrics",
                    "inspect_environment",
                    "inspect_runtime_history",
                    "inspect_browser",
                    "remember",
                    "write_file",
                    "write_document",
                    "patch_file",
                    "run_command",
                    "request_authorization",
                    "configure_permissions",
                    "schedule_proactive_file_write",
                    "fetch_url",
                    "list_skills",
                    "read_skill",
                    "submit_result",
                },
            )

    def test_duplicate_tool_names_are_rejected(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec(
            name="echo",
            description="Echo.",
            parameters={"type": "object", "properties": {}},
            handler=lambda _arguments: ToolResult(True, "ok"),
        )
        registry.register(spec)
        with self.assertRaises(ValueError):
            registry.register(spec)


if __name__ == "__main__":
    unittest.main()
