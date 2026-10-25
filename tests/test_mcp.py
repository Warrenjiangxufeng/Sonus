from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from sunus.config import MCPServerConfig
from sunus.events import EventBus
from sunus.mcp import MCPManager
from sunus.policy import ApprovalManager, PathGuard
from sunus.storage import ArtifactStore
from sunus.tools import ToolExecutor


class MCPTests(unittest.TestCase):
    def test_stdio_server_discovery_and_call(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"
        with tempfile.TemporaryDirectory() as temp:
            config = MCPServerConfig(
                name="fake",
                command=sys.executable,
                args=(str(fixture),),
                startup_timeout_sec=5,
                tool_timeout_sec=5,
            )
            manager = MCPManager(
                (config,),
                events=EventBus(Path(temp)),
            )
            try:
                specs = manager.discover_specs()
                self.assertEqual(len(specs), 1)
                spec = specs[0]
                self.assertEqual(spec.name, "mcp__fake__echo")
                self.assertEqual(spec.source, "mcp")
                self.assertEqual(spec.approval_mode, "ask")
                result = spec.handler({"text": "hello"})
                self.assertTrue(result.ok)
                self.assertEqual(result.content, "hello")
            finally:
                manager.close()

    def test_tool_filters_are_applied(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"
        with tempfile.TemporaryDirectory() as temp:
            config = MCPServerConfig(
                name="fake",
                command=sys.executable,
                args=(str(fixture),),
                enabled_tools=("missing",),
            )
            manager = MCPManager(
                (config,),
                events=EventBus(Path(temp)),
            )
            try:
                self.assertEqual(manager.discover_specs(), [])
            finally:
                manager.close()

    def test_tool_executor_requires_approval_for_mcp(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            config = MCPServerConfig(
                name="fake",
                command=sys.executable,
                args=(str(fixture),),
                startup_timeout_sec=5,
                tool_timeout_sec=5,
            )
            events = EventBus(data)
            approvals = ApprovalManager(
                auto_approve=False,
                interactive=False,
                workspace=root,
            )
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=approvals,
                artifacts=ArtifactStore(root, data),
                events=events,
                mcp_servers=(config,),
            )
            try:
                self.assertIn("mcp__fake__echo", tools.registry.names())
                denied = tools.execute("mcp__fake__echo", {"text": "hello"})
                self.assertFalse(denied.ok)
                self.assertIn("拒绝", denied.content)
                approvals.session_allow = True
                allowed = tools.execute("mcp__fake__echo", {"text": "hello"})
                self.assertTrue(allowed.ok)
                self.assertEqual(allowed.content, "hello")
            finally:
                tools.close()


if __name__ == "__main__":
    unittest.main()
