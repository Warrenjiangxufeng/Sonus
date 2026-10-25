from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from sunus.config import MCPServerConfig, redact_text
from sunus.events import EventBus
from sunus.tool_registry import ToolSpec
from sunus.tools import ToolResult


class MCPError(RuntimeError):
    pass


class StdioMCPServer:
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        events: EventBus,
        secret: str = "",
    ) -> None:
        self.config = config
        self.events = events
        self.secret = secret
        self.process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr: threading.Thread | None = None
        self._stderr_lines: list[str] = []
        self._request_id = 0
        self._lock = threading.RLock()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        env = {
            "PATH": os.environ.get(
                "PATH",
                "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            ),
            "HOME": os.environ.get("HOME", str(Path.home())),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        }
        env.update(dict(self.config.env))
        cwd = (
            Path(self.config.cwd).expanduser()
            if self.config.cwd
            else Path.home()
        )
        try:
            self.process = subprocess.Popen(
                [self.config.command, *self.config.args],
                cwd=str(cwd),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            raise MCPError(f"MCP 服务器 {self.config.name} 启动失败: {error}") from error
        self._reader = threading.Thread(
            target=self._read_stdout,
            name=f"mcp-{self.config.name}-stdout",
            daemon=True,
        )
        self._reader.start()
        self._stderr = threading.Thread(
            target=self._read_stderr,
            name=f"mcp-{self.config.name}-stderr",
            daemon=True,
        )
        self._stderr.start()
        self.events.emit(
            "mcp_server_started",
            server=self.config.name,
            summary=f"MCP 服务器已启动 · {self.config.name}",
        )

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            self._responses.put(None)
            return
        try:
            for line in process.stdout:
                if len(line) > 2 * 1024 * 1024:
                    self._responses.put(
                        {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32000,
                                "message": "MCP 单条响应超过 2 MB",
                            },
                        }
                    )
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    self._responses.put(payload)
        finally:
            self._responses.put(None)

    def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = redact_text(line.strip(), self.secret)[:1000]
            if text:
                self._stderr_lines.append(text)
                del self._stderr_lines[:-30]

    def _send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise MCPError(f"MCP 服务器 {self.config.name} 未运行")
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise MCPError(
                f"MCP 服务器 {self.config.name} 连接中断: {error}"
            ) from error

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_sec: float,
    ) -> dict[str, Any]:
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            deadline = time.monotonic() + timeout_sec
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(
                        f"MCP 请求超时 · {self.config.name} · {method}"
                    )
                try:
                    response = self._responses.get(timeout=remaining)
                except queue.Empty as error:
                    raise MCPError(
                        f"MCP 请求超时 · {self.config.name} · {method}"
                    ) from error
                if response is None:
                    detail = self._stderr_lines[-1] if self._stderr_lines else ""
                    raise MCPError(
                        f"MCP 服务器 {self.config.name} 意外退出"
                        + (f": {detail}" if detail else "")
                    )
                if response.get("id") != request_id:
                    continue
                error = response.get("error")
                if error:
                    raise MCPError(
                        f"MCP {method} 失败: "
                        f"{error.get('message') or error}"
                    )
                result = response.get("result")
                return result if isinstance(result, dict) else {}

    def initialize(self) -> None:
        self.start()
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "Sunus", "version": "0.1.0"},
            },
            timeout_sec=self.config.startup_timeout_sec,
        )
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
        )

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = ""
        while True:
            result = self._request(
                "tools/list",
                {"cursor": cursor} if cursor else {},
                timeout_sec=self.config.startup_timeout_sec,
            )
            tools.extend(
                item
                for item in result.get("tools", [])
                if isinstance(item, dict)
            )
            cursor = str(result.get("nextCursor") or "")
            if not cursor:
                return tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_sec: float,
    ) -> ToolResult:
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout_sec=timeout_sec,
        )
        texts: list[str] = []
        for item in result.get("content", []):
            if not isinstance(item, dict):
                texts.append(str(item))
                continue
            if item.get("type") == "text":
                texts.append(str(item.get("text") or ""))
            else:
                texts.append(json.dumps(item, ensure_ascii=False))
        content = "\n".join(text for text in texts if text)
        if not content and result:
            content = json.dumps(result, ensure_ascii=False)
        return ToolResult(
            not bool(result.get("isError")),
            redact_text(content, self.secret),
        )

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            except OSError:
                pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self.events.emit(
            "mcp_server_stopped",
            server=self.config.name,
            summary=f"MCP 服务器已停止 · {self.config.name}",
        )


class MCPManager:
    def __init__(
        self,
        configs: tuple[MCPServerConfig, ...],
        *,
        events: EventBus,
        secret: str = "",
    ) -> None:
        self.events = events
        self.servers = [
            StdioMCPServer(config, events=events, secret=secret)
            for config in configs
            if config.enabled
        ]

    def discover_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for server in self.servers:
            try:
                server.initialize()
                tools = server.list_tools()
            except Exception as error:
                detail = redact_text(str(error), server.secret)
                self.events.emit(
                    "mcp_server_failed",
                    server=server.config.name,
                    summary=f"MCP 服务器不可用 · {server.config.name}: {detail[:500]}",
                )
                server.close()
                continue
            enabled = set(server.config.enabled_tools)
            disabled = set(server.config.disabled_tools)
            approval_modes = dict(server.config.approval_modes)
            discovered = 0
            for tool in tools:
                original_name = str(tool.get("name") or "").strip()
                if not original_name:
                    continue
                if enabled and original_name not in enabled:
                    continue
                if original_name in disabled:
                    continue
                public_name = (
                    f"mcp__{self._safe_name(server.config.name)}__"
                    f"{self._safe_name(original_name)}"
                )
                schema = tool.get("inputSchema") or tool.get("parameters") or {}
                if not isinstance(schema, dict):
                    schema = {}
                schema = dict(schema)
                schema.setdefault("type", "object")
                schema.setdefault("properties", {})
                schema.setdefault("required", [])

                def handler(
                    arguments: dict[str, Any],
                    *,
                    target: StdioMCPServer = server,
                    target_name: str = original_name,
                    default_timeout: int = server.config.tool_timeout_sec,
                ) -> ToolResult:
                    timeout_sec = max(
                        1,
                        min(
                            default_timeout,
                            int(arguments.pop("timeout_sec", default_timeout)),
                        ),
                    )
                    return target.call_tool(
                        target_name,
                        arguments,
                        timeout_sec=timeout_sec,
                    )

                specs.append(
                    ToolSpec(
                        name=public_name,
                        description=str(tool.get("description") or ""),
                        parameters=schema,
                        handler=handler,
                        source="mcp",
                        risk="R2",
                        read_only=False,
                        parallel_safe=False,
                        timeout_sec=server.config.tool_timeout_sec,
                        approval_mode=approval_modes.get(original_name, "ask"),
                    )
                )
                discovered += 1
            self.events.emit(
                "mcp_tools_discovered",
                server=server.config.name,
                count=discovered,
                summary=f"MCP 发现 {discovered} 个工具 · {server.config.name}",
            )
        return specs

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in value
        )

    def close(self) -> None:
        for server in self.servers:
            server.close()
