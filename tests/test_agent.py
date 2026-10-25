from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sunus.agent import Agent
from sunus.config import LoopBudgetConfig, PersonaConfig, RuntimeConfig
from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.model_events import ModelEvent
from sunus.model import ModelCancelled, ModelReply
from sunus.policy import ApprovalManager, PathGuard
from sunus.storage import ArtifactStore, SessionLogger
from sunus.tools import ToolExecutor


class FakeAgentModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages = []
        self.last_max_tokens = None

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        self.last_messages = messages
        self.last_max_tokens = max_tokens
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": "我先读取文件。",
                    "reasoning_content": "provider reasoning",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"note.txt"}',
                            },
                        }
                    ],
                },
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_cache_hit_tokens": 7,
                    "prompt_cache_miss_tokens": 3,
                },
                finish_reason="tool_calls",
            )
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        assert tool_messages and "sunus test" in tool_messages[-1]["content"]
        return ModelReply(
            message={"role": "assistant", "content": "文件内容已确认。"},
            usage={
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "prompt_cache_hit_tokens": 6,
                "prompt_cache_miss_tokens": 2,
            },
            finish_reason="stop",
        )


class ExplodingModel:
    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        raise RuntimeError("unexpected transport failure")


class EmptyTruncatedModel:
    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        return ModelReply(
            message={"role": "assistant", "content": ""},
            usage={"total_tokens": 2048},
            finish_reason="length",
        )


class RetryAfterEmptyModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_max_tokens: list[int | None] = []
        self.last_messages: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        self.last_max_tokens.append(max_tokens)
        self.last_messages = messages
        if self.calls == 1:
            return ModelReply(
                message={"role": "assistant", "content": "", "reasoning_content": "权衡"},
                usage={"prompt_tokens": 10, "completion_tokens": 512},
                finish_reason="length",
            )
        return ModelReply(
            message={"role": "assistant", "content": "直接回答。"},
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )


class CapturingReplyModel:
    def __init__(self, text: str) -> None:
        self.text = text
        self.last_messages = []
        self.last_max_tokens = None
        self.tool_names: set[str] = set()

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.last_messages = messages
        self.last_max_tokens = max_tokens
        self.tool_names = {
            item["function"]["name"]
            for item in tools or []
        }
        return ModelReply(
            message={"role": "assistant", "content": self.text},
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )


class TextualProtocolRecoveryModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        self.last_messages = messages
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": (
                        "<｜｜DSML｜｜ calls>\n"
                        '<｜｜DSML｜｜ invoke name="list_files">\n'
                        '<｜｜DSML｜｜ parameter name="path" string="true">'
                        ".</｜｜DSML｜｜ parameter>\n"
                        "</｜｜DSML｜｜ invoke>\n"
                        "</｜｜DSML｜｜ calls>"
                    ),
                },
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                finish_reason="stop",
            )
        return ModelReply(
            message={"role": "assistant", "content": "这次正常回复。"},
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )

    def chat_stream(
        self,
        messages,
        *,
        tools=None,
        max_tokens=None,
        temperature=0.2,
        on_event=None,
    ):
        reply = self.chat(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if on_event and reply.message.get("content"):
            on_event(
                ModelEvent(
                    "text_delta",
                    text=str(reply.message["content"]),
                )
            )
        return reply


class ClassifyingReplyModel(CapturingReplyModel):
    def __init__(self, text: str, mode: str) -> None:
        super().__init__(text)
        self.mode = mode
        self.classification_calls = 0
        self.classification_previous = ""

    def classify_turn(self, user_input, *, previous_assistant=""):
        self.classification_calls += 1
        self.classification_previous = previous_assistant
        return self.mode, {"prompt_tokens": 20, "completion_tokens": 1}


class CancelledAfterWriteModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": "先写入文件。",
                    "tool_calls": [
                        {
                            "id": "call_write",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path":"changed.txt","content":"new"}',
                            },
                        }
                    ],
                },
                usage={},
                finish_reason="tool_calls",
            )
        raise ModelCancelled("思考已中断")


class NoProgressModel:
    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        return ModelReply(
            message={
                "role": "assistant",
                "content": "继续检查。",
                "tool_calls": [
                    {
                        "id": "call_repeat",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": '{"path":"."}',
                        },
                    }
                ],
            },
            usage={"total_tokens": 10},
            finish_reason="tool_calls",
        )


class FinalizingResultModel:
    def __init__(self) -> None:
        self.calls = 0
        self.tool_name_sets: list[set[str]] = []

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        self.tool_name_sets.append(
            {item["function"]["name"] for item in tools or []}
        )
        if self.calls < 4:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": "继续执行。",
                    "tool_calls": [
                        {
                            "id": f"call_continue_{self.calls}",
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "arguments": '{"path":"."}',
                            },
                        }
                    ],
                },
                usage={"total_tokens": 10},
                finish_reason="tool_calls",
            )
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_final_result",
                        "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": (
                                '{"kind":"answer","status":"partial",'
                                '"summary":"已基于现有证据给出部分结论。",'
                                '"risks":["缺少精确 tokenizer"]}'
                            ),
                        },
                    }
                ],
            },
            usage={"total_tokens": 5},
            finish_reason="tool_calls",
        )


class TokenBudgetFinalizingModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": "先读取目录。",
                    "tool_calls": [
                        {
                            "id": "call_budget_tool",
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "arguments": '{"path":"."}',
                            },
                        }
                    ],
                },
                usage={"prompt_tokens": 70, "completion_tokens": 10},
                finish_reason="tool_calls",
            )
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_budget_result",
                        "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": (
                                '{"kind":"blocked","status":"blocked",'
                                '"summary":"Token 预算已收口。",'
                                '"next_action":"提高预算后继续"}'
                            ),
                        },
                    }
                ],
            },
            usage={"prompt_tokens": 40, "completion_tokens": 10},
            finish_reason="tool_calls",
        )


class MemoryMetricsQuestionModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": "先统计本轮记忆上下文。",
                    "tool_calls": [
                        {
                            "id": "call_metrics",
                            "type": "function",
                            "function": {
                                "name": "memory_context_metrics",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                usage={"total_tokens": 10},
                finish_reason="tool_calls",
            )
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        assert tool_messages and "exact=false" in tool_messages[-1]["content"]
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_answer",
                        "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": (
                                '{"kind":"answer","status":"completed",'
                                '"summary":"已统计记忆上下文；字符数精确，'
                                'Token 为估算值。"}'
                            ),
                        },
                    }
                ],
            },
            usage={"total_tokens": 5},
            finish_reason="tool_calls",
        )


class SubmitThenContinueModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_submit_first",
                            "type": "function",
                            "function": {
                                "name": "submit_result",
                                "arguments": (
                                    '{"kind":"answer","status":"completed",'
                                    '"summary":"第一轮完成。"}'
                                ),
                            },
                        }
                    ],
                },
                usage={},
                finish_reason="tool_calls",
            )
        for index, message in enumerate(messages):
            if not message.get("tool_calls"):
                continue
            expected = {
                call.get("id")
                for call in message["tool_calls"]
                if call.get("id")
            }
            actual = {
                item.get("tool_call_id")
                for item in messages[index + 1 :]
                if item.get("role") == "tool"
            }
            assert expected <= actual, (expected, actual)
        return ModelReply(
            message={"role": "assistant", "content": "第二轮继续成功。"},
            usage={},
            finish_reason="stop",
        )


class StructuredResultModel:
    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_result",
                        "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": (
                                '{"kind":"answer","status":"completed",'
                                '"summary":"结构化回答。",'
                                '"next_action":"顺手再优化其他功能",'
                                '"verification":[{"command":"unit test",'
                                '"status":"passed","summary":"2/2"}]}'
                            ),
                        },
                    }
                ],
            },
            usage={"total_tokens": 5},
            finish_reason="tool_calls",
        )


class TextualDSMLResultModel:
    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        return ModelReply(
            message={
                "role": "assistant",
                "content": (
                    "<｜｜DSML｜｜ calls>\n"
                    '<｜｜DSML｜｜ invoke name="submit_result">\n'
                    '<｜｜DSML｜｜ parameter name="kind" string="true">'
                    "answer</｜｜DSML｜｜ parameter>\n"
                    '<｜｜DSML｜｜ parameter name="status" string="true">'
                    "completed</｜｜DSML｜｜ parameter>\n"
                    '<｜｜DSML｜｜ parameter name="summary" string="true">'
                    "DSML 已转换。</｜｜DSML｜｜ parameter>\n"
                    "</｜｜DSML｜｜ invoke>\n"
                    "</｜｜DSML｜｜ calls>"
                ),
            },
            usage={"total_tokens": 5},
            finish_reason="stop",
        )


class ParallelReadModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": f'{{"path":"file_{index}.txt"}}',
                            },
                        }
                        for index in range(3)
                    ],
                },
                usage={"total_tokens": 5},
                finish_reason="tool_calls",
            )
        return ModelReply(
            message={"role": "assistant", "content": "三个文件已读取。"},
            usage={"total_tokens": 2},
            finish_reason="stop",
        )


class UnrequestedWriteModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_unrequested",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path":"extra.txt","content":"extra"}',
                            },
                        }
                    ],
                },
                usage={},
                finish_reason="tool_calls",
            )
        return StructuredResultModel().chat(messages)


class AgentTests(unittest.TestCase):
    def test_tool_loop_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            (root / "note.txt").write_text("sunus test", encoding="utf-8")
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_test", root, data)
            tools = ToolExecutor(
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
            model = FakeAgentModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            result = agent.run_turn("读取 note.txt")
            self.assertEqual(result.reason, "completed")
            self.assertEqual(result.text, "文件内容已确认。")
            self.assertEqual(model.calls, 2)
            self.assertEqual(model.last_max_tokens, 8192)
            self.assertEqual(len(result.tool_traces), 1)
            self.assertTrue(result.receipt_id.startswith("r_"))
            self.assertTrue(session.path.exists())
            self.assertTrue(memory.recent(scope=str(root.resolve()), limit=5))
            self.assertEqual(result.structured_result["kind"], "answer")
            self.assertEqual(result.structured_result["next_action"], "")
            self.assertIn('"result"', session.path.read_text(encoding="utf-8"))
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn("task_grant_started", events_text)
            self.assertIn("task_grant_ended", events_text)
            self.assertIn("读取文件 · note.txt", events_text)
            self.assertIn("assistant_progress", events_text)
            self.assertIn("我先读取文件。", events_text)
            self.assertIn("已读取 1 行", events_text)
            self.assertIn("输出规范", model.last_messages[0]["content"])
            self.assertIn("结论先行", model.last_messages[0]["content"])
            self.assertIn("任务边界", model.last_messages[0]["content"])
            self.assertIn(
                "多个互相独立的只读检查必须在同一轮 `tool_calls` 中批量发起",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "诊断、解释和评审类回答先给结论",
                model.last_messages[0]["content"],
            )
            self.assertEqual(result.usage["prompt_cache_hit_tokens"], 13)
            self.assertEqual(result.usage["prompt_cache_miss_tokens"], 5)
            self.assertIn('"prompt_cache_hit_tokens":7', events_text)
            self.assertIn('"prompt_cache_miss_tokens":3', events_text)
            assistant_messages = [
                item
                for item in model.last_messages
                if item.get("role") == "assistant"
            ]
            self.assertEqual(
                assistant_messages[0]["reasoning_content"],
                "provider reasoning",
            )
            self.assertEqual(tools.approvals.grant.risk_ceiling, "R0")
            memory.close()

    def test_submit_result_is_validated_and_ends_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_result", root, data)
            tools = ToolExecutor(
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
            agent = Agent(
                config=config,
                model=StructuredResultModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("给出结构化结果")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(result.structured_result["kind"], "answer")
            self.assertIn("结构化回答。", result.text)
            self.assertNotIn("unit test", result.text)
            self.assertNotIn("**验证**", result.text)
            self.assertEqual(result.tool_traces[0]["name"], "submit_result")
            self.assertNotIn(
                "result_schema_fallback",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_textual_dsml_result_is_converted_and_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_dsml_result", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            agent = Agent(
                config=config,
                model=TextualDSMLResultModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("检查并分析当前项目结构")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(result.text, "DSML 已转换。")
            self.assertEqual(result.tool_traces[0]["name"], "submit_result")
            self.assertNotIn(
                "<｜｜DSML｜｜",
                session.path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "model_protocol_normalized",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_parallel_safe_tools_run_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            for index in range(3):
                (root / f"file_{index}.txt").write_text(
                    f"content {index}",
                    encoding="utf-8",
                )
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_parallel", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            original_execute = tools.execute
            lock = threading.Lock()
            active = 0
            maximum = 0

            def tracked_execute(name, arguments):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    time.sleep(0.05)
                    return original_execute(name, arguments)
                finally:
                    with lock:
                        active -= 1

            agent = Agent(
                config=config,
                model=ParallelReadModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            with patch.object(
                tools,
                "execute",
                side_effect=tracked_execute,
            ):
                result = agent.run_turn("读取这三个文件")

            self.assertEqual(result.text, "三个文件已读取。")
            self.assertGreaterEqual(maximum, 2)
            self.assertIn(
                "parallel_tool_batch",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_casual_chat_uses_short_no_tool_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            memory.record_turn(
                user_input="我现在设定你的性别为女性，性格温柔体贴",
                assistant_text="好。",
                scope=str(root.resolve()),
                receipt_id="r_casual_persona",
            )
            events = EventBus(data)
            session = SessionLogger("s_chat", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = CapturingReplyModel("我在呢，今天过得怎么样？")
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("哈喽在干嘛")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(model.last_max_tokens, 512)
            self.assertEqual(model.tool_names, set())
            self.assertIn("本轮回应模式：日常对话", model.last_messages[0]["content"])
            self.assertIn("不要调用任何工具", model.last_messages[0]["content"])
            self.assertNotIn(
                "必须调用 submit_result",
                model.last_messages[0]["content"],
            )
            self.assertIn("可演化互动习惯", model.last_messages[0]["content"])
            self.assertIn("性格温柔", model.last_messages[0]["content"])
            self.assertEqual(result.tool_traces, [])
            memory.close()

    def test_dsml_protocol_output_is_intercepted_and_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_dsml", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = TextualProtocolRecoveryModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("哈喽在干嘛")

            self.assertEqual(result.text, "这次正常回复。")
            self.assertEqual(model.calls, 2)
            self.assertNotIn("DSML", result.text)
            self.assertNotIn("DSML", session.path.read_text(encoding="utf-8"))
            self.assertIn(
                "本轮不要输出任何 DSML",
                model.last_messages[0]["content"],
            )
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn("model_protocol_normalized", events_text)
            self.assertIn("model_protocol_retry", events_text)
            self.assertNotIn("<｜｜DSML｜｜", events_text)
            memory.close()

    def test_configured_persona_is_injected_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
                persona=PersonaConfig(
                    name="苏诺",
                    prompt="苏诺语气甜软，会自然地撒娇，也会认真解决问题。",
                    examples=("这句夸我收下啦，正事也会认真做。",),
                    style=(("sweetness", 0.9), ("playfulness", 0.8)),
                    context_intensity=(("chat", 1.0), ("task", 0.6)),
                ),
            )
            memory = MemoryStore(data / "memory.db")
            memory.record_turn(
                user_input="我现在设定你的性别为男性，性格冷淡",
                assistant_text="旧设定。",
                scope=str(root.resolve()),
                receipt_id="r_old_identity",
            )
            events = EventBus(data)
            session = SessionLogger("s_persona_config", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = CapturingReplyModel("我在呀，今天想聊点什么？")
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("哈喽在干嘛")

            system_prompt = model.last_messages[0]["content"]
            self.assertEqual(result.reason, "completed")
            self.assertIn("你是苏诺", system_prompt)
            self.assertIn("语气甜软", system_prompt)
            self.assertIn("正事也会认真做", system_prompt)
            self.assertIn("甜度: 90%", system_prompt)
            self.assertIn("闲聊场景", system_prompt)
            self.assertNotIn("Sunus 核心性别设定为男性", system_prompt)
            self.assertGreater(result.metrics["context"]["persona_config"]["characters"], 0)
            memory.close()

    def test_model_classifies_ambiguous_short_followup(self) -> None:
        cases = (
            ("task", "加点卖萌的元素", 8192),
            ("casual", "你在想我吗", 512),
        )
        with tempfile.TemporaryDirectory() as temp:
            for mode, user_input, expected_limit in cases:
                with self.subTest(mode=mode):
                    root = Path(temp) / f"workspace_{mode}"
                    root.mkdir()
                    data = Path(temp) / f"data_{mode}"
                    config = RuntimeConfig(
                        workspace=root,
                        data_dir=data,
                        api_key="sk-test",
                        base_url="https://example.invalid",
                        model="fake",
                        max_steps=4,
                    )
                    memory = MemoryStore(data / "memory.db")
                    events = EventBus(data)
                    session = SessionLogger(f"s_classify_{mode}", root, data)
                    tools = ToolExecutor(
                        workspace=root,
                        path_guard=PathGuard(root),
                        approvals=ApprovalManager(
                            auto_approve=False,
                            interactive=False,
                            workspace=root,
                        ),
                        artifacts=ArtifactStore(root, data),
                        events=events,
                        memory=memory,
                    )
                    model = ClassifyingReplyModel("本轮已处理。", mode)
                    agent = Agent(
                        config=config,
                        model=model,  # type: ignore[arg-type]
                        memory=memory,
                        tools=tools,
                        events=events,
                        session=session,
                    )
                    if mode == "task":
                        agent.messages.append(
                            {
                                "role": "assistant",
                                "content": "上一轮生成的角色提示词正文。",
                            }
                        )

                    result = agent.run_turn(user_input)

                    self.assertEqual(result.reason, "completed")
                    self.assertEqual(model.classification_calls, 1)
                    self.assertEqual(model.last_max_tokens, expected_limit)
                    self.assertEqual(
                        "本轮回应模式：日常对话"
                        in model.last_messages[0]["content"],
                        mode == "casual",
                    )
                    if mode == "task":
                        self.assertEqual(
                            model.classification_previous,
                            "上一轮生成的角色提示词正文。",
                        )
                    events_text = events.path.read_text(encoding="utf-8")
                    self.assertIn("response_mode_classified", events_text)
                    self.assertIn('"source":"model"', events_text)
                    memory.close()

    def test_consultation_filters_unauthorized_command_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_consult", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = CapturingReplyModel("可以，不同窗口各自是独立会话。")
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("codex CLI 能同时开两个窗口吗")

            self.assertEqual(result.reason, "completed")
            self.assertIn("read_file", model.tool_names)
            self.assertIn("submit_result", model.tool_names)
            self.assertNotIn("run_command", model.tool_names)
            self.assertNotIn("write_file", model.tool_names)
            system_prompt = model.last_messages[0]["content"]
            self.assertIn(f"权威运行数据目录：{data.resolve()}", system_prompt)
            self.assertIn("本轮未注入某项工具", system_prompt)
            self.assertIn("不代表 Sunus 永久没有该能力", system_prompt)
            self.assertIn("未开放: fetch_url, patch_file, run_command", system_prompt)
            memory.close()

    def test_mutation_exposes_command_tool_for_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_verify", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = CapturingReplyModel("修改完成，并已运行验证。")
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            agent.run_turn("修复记忆系统的写入逻辑")

            self.assertIn("patch_file", model.tool_names)
            self.assertIn("run_command", model.tool_names)
            self.assertIn(
                "本轮允许修改，因此已开放本地命令用于必要验证",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "不要把 `grep -r`/`rg` 扫描用户主目录、`Documents` 等大目录作为首选",
                model.last_messages[0]["content"],
            )
            memory.close()

    def test_diagnostic_turn_injects_evidence_gated_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_diagnose", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = CapturingReplyModel("先核对实际环境。")
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            agent.run_turn("为什么它说没有 shell 工具？")

            system_prompt = model.last_messages[0]["content"]
            self.assertIn("至少形成两个竞争假设", system_prompt)
            self.assertIn("单个失败入口不能证明", system_prompt)
            self.assertIn("inspect_environment", model.tool_names)
            self.assertIn("本地执行环境", system_prompt)
            memory.close()

    def test_short_confirmation_inherits_previous_network_offer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_followup", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = CapturingReplyModel("我现在去查。")
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            agent.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "我手里确实有联网工具。"
                        "你要是愿意，我现在试一次，去搜搜看。"
                    ),
                }
            )

            result = agent.run_turn("试一下，特别是江浙沪ip的")

            self.assertEqual(result.reason, "completed")
            self.assertIn("fetch_url", model.tool_names)
            self.assertEqual(model.last_max_tokens, 8192)
            memory.close()

    def test_question_cannot_trigger_unrequested_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_scope", root, data)
            tools = ToolExecutor(
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
            agent = Agent(
                config=config,
                model=UnrequestedWriteModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("这个文件为什么有问题？只分析原因。")

            self.assertEqual(result.reason, "completed")
            self.assertFalse((root / "extra.txt").exists())
            write_trace = next(
                item for item in result.tool_traces if item["name"] == "write_file"
            )
            self.assertFalse(write_trace["ok"])
            self.assertIn("没有明确要求", write_trace["content"])
            memory.close()

    def test_unexpected_exception_still_writes_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_error", root, data)
            tools = ToolExecutor(
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
            agent = Agent(
                config=config,
                model=ExplodingModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("触发异常")

            self.assertEqual(result.reason, "internal_error")
            self.assertIn("Sunus 内部错误", result.text)
            self.assertTrue(result.receipt_id.startswith("r_"))
            session_text = session.path.read_text(encoding="utf-8")
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn('"reason": "internal_error"', session_text)
            self.assertIn("internal_error", events_text)
            self.assertIn("turn_end", events_text)
            self.assertEqual(tools.approvals.grant.risk_ceiling, "R0")
            self.assertEqual(memory.recent(scope=str(root.resolve()), limit=5), [])
            memory.close()

    def test_empty_truncated_output_is_labeled_as_runtime_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_truncated", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            agent = Agent(
                config=config,
                model=EmptyTruncatedModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("给出上下文 token 数量")

            self.assertEqual(result.reason, "output_truncated")
            self.assertEqual(result.structured_result["status"], "partial")
            self.assertIn("运行时回退", result.text)
            self.assertIn("不是模型回答", result.text)
            self.assertIn("finish_reason=length", result.text)
            self.assertEqual(result.metrics["last_finish_reason"], "length")
            self.assertIn("model_output_truncated", events.path.read_text(encoding="utf-8"))
            memory.close()

    def test_empty_length_response_retries_with_full_output_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_output_retry", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = RetryAfterEmptyModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("修复测试并提交")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(result.text, "直接回答。")
            self.assertEqual(model.calls, 2)
            self.assertEqual(model.last_max_tokens, [8192, 16_384])
            event_rows = [
                json.loads(line)
                for line in events.path.read_text(encoding="utf-8").splitlines()
            ]
            requests = [
                row for row in event_rows if row.get("type") == "model_request"
            ]
            self.assertEqual(
                [row["max_tokens"] for row in requests],
                [8192, 16_384],
            )
            retries = [
                row for row in event_rows if row.get("type") == "model_output_retry"
            ]
            self.assertEqual(len(retries), 1)
            self.assertEqual(retries[0]["previous_max_tokens"], 8192)
            self.assertEqual(retries[0]["next_max_tokens"], 16_384)
            self.assertTrue(retries[0]["expanded"])
            memory.close()

    def test_thinking_interrupt_preserves_prior_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_cancel", root, data)
            tools = ToolExecutor(
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
            agent = Agent(
                config=config,
                model=CancelledAfterWriteModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("修改文件后继续思考")

            self.assertEqual(result.reason, "user_interrupt")
            self.assertEqual(result.changed_files, ["changed.txt"])
            self.assertTrue((root / "changed.txt").is_file())
            self.assertEqual(
                (root / "changed.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertIn("变更已保留", result.text)
            self.assertIn("不会自动回滚", result.text)
            receipt = [
                line
                for line in session.path.read_text(encoding="utf-8").splitlines()
                if '"type": "receipt"' in line
            ][-1]
            self.assertIn("changed.txt", receipt)
            self.assertIn("thinking_interrupted", events.path.read_text(encoding="utf-8"))
            memory.close()

    def test_no_progress_opens_circuit_with_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=10,
                loop_budget=LoopBudgetConfig(
                    max_model_calls=10,
                    max_tool_calls=20,
                    max_wall_seconds=60,
                    max_total_tokens=10_000,
                    no_progress_soft=2,
                    no_progress_hard=3,
                ),
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_circuit", root, data)
            tools = ToolExecutor(
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
            agent = Agent(
                config=config,
                model=NoProgressModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("持续检查目录")

            self.assertEqual(result.reason, "circuit_open")
            self.assertIn("熔断", result.text)
            self.assertGreaterEqual(result.metrics["no_progress_rounds"], 3)
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn("recovery_requested", events_text)
            self.assertIn("circuit_open", events_text)
            receipt_lines = [
                line
                for line in session.path.read_text(encoding="utf-8").splitlines()
                if '"type": "receipt"' in line
            ]
            self.assertTrue(receipt_lines)
            self.assertIn("circuit_open", receipt_lines[-1])
            self.assertIn('"metrics"', receipt_lines[-1])
            memory.close()

    def test_reserved_finalization_returns_result_at_model_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
                loop_budget=LoopBudgetConfig(
                    max_model_calls=4,
                    max_tool_calls=20,
                    max_wall_seconds=60,
                    max_total_tokens=10_000,
                    finalization_reserve=2,
                    no_progress_soft=10,
                    no_progress_hard=11,
                ),
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_finalize", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = FinalizingResultModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("持续检查目录")

            self.assertEqual(result.reason, "partial_result")
            self.assertEqual(model.calls, 4)
            self.assertIn("submit_result", model.tool_name_sets[2])
            self.assertEqual(model.tool_name_sets[2], {"submit_result"})
            self.assertIn("部分结论", result.text)
            self.assertIn("缺少精确 tokenizer", result.text)
            self.assertIn("finalization_started", events.path.read_text(encoding="utf-8"))
            self.assertIn("finalization_tool_blocked", events.path.read_text(encoding="utf-8"))
            self.assertFalse(result.metrics["context"]["exact"])
            self.assertGreater(
                result.metrics["context"]["request"]["estimated_input_tokens"],
                0,
            )
            memory.close()

    def test_submit_result_is_accepted_after_token_soft_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=6,
                loop_budget=LoopBudgetConfig(
                    max_model_calls=6,
                    max_tool_calls=20,
                    max_wall_seconds=60,
                    max_total_tokens=100,
                    warning_ratio=0.8,
                    finalization_reserve=2,
                    no_progress_soft=10,
                    no_progress_hard=11,
                ),
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_token_budget", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            agent = Agent(
                config=config,
                model=TokenBudgetFinalizingModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("分析架构")

            self.assertEqual(result.reason, "blocked")
            self.assertIn("Token 预算已收口", result.text)
            self.assertGreaterEqual(result.metrics["total_tokens"], 130)
            self.assertIn("finalization_started", events.path.read_text(encoding="utf-8"))
            memory.close()

    def test_memory_token_question_uses_builtin_metrics_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            memory.add(
                content="这是一条会被检索并送入上下文的相关记忆。",
                scope=str(root.resolve()),
                kind="semantic",
                status="supported",
            )
            events = EventBus(data)
            session = SessionLogger("s_metrics_question", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = MemoryMetricsQuestionModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("本地记忆每次送入上下文的 token 大概有多少？")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(model.calls, 2)
            self.assertEqual(
                [trace["name"] for trace in result.tool_traces],
                ["memory_context_metrics", "submit_result"],
            )
            self.assertIn("Token 为估算值", result.text)
            self.assertFalse(result.metrics["context"]["exact"])
            self.assertGreater(
                result.metrics["context"]["memory"]["recall"]["items"],
                0,
            )
            memory.close()

    def test_submit_result_message_is_paired_before_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_pair", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            agent = Agent(
                config=config,
                model=SubmitThenContinueModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            first = agent.run_turn("第一轮")
            second = agent.run_turn("第二轮")

            self.assertEqual(first.reason, "completed")
            self.assertEqual(second.reason, "completed")
            self.assertEqual(second.text, "第二轮继续成功。")
            self.assertIn("call_submit_first", session.path.read_text(encoding="utf-8"))
            memory.close()

    def test_context_compaction_keeps_early_requirement_and_recent_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=4,
                loop_budget=LoopBudgetConfig(
                    context_soft_tokens=180,
                    context_hard_tokens=260,
                    context_keep_recent=4,
                    context_summary_chars=4_000,
                ),
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_compact", root, data)
            tools = ToolExecutor(
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
            agent = Agent(
                config=config,
                model=StructuredResultModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            agent.messages.extend(
                [
                    {"role": "user", "content": "必须保留：只修改 src/。"},
                    {"role": "assistant", "content": "已记录边界。"},
                ]
            )
            for index in range(6):
                agent.messages.extend(
                    [
                        {
                            "role": "user",
                            "content": f"第 {index} 轮：" + ("检查证据" * 100),
                        },
                        {
                            "role": "assistant",
                            "content": f"第 {index} 轮完成：" + ("已检查" * 100),
                        },
                    ]
                )
            agent.messages.append(
                {"role": "user", "content": "最新要求：只输出最终结论。"}
            )

            agent._compact_context([], turn_id="t_compact")

            self.assertIn("只修改 src/", agent._conversation_summary)
            self.assertEqual(agent.messages[-1]["content"], "最新要求：只输出最终结论。")
            self.assertIn(
                "context_compacted",
                events.path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "context_checkpoint",
                session.path.read_text(encoding="utf-8"),
            )
            agent.restore_messages(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_restored",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"missing.txt"}',
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": "恢复后继续"},
                ],
                summary="- 旧上下文摘要",
            )
            self.assertEqual(
                [message["role"] for message in agent.messages],
                ["system", "assistant", "tool", "user"],
            )
            self.assertEqual(agent.messages[2]["tool_call_id"], "call_restored")
            self.assertIn("旧上下文摘要", agent._system_context)
            memory.close()


if __name__ == "__main__":
    unittest.main()

