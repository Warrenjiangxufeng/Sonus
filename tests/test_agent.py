from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sonus.agent import Agent
from sonus.config import LoopBudgetConfig, RuntimeConfig
from sonus.events import EventBus
from sonus.memory import MemoryStore
from sonus.model import ModelCancelled, ModelReply
from sonus.policy import ApprovalManager, PathGuard
from sonus.storage import ArtifactStore, SessionLogger
from sonus.tools import ToolExecutor


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
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                finish_reason="tool_calls",
            )
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        assert tool_messages and "sonus test" in tool_messages[-1]["content"]
        return ModelReply(
            message={"role": "assistant", "content": "文件内容已确认。"},
            usage={"prompt_tokens": 8, "completion_tokens": 4},
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
            (root / "note.txt").write_text("sonus test", encoding="utf-8")
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
            self.assertEqual(model.last_max_tokens, 2048)
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
            self.assertIn("Sonus 内部错误", result.text)
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

