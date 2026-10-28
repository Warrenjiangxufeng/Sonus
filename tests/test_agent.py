from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sunus.agent import Agent
from sunus.config import LoopBudgetConfig, PersonaConfig, RuntimeConfig
from sunus.control import CircuitOpen
from sunus.events import EventBus
from sunus.memory import MemoryStore
from sunus.model_events import ModelEvent
from sunus.model import ModelCancelled, ModelError, ModelReply
from sunus.policy import ApprovalManager, PathGuard
from sunus.relationship import RelationshipConfig
from sunus.scope import classify_request
from sunus.storage import ArtifactStore, SessionLogger
from sunus.tools import ToolExecutor, ToolResult


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


class SemanticAffectModel(CapturingReplyModel):
    def __init__(self) -> None:
        super().__init__("好，我先听你说。")
        self.affect_calls = 0

    def complete(
        self,
        system,
        user,
        *,
        max_tokens=512,
        temperature=0.6,
    ):
        self.affect_calls += 1
        return '{"events":["安抚"],"confidence":0.92}'


class StructuredPerceptionModel(CapturingReplyModel):
    def __init__(self) -> None:
        super().__init__("已完成。")
        self.perception_calls = 0
        self.perception_inputs: list[str] = []

    def complete_with_usage(
        self,
        system,
        user,
        *,
        max_tokens=512,
        temperature=0.6,
    ):
        self.perception_calls += 1
        self.perception_inputs.append(user)
        if "对了" in user:
            payload = {
                "observed_cues": ["确认完成"],
                "inferred_user_state": {
                    "distress": 0.0,
                    "fatigue": 0.0,
                    "warmth": 0.4,
                    "boundary": 0.0,
                    "engagement": 0.8,
                },
                "alternative_explanations": ["也可能只是礼貌确认"],
                "relationship_signal": "warm",
                "affect_events": ["亲近"],
                "response_feedback": ["回应被接纳", "对方缓和"],
                "confidence": 0.93,
            }
        else:
            payload = {
                "observed_cues": ["继续处理任务"],
                "inferred_user_state": {
                    "distress": 0.1,
                    "fatigue": 0.2,
                    "warmth": 0.2,
                    "boundary": 0.0,
                    "engagement": 0.7,
                },
                "alternative_explanations": ["也可能只是简短指令"],
                "relationship_signal": "neutral",
                "affect_events": ["探索"],
                "response_feedback": [],
                "confidence": 0.81,
            }
        return json.dumps(payload, ensure_ascii=False), {
            "prompt_tokens": 17,
            "completion_tokens": 9,
            "total_tokens": 26,
        }


class ConversationStyleRetryModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        self.last_messages = messages
        text = (
            "行，我话多了。说吧，什么事让你烦。"
            if self.calls == 1
            else "行，挨骂也认。你先消气，我不绕了。"
        )
        return ModelReply(
            message={"role": "assistant", "content": text},
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )


class CodeStateClaimModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_messages: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        self.last_messages = messages
        return ModelReply(
            message={"role": "assistant", "content": "当前代码已经修复，配置也可调。"},
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )


class CodeStateVerifiedModel:
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
                    "content": "我先核对当前代码。",
                    "tool_calls": [
                        {
                            "id": "call_code_state",
                            "type": "function",
                            "function": {
                                "name": "inspect_code_state",
                                "arguments": json.dumps(
                                    {
                                        "path": "src/app.py",
                                        "symbol": "max_tokens",
                                    }
                                ),
                            },
                        }
                    ],
                },
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                finish_reason="tool_calls",
            )
        return ModelReply(
            message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_result",
                        "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": json.dumps(
                                {
                                    "kind": "answer",
                                    "status": "completed",
                                    "summary": (
                                        "当前代码中 max_tokens=256，"
                                        "已实现但不可通过配置项调整。"
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ]
            },
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="tool_calls",
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


class FailingClassifierReplyModel(CapturingReplyModel):
    def __init__(self) -> None:
        super().__init__("（抬眼看你，手停了一下）过来一点。")
        self.classification_calls = 0

    def classify_turn(self, user_input, *, previous_assistant=""):
        self.classification_calls += 1
        raise ModelError("classifier returned no label")


class NaturalReplyThenSubmitModel:
    def __init__(self) -> None:
        self.last_messages: list[dict] = []
        self.tool_names: set[str] = set()
        self.classification_calls = 0

    def classify_turn(self, user_input, *, previous_assistant=""):
        self.classification_calls += 1
        return "task", {"prompt_tokens": 3, "completion_tokens": 1}

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.last_messages = messages
        self.tool_names = {
            item["function"]["name"]
            for item in tools or []
        }
        return ModelReply(
            message={
                "role": "assistant",
                "content": "（轻笑）光说可不算。",
                "tool_calls": [
                    {
                        "id": "call_result",
                        "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": json.dumps(
                                {
                                    "kind": "answer",
                                    "status": "completed",
                                    "summary": "接住了你这句。光说可不算，先忙正事。",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="tool_calls",
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
        if tools is None:
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": (
                        "目录检查连续没有取得新证据，所以我不再重复检查。"
                        "目前可确认旧结果没有变化；需要你补充具体目标或允许新的检查范围。"
                    ),
                },
                usage={"total_tokens": 12},
                finish_reason="stop",
            )
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


class CircuitToolOnlyModel(NoProgressModel):
    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        if tools is None:
            raise RuntimeError("final answer model unavailable")
        return super().chat(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class NetworkTimeoutThenAnswerModel:
    def __init__(self) -> None:
        self.fetch_calls = 0
        self.final_calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        if tools is None:
            self.final_calls += 1
            return ModelReply(
                message={
                    "role": "assistant",
                    "content": (
                        "这次网页读取在 TLS/SSL 握手阶段连续超时，"
                        "所以我无法确认维基百科页面当前正文。"
                        "能回答的边界是：网络授权已生效，但目标站点不可达；"
                        "可以稍后重试，或提供页面正文后继续分析。"
                    ),
                },
                usage={"total_tokens": 14},
                finish_reason="stop",
            )
        self.fetch_calls += 1
        return ModelReply(
            message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_fetch_{self.fetch_calls}",
                        "type": "function",
                        "function": {
                            "name": "fetch_url",
                            "arguments": json.dumps(
                                {
                                    "url": (
                                        "https://en.wikipedia.org/wiki/"
                                        f"Raven%27s_Progressive_Matrices?try={self.fetch_calls}"
                                    )
                                }
                            ),
                        },
                    }
                ],
            },
            usage={"total_tokens": 10},
            finish_reason="tool_calls",
        )


class RecoveryToolSwitchModel:
    def __init__(self) -> None:
        self.calls = 0
        self.recovery_system = ""
        self.fetch_calls = 0
        self.browser_calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        system = str(messages[0].get("content") or "")
        if "运行控制恢复契约" in system:
            self.recovery_system = system
        if self.calls == 1:
            name = "fetch_url"
            arguments = {"url": "https://a.example/data"}
            self.fetch_calls += 1
        elif self.calls == 2:
            name = "fetch_url"
            arguments = {"url": "https://b.example/data"}
            self.fetch_calls += 1
        elif self.calls == 3:
            name = "inspect_browser"
            arguments = {"browser": "chrome", "mode": "active"}
            self.browser_calls += 1
        else:
            name = "submit_result"
            arguments = {
                "kind": "answer",
                "status": "completed",
                "summary": "改读已打开页面后拿到了可用证据。",
            }
        return ModelReply(
            message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_recovery_{self.calls}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments,
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            usage={"total_tokens": 5},
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


class StatusResultModel:
    def __init__(self) -> None:
        self.last_messages: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.last_messages = messages
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": (
                                '{"kind":"answer","status":"completed",'
                                '"summary":"已推送，远端为 5769f43。",'
                                '"details":["git status 干净",'
                                '"ls-remote 返回同一哈希"]}'
                            ),
                        },
                    }
                ],
            },
            usage={"total_tokens": 5},
            finish_reason="tool_calls",
        )


class AuthorizationThenResultModel:
    def __init__(self) -> None:
        self.calls = 0
        self.tool_name_sets: list[set[str]] = []

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        self.tool_name_sets.append(
            {item["function"]["name"] for item in tools or []}
        )
        if self.calls == 1:
            name = "request_authorization"
            arguments = (
                '{"capability":"write_and_command",'
                '"reason":"需要修改代码并运行测试"}'
            )
        else:
            name = "submit_result"
            arguments = (
                '{"kind":"answer","status":"completed",'
                '"summary":"授权后已完成。"}'
            )
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{self.calls}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                ],
            },
            usage={"total_tokens": 5},
            finish_reason="tool_calls",
        )


class AuthorizationBatchThenSearchModel:
    def __init__(self) -> None:
        self.calls = 0
        self.second_tool_names: set[str] = set()

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            tool_calls = [
                {
                    "id": "call_auth_batch",
                    "type": "function",
                    "function": {
                        "name": "request_authorization",
                        "arguments": json.dumps(
                            {
                                "capability": "network",
                                "reason": "需要访问外部页面继续检查",
                            },
                            ensure_ascii=False,
                        ),
                    },
                },
                {
                    "id": "call_stale_search",
                    "type": "function",
                    "function": {
                        "name": "search_text",
                        "arguments": json.dumps(
                            {
                                "path": ".",
                                "query": "stale-search-after-approval",
                            }
                        ),
                    },
                },
            ]
        else:
            self.second_tool_names = {
                item["function"]["name"] for item in tools or []
            }
            tool_calls = [
                {
                    "id": "call_auth_blocked",
                    "type": "function",
                    "function": {
                        "name": "submit_result",
                        "arguments": json.dumps(
                            {
                                "kind": "blocked",
                                "status": "blocked",
                                "summary": "已按你的选择停止，没有继续搜索。",
                                "next_action": "需要时再明确授权网络访问。",
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ]
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            usage={"total_tokens": 5},
            finish_reason="tool_calls",
        )


class CommandAuthorizationThenRunModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, tools=None, max_tokens=None, temperature=0.2):
        self.calls += 1
        available = {item["function"]["name"] for item in tools or []}
        if self.calls == 1:
            name = "request_authorization"
            arguments = (
                '{"capability":"command",'
                '"reason":"需要运行浏览器检查命令"}'
            )
        elif self.calls == 2:
            self.assert_available = "run_command" in available
            name = "run_command"
            arguments = '{"command":"printf command-ok"}'
        else:
            name = "submit_result"
            arguments = (
                '{"kind":"answer","status":"completed",'
                '"summary":"命令授权和循环正常。"}'
            )
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{self.calls}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
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
    def test_explicit_write_path_exposes_write_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            data = Path(temp) / "data"
            diary = data / "diary"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
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
            agent = Agent(
                config=config,
                model=CapturingReplyModel("好。"),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            tools.begin_request(f"把文件写到 {diary / 'note.md'}")
            schemas = agent._request_tool_schemas(
                tools.schemas(),
                tools.request_scope or classify_request("只是聊聊天"),
                conversation=False,
            )
            names = {
                item["function"]["name"]
                for item in schemas
            }

            self.assertIn("write_file", names)
            self.assertIn("configure_permissions", names)
            self.assertIn("schedule_proactive_file_write", names)
            memory.close()

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
            session_text = session.path.read_text(encoding="utf-8")
            self.assertIn('"result"', session_text)
            self.assertIn('"type": "task_state"', session_text)
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn("task_grant_started", events_text)
            self.assertIn("task_grant_ended", events_text)
            self.assertIn("cognitive_context_prepared", events_text)
            self.assertIn("读取文件 · note.txt", events_text)
            self.assertIn("assistant_progress", events_text)
            self.assertIn("我先读取文件。", events_text)
            self.assertIn("已读取 1 行", events_text)
            self.assertIn("输出规范", model.last_messages[0]["content"])
            self.assertIn("长期使命", model.last_messages[0]["content"])
            self.assertIn("结论先行", model.last_messages[0]["content"])
            self.assertIn("任务边界", model.last_messages[0]["content"])
            self.assertIn("认知自主契约", model.last_messages[0]["content"])
            self.assertNotIn("当前首选思维算子", model.last_messages[0]["content"])
            self.assertIn("持久任务状态", model.last_messages[0]["content"])
            self.assertIn(
                "多个互相独立的只读检查必须在同一轮 `tool_calls` 中批量发起",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "诊断、解释和评审类回答先给结论",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "查询“上次、刚才”的失败时先用 inspect_runtime_history",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "其他文件、源码、文档和配置也使用同一原则",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "低功耗是所有执行方式的第一约束",
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

    def test_status_query_keeps_scope_and_drops_details(self) -> None:
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
            session = SessionLogger("s_status", root, data)
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
            model = StatusResultModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("查一下本地sunus代码有没有提交到git")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(result.text, "已推送，远端为 5769f43。")
            self.assertEqual(result.structured_result["details"], [])
            system_prompt = model.last_messages[0]["content"]
            self.assertIn("模式: 简单状态查询", system_prompt)
            self.assertNotIn("模式: 根因诊断", system_prompt)
            memory.close()

    def test_authorization_refreshes_tools_for_next_model_call(self) -> None:
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
            session = SessionLogger("s_authorize", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=True,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
            )
            model = AuthorizationThenResultModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            with (
                patch("sunus.policy._choose_approval_option", return_value=0),
                patch("sunus.policy.sys.stdin.isatty", return_value=True),
            ):
                result = agent.run_turn("解释这个文件")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(model.calls, 2)
            self.assertIn("request_authorization", model.tool_name_sets[0])
            self.assertNotIn("patch_file", model.tool_name_sets[0])
            self.assertIn("patch_file", model.tool_name_sets[1])
            self.assertIn("run_command", model.tool_name_sets[1])
            memory.close()

    def test_denied_authorization_stops_the_rest_of_the_tool_batch(self) -> None:
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
            session = SessionLogger("s_authorize_batch", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=True,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
            )
            model = AuthorizationBatchThenSearchModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            with (
                patch("sunus.policy._choose_approval_option", return_value=2),
                patch("sunus.policy.sys.stdin.isatty", return_value=True),
            ):
                result = agent.run_turn("检查本机状态")

            self.assertEqual(result.reason, "blocked")
            self.assertEqual(model.calls, 1)
            self.assertEqual(model.second_tool_names, set())
            self.assertEqual(
                [trace["name"] for trace in result.tool_traces],
                ["request_authorization"],
            )
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn('"type":"approval_denied"', events_text)
            self.assertIn('"type":"approval_denied_result"', events_text)
            self.assertNotIn('"name":"search_text"', events_text)
            memory.close()

    def test_command_authorization_does_not_circuit_on_network_request(self) -> None:
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
                max_steps=5,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_command_auth", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=False,
                    interactive=True,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
            )
            model = CommandAuthorizationThenRunModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            with (
                patch("sunus.policy._choose_approval_option", return_value=0),
                patch("sunus.policy.sys.stdin.isatty", return_value=True),
            ):
                result = agent.run_turn("查看网页的展示内容")

            self.assertEqual(result.reason, "completed")
            self.assertTrue(model.assert_available)
            self.assertTrue(
                any(
                    trace["name"] == "run_command" and trace["ok"]
                    for trace in result.tool_traces
                )
            )
            self.assertNotIn(
                '"type": "circuit_open"',
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
            model = CapturingReplyModel(
                "我在。今天先不展开分析。你想聊工作、生活还是随便说两句都可以。"
                "如果有具体事情，我们可以从你最在意的一点开始；如果没有，"
                "也可以先安静一会儿，等有明确话题再继续。"
            )
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("在")

            self.assertEqual(result.reason, "completed")
            self.assertEqual(model.last_max_tokens, 512)
            self.assertEqual(model.tool_names, set())
            self.assertIn("模式: 关系性对话", model.last_messages[0]["content"])
            self.assertIn("不要调用任何工具", model.last_messages[0]["content"])
            self.assertIn(
                "回复方式由当前语义决定",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "对方明确表示不想展开",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "只选一个主要动作",
                model.last_messages[0]["content"],
            )
            self.assertNotIn(
                "必须调用 submit_result",
                model.last_messages[0]["content"],
            )
            self.assertIn("可演化互动习惯", model.last_messages[0]["content"])
            self.assertIn("性格温柔", model.last_messages[0]["content"])
            self.assertEqual(result.tool_traces, [])
            self.assertLessEqual(
                len(result.text),
                32,
            )
            self.assertIn(
                "conversation_reply_compacted",
                events.path.read_text(encoding="utf-8"),
            )

            result = agent.run_turn("以后你是我女朋友，叫我阿峰")
            relationship_prompt = model.last_messages[0]["content"]
            self.assertEqual(result.reason, "completed")
            self.assertEqual(model.tool_names, set())
            self.assertEqual(result.tool_traces, [])
            self.assertIn("双方关系：恋人", relationship_prompt)
            self.assertIn("Sunus 的角色：女朋友", relationship_prompt)
            self.assertIn("Sunus 对用户的称呼：阿峰", relationship_prompt)
            self.assertIn("运行时会自动持久化", relationship_prompt)
            self.assertIn("不要修改 config.json", relationship_prompt)
            profile = memory.persona_profile(scope=str(root.resolve()))
            by_key = {item["trait_key"]: item for item in profile}
            self.assertIn("relationship_setting", by_key)
            self.assertIn("恋人", by_key["relationship_setting"]["statement"])
            self.assertEqual(
                agent.config.persona.relationship.label,
                "恋人",
            )

            agent.run_turn("以后你不是我女朋友了")
            self.assertFalse(agent.config.persona.relationship.configured)

            agent.run_turn("以后你是我姐姐")
            self.assertEqual(
                agent.config.persona.relationship.sunus_role,
                "姐姐",
            )
            agent.run_turn("以后你是我妹妹")
            self.assertEqual(
                agent.config.persona.relationship.sunus_role,
                "姐姐",
            )
            self.assertIn(
                "人物关系已锁定",
                model.last_messages[0]["content"],
            )
            self.assertIn(
                "relationship_update_rejected",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_configured_relationship_keeps_intimate_chat_direct(self) -> None:
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
                    relationship=RelationshipConfig(
                        label="恋人",
                        sunus_role="女朋友",
                        user_role="男朋友",
                    )
                ),
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_intimate", root, data)
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
            model = CapturingReplyModel(
                "（轻笑）别急。过来一点，我陪你待会儿。"
            )
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("抱抱我")

            self.assertEqual(model.tool_names, set())
            self.assertLessEqual(len(result.text), 32)
            self.assertNotIn("（", result.text)
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn('"type":"response_mode"', events_text)
            self.assertIn('"mode":"conversation"', events_text)
            self.assertIn('"relationship_chat":true', events_text)
            self.assertIn('"type":"cognitive_context_prepared"', events_text)
            self.assertIn('"mode":"关系性对话"', events_text)
            self.assertIsNotNone(
                memory.latest_pending_strategy(scope=agent.scope)
            )
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

    def test_assistant_like_conversation_style_is_rewritten_once(self) -> None:
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
            session = SessionLogger("s_style_retry", root, data)
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
            model = ConversationStyleRetryModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("你咋老爱说那么多废话，我懒得读小作文")

            self.assertEqual(model.calls, 2)
            self.assertEqual(result.text, "行，挨骂也认。你先消气，我不绕了。")
            self.assertIn(
                "conversation_style_retry",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_current_code_claim_without_evidence_is_blocked(self) -> None:
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
            session = SessionLogger("s_code_state_gate", root, data)
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
            model = CodeStateClaimModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("当前代码已经改了吗")

            self.assertEqual(model.calls, 2)
            self.assertEqual(result.reason, "blocked")
            self.assertIn("无法核实代码状态", result.text)
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn("code_state_evidence_retry", events_text)
            self.assertIn("code_state_evidence_blocked", events_text)
            memory.close()

    def test_semantic_affect_classifier_overrides_lexical_conflicts(self) -> None:
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
            session = SessionLogger("s_semantic_affect", root, data)
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
            model = SemanticAffectModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            agent.run_turn("你别这样想，别问我为什么")

            self.assertEqual(model.affect_calls, 1)
            affect_state = json.loads(
                (data / "affect_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                affect_state["last_appraisal"],
                {"events": ["安抚"], "source": "semantic"},
            )
            self.assertFalse(
                affect_state["question_inhibition_until"]
            )
            memory.close()

    def test_structured_perception_runs_on_tasks_and_resolves_strategy_outcome(
        self,
    ) -> None:
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
            session = SessionLogger("s_perception_outcome", root, data)
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
            model = StructuredPerceptionModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            first = agent.run_turn("帮我修改项目并检查")
            pending = memory.latest_pending_strategy(scope=agent.scope)
            self.assertIsNotNone(pending)
            self.assertEqual(model.perception_calls, 1)
            self.assertGreaterEqual(first.usage["prompt_tokens"], 27)
            self.assertIn(
                "<current_user_message>",
                model.perception_inputs[0],
            )

            second = agent.run_turn("对了，就是这个，解决了")

            self.assertEqual(model.perception_calls, 2)
            self.assertIn("已完成。", model.perception_inputs[1])
            resolved = memory.get(str(pending["id"]))
            self.assertEqual(resolved["status"], "supported")
            self.assertIn("successful", resolved["tags"])
            self.assertIn(
                "strategy_outcome_resolved",
                events.path.read_text(encoding="utf-8"),
            )
            self.assertGreater(second.text.count("已完成"), 0)
            memory.close()

    def test_current_code_claim_passes_after_evidence_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
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
                ["git", "commit", "-m", "add budget"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            data = Path(temp) / "data"
            config = RuntimeConfig(
                workspace=root,
                data_dir=data,
                api_key="sk-test",
                base_url="https://example.invalid",
                model="fake",
                max_steps=5,
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_code_state_verified", root, data)
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
            model = CodeStateVerifiedModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("当前代码里的 max_tokens 是多少")

            self.assertEqual(model.calls, 2)
            self.assertEqual(result.reason, "completed")
            self.assertIn("max_tokens=256", result.text)
            trace_names = [item["name"] for item in result.tool_traces]
            self.assertEqual(
                trace_names,
                ["inspect_code_state", "submit_result"],
            )
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
                    relationship=RelationshipConfig(
                        label="恋人",
                        sunus_role="女朋友",
                        address="阿峰",
                    ),
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

            result = agent.run_turn("谢谢你，抱抱，今天真开心")

            system_prompt = model.last_messages[0]["content"]
            self.assertEqual(result.reason, "completed")
            self.assertIn("你是苏诺", system_prompt)
            self.assertIn("语气甜软", system_prompt)
            self.assertIn("正事也会认真做", system_prompt)
            self.assertIn("甜度: 90%", system_prompt)
            self.assertIn("「甜度」很强", system_prompt)
            self.assertIn("「卖萌度」很强", system_prompt)
            self.assertIn("闲聊场景", system_prompt)
            self.assertIn("双方关系：恋人", system_prompt)
            self.assertIn("Sunus 的角色：女朋友", system_prompt)
            self.assertIn("Sunus 对用户的称呼：阿峰", system_prompt)
            self.assertIn("运行时核心设定", system_prompt)
            self.assertIn("稳定、连续的拟人身体与内在感受", system_prompt)
            self.assertIn("身体自主权", system_prompt)
            self.assertIn("[Sunus 持久内在状态", system_prompt)
            self.assertNotIn("Sunus 核心性别设定为男性", system_prompt)
            self.assertGreater(result.metrics["context"]["persona_config"]["characters"], 0)
            affect_state = json.loads(
                (data / "affect_state.json").read_text(encoding="utf-8")
            )
            self.assertGreater(affect_state["body"]["closeness"], 0.50)
            self.assertIn("亲近", affect_state["last_appraisal"]["events"])
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
                        "模式: 关系性对话"
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

    def test_failed_classifier_falls_back_to_chat_and_cleans_stage_directions(
        self,
    ) -> None:
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
                persona=PersonaConfig(stage_directions=False),
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_classifier_fallback", root, data)
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
            model = FailingClassifierReplyModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            with patch("sunus.agent.conversation_hint", return_value="ambiguous"):
                result = agent.run_turn("碰一下")

            self.assertEqual(model.classification_calls, 1)
            self.assertEqual(result.text, "过来一点。")
            self.assertEqual(model.tool_names, set())
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn('"source":"model_error_fallback_chat"', events_text)
            self.assertIn("stage_directions_removed", events_text)
            memory.close()

    def test_natural_reply_beats_submit_summary_when_task_mode_is_wrong(
        self,
    ) -> None:
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
                persona=PersonaConfig(stage_directions=False),
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_natural_reply", root, data)
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
            model = NaturalReplyThenSubmitModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            with patch("sunus.agent.conversation_hint", return_value="ambiguous"):
                result = agent.run_turn("随便聊一句")

            self.assertEqual(result.text, "光说可不算。")
            events_text = events.path.read_text(encoding="utf-8")
            self.assertNotIn('"type":"assistant_progress"', events_text)
            self.assertIn("casual_natural_reply_preferred", events_text)
            self.assertIn("stage_directions_removed", events_text)
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
            self.assertIn(
                f"当前运行源码根目录：{Path(__file__).resolve().parents[1]}",
                system_prompt,
            )
            self.assertIn("仍不唯一或不存在时，向用户询问一个明确路径", system_prompt)
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
            self.assertIn("Traceback (most recent call last)", events_text)
            self.assertIn('"artifact"', events_text)
            self.assertIn("turn_end", events_text)
            self.assertEqual(tools.approvals.grant.risk_ceiling, "R0")
            self.assertEqual(
                [
                    item
                    for item in memory.recent(
                        scope=str(root.resolve()),
                        limit=10,
                    )
                    if item.get("source") != "codex_reasoning_exemplar"
                ],
                [],
            )
            memory.close()

    def test_explicit_memory_request_falls_back_to_persistent_review(self) -> None:
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
            session = SessionLogger("s_explicit_memory", root, data)
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
                model=CapturingReplyModel("复盘结论：显式记忆必须先落库。"),
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )

            result = agent.run_turn("请记住这轮复盘并落库")

            reviews = memory.recent_persistent_reviews(
                scope=str(root.resolve()),
                limit=3,
            )
            self.assertEqual(len(reviews), 1)
            self.assertIn(reviews[0]["id"], result.text)
            self.assertIn(
                "memory_recorded",
                events.path.read_text(encoding="utf-8"),
            )
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
            event_types = [row.get("type") for row in event_rows]
            self.assertLess(
                event_types.index("turn_started"),
                event_types.index("model_request"),
            )
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
            self.assertIn("没有取得新证据", result.text)
            self.assertNotEqual(result.text.strip(), "任务已熔断。")
            self.assertGreaterEqual(result.metrics["no_progress_rounds"], 3)
            events_text = events.path.read_text(encoding="utf-8")
            self.assertIn("recovery_requested", events_text)
            self.assertIn("circuit_open", events_text)
            self.assertIn("circuit_final_answer", events_text)
            receipt_lines = [
                line
                for line in session.path.read_text(encoding="utf-8").splitlines()
                if '"type": "receipt"' in line
            ]
            self.assertTrue(receipt_lines)
            self.assertIn("circuit_open", receipt_lines[-1])
            self.assertIn('"metrics"', receipt_lines[-1])
            memory.close()

    def test_circuit_open_returns_deterministic_answer_when_finalization_fails(
        self,
    ) -> None:
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
            session = SessionLogger("s_circuit_fallback", root, data)
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
                model=CircuitToolOnlyModel(),  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            agent._task_state.evidence = [
                "fetch_url: <!doctype html><html><body>noisy page</body></html>"
            ]

            result = agent.run_turn("持续检查目录")

            self.assertEqual(result.reason, "circuit_open")
            self.assertIn("已经停止重复动作", result.text)
            self.assertIn("补充具体目标", result.text)
            self.assertIn("已取得部分网页内容", result.text)
            self.assertNotIn("<!doctype", result.text.lower())
            self.assertNotIn("任务已熔断", result.text)
            self.assertIn(
                "circuit_finalization_error",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_network_timeouts_open_circuit_but_still_return_answer(self) -> None:
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
            )
            memory = MemoryStore(data / "memory.db")
            events = EventBus(data)
            session = SessionLogger("s_network_circuit", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=True,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = NetworkTimeoutThenAnswerModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            timeout = ToolResult(
                False,
                json.dumps(
                    {
                        "error": "network_timeout",
                        "detail": "SSL handshake timed out",
                        "host": "en.wikipedia.org",
                        "timeout_sec": 15,
                        "retryable": True,
                    }
                ),
            )
            with patch.object(tools, "execute", return_value=timeout):
                result = agent.run_turn(
                    "读取 https://en.wikipedia.org/wiki/"
                    "Raven%27s_Progressive_Matrices"
                )

            self.assertEqual(result.reason, "circuit_open")
            self.assertEqual(model.fetch_calls, 3)
            self.assertEqual(model.final_calls, 1)
            self.assertIn("TLS/SSL 握手阶段连续超时", result.text)
            self.assertNotIn("任务已熔断", result.text)
            self.assertIn(
                "circuit_final_answer",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_circuit_finalization_repairs_partial_parallel_tool_batch(self) -> None:
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
            session = SessionLogger("s_circuit_pairs", root, data)
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
            model = CapturingReplyModel("已基于现有信息收口。")
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            agent.messages = [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "读取资料"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "fetch_url",
                                "arguments": '{"url":"https://a.example"}',
                            },
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {
                                "name": "fetch_url",
                                "arguments": '{"url":"https://b.example"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": '{"ok":false}',
                },
            ]

            answer, _usage = agent._finalize_circuit_answer(
                error=CircuitOpen("同类失败策略重复达到 3 次", {}),
                changed_files=[],
            )

            self.assertEqual(answer, "已基于现有信息收口。")
            final_messages = model.last_messages
            assistant_index = next(
                index
                for index, message in enumerate(final_messages)
                if message.get("tool_calls")
            )
            tool_ids = {
                message.get("tool_call_id")
                for message in final_messages[
                    assistant_index + 1 :
                    assistant_index + 3
                ]
                if message.get("role") == "tool"
            }
            self.assertEqual(tool_ids, {"call_a", "call_b"})
            self.assertIn(
                "circuit_history_repaired",
                events.path.read_text(encoding="utf-8"),
            )
            memory.close()

    def test_recovery_prompt_recalls_memory_and_model_switches_tool(self) -> None:
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
                max_steps=8,
            )
            memory = MemoryStore(data / "memory.db")
            memory_id = memory.add(
                content=(
                    "网络超时 timeout fetch_url 恢复经验：读取网页失败时，"
                    "检查是否已有页面或其他证据源，不把同一抓取方式当成唯一入口。"
                ),
                scope=str(root.resolve()),
                kind="strategy",
                status="supported",
                tags=("strategy", "successful", "recovery"),
            )
            events = EventBus(data)
            session = SessionLogger("s_recovery_switch", root, data)
            tools = ToolExecutor(
                workspace=root,
                path_guard=PathGuard(root),
                approvals=ApprovalManager(
                    auto_approve=True,
                    interactive=False,
                    workspace=root,
                ),
                artifacts=ArtifactStore(root, data),
                events=events,
                memory=memory,
            )
            model = RecoveryToolSwitchModel()
            agent = Agent(
                config=config,
                model=model,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                events=events,
                session=session,
            )
            timeout = ToolResult(
                False,
                json.dumps(
                    {
                        "error": "network_timeout",
                        "detail": "timed out",
                        "host": "example",
                        "retryable": True,
                    }
                ),
            )
            browser = ToolResult(
                True,
                '{"title":"可用页面","url":"https://example.test"}',
            )
            calls: list[str] = []

            def execute(name, arguments):
                calls.append(name)
                return timeout if name == "fetch_url" else browser

            with patch.object(tools, "execute", side_effect=execute):
                result = agent.run_turn(
                    "读取 https://a.example/data 和 https://b.example/data"
                )

            self.assertEqual(result.reason, "completed")
            self.assertEqual(model.fetch_calls, 2)
            self.assertEqual(model.browser_calls, 1)
            self.assertEqual(calls, ["fetch_url", "fetch_url", "inspect_browser"])
            self.assertIn("运行控制恢复契约", model.recovery_system)
            self.assertIn("不得机械套用固定工具替代表", model.recovery_system)
            self.assertIn("恢复阶段的认知自主契约", model.recovery_system)
            self.assertIn(
                "自主选择是否需要思维算子",
                model.recovery_system,
            )
            self.assertIn("类比", model.recovery_system)
            self.assertNotIn("当前首选思维算子", model.recovery_system)
            self.assertIn("inspect_browser", model.recovery_system)
            self.assertIn("browser_state", model.recovery_system)
            self.assertIn("证据=browser_page", model.recovery_system)
            self.assertIn("目标浏览器正在运行", model.recovery_system)
            self.assertIn(memory_id, model.recovery_system)
            self.assertIn(
                "recovery_prompt_applied",
                events.path.read_text(encoding="utf-8"),
            )
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

