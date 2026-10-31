from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sunus.memory import MemoryStore
from sunus.model import BaseModelClient, ModelReply
from sunus.reasoning_traces import (
    record_reasoning_trace,
    record_strategy_reflection,
    seed_reasoning_exemplars,
)


class ReasoningTraceTests(unittest.TestCase):
    def test_model_reflection_parses_candidate_strategy(self) -> None:
        class ReflectionModel:
            def chat(self, messages, **kwargs):
                return ModelReply(
                    message={
                        "role": "assistant",
                        "content": (
                            "```json\n"
                            '{"strategy":"持续为空的检查应触发证据层切换",'
                            '"when_to_use":["同一目标多次没有状态变化"],'
                            '"avoid_when":["一次网络等待尚未结束"],'
                            '"decisive_evidence":["目标状态发生变化"],'
                            '"unproductive_patterns":["只换同类选择器"],'
                            '"confidence":0.78}\n'
                            "```"
                        ),
                    },
                    usage={"prompt_tokens": 20, "completion_tokens": 10},
                    finish_reason="stop",
                )

        reflection, usage = BaseModelClient.reflect_strategy(
            ReflectionModel(),  # type: ignore[arg-type]
            {"objective": "定位目标"},
        )

        self.assertIn("证据层切换", reflection["strategy"])
        self.assertEqual(reflection["confidence"], 0.78)
        self.assertEqual(usage["prompt_tokens"], 20)

    def test_reflection_is_stored_as_candidate_with_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory_id = record_strategy_reflection(
                memory,
                scope="/project/example",
                receipt_id="r_reflection",
                trace_id="m_trace",
                objective="定位目标并验证结果",
                reason="completed",
                reflection={
                    "strategy": "负结果持续重复时切换证据层",
                    "when_to_use": ["同一假设连续没有增量"],
                    "avoid_when": ["目标状态仍在正常加载"],
                    "decisive_evidence": ["新的状态变化"],
                    "unproductive_patterns": ["只更换同类查询参数"],
                    "confidence": 0.8,
                },
            )

            self.assertIsNotNone(memory_id)
            row = memory.get(str(memory_id))
            self.assertEqual(row["kind"], "strategy")
            self.assertEqual(row["status"], "candidate")
            self.assertIn("reflection", row["tags"])
            self.assertIn("promising", row["tags"])
            self.assertIn("适用条件=同一假设连续没有增量", row["content"])
            self.assertIn('"trace_id": "m_trace"', row["metadata"])
            self.assertIn('"outcome": "untested"', row["metadata"])
            memory.close()

    def test_exemplars_are_seeded_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            first = seed_reasoning_exemplars(memory)
            second = seed_reasoning_exemplars(memory)

            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertTrue(
                all(memory.get(memory_id)["kind"] == "strategy" for memory_id in first)
            )
            memory.close()

    def test_completed_trace_records_real_tool_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory_id = record_reasoning_trace(
                memory,
                scope="/project/example",
                receipt_id="r_completed",
                objective="读取当前页面标题",
                mode="浏览器只读查证",
                capabilities=("command",),
                preferred_tools=("inspect_browser",),
                tool_traces=[
                    {"name": "inspect_browser"},
                    {"name": "submit_result"},
                ],
                blocker="",
                reason="completed",
                operators=("演绎", "批判"),
                contrast_checks=("检查前提和边界是否成立",),
                operator_checkpoints=("锁定当前对象", "证据足够立即停止"),
            )

            self.assertIsNotNone(memory_id)
            row = memory.get(str(memory_id))
            self.assertEqual(row["status"], "candidate")
            self.assertIn("completed", row["tags"])
            self.assertIn('"outcome": "pending"', row["metadata"])
            self.assertIn('"tool_sequence": ["inspect_browser"]', row["metadata"])
            self.assertIn('"operators": ["演绎", "批判"]', row["metadata"])
            self.assertIn(
                '"operator_checkpoints": ["锁定当前对象", "证据足够立即停止"]',
                row["metadata"],
            )
            self.assertIn("复核=检查前提和边界是否成立", row["content"])
            self.assertIn("检查点=锁定当前对象,证据足够立即停止", row["content"])

            memory.resolve_strategy_outcome(
                str(memory_id),
                "positive",
                "用户确认解决",
            )
            resolved = memory.get(str(memory_id))
            self.assertEqual(resolved["status"], "supported")
            self.assertIn("successful", resolved["tags"])
            self.assertIn('"outcome": "positive"', resolved["metadata"])
            memory.close()

    def test_trace_and_turn_episode_do_not_overwrite_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            trace_id = record_reasoning_trace(
                memory,
                scope="/project/example",
                receipt_id="r_shared",
                objective="读取浏览器余额",
                mode="浏览器只读查证",
                capabilities=("command",),
                preferred_tools=("run_command",),
                tool_traces=[{"name": "run_command"}],
                blocker="",
                reason="completed",
            )

            episode_ids = memory.record_turn(
                user_input="读取浏览器余额",
                assistant_text="已通过 pw.sh 读取。",
                scope="/project/example",
                receipt_id="r_shared",
            )

            self.assertIsNotNone(trace_id)
            self.assertNotIn(trace_id, episode_ids)
            trace = memory.get(str(trace_id))
            episode = memory.get(episode_ids[0])
            self.assertEqual(trace["kind"], "strategy")
            self.assertIn("策略轨迹：", trace["content"])
            self.assertEqual(episode["kind"], "episode")
            self.assertIn("pw.sh", episode["content"])
            context = memory.build_context(
                scope="/project/example",
                query="还记得浏览器读取余额的操作吗",
                recall_limit=5,
            )
            recalled_ids = {item["id"] for item in context.recalled}
            self.assertIn(trace_id, recalled_ids)
            self.assertIn(episode["id"], recalled_ids)
            memory.close()

    def test_failed_trace_is_candidate_and_not_reused_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory_id = record_reasoning_trace(
                memory,
                scope="/project/example",
                receipt_id="r_failed",
                objective="继续探测登录页",
                mode="Playwright CLI 工具链",
                capabilities=("command",),
                preferred_tools=("inspect_environment",),
                tool_traces=[{"name": "inspect_environment"}],
                blocker="authorization",
                reason="circuit_open",
            )

            self.assertIsNotNone(memory_id)
            row = memory.get(str(memory_id))
            self.assertEqual(row["status"], "candidate")
            self.assertIn("failed", row["tags"])
            strategies = memory.relevant_strategies(
                "继续探测登录页",
                scope="/project/example",
                limit=2,
            )
            self.assertNotIn(memory_id, {item["id"] for item in strategies})
            memory.close()

    def test_completed_but_corrected_trace_is_not_reused_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            memory_id = record_reasoning_trace(
                memory,
                scope="/project/example",
                receipt_id="r_corrected",
                objective="修复同步失败",
                mode="根因诊断",
                capabilities=("command",),
                preferred_tools=("run_command",),
                tool_traces=[{"name": "run_command"}],
                blocker="",
                reason="completed",
            )
            self.assertIsNotNone(memory_id)
            memory.resolve_strategy_outcome(
                str(memory_id),
                "negative",
                "用户指出没有找到根因",
            )
            row = memory.get(str(memory_id))
            self.assertEqual(row["status"], "candidate")
            self.assertIn("failed", row["tags"])
            self.assertNotIn(
                memory_id,
                {
                    item["id"]
                    for item in memory.relevant_strategies(
                        "修复同步失败",
                        scope="/project/example",
                    )
                },
            )
            memory.close()

    def test_strategy_recall_is_bounded_and_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            for index in range(5):
                memory.add(
                    content=f"策略轨迹：登录页检查路径 {index}",
                    scope="/project/example",
                    kind="strategy",
                    status="supported",
                    tags=("strategy", "successful"),
                )
            memory.add(
                content="登录页需要权限",
                scope="/project/example",
                kind="semantic",
                status="supported",
            )

            strategies = memory.relevant_strategies(
                "登录页检查",
                scope="/project/example",
                limit=2,
            )
            context = memory.build_context(
                scope="/project/example",
                query="登录页检查",
                recall_limit=3,
            )

            self.assertEqual(len(strategies), 2)
            self.assertLessEqual(len(context.recalled), 3)
            self.assertEqual(
                sum(item["kind"] == "strategy" for item in context.recalled),
                2,
            )
            memory.close()


if __name__ == "__main__":
    unittest.main()
