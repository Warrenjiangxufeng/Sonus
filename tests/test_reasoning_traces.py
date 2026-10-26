from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sunus.memory import MemoryStore
from sunus.reasoning_traces import (
    record_reasoning_trace,
    seed_reasoning_exemplars,
)


class ReasoningTraceTests(unittest.TestCase):
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
            )

            self.assertIsNotNone(memory_id)
            row = memory.get(str(memory_id))
            self.assertEqual(row["status"], "supported")
            self.assertIn("successful", row["tags"])
            self.assertIn('"tool_sequence": ["inspect_browser"]', row["metadata"])
            self.assertIn('"operators": ["演绎", "批判"]', row["metadata"])
            self.assertIn("复核=检查前提和边界是否成立", row["content"])
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
