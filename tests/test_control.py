from __future__ import annotations

import unittest
from dataclasses import replace

from sunus.config import LoopBudgetConfig
from sunus.control import CircuitOpen, TurnController


class RecordingEvents:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, **payload):
        self.events.append((event_type, payload))


class TurnControllerTests(unittest.TestCase):
    def test_model_call_budget_opens_circuit(self) -> None:
        controller = TurnController(
            LoopBudgetConfig(
                max_model_calls=2,
                no_progress_soft=5,
                no_progress_hard=6,
                finalization_reserve=0,
            )
        )
        controller.before_model_call()
        controller.before_model_call()
        with self.assertRaises(CircuitOpen) as raised:
            controller.before_model_call()
        self.assertIn("模型调用达到上限 2", raised.exception.reason)

    def test_finalization_uses_reserved_model_calls(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(
                max_model_calls=4,
                finalization_reserve=2,
                no_progress_soft=10,
                no_progress_hard=11,
            ),
            events=events,  # type: ignore[arg-type]
        )
        self.assertIsNone(controller.before_model_call())
        controller.before_tool("list_files", {"path": "."})
        controller.record_tool_result(
            "list_files",
            {"path": "."},
            ok=True,
            content="a.txt",
            changed_files=[],
        )
        self.assertIsNone(controller.before_model_call())
        controller.before_tool("list_files", {"path": "."})
        controller.record_tool_result(
            "list_files",
            {"path": "."},
            ok=True,
            content="b.txt",
            changed_files=[],
        )

        prompt = controller.before_model_call()
        self.assertTrue(controller.finalizing)
        self.assertIn("终态阶段", prompt or "")
        self.assertIn("submit_result", prompt or "")
        self.assertIn(
            "finalization_started",
            [event_type for event_type, _ in events.events],
        )
        self.assertIn(
            "终态阶段",
            controller.before_tool("run_command", {"command": "ls"}) or "",
        )
        self.assertEqual(controller.tool_calls, 2)

    def test_token_soft_limit_enters_finalization_before_hard_limit(self) -> None:
        controller = TurnController(
            LoopBudgetConfig(
                max_model_calls=20,
                max_total_tokens=100,
                warning_ratio=0.8,
                finalization_reserve=2,
            )
        )
        controller.before_model_call()
        controller.record_usage({"total_tokens": 80})

        prompt = controller.before_model_call()

        self.assertTrue(controller.finalizing)
        self.assertIn("Token 预算", prompt or "")

    def test_distinct_evidence_resets_no_progress(self) -> None:
        controller = TurnController(LoopBudgetConfig())
        controller.before_model_call()
        controller.record_tool_result(
            "read_file",
            {"path": "a.txt"},
            ok=True,
            content="alpha",
            changed_files=[],
        )
        controller.before_model_call()
        self.assertEqual(controller.no_progress_rounds, 0)
        controller.record_tool_result(
            "read_file",
            {"path": "b.txt"},
            ok=True,
            content="beta",
            changed_files=[],
        )
        controller.before_model_call()
        self.assertEqual(controller.no_progress_rounds, 0)

    def test_same_error_is_normalized_and_triggers_recovery(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(max_recoveries=2),
            events=events,  # type: ignore[arg-type]
            turn_id="t_test",
            clock=lambda: 0.0,
        )
        controller.start()
        controller.before_model_call()
        controller.before_tool("run_command", {"command": "pytest"})
        controller.record_tool_result(
            "run_command",
            {"command": "pytest"},
            ok=False,
            content=(
                "exit_code=1\n"
                "2026-10-24T12:00:01Z ImportError: missing module x\n"
                "path=/private/tmp/run-1234"
            ),
            changed_files=[],
        )
        controller.before_model_call()
        controller.before_tool("run_command", {"command": "pytest"})
        controller.record_tool_result(
            "run_command",
            {"command": "pytest"},
            ok=False,
            content=(
                "exit_code=1\n"
                "2026-10-24T12:00:09Z ImportError: missing module x\n"
                "path=/private/tmp/run-5678"
            ),
            changed_files=[],
        )

        recovery = controller.before_model_call()
        self.assertIsNotNone(recovery)
        self.assertIn("ImportError", recovery or "")
        self.assertEqual(controller.recovery_count, 1)
        self.assertIn(
            "recovery_requested",
            [event_type for event_type, _ in events.events],
        )
        self.assertIsNotNone(
            controller.before_tool("run_command", {"command": "pytest"})
        )
        controller.record_tool_result(
            "read_file",
            {"path": "README.md"},
            ok=True,
            content="project readme",
            changed_files=[],
        )
        self.assertIsNone(controller.before_model_call())
        self.assertEqual(controller.recovery_count, 1)

    def test_repeated_full_text_search_must_converge(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(
                no_progress_soft=10,
                no_progress_hard=20,
            ),
            events=events,  # type: ignore[arg-type]
        )
        controller.before_model_call()
        for index in range(4):
            args = {"query": f"term-{index}"}
            controller.before_tool("search_text", args)
            controller.record_tool_result(
                "search_text",
                args,
                ok=True,
                content=f"result-{index}",
                changed_files=[],
            )

        checkpoint = controller.before_model_call()
        self.assertIn("检索收束检查点", checkpoint or "")
        self.assertIn(
            "search_checkpoint",
            [event_type for event_type, _payload in events.events],
        )
        self.assertIsNone(
            controller.before_tool("search_text", {"query": "term-5"})
        )
        self.assertIsNone(
            controller.before_tool("search_text", {"query": "term-6"})
        )
        blocked = controller.before_tool(
            "search_text",
            {"query": "term-7"},
        )
        self.assertIn("超过 6 次", blocked or "")

    def test_repeated_tool_failure_is_blocked(self) -> None:
        controller = TurnController(
            LoopBudgetConfig(
                same_error_soft=2,
                same_error_hard=5,
                no_progress_soft=10,
                no_progress_hard=11,
            )
        )
        controller.before_model_call()
        args = {"path": "missing.txt"}
        for _ in range(2):
            controller.before_tool("read_file", args)
            controller.record_tool_result(
                "read_file",
                args,
                ok=False,
                content="FileNotFoundError: missing.txt",
                changed_files=[],
            )
        controller.before_model_call()
        self.assertIn(
            "重复失败",
            controller.before_tool("read_file", args) or "",
        )

    def test_time_and_token_budgets_are_enforced(self) -> None:
        now = [0.0]
        controller = TurnController(
            replace(
                LoopBudgetConfig(),
                max_wall_seconds=10,
                max_total_tokens=100,
            ),
            clock=lambda: now[0],
        )
        controller.before_model_call()
        controller.record_usage({"total_tokens": 101})
        with self.assertRaises(CircuitOpen):
            controller.ensure_within_budget()

    def test_error_fingerprint_ignores_transient_details(self) -> None:
        first = (
            "2026-10-24T12:00:01Z Error: request failed "
            "pid=1234 /private/tmp/run-1234 0xdeadbeef"
        )
        second = (
            "2026-10-24T12:05:09Z Error: request failed "
            "pid=9876 /private/tmp/run-9876 0xcafebabe"
        )
        left = TurnController._error_fingerprint("fetch_url", first)
        right = TurnController._error_fingerprint("fetch_url", second)
        self.assertEqual(left, right)

    def test_error_fingerprint_attributes_failure_before_recovery(self) -> None:
        family, _detail, _label = TurnController._error_fingerprint(
            "run_command",
            "当前用户消息没有明确要求该副作用操作",
        )
        self.assertEqual(family, "authorization")
        self.assertEqual(
            TurnController._error_fingerprint(
                "read_file",
                "FileNotFoundError: no such file",
            )[0],
            "path",
        )
        self.assertEqual(
            TurnController._error_fingerprint(
                "run_command",
                "exit_code=1\nAssertionError: expected 2",
            )[0],
            "verification",
        )

    def test_transient_output_noise_does_not_count_as_progress(self) -> None:
        controller = TurnController(LoopBudgetConfig())
        controller.before_model_call()
        controller.record_tool_result(
            "run_command",
            {"command": "date"},
            ok=True,
            content="ready at 2026-10-24T12:00:01Z pid=1234",
            changed_files=[],
        )
        controller.before_model_call()
        controller.record_tool_result(
            "run_command",
            {"command": "date"},
            ok=True,
            content="ready at 2026-10-24T12:05:01Z pid=9876",
            changed_files=[],
        )
        controller.before_model_call()
        self.assertEqual(controller.no_progress_rounds, 1)


if __name__ == "__main__":
    unittest.main()
