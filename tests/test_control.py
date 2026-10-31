from __future__ import annotations

import json
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

    def test_base_budget_requires_explicit_extension_before_hard_limit(
        self,
    ) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(
                base_model_calls=2,
                max_model_calls=5,
                finalization_reserve=1,
                no_progress_soft=10,
                no_progress_hard=11,
            ),
            events=events,  # type: ignore[arg-type]
        )

        self.assertIsNone(controller.before_model_call())
        self.assertIsNone(controller.before_model_call())
        prompt = controller.before_model_call()

        self.assertTrue(controller.reviewing_budget)
        self.assertIn("request_budget_extension", prompt or "")
        blocked = controller.before_tool("list_files", {"path": "."})
        self.assertIn("只能申请", blocked or "")

        ok, content = controller.request_budget_extension(
            2,
            reason="已确认候选文件，下一步批量读取并比较。",
        )

        self.assertTrue(ok)
        self.assertIn("额外 2 次", content)
        self.assertFalse(controller.reviewing_budget)
        self.assertEqual(controller.effective_model_limit, 4)
        self.assertIn(
            "budget_extension_granted",
            [event_type for event_type, _ in events.events],
        )

    def test_budget_extension_requires_reason_and_respects_hard_limit(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(
                base_model_calls=1,
                max_model_calls=3,
                finalization_reserve=1,
                no_progress_soft=10,
                no_progress_hard=11,
            ),
            events=events,  # type: ignore[arg-type]
        )
        controller.before_model_call()
        controller.before_model_call()

        ok, content = controller.request_budget_extension(
            10,
            reason="",
        )
        self.assertFalse(ok)
        self.assertIn("必须说明", content)
        self.assertTrue(controller.reviewing_budget)

        ok, content = controller.request_budget_extension(
            10,
            reason="剩余工作必须依赖一次新的验证结果。",
        )
        self.assertTrue(ok)
        self.assertIn("额外 1 次", content)
        self.assertEqual(controller.effective_model_limit, 2)

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

    def test_distinct_results_without_state_change_trigger_cognition_checkpoint(
        self,
    ) -> None:
        events = RecordingEvents()
        controller = TurnController(
            replace(
                LoopBudgetConfig(),
                cognitive_checkpoint_after=2,
                cognitive_checkpoint_repeat=1,
            ),
            events=events,  # type: ignore[arg-type]
        )
        controller.before_model_call()
        for index in range(2):
            args = {"command": f"inspect-{index}"}
            controller.before_tool("run_command", args)
            controller.record_tool_result(
                "run_command",
                args,
                ok=True,
                content=f"different output {index}",
                changed_files=[],
            )

        checkpoint = controller.before_model_call()

        self.assertEqual(controller.no_progress_rounds, 0)
        self.assertEqual(controller.steps_without_state_change, 2)
        self.assertIn("认知收束检查点", checkpoint or "")
        self.assertIn("不要把“工具返回了不同内容”", checkpoint or "")
        self.assertEqual(controller.cognitive_checkpoint_count, 1)
        self.assertIn(
            "cognitive_checkpoint",
            [event_type for event_type, _payload in events.events],
        )

    def test_target_state_change_resets_cognition_stall_counter(self) -> None:
        controller = TurnController(
            replace(
                LoopBudgetConfig(),
                cognitive_checkpoint_after=2,
                cognitive_checkpoint_repeat=1,
            )
        )
        controller.before_model_call()
        for index in range(2):
            args = {"command": f"inspect-{index}"}
            controller.before_tool("run_command", args)
            controller.record_tool_result(
                "run_command",
                args,
                ok=True,
                content=f"output {index}",
                changed_files=[],
            )
        args = {"command": "publish"}
        controller.before_tool("run_command", args)
        controller.record_tool_result(
            "run_command",
            args,
            ok=True,
            content="published=true",
            changed_files=[],
            state_changed=True,
        )

        self.assertIsNone(controller.before_model_call())
        self.assertEqual(controller.steps_without_state_change, 0)

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

    def test_identical_successful_action_is_blocked_after_duplicate_evidence(
        self,
    ) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(
                no_progress_soft=10,
                no_progress_hard=20,
            ),
            events=events,  # type: ignore[arg-type]
        )
        args = {"path": "Desktop/sunus"}
        for _ in range(2):
            self.assertIsNone(controller.before_tool("list_files", args))
            controller.record_tool_result(
                "list_files",
                args,
                ok=True,
                content="a.txt\nb.txt",
                changed_files=[],
            )

        blocked = controller.before_tool("list_files", args)

        self.assertIn("继续重复不会增加信息", blocked or "")
        self.assertIn(
            "repeat_evidence_blocked",
            [event_type for event_type, _ in events.events],
        )

    def test_new_evidence_or_state_change_clears_duplicate_block(self) -> None:
        controller = TurnController(
            LoopBudgetConfig(
                no_progress_soft=10,
                no_progress_hard=20,
            )
        )
        args = {"path": "Desktop/sunus"}
        controller.record_tool_result(
            "list_files",
            args,
            ok=True,
            content="a.txt",
            changed_files=[],
        )
        controller.record_tool_result(
            "list_files",
            args,
            ok=True,
            content="a.txt",
            changed_files=[],
        )
        self.assertIsNotNone(controller.before_tool("list_files", args))

        controller.record_tool_result(
            "write_file",
            {"path": "Desktop/sunus/new.txt", "content": "new"},
            ok=True,
            content="已写入",
            changed_files=["Desktop/sunus/new.txt"],
        )

        self.assertIsNone(controller.before_tool("list_files", args))

    def test_failure_output_is_evidence_but_not_progress(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(no_progress_soft=2, no_progress_hard=4),
            events=events,  # type: ignore[arg-type]
        )
        controller.before_model_call()
        controller.record_tool_result(
            "run_command",
            {"command": "first broken probe"},
            ok=False,
            content="Error: first failure",
            changed_files=[],
        )
        controller.before_model_call()

        self.assertEqual(controller.no_progress_rounds, 1)
        event_types = [event_type for event_type, _ in events.events]
        self.assertNotIn("progress_marked", event_types)
        self.assertIn("diagnostic_evidence", event_types)

    def test_varied_runtime_errors_share_one_failure_strategy(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(
                same_error_soft=2,
                same_error_hard=3,
                no_progress_soft=20,
                no_progress_hard=21,
                max_recoveries=2,
            ),
            events=events,  # type: ignore[arg-type]
            clock=lambda: 0.0,
        )
        controller.before_model_call()
        controller.record_tool_result(
            "run_command",
            {"command": "playwright-cli --help"},
            ok=False,
            content="SyntaxError: Unexpected token '??='",
            changed_files=[],
        )
        controller.before_model_call()
        controller.record_tool_result(
            "run_command",
            {"command": "node18 playwright-cli --help"},
            ok=False,
            content="Playwright requires Node.js 20 or higher.",
            changed_files=[],
        )

        recovery = controller.before_model_call()
        self.assertIn("同类失败策略", recovery or "")
        self.assertIn("运行时或依赖版本不兼容", recovery or "")
        self.assertEqual(controller.recovery_count, 1)
        with self.assertRaises(CircuitOpen) as raised:
            controller.record_tool_result(
                "run_command",
                {"command": "node18 shim playwright-cli"},
                ok=False,
                content="TypeError: URL.canParse is not a function",
                changed_files=[],
            )
        self.assertIn("同类失败策略重复达到 3 次", raised.exception.reason)
        self.assertEqual(len(controller.errors), 3)
        self.assertEqual(len(controller.failure_strategies), 1)

    def test_browser_failures_with_different_causes_stay_separate(self) -> None:
        controller = TurnController(
            LoopBudgetConfig(
                same_error_soft=2,
                same_error_hard=3,
                no_progress_soft=20,
                no_progress_hard=21,
            )
        )
        controller.before_model_call()
        failures = (
            (
                {"browser": "chrome", "mode": "active"},
                {
                    "ok": False,
                    "browser": "chrome",
                    "error": "browser_access_failed",
                    "detail": "不能获得 window 1；无效的索引",
                },
            ),
            (
                {"browser": "safari", "mode": "active"},
                {
                    "ok": False,
                    "browser": "safari",
                    "error": "browser_not_running",
                    "detail": "Safari 未运行",
                },
            ),
            (
                {"browser": "chrome", "mode": "content"},
                {
                    "ok": False,
                    "browser": "chrome",
                    "error": "browser_access_failed",
                    "detail": "通过 AppleScript 执行 JavaScript 的功能已关闭",
                },
            ),
        )
        for arguments, payload in failures:
            controller.record_tool_result(
                "inspect_browser",
                arguments,
                ok=False,
                content=json.dumps(payload, ensure_ascii=False),
                changed_files=[],
            )

        self.assertEqual(len(controller.failure_strategies), 3)
        self.assertTrue(
            all(
                record.count == 1
                for record in controller.failure_strategies.values()
            )
        )

    def test_external_state_change_resets_failure_strategy(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            LoopBudgetConfig(
                same_error_soft=2,
                same_error_hard=3,
                no_progress_soft=20,
                no_progress_hard=21,
            ),
            events=events,  # type: ignore[arg-type]
        )
        payload = json.dumps(
            {
                "ok": False,
                "browser": "chrome",
                "error": "browser_access_failed",
                "detail": "不能获得 window 1；无效的索引",
            },
            ensure_ascii=False,
        )
        for _ in range(2):
            controller.record_tool_result(
                "inspect_browser",
                {"browser": "chrome", "mode": "active"},
                ok=False,
                content=payload,
                changed_files=[],
            )
        self.assertTrue(controller.failure_strategies)

        controller.record_tool_result(
            "run_command",
            {"command": "osascript -e 'make new window'"},
            ok=True,
            content="已打开 DeepSeek 页面",
            changed_files=[],
            state_changed=True,
        )

        self.assertEqual(controller.failure_strategies, {})
        self.assertIn(
            "failure_state_reset",
            [event_type for event_type, _ in events.events],
        )

    def test_state_change_resets_previous_failure_strategy(self) -> None:
        controller = TurnController(
            LoopBudgetConfig(
                same_error_soft=2,
                same_error_hard=3,
                no_progress_soft=20,
                no_progress_hard=21,
            )
        )
        controller.before_model_call()
        for _ in range(2):
            controller.record_tool_result(
                "run_command",
                {"command": "pytest"},
                ok=False,
                content="exit_code=1\nAssertionError: first",
                changed_files=[],
            )
        controller.record_tool_result(
            "patch_file",
            {"path": "code.py"},
            ok=True,
            content="patched",
            changed_files=["code.py"],
        )

        self.assertEqual(controller.failure_strategies, {})
        self.assertEqual(controller.errors, {})

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

    def test_structured_network_timeout_has_stable_strategy_without_target(
        self,
    ) -> None:
        first = json.dumps(
            {
                "error": "network_timeout",
                "detail": "handshake timed out after 7 seconds",
                "host": "en.wikipedia.org",
            }
        )
        second = json.dumps(
            {
                "error": "network_timeout",
                "detail": "timed out after 9 seconds",
                "host": "example.org",
            }
        )
        left = TurnController._failure_strategy_key(
            "fetch_url",
            "timeout",
            "timed out",
            first,
        )
        right = TurnController._failure_strategy_key(
            "fetch_url",
            "timeout",
            "timed out",
            second,
        )
        self.assertEqual(left, right)
        self.assertIn("network-timeout", left)

    def test_recovery_blocks_exact_failed_actions_but_allows_new_paths(self) -> None:
        events = RecordingEvents()
        controller = TurnController(
            replace(
                LoopBudgetConfig(),
                same_error_soft=2,
                same_error_hard=4,
            ),
            events=events,  # type: ignore[arg-type]
        )
        controller.before_model_call()
        controller.record_tool_result(
            "fetch_url",
            {"url": "https://a.example"},
            ok=False,
            content=(
                '{"error":"network_timeout","detail":"timed out",'
                '"host":"a.example"}'
            ),
            changed_files=[],
        )
        controller.before_model_call()
        controller.record_tool_result(
            "fetch_url",
            {"url": "https://b.example"},
            ok=False,
            content=(
                '{"error":"network_timeout","detail":"timed out",'
                '"host":"b.example"}'
            ),
            changed_files=[],
        )
        recovery = controller.before_model_call()

        self.assertIsNotNone(recovery)
        prompt = recovery or ""
        self.assertIn("由你结合用户目标", prompt)
        self.assertIn("不得机械套用固定工具替代表", prompt)
        self.assertNotIn("inspect_browser", prompt)
        self.assertNotIn("request_authorization", prompt)
        self.assertIn("https://a.example", prompt)
        self.assertIn("https://b.example", prompt)
        self.assertIsNotNone(
            controller.before_tool(
                "fetch_url",
                {"url": "https://a.example"},
            )
        )
        self.assertIsNotNone(
            controller.before_tool(
                "fetch_url",
                {"url": "https://b.example"},
            )
        )
        self.assertIsNone(
            controller.before_tool(
                "fetch_url",
                {"url": "https://c.example"},
            )
        )
        self.assertIsNone(
            controller.before_tool(
                "inspect_browser",
                {"browser": "chrome", "mode": "active"},
            )
        )

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
