from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sunus.scope import classify_request
from sunus.storage import SessionLogger, load_session
from sunus.task_state import TaskState


class TaskStateTests(unittest.TestCase):
    def test_followup_preserves_objective_and_constraints(self) -> None:
        state = TaskState()
        first = "查看网页展示内容，不是让你自己算"
        first_scope = classify_request(first)
        state.update(first, first_scope)
        state.update("按刚才的来", first_scope)

        self.assertEqual(state.objective, first)
        self.assertTrue(
            any("不是让你自己算" in item for item in state.constraints)
        )
        self.assertIn("按刚才的来", state.clarifications)
        self.assertIn("command", state.required_capabilities)
        self.assertNotIn("network", state.required_capabilities)
        self.assertEqual(state.turn_count, 2)
        self.assertEqual(state.current_subgoal, first)
        self.assertTrue(state.success_criteria)
        self.assertIn("证据", state.evidence_gap)

    def test_tool_outcomes_update_evidence_and_blocker(self) -> None:
        state = TaskState(objective="修改 cli.py")
        state.record_tool(
            "read_file",
            {"path": "cli.py"},
            ok=True,
            content="content",
            changed_files=[],
        )
        state.record_tool(
            "patch_file",
            {"path": "cli.py"},
            ok=False,
            content="当前用户消息没有明确要求该副作用操作",
            changed_files=[],
        )

        self.assertIn("read_file", state.evidence[-1])
        self.assertEqual(state.blocker, "authorization")
        self.assertIn("当前用户消息没有明确要求", state.last_error)
        self.assertEqual(state.phase, "blocked")
        self.assertGreaterEqual(state.stalled_steps, 1)
        self.assertEqual(state.evidence_layers, ["read_file", "patch_file"])
        self.assertEqual(len(state.decision_log), 2)

    def test_task_state_is_persisted_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = SessionLogger("s_state", root, root / ".data")
            state = TaskState(
                objective="查看网页内容",
                constraints=["不是让你自己算"],
                required_capabilities={"command"},
                success_criteria=["页面内容与目标对象一致"],
                current_subgoal="打开目标页面",
                working_hypothesis="当前标签页就是目标页面",
                evidence_gap="尚未确认页面标题",
                evidence_layers=["inspect_browser"],
                decision_log=["inspect_browser: failed; wrong tab"],
                stalled_steps=2,
                phase="planning",
                turn_count=1,
            )
            session.append("task_state", state.to_dict())

            transcript = load_session(session.path)
            restored = TaskState.from_mapping(transcript.task_state)

            self.assertEqual(restored.objective, state.objective)
            self.assertEqual(restored.constraints, state.constraints)
            self.assertEqual(restored.required_capabilities, {"command"})
            self.assertEqual(restored.success_criteria, state.success_criteria)
            self.assertEqual(restored.current_subgoal, state.current_subgoal)
            self.assertEqual(restored.working_hypothesis, state.working_hypothesis)
            self.assertEqual(restored.evidence_gap, state.evidence_gap)
            self.assertEqual(restored.evidence_layers, state.evidence_layers)
            self.assertEqual(restored.decision_log, state.decision_log)
            self.assertEqual(restored.stalled_steps, 2)

    def test_state_change_resets_stalled_step_counter(self) -> None:
        state = TaskState(objective="发布内容")
        state.record_tool(
            "run_command",
            {"command": "inspect"},
            ok=True,
            content="no actionable target",
            changed_files=[],
        )
        state.record_tool(
            "run_command",
            {"command": "click"},
            ok=True,
            content="published=true",
            changed_files=[],
            state_changed=True,
        )

        self.assertEqual(state.stalled_steps, 0)
        self.assertIn("成功判据", state.evidence_gap)


if __name__ == "__main__":
    unittest.main()
