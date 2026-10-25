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
        self.assertEqual(state.phase, "blocked")

    def test_task_state_is_persisted_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = SessionLogger("s_state", root, root / ".data")
            state = TaskState(
                objective="查看网页内容",
                constraints=["不是让你自己算"],
                required_capabilities={"command"},
                phase="planning",
                turn_count=1,
            )
            session.append("task_state", state.to_dict())

            transcript = load_session(session.path)
            restored = TaskState.from_mapping(transcript.task_state)

            self.assertEqual(restored.objective, state.objective)
            self.assertEqual(restored.constraints, state.constraints)
            self.assertEqual(restored.required_capabilities, {"command"})


if __name__ == "__main__":
    unittest.main()
