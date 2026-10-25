from __future__ import annotations

import unittest

from sunus.result import (
    ResultEnvelope,
    fallback_result,
    normalize_result,
    render_result,
)


class ResultEnvelopeTests(unittest.TestCase):
    def test_validates_and_renders_change_report(self) -> None:
        envelope = ResultEnvelope.from_mapping(
            {
                "kind": "change_report",
                "status": "completed",
                "summary": "修复完成。",
                "details": ["结果可复现。"],
                "changes": [
                    {
                        "path": "src/sunus/model.py",
                        "action": "modified",
                        "purpose": "处理断连",
                    }
                ],
                "verification": [
                    {
                        "command": "python3 -m unittest",
                        "status": "passed",
                        "summary": "66 项通过",
                    }
                ],
                "risks": [],
                "artifacts": [],
                "next_action": "",
            }
        )
        rendered = render_result(envelope)
        self.assertTrue(rendered.startswith("修复完成。"))
        self.assertIn("`src/sunus/model.py` · modified", rendered)
        self.assertIn("✓ `python3 -m unittest` · 66 项通过", rendered)

    def test_rejects_unknown_fields_and_blocked_without_next_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "未知字段"):
            ResultEnvelope.from_mapping(
                {
                    "kind": "answer",
                    "status": "completed",
                    "summary": "ok",
                    "extra": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "next_action"):
            ResultEnvelope.from_mapping(
                {
                    "kind": "blocked",
                    "status": "blocked",
                    "summary": "缺少依赖",
                }
            )

    def test_fallback_marks_model_error_as_blocked(self) -> None:
        envelope = fallback_result(
            "模型调用失败。",
            changed_files=[],
            reason="model_error",
        )
        self.assertEqual(envelope.kind, "blocked")
        self.assertEqual(envelope.status, "blocked")
        self.assertIn("恢复条件", render_result(envelope))

    def test_read_only_answer_drops_audit_sections(self) -> None:
        raw = {
            "kind": "analysis",
            "status": "completed",
            "summary": "记忆体系分三层。",
            "details": ["注入层", "存储层", "记录层", "第四条"],
            "changes": [{"path": ".sunus", "action": "read"}],
            "verification": [
                {
                    "command": "read_file .sunus/memory.db",
                    "status": "passed",
                }
            ],
            "risks": ["未查证 schema", "未查证淘汰", "未查证向量"],
        }
        envelope = ResultEnvelope.from_mapping(raw)
        normalized = normalize_result(
            envelope,
            tool_traces=[],
            changed_files=[],
        )
        rendered = render_result(normalized)

        self.assertEqual(normalized.details, ("注入层", "存储层", "记录层"))
        self.assertEqual(normalized.changes, ())
        self.assertEqual(normalized.verification, ())
        self.assertEqual(normalized.risks, ("未查证 schema",))
        self.assertNotIn("**改动**", rendered)
        self.assertNotIn("**验证**", rendered)
        self.assertIn("**限制**：未查证 schema", rendered)

    def test_change_report_only_keeps_actual_mutations_and_commands(self) -> None:
        envelope = ResultEnvelope.from_mapping(
            {
                "kind": "change_report",
                "status": "completed",
                "summary": "修复完成。",
                "changes": [
                    {"path": "src/a.py", "action": "modified"},
                    {"path": "src/not-changed.py", "action": "modified"},
                ],
                "verification": [
                    {"command": "python3 -m unittest", "status": "failed"},
                    {"command": "read_file src/a.py", "status": "passed"},
                ],
            }
        )
        normalized = normalize_result(
            envelope,
            tool_traces=[
                {
                    "name": "run_command",
                    "arguments": {"command": "python3 -m unittest"},
                    "ok": True,
                }
            ],
            changed_files=["src/a.py"],
        )

        self.assertEqual([item.path for item in normalized.changes], ["src/a.py"])
        self.assertEqual(len(normalized.verification), 1)
        self.assertEqual(normalized.verification[0].status, "passed")

    def test_change_report_without_real_changes_renders_as_answer(self) -> None:
        envelope = ResultEnvelope.from_mapping(
            {
                "kind": "change_report",
                "status": "completed",
                "summary": "只是分析，没有修改。",
                "changes": [{"path": "src/a.py", "action": "modified"}],
            }
        )

        normalized = normalize_result(
            envelope,
            tool_traces=[],
            changed_files=[],
        )

        self.assertEqual(normalized.kind, "answer")
        self.assertNotIn("**改动**", render_result(normalized))


if __name__ == "__main__":
    unittest.main()
