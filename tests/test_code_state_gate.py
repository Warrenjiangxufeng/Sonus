from __future__ import annotations

import json
import unittest

from sunus.code_state_gate import (
    CodeStateEvidenceGate,
    claims_current_code_state,
)


class CodeStateEvidenceGateTests(unittest.TestCase):
    def test_claim_detection_is_based_on_the_answer_not_only_the_prompt(self) -> None:
        self.assertTrue(claims_current_code_state("当前代码已经修复"))
        self.assertTrue(claims_current_code_state("这里没有独立配置项"))
        self.assertTrue(claims_current_code_state("这属于旧版本的问题"))
        self.assertFalse(claims_current_code_state("今天先早点休息"))

    def test_gate_accepts_only_verified_code_state_evidence(self) -> None:
        gate = CodeStateEvidenceGate(enabled=True)
        gate.record_tool(
            "inspect_code_state",
            {"path": "src/sunus/proactive.py", "symbol": "max_tokens"},
            ok=True,
            content=json.dumps(
                {
                    "head": "abc123",
                    "target_exists": True,
                    "version_verified": True,
                }
            ),
        )
        self.assertTrue(gate.verified)
        self.assertEqual(gate.missing_instruction("当前代码已经修复"), "")

    def test_gate_requires_evidence_for_a_current_code_claim(self) -> None:
        gate = CodeStateEvidenceGate(enabled=True)
        missing = gate.missing_instruction("这里没有独立配置项")
        self.assertIn("inspect_code_state", missing)
        self.assertNotEqual(gate.unverified_answer(), "")


if __name__ == "__main__":
    unittest.main()
