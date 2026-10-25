from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sunus.memory import MemoryStore
from sunus.memory_import import import_session_memory


class MemoryImportTests(unittest.TestCase):
    def test_imports_turn_receipts_and_explicit_reviews_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sessions = root / "sessions"
            sessions.mkdir()
            path = sessions / "s_20261025-120000_test.jsonl"
            rows = [
                {
                    "type": "meta",
                    "workspace": "/project/example",
                },
                {
                    "type": "user",
                    "turn_id": "t1",
                    "text": "修复记忆逻辑",
                },
                {
                    "type": "receipt",
                    "turn_id": "t1",
                    "receipt_id": "r1",
                    "user_input": "修复记忆逻辑",
                    "final_text": "已修复。",
                    "reason": "completed",
                },
                {
                    "type": "user",
                    "turn_id": "t2",
                    "text": "请记住这轮复盘并落库",
                },
                {
                    "type": "receipt",
                    "turn_id": "t2",
                    "receipt_id": "r2",
                    "user_input": "请记住这轮复盘并落库",
                    "final_text": "复盘结论：显式记忆工具已补上。",
                    "reason": "completed",
                },
                {
                    "type": "user",
                    "turn_id": "t3",
                    "text": "检查导入兜底",
                },
                {
                    "type": "message",
                    "turn_id": "t3",
                    "message": {
                        "role": "assistant",
                        "content": "assistant-only fallback",
                    },
                },
                {
                    "type": "receipt",
                    "turn_id": "t3",
                    "receipt_id": "r3",
                    "user_input": "检查导入兜底",
                    "final_text": "",
                    "reason": "completed",
                },
            ]
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            memory = MemoryStore(root / "memory.db")

            first = import_session_memory(
                memory,
                data_dir=root,
                date="2026-10-25",
            )
            second = import_session_memory(
                memory,
                data_dir=root,
                date="2026-10-25",
            )

            self.assertEqual(first["turns"], 3)
            self.assertEqual(len(first["review_ids"]), 1)
            self.assertEqual(first["review_ids"], second["review_ids"])
            reviews = memory.recent_persistent_reviews(
                scope="/project/example",
                limit=3,
            )
            self.assertEqual(len(reviews), 1)
            fallback = memory.search(
                "assistant-only",
                scope="/project/example",
                limit=3,
            )
            self.assertTrue(fallback)
            memory.close()


if __name__ == "__main__":
    unittest.main()
