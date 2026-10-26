from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sunus.memory import MemoryStore
from sunus.memory_import import import_session_memory


class CrossTurnMemoryTests(unittest.TestCase):
    def test_followup_answers_are_retrievable_across_domains(self) -> None:
        cases = (
            ("你养的是什么宠物？", "团子", "宠物"),
            ("你最喜欢哪座城市？", "杭州", "城市"),
            ("你平时用哪个命令行工具？", "Warp", "工具"),
        )
        with tempfile.TemporaryDirectory() as temp:
            memory = MemoryStore(Path(temp) / "memory.db")
            for index, (question, answer, topic) in enumerate(cases):
                ids = memory.record_turn(
                    user_input=answer,
                    assistant_text="明白。",
                    previous_assistant_text=question,
                    scope="/project/example",
                    receipt_id=f"r_followup_{index}",
                )
                context = memory.build_context(
                    scope="/project/example",
                    query=f"我之前说过的{topic}是什么",
                    recall_limit=5,
                )
                self.assertIn(ids[0], {item["id"] for item in context.recalled})
            memory.close()

    def test_session_import_preserves_previous_turn_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sessions = root / "sessions"
            sessions.mkdir()
            path = sessions / "s_20261026-120000_test.jsonl"
            records = (
                {
                    "type": "meta",
                    "workspace": "/project/example",
                    "session_id": path.stem,
                },
                {
                    "type": "receipt",
                    "turn_id": "t1",
                    "receipt_id": "r_context_1",
                    "user_input": "你最近在读哪本书？",
                    "final_text": "你最近在读哪本书？",
                    "reason": "completed",
                },
                {
                    "type": "receipt",
                    "turn_id": "t2",
                    "receipt_id": "r_context_2",
                    "user_input": "三体",
                    "final_text": "好，我记住了。",
                    "reason": "completed",
                },
            )
            path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            memory = MemoryStore(root / "memory.db")
            imported = import_session_memory(
                memory,
                data_dir=root,
                date="2026-10-26",
            )
            context = memory.build_context(
                scope="/project/example",
                query="上次我说在读什么书",
                recall_limit=5,
            )

            self.assertEqual(imported["turns"], 2)
            self.assertTrue(
                any("三体" in item["content"] for item in context.recalled)
            )
            memory.close()

    def test_context_expands_followup_neighbors_across_domains(self) -> None:
        cases = (
            ("你养的是什么宠物？", "团子", "宠物"),
            ("你最喜欢哪座城市？", "杭州", "城市"),
            ("你平时用哪个命令行工具？", "Warp", "工具"),
        )
        for index, (question, answer, topic) in enumerate(cases):
            with self.subTest(topic=topic), tempfile.TemporaryDirectory() as temp:
                memory = MemoryStore(Path(temp) / "memory.db")
                memory.record_turn(
                    user_input=f"我想聊一个关于{topic}的话题",
                    assistant_text=question,
                    scope="/project/example",
                    receipt_id=f"r_anchor_{index}",
                )
                answer_ids = memory.record_turn(
                    user_input=answer,
                    assistant_text="明白。",
                    scope="/project/example",
                    receipt_id=f"r_answer_{index}",
                )
                context = memory.build_context(
                    scope="/project/example",
                    query=f"我之前说过的{topic}是什么",
                    recall_limit=5,
                )

                self.assertIn(
                    answer_ids[0],
                    {item["id"] for item in context.recalled},
                )
                memory.close()


if __name__ == "__main__":
    unittest.main()
