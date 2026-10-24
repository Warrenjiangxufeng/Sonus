from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from sonus.cli import copy_transcript_messages
from sonus.storage import (
    SessionLogger,
    list_sessions,
    load_session,
    resolve_session_path,
)


class SessionTests(unittest.TestCase):
    def test_checkpoint_loads_compact_history_and_later_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = SessionLogger("s_test", root, root / ".data")
            session.append("user", {"turn_id": "t1", "text": "第一轮"})
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
            tool = {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok":true,"content":"a.txt"}',
            }
            session.append(
                "message",
                {"turn_id": "t1", "message": assistant},
            )
            session.append("message", {"turn_id": "t1", "message": tool})
            checkpoint = [
                {"role": "user", "content": "第一轮摘要保留"},
                assistant,
                tool,
            ]
            session.append(
                "context_checkpoint",
                {
                    "turn_id": "t1",
                    "summary": "- 用户要求：第一轮",
                    "messages": checkpoint,
                },
            )
            session.append(
                "message",
                {
                    "turn_id": "t2",
                    "message": {
                        "role": "assistant",
                        "content": "检查点之后的消息",
                    },
                },
            )

            loaded = load_session(session.path)

            self.assertEqual(loaded.session_id, "s_test")
            self.assertEqual(loaded.summary, "- 用户要求：第一轮")
            self.assertEqual(len(loaded.messages), 4)
            self.assertEqual(loaded.messages[0]["content"], "第一轮摘要保留")
            self.assertEqual(loaded.messages[-1]["content"], "检查点之后的消息")

    def test_list_and_resolve_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            older = SessionLogger("s_20260101-000000_aaaaaa", root, root / ".data")
            older.append("user", {"turn_id": "t1", "text": "旧会话"})
            newer = SessionLogger("s_20260102-000000_bbbbbb", root, root / ".data")
            newer.append("user", {"turn_id": "t2", "text": "新会话"})
            os.utime(older.path, (1_700_000_000, 1_700_000_000))
            os.utime(newer.path, (1_800_000_000, 1_800_000_000))

            rows = list_sessions(root, root / ".data")
            resolved_id, resolved_path = resolve_session_path(
                root,
                root / ".data",
                None,
                last=True,
            )

            self.assertEqual({row["session_id"] for row in rows}, {
                older.session_id,
                newer.session_id,
            })
            self.assertEqual(resolved_id, newer.session_id)
            self.assertTrue(resolved_path.is_file())

    def test_fork_copies_transcript_into_new_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = SessionLogger("s_source", root, root / ".data")
            source.append("user", {"turn_id": "t1", "text": "原始问题"})
            source.append(
                "message",
                {
                    "turn_id": "t1",
                    "message": {"role": "assistant", "content": "原始回答"},
                },
            )
            transcript = load_session(source.path)

            forked = SessionLogger("s_forked", root, root / ".data")
            copy_transcript_messages(
                forked,
                transcript,
                forked_from=source.session_id,
            )
            loaded = load_session(forked.path)

            self.assertEqual(loaded.messages, transcript.messages)
            self.assertIn('"type": "fork"', forked.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
