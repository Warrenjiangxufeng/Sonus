from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sunus.runtime_history import inspect_runtime_history


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class RuntimeHistoryTests(unittest.TestCase):
    def test_latest_failure_returns_only_newest_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            _write_jsonl(
                data / "events" / "2026-10-24.jsonl",
                [
                    {
                        "type": "internal_error",
                        "ts": "2026-10-24T10:00:00+00:00",
                        "turn_id": "old",
                        "summary": "old failure",
                    }
                ],
            )
            _write_jsonl(
                data / "events" / "2026-10-25.jsonl",
                [
                    {
                        "type": "model_error",
                        "ts": "2026-10-25T10:00:00+00:00",
                        "turn_id": "new",
                        "summary": "new failure",
                    }
                ],
            )

            result = inspect_runtime_history(
                data,
                mode="latest_failure",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["returned"], 1)
            self.assertEqual(result["rows"][0]["turn_id"], "new")

    def test_trend_aggregates_365_days_instead_of_returning_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            rows = []
            for index in range(40):
                rows.append(
                    {
                        "type": "internal_error",
                        "ts": f"2026-10-25T10:{index:02d}:00+00:00",
                        "turn_id": f"t_{index}",
                        "summary": (
                            "Sunus 内部错误：TypeError: "
                            f"request {index} failed at line {100 + index}"
                        ),
                    }
                )
            _write_jsonl(data / "events" / "2026-10-25.jsonl", rows)

            result = inspect_runtime_history(
                data,
                mode="trend",
                days=365,
                limit=5,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["total_failures"], 40)
            self.assertEqual(result["top_errors"][0]["count"], 40)
            self.assertNotIn("rows", result)

    def test_turn_timeline_includes_traceback_only_for_selected_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            _write_jsonl(
                data / "events" / "2026-10-25.jsonl",
                [
                    {
                        "type": "internal_error",
                        "ts": "2026-10-25T10:00:00+00:00",
                        "turn_id": "target",
                        "reason": "internal_error",
                        "summary": "failed",
                        "traceback": "Traceback (most recent call last): ValueError",
                    },
                    {
                        "type": "turn_end",
                        "ts": "2026-10-25T10:00:01+00:00",
                        "turn_id": "other",
                        "reason": "completed",
                    },
                ],
            )

            result = inspect_runtime_history(
                data,
                mode="turn",
                turn_id="target",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(len(result["rows"]), 1)
            self.assertIn("Traceback", result["rows"][0]["traceback"])


if __name__ == "__main__":
    unittest.main()
