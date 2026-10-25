from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sunus.environment import (
    discover_environment,
    environment_prompt,
    shell_invocation,
)


class EnvironmentTests(unittest.TestCase):
    def test_discovers_arbitrary_command_from_live_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "sunus-capability-probe"
            executable.write_text(
                "#!/bin/sh\nprintf '9.8.7\\n'\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            current_path = os.environ.get("PATH", "")
            with patch.dict(
                os.environ,
                {"PATH": f"{root}{os.pathsep}{current_path}"},
            ):
                report = discover_environment(
                    ["sunus-capability-probe"],
                    refresh=True,
                )

        matches = report["requested_commands"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["name"], "sunus-capability-probe")
        self.assertEqual(
            Path(matches[0]["command"]).resolve(),
            executable.resolve(),
        )
        self.assertEqual(matches[0]["version"], "9.8.7")

    def test_environment_prompt_is_based_on_runtime_discovery(self) -> None:
        prompt = environment_prompt()
        self.assertIn("实时扫描", prompt)
        self.assertIn("PATH", prompt)
        self.assertIn("inspect_environment", prompt)

    def test_shell_invocation_uses_detected_shell_family(self) -> None:
        self.assertEqual(
            shell_invocation("/somewhere/zsh", "pytest -q"),
            ["/somewhere/zsh", "-lc", "pytest -q"],
        )
        self.assertEqual(
            shell_invocation("/somewhere/pwsh", "pytest"),
            ["/somewhere/pwsh", "-NoProfile", "-Command", "pytest"],
        )


if __name__ == "__main__":
    unittest.main()
