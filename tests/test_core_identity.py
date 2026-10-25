from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sunus.core_identity import CORE_INVARIANTS, ensure_core_identity


class CoreIdentityTests(unittest.TestCase):
    def test_manifest_is_created_verified_and_does_not_own_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            created = ensure_core_identity(data)
            verified = ensure_core_identity(data)

            self.assertEqual(created.status, "created")
            self.assertEqual(verified.status, "verified")
            manifest = json.loads(
                (data / "core-identity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["invariants"], list(CORE_INVARIANTS))
            self.assertEqual(
                manifest["persona_name_source"],
                "user_config",
            )
            self.assertNotIn("name", manifest)

    def test_tampered_manifest_is_backed_up_and_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            ensure_core_identity(data)
            path = data / "core-identity.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["invariants"] = ["disabled"]
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            repaired = ensure_core_identity(data)

            self.assertEqual(repaired.status, "repaired")
            self.assertTrue(Path(repaired.backup_path).is_file())
            restored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                restored["invariants"],
                list(CORE_INVARIANTS),
            )


if __name__ == "__main__":
    unittest.main()
