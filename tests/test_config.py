from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sunus.config import (
    ConfigError,
    load_sunus_deepseek_config,
    redact_text,
    redact_value,
    resolve_config,
    save_user_config,
)


class ConfigTests(unittest.TestCase):
    def test_reads_sunus_deepseek_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                """
model = "deepseek-flash"
[model_providers.deepseek]
base_url = "https://api.deepseek.com/"
experimental_bearer_token ="sk-test-secret-value"
""".strip(),
                encoding="utf-8",
            )
            key, base = load_sunus_deepseek_config(path)
            self.assertEqual(key, "sk-test-secret-value")
            self.assertEqual(base, "https://api.deepseek.com")

    def test_env_overrides_user_config_without_writing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "config.json").write_text(
                '{"model":"deepseek-v4-pro","proactive":{"enabled":false}}',
                encoding="utf-8",
            )
            env = {
                "SUNUS_DATA_DIR": str(data),
                "SUNUS_API_KEY": "sk-test-env",
                "SUNUS_MODEL": "deepseek-flash",
            }
            with patch.dict(os.environ, env, clear=False):
                config = resolve_config(root / "workspace")
            self.assertEqual(config.api_key, "sk-test-env")
            self.assertEqual(config.model, "deepseek-flash")
            self.assertFalse(config.proactive.enabled)
            self.assertEqual(config.proactive.base_interval_sec, 300)
            self.assertEqual(config.proactive.backoff_factor, 2)
            self.assertEqual(
                dict(config.proactive.topic_weights),
                {
                    "daily_emotion": 20,
                    "work_coding": 20,
                    "self_evolution": 20,
                    "user_gender": 20,
                    "user_custom": 20,
                },
            )
            self.assertEqual(config.proactive.user_gender, "")
            self.assertEqual(config.proactive.sunus_gender, "")
            self.assertEqual(config.proactive.custom_topics, ())
            self.assertEqual(config.proactive.custom_quirks, ())
            self.assertEqual(config.loop_budget.profile, "standard")
            self.assertEqual(config.loop_budget.max_model_calls, 60)
            self.assertEqual(config.loop_budget.max_response_tokens, 8192)
            self.assertEqual(config.loop_budget.finalization_reserve, 2)
            self.assertEqual(config.loop_budget.context_soft_tokens, 200_000)
            self.assertEqual(config.loop_budget.context_hard_tokens, 960_000)
            self.assertEqual(config.loop_budget.max_total_tokens, 2_000_000)
            self.assertNotIn("sk-test-env", repr(config))
            mode = stat.S_IMODE(data.stat().st_mode)
            self.assertEqual(mode, 0o700)

    def test_redaction_masks_common_secret_shapes(self) -> None:
        value = "Authorization: Bearer abcdefghijklmnop api_key=very-secret sk-123456789"
        redacted = redact_text(value)
        self.assertNotIn("abcdefghijklmnop", redacted)
        self.assertNotIn("very-secret", redacted)
        self.assertNotIn("123456789", redacted)

    def test_redaction_preserves_numeric_token_metrics(self) -> None:
        value = redact_value(
            {
                "total_tokens": 321,
                "max_total_tokens": 1000,
                "token_parameter": "max_completion_tokens",
                "api_key": "sk-secret-value",
                "access_token": "secret-access-value",
            }
        )
        self.assertEqual(value["total_tokens"], 321)
        self.assertEqual(value["max_total_tokens"], 1000)
        self.assertEqual(value["token_parameter"], "max_completion_tokens")
        self.assertEqual(value["api_key"], "<redacted>")
        self.assertEqual(value["access_token"], "<redacted>")

    def test_minimax_provider_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = {
                "SUNUS_DATA_DIR": str(root / "data"),
                "MINIMAX_API_KEY": "mx-test",
            }
            with patch.dict(os.environ, env, clear=True):
                config = resolve_config(root / "workspace", provider="minimax")
            self.assertEqual(config.api_key, "mx-test")
            self.assertEqual(config.base_url, "https://api.minimax.io/v1")
            self.assertEqual(config.model, "MiniMax-M3.1-Flash")
            self.assertEqual(
                config.model_settings.token_parameter,
                "max_completion_tokens",
            )
            self.assertTrue(config.model_settings.reasoning_split)

    def test_openai_requires_explicit_model_and_supports_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = {
                "SUNUS_DATA_DIR": str(root / "data"),
                "OPENAI_API_KEY": "sk-openai-test",
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ConfigError):
                    resolve_config(root / "workspace", provider="openai")
                config = resolve_config(
                    root / "workspace",
                    provider="openai",
                    api_mode="responses",
                    model="test-model",
                )
            self.assertEqual(config.base_url, "https://api.openai.com/v1")
            self.assertEqual(config.model_settings.provider, "openai")
            self.assertEqual(config.model_settings.api_mode, "responses")
            self.assertEqual(
                config.model_settings.token_parameter,
                "max_completion_tokens",
            )

    def test_budget_profile_and_custom_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "config.json").write_text(
                '{"budget":{"profile":"standard","max_tool_calls":77}}',
                encoding="utf-8",
            )
            env = {
                "SUNUS_DATA_DIR": str(data),
                "SUNUS_API_KEY": "sk-test",
            }
            with patch.dict(os.environ, env, clear=True):
                config = resolve_config(
                    root / "workspace",
                    budget_profile="quick",
                )
            self.assertEqual(config.loop_budget.profile, "quick")
            self.assertEqual(config.loop_budget.max_model_calls, 8)
            self.assertEqual(config.loop_budget.max_tool_calls, 77)
            self.assertEqual(config.loop_budget.max_response_tokens, 1024)
            self.assertEqual(config.loop_budget.finalization_reserve, 1)
            self.assertEqual(config.loop_budget.context_keep_recent, 10)
            self.assertEqual(config.max_steps, 8)

    def test_deep_profile_supports_million_token_context_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = {
                "SUNUS_DATA_DIR": str(root / "data"),
                "SUNUS_API_KEY": "sk-test",
            }
            with patch.dict(os.environ, env, clear=True):
                config = resolve_config(
                    root / "workspace",
                    budget_profile="deep",
                )
            self.assertEqual(config.loop_budget.context_soft_tokens, 400_000)
            self.assertEqual(config.loop_budget.context_hard_tokens, 960_000)
            self.assertEqual(config.loop_budget.max_total_tokens, 4_000_000)
            self.assertEqual(config.loop_budget.max_response_tokens, 16_384)
            self.assertEqual(config.max_tokens, 16_384)

    def test_proactive_topic_gender_and_custom_focus(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "config.json").write_text(
                json.dumps(
                    {
                        "proactive": {
                            "topic_weights": {
                                "daily_emotion": 35,
                                "work_coding": 20,
                                "self_evolution": 10,
                                "user_gender": 10,
                                "user_custom": 25,
                            },
                            "sunus_gender": "女",
                            "user_gender": "男",
                            "custom_quirks": ["收集钢笔", "雨天散步"],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            env = {
                "SUNUS_DATA_DIR": str(data),
                "SUNUS_API_KEY": "sk-test",
            }
            with patch.dict(os.environ, env, clear=True):
                config = resolve_config(root / "workspace")

            self.assertEqual(dict(config.proactive.topic_weights)["daily_emotion"], 35)
            self.assertEqual(config.proactive.user_gender, "男")
            self.assertEqual(config.proactive.sunus_gender, "女")
            self.assertEqual(
                config.proactive.custom_quirks,
                ("收集钢笔", "雨天散步"),
            )
            self.assertEqual(
                config.proactive.custom_topics,
                ("收集钢笔", "雨天散步"),
            )

    def test_proactive_topic_weights_must_total_100(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "config.json").write_text(
                json.dumps(
                    {
                        "proactive": {
                            "topic_weights": {"daily_emotion": 99}
                        }
                    }
                ),
                encoding="utf-8",
            )
            env = {
                "SUNUS_DATA_DIR": str(data),
                "SUNUS_API_KEY": "sk-test",
            }
            with (
                patch.dict(os.environ, env, clear=True),
                self.assertRaisesRegex(ConfigError, "合计必须为 100"),
            ):
                resolve_config(root / "workspace")

    def test_user_config_update_preserves_other_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "config.json").write_text(
                json.dumps(
                    {
                        "model": "deepseek-v4-pro",
                        "proactive": {
                            "base_interval_sec": 420,
                            "custom_quirks": ["摄影"],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            path = save_user_config(
                data,
                {
                    "proactive": {
                        "topic_weights": {
                            "daily_emotion": 45,
                            "work_coding": 20,
                            "self_evolution": 10,
                            "user_gender": 5,
                            "user_custom": 20,
                        }
                    }
                },
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["model"], "deepseek-v4-pro")
            self.assertEqual(payload["proactive"]["base_interval_sec"], 420)
            self.assertEqual(payload["proactive"]["custom_quirks"], ["摄影"])
            self.assertEqual(
                payload["proactive"]["topic_weights"]["daily_emotion"],
                45,
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

if __name__ == "__main__":
    unittest.main()

