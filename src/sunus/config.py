from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


DEFAULT_PROACTIVE_BASE_INTERVAL_SEC = 5 * 60
PROACTIVE_TOPIC_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("daily_emotion", 20),
    ("work_coding", 20),
    ("self_evolution", 20),
    ("user_gender", 20),
    ("user_custom", 20),
)
MODEL_API_MODES = {"chat_completions", "responses"}
LOOP_BUDGET_PROFILES: dict[str, dict[str, int]] = {
    "quick": {
        "max_model_calls": 8,
        "max_tool_calls": 24,
        "max_wall_seconds": 180,
        "max_total_tokens": 50_000,
        "max_response_tokens": 1024,
        "finalization_reserve": 1,
        "context_soft_tokens": 12_000,
        "context_hard_tokens": 24_000,
        "context_keep_recent": 10,
        "context_summary_chars": 8_000,
    },
    "standard": {
        "max_model_calls": 60,
        "max_tool_calls": 180,
        "max_wall_seconds": 3600,
        "max_total_tokens": 2_000_000,
        "max_response_tokens": 8192,
        "finalization_reserve": 2,
        "context_soft_tokens": 200_000,
        "context_hard_tokens": 960_000,
        "context_keep_recent": 48,
        "context_summary_chars": 48_000,
    },
    "deep": {
        "max_model_calls": 120,
        "max_tool_calls": 360,
        "max_wall_seconds": 7200,
        "max_total_tokens": 4_000_000,
        "max_response_tokens": 16_384,
        "finalization_reserve": 2,
        "context_soft_tokens": 400_000,
        "context_hard_tokens": 960_000,
        "context_keep_recent": 64,
        "context_summary_chars": 64_000,
    },
}


PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_mode": "chat_completions",
        "token_parameter": "max_tokens",
        "model": "deepseek-flash",
        "options": {},
        "key_envs": ("DEEPSEEK_API_KEY",),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_mode": "chat_completions",
        "token_parameter": "max_completion_tokens",
        "model": "",
        "options": {},
        "key_envs": ("OPENAI_API_KEY",),
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "api_mode": "chat_completions",
        "token_parameter": "max_completion_tokens",
        "model": "MiniMax-M3.1-Flash",
        "options": {"reasoning_split": True},
        "key_envs": ("MINIMAX_API_KEY",),
    },
    "openai_compatible": {
        "base_url": "",
        "api_mode": "chat_completions",
        "token_parameter": "max_tokens",
        "model": "",
        "options": {},
        "key_envs": ("OPENAI_API_KEY",),
    },
}


@dataclass(frozen=True)
class ModelSettings:
    provider: str = "deepseek"
    api_mode: str = "chat_completions"
    token_parameter: str = "max_tokens"
    reasoning_effort: str | None = None
    reasoning_split: bool | None = None
    options: tuple[tuple[str, Any], ...] = ()

    def options_as_dict(self) -> dict[str, Any]:
        return dict(self.options)


@dataclass(frozen=True)
class LoopBudgetConfig:
    profile: str = "standard"
    max_model_calls: int = 60
    max_tool_calls: int = 180
    max_wall_seconds: int = 3600
    max_total_tokens: int = 2_000_000
    max_response_tokens: int = 8192
    warning_ratio: float = 0.8
    no_progress_soft: int = 2
    no_progress_hard: int = 3
    same_error_soft: int = 2
    same_error_hard: int = 3
    consecutive_failure_hard: int = 4
    max_recoveries: int = 2
    finalization_reserve: int = 2
    context_soft_tokens: int = 200_000
    context_hard_tokens: int = 960_000
    context_keep_recent: int = 48
    context_summary_chars: int = 48_000


@dataclass(frozen=True)
class ProactiveConfig:
    enabled: bool = True
    base_interval_sec: int = DEFAULT_PROACTIVE_BASE_INTERVAL_SEC
    backoff_factor: int = 2
    recent_memory_window_hours: int = 168
    recent_memory_limit: int = 32
    preference_items_in_context: int = 8
    topic_weights: tuple[tuple[str, int], ...] = PROACTIVE_TOPIC_WEIGHTS
    sunus_gender: str = ""
    user_gender: str = ""
    custom_quirks: tuple[str, ...] = ()
    custom_topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    workspace: Path
    data_dir: Path
    api_key: str = field(repr=False)
    base_url: str
    model: str
    max_steps: int = 20
    max_tokens: int = 16_384
    request_timeout_sec: int = 120
    auto_approve: bool = False
    allow_outside: bool = False
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    loop_budget: LoopBudgetConfig = field(default_factory=LoopBudgetConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


DEFAULT_USER_CONFIG: dict[str, Any] = {
    "max_tokens": 16_384,
    "request_timeout_sec": 120,
    "budget": {"profile": "standard"},
    "proactive": {
        "enabled": True,
        "base_interval_sec": DEFAULT_PROACTIVE_BASE_INTERVAL_SEC,
        "backoff_factor": 2,
        "recent_memory_window_hours": 168,
        "recent_memory_limit": 32,
        "preference_items_in_context": 8,
        "topic_weights": dict(PROACTIVE_TOPIC_WEIGHTS),
        "sunus_gender": "",
        "user_gender": "",
        "custom_quirks": [],
        "custom_topics": [],
    },
}


def _sunus_provider_config_path() -> Path:
    explicit = os.environ.get("SUNUS_PROVIDER_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    provider_home = os.environ.get("SUNUS_PROVIDER_HOME") or os.environ.get("CODEX_HOME")
    if provider_home:
        return Path(provider_home).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value.split("#", 1)[0].strip()


def load_sunus_deepseek_config(path: Path | None = None) -> tuple[str, str]:
    """Read the local Sunus provider token without copying or logging it."""
    path = path or _sunus_provider_config_path()
    if not path.is_file():
        return "", ""

    section = ""
    key = ""
    base_url = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section != "model_providers.deepseek":
            continue
        match = re.match(
            r"^(experimental_bearer_token|base_url)\s*=\s*(.+)$",
            stripped,
        )
        if not match:
            continue
        if match.group(1) == "experimental_bearer_token":
            key = _unquote(match.group(2))
        else:
            base_url = _unquote(match.group(2))
    return key, base_url.rstrip("/")


def _load_user_config(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "config.json"
    if not path.is_file():
        return json.loads(json.dumps(DEFAULT_USER_CONFIG))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"无法读取 {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{path} 必须是 JSON object")
    result = json.loads(json.dumps(DEFAULT_USER_CONFIG))
    result.update(payload)
    if isinstance(payload.get("proactive"), dict):
        proactive = dict(DEFAULT_USER_CONFIG["proactive"])
        proactive.update(payload["proactive"])
        result["proactive"] = proactive
    return result


def save_user_config(data_dir: Path, updates: dict[str, Any]) -> Path:
    """Deep-merge updates into the user JSON config using an atomic replace."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "config.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(f"无法读取 {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ConfigError(f"{path} 必须是 JSON object")
    else:
        payload = {}

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(payload, updates)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def resolve_config(
    workspace: Path,
    *,
    provider: str | None = None,
    api_mode: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    auto_approve: bool = False,
    allow_outside: bool = False,
    proactive_enabled: bool | None = None,
    max_steps: int | None = None,
    budget_profile: str | None = None,
) -> RuntimeConfig:
    workspace = workspace.expanduser().resolve()
    data_dir = Path(
        os.environ.get("SUNUS_DATA_DIR", Path.home() / ".sunus")
    ).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_dir.chmod(0o700)
    except OSError:
        pass
    user = _load_user_config(data_dir)

    sunus_key, sunus_base = load_sunus_deepseek_config()
    provider_name = str(
        provider
        or os.environ.get("SUNUS_PROVIDER")
        or user.get("provider")
        or "deepseek"
    ).strip().lower()
    if provider_name not in PROVIDER_DEFAULTS:
        raise ConfigError(
            f"不支持的 provider: {provider_name}；"
            f"可选: {', '.join(sorted(PROVIDER_DEFAULTS))}"
        )
    provider_defaults = PROVIDER_DEFAULTS[provider_name]
    provider_key = next(
        (
            os.environ[name]
            for name in provider_defaults["key_envs"]
            if os.environ.get(name)
        ),
        "",
    )
    api_key = os.environ.get("SUNUS_API_KEY") or provider_key
    if provider_name == "deepseek" and not api_key:
        api_key = sunus_key

    resolved_api_mode = str(
        api_mode
        or os.environ.get("SUNUS_API_MODE")
        or user.get("api_mode")
        or provider_defaults["api_mode"]
    ).strip().lower()
    if resolved_api_mode not in MODEL_API_MODES:
        raise ConfigError(
            f"不支持的 api_mode: {resolved_api_mode}；"
            f"可选: {', '.join(sorted(MODEL_API_MODES))}"
        )
    resolved_base = (
        base_url
        or os.environ.get("SUNUS_BASE_URL")
        or user.get("base_url")
        or (sunus_base if provider_name == "deepseek" else "")
        or provider_defaults["base_url"]
    ).rstrip("/")
    if not resolved_base:
        raise ConfigError("openai_compatible 需要显式 base_url 或 SUNUS_BASE_URL")
    resolved_model = (
        model
        or os.environ.get("SUNUS_MODEL")
        or user.get("model")
        or provider_defaults["model"]
    )
    if not resolved_model:
        raise ConfigError(
            f"{provider_name} 需要显式模型：请使用 --model、SUNUS_MODEL "
            "或用户配置中的 model"
        )
    token_parameter = str(
        os.environ.get("SUNUS_TOKEN_PARAMETER")
        or user.get("token_parameter")
        or provider_defaults["token_parameter"]
    ).strip()
    if token_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise ConfigError(
            "token_parameter 必须是 max_tokens 或 max_completion_tokens"
        )
    resolved_reasoning_effort = (
        reasoning_effort
        or os.environ.get("SUNUS_REASONING_EFFORT")
        or user.get("reasoning_effort")
        or None
    )
    options = dict(provider_defaults["options"])
    configured_options = user.get("model_options", {})
    if configured_options:
        if not isinstance(configured_options, dict):
            raise ConfigError("model_options 必须是 JSON object")
        options.update(configured_options)
    model_settings = ModelSettings(
        provider=provider_name,
        api_mode=resolved_api_mode,
        token_parameter=token_parameter,
        reasoning_effort=(
            str(resolved_reasoning_effort)
            if resolved_reasoning_effort
            else None
        ),
        reasoning_split=(
            bool(options["reasoning_split"])
            if "reasoning_split" in options
            else None
        ),
        options=tuple(sorted(options.items())),
    )
    proactive_payload = user.get("proactive", {})
    raw_topic_weights = proactive_payload.get(
        "topic_weights",
        dict(PROACTIVE_TOPIC_WEIGHTS),
    )
    if not isinstance(raw_topic_weights, dict):
        raise ConfigError("proactive.topic_weights 必须是 JSON object")
    topic_weights: list[tuple[str, int]] = []
    for topic, default_weight in PROACTIVE_TOPIC_WEIGHTS:
        try:
            weight = int(raw_topic_weights.get(topic, default_weight))
        except (TypeError, ValueError) as error:
            raise ConfigError(f"主动消息主题权重无效: {topic}") from error
        if weight < 0:
            raise ConfigError(f"主动消息主题权重不能为负数: {topic}")
        topic_weights.append((topic, weight))
    unknown_topics = sorted(
        set(raw_topic_weights) - {topic for topic, _ in PROACTIVE_TOPIC_WEIGHTS}
    )
    if unknown_topics:
        raise ConfigError(
            "未知主动消息主题: " + ", ".join(unknown_topics)
        )
    if sum(weight for _, weight in topic_weights) != 100:
        raise ConfigError("proactive.topic_weights 权重合计必须为 100")

    raw_custom_topics = (
        os.environ.get("SUNUS_PROACTIVE_CUSTOM_QUIRKS")
        or proactive_payload.get("custom_quirks")
        or os.environ.get("SUNUS_PROACTIVE_CUSTOM_TOPICS")
        or proactive_payload.get("custom_topics", [])
    )
    if isinstance(raw_custom_topics, str):
        custom_topics = tuple(
            item.strip()
            for item in re.split(r"[,，\n]", raw_custom_topics)
            if item.strip()
        )
    elif isinstance(raw_custom_topics, list):
        custom_topics = tuple(
            str(item).strip() for item in raw_custom_topics if str(item).strip()
        )
    else:
        raise ConfigError("proactive.custom_topics 必须是字符串数组")

    proactive = ProactiveConfig(
        enabled=(
            bool(proactive_payload.get("enabled", True))
            if proactive_enabled is None
            else proactive_enabled
        ),
        base_interval_sec=max(
            1,
            int(
                proactive_payload.get(
                    "base_interval_sec",
                    DEFAULT_PROACTIVE_BASE_INTERVAL_SEC,
                )
            ),
        ),
        backoff_factor=max(2, int(proactive_payload.get("backoff_factor", 2))),
        recent_memory_window_hours=max(
            1, int(proactive_payload.get("recent_memory_window_hours", 168))
        ),
        recent_memory_limit=max(
            1, int(proactive_payload.get("recent_memory_limit", 32))
        ),
        preference_items_in_context=max(
            1, int(proactive_payload.get("preference_items_in_context", 8))
        ),
        topic_weights=tuple(topic_weights),
        sunus_gender=(
            os.environ.get("SUNUS_GENDER")
            or str(proactive_payload.get("sunus_gender") or "")
        ).strip(),
        user_gender=(
            os.environ.get("SUNUS_USER_GENDER")
            or str(proactive_payload.get("user_gender") or "")
        ).strip(),
        custom_quirks=custom_topics,
        custom_topics=custom_topics,
    )
    budget_payload = user.get("budget", {})
    if not isinstance(budget_payload, dict):
        raise ConfigError("budget 必须是 JSON object")
    profile = str(
        budget_profile
        or os.environ.get("SUNUS_BUDGET_PROFILE")
        or budget_payload.get("profile")
        or "standard"
    ).strip().lower()
    if profile not in LOOP_BUDGET_PROFILES:
        raise ConfigError(
            f"不支持的 budget profile: {profile}；"
            f"可选: {', '.join(sorted(LOOP_BUDGET_PROFILES))}"
        )
    budget_values = asdict(LoopBudgetConfig(profile=profile))
    budget_values.update(LOOP_BUDGET_PROFILES[profile])
    budget_values.update(
        {
            key: value
            for key, value in budget_payload.items()
            if key
            in {
                "max_model_calls",
                "max_tool_calls",
                "max_wall_seconds",
                "max_total_tokens",
                "max_response_tokens",
                "warning_ratio",
                "no_progress_soft",
                "no_progress_hard",
                "same_error_soft",
                "same_error_hard",
                "consecutive_failure_hard",
                "max_recoveries",
                "finalization_reserve",
                "context_soft_tokens",
                "context_hard_tokens",
                "context_keep_recent",
                "context_summary_chars",
            }
        }
    )
    no_progress_soft = max(1, int(budget_values["no_progress_soft"]))
    no_progress_hard = max(
        no_progress_soft + 1,
        int(budget_values["no_progress_hard"]),
    )
    same_error_soft = max(1, int(budget_values["same_error_soft"]))
    same_error_hard = max(
        same_error_soft + 1,
        int(budget_values["same_error_hard"]),
    )
    context_soft_tokens = max(1_000, int(budget_values["context_soft_tokens"]))
    context_hard_tokens = max(
        context_soft_tokens,
        int(budget_values["context_hard_tokens"]),
    )
    loop_budget = LoopBudgetConfig(
        profile=profile,
        max_model_calls=max(1, int(budget_values["max_model_calls"])),
        max_tool_calls=max(1, int(budget_values["max_tool_calls"])),
        max_wall_seconds=max(1, int(budget_values["max_wall_seconds"])),
        max_total_tokens=max(1, int(budget_values["max_total_tokens"])),
        max_response_tokens=max(256, int(budget_values["max_response_tokens"])),
        warning_ratio=min(1.0, max(0.1, float(budget_values["warning_ratio"]))),
        no_progress_soft=no_progress_soft,
        no_progress_hard=no_progress_hard,
        same_error_soft=same_error_soft,
        same_error_hard=same_error_hard,
        consecutive_failure_hard=max(
            1, int(budget_values["consecutive_failure_hard"])
        ),
        max_recoveries=max(0, int(budget_values["max_recoveries"])),
        finalization_reserve=max(
            0,
            int(budget_values["finalization_reserve"]),
        ),
        context_soft_tokens=context_soft_tokens,
        context_hard_tokens=context_hard_tokens,
        context_keep_recent=max(2, int(budget_values["context_keep_recent"])),
        context_summary_chars=max(
            1_000,
            int(budget_values["context_summary_chars"]),
        ),
    )
    return RuntimeConfig(
        workspace=workspace,
        data_dir=data_dir,
        api_key=api_key,
        base_url=resolved_base,
        model=resolved_model,
        max_steps=max(
            1,
            int(
                max_steps
                or user.get("max_steps")
                or loop_budget.max_model_calls
            ),
        ),
        max_tokens=max(256, int(user.get("max_tokens", 16_384))),
        request_timeout_sec=max(10, int(user.get("request_timeout_sec", 120))),
        auto_approve=auto_approve,
        allow_outside=allow_outside,
        model_settings=model_settings,
        loop_budget=loop_budget,
        proactive=proactive,
    )


def make_ssl_context() -> ssl.SSLContext:
    cafile = Path("/etc/ssl/cert.pem")
    if cafile.is_file():
        return ssl.create_default_context(cafile=str(cafile))
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def redact_text(text: Any, secret: str = "") -> str:
    value = str(text)
    if secret:
        value = value.replace(secret, "<redacted>")
    value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer <redacted>", value)
    value = re.sub(r"\bsk-[A-Za-z0-9._-]{8,}\b", "sk-<redacted>", value)
    value = re.sub(
        r"(?i)((?:api[_-]?key|bearer[_-]?token|token)\s*[=:]\s*)[^\s,;\]}]+",
        r"\1<redacted>",
        value,
    )
    return value


def redact_value(value: Any, secret: str = "") -> Any:
    if isinstance(value, str):
        return redact_text(value, secret)
    if isinstance(value, list):
        return [redact_value(item, secret) for item in value]
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if _sensitive_key(str(key))
                else redact_value(item, secret)
            )
            for key, item in value.items()
        }
    return value


def _sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in {
        "token",
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "private_key",
        "access_token",
        "refresh_token",
        "bearer_token",
        "auth_token",
        "client_secret",
    }:
        return True
    return normalized.endswith(
        ("_api_key", "_access_token", "_refresh_token", "_bearer_token")
    )


def config_summary(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "workspace": str(config.workspace),
        "data_dir": str(config.data_dir),
        "provider": config.model_settings.provider,
        "api_mode": config.model_settings.api_mode,
        "token_parameter": config.model_settings.token_parameter,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_loaded": config.has_api_key,
        "max_steps": config.max_steps,
        "max_tokens": config.max_tokens,
        "budget": {
            "profile": config.loop_budget.profile,
            "max_model_calls": config.loop_budget.max_model_calls,
            "max_tool_calls": config.loop_budget.max_tool_calls,
            "max_wall_seconds": config.loop_budget.max_wall_seconds,
            "max_total_tokens": config.loop_budget.max_total_tokens,
            "max_response_tokens": config.loop_budget.max_response_tokens,
            "finalization_reserve": config.loop_budget.finalization_reserve,
            "context_soft_tokens": config.loop_budget.context_soft_tokens,
            "context_hard_tokens": config.loop_budget.context_hard_tokens,
            "context_keep_recent": config.loop_budget.context_keep_recent,
            "context_summary_chars": config.loop_budget.context_summary_chars,
        },
        "proactive": {
            "enabled": config.proactive.enabled,
            "base_interval_sec": config.proactive.base_interval_sec,
            "backoff_factor": config.proactive.backoff_factor,
            "recent_memory_window_hours": config.proactive.recent_memory_window_hours,
            "recent_memory_limit": config.proactive.recent_memory_limit,
            "topic_weights": dict(config.proactive.topic_weights),
            "sunus_gender": config.proactive.sunus_gender,
            "user_gender": config.proactive.user_gender,
            "custom_quirks": list(
                config.proactive.custom_quirks or config.proactive.custom_topics
            ),
            "custom_topics": list(config.proactive.custom_topics),
        },
    }

