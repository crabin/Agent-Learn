from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

SYSTEM_ONLY_KEYS = {"TAVILY_API_KEY", "GITHUB_TOKEN", "WANDB_API_KEY"}


@dataclass(frozen=True)
class LLMConfig:
    model_id: str
    api_key: str
    base_url: str
    timeout: int


def find_code_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in [current, *current.parents]:
        if candidate.name == "code" and candidate.parent.name == "hello-agents":
            return candidate
    raise ValueError("Unable to locate hello-agents/code root")


def load_shared_dotenv(code_root: Path | None = None) -> Path:
    root = code_root or find_code_root()
    env_path = root / ".env"
    load_dotenv(env_path, override=False)
    return env_path


def get_env_value(
    name: str,
    *,
    default: str | None = None,
    required: bool = False,
    code_root: Path | None = None,
) -> str | None:
    if name in SYSTEM_ONLY_KEYS:
        raise ValueError(f"{name} must be read with get_system_env()")

    env_path = load_shared_dotenv(code_root=code_root)
    env_values = dotenv_values(env_path)
    raw_value = env_values.get(name)
    value = raw_value if raw_value not in (None, "") else default

    if required and not value:
        raise ValueError(f"Missing required shared .env value: {name}")
    return value


def get_system_env(name: str) -> str:
    if name not in SYSTEM_ONLY_KEYS:
        raise ValueError(f"{name} is not a system-only key")
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required system environment variable: {name}")
    return value


def get_llm_config(*, code_root: Path | None = None) -> LLMConfig:
    return LLMConfig(
        model_id=get_env_value("MODEL_ID", required=True, code_root=code_root) or "",
        api_key=get_env_value("API_KEY", required=True, code_root=code_root) or "",
        base_url=get_env_value("BASE_URL", required=True, code_root=code_root) or "",
        timeout=int(get_env_value("LLM_TIMEOUT", default="60", code_root=code_root) or "60"),
    )
