import os
from typing import Any

from langchain_openai import ChatOpenAI


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    if fallback:
        return os.getenv(fallback)
    return None


def _env_float(name: str, fallback: str | None = None, default: float = 0.0) -> float:
    value = _env(name, fallback)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def llm_config() -> dict[str, Any]:
    """Resolve LLM config from env with backward-compatible fallbacks."""
    return {
        "model": _env("LLM_MODEL", "ORCH_MODEL") or "gpt-4o-mini",
        "api_key": _env("LLM_API_KEY", "GATOR_API_KEY"),
        "base_url": _env("LLM_BASE_URL", "GATOR_BASE_URL"),
        "temperature": _env_float("LLM_TEMPERATURE", default=0.0),
    }


def get_chat_model(**overrides: Any) -> ChatOpenAI:
    """Create a ChatOpenAI client using centralized config."""
    config = llm_config()
    config.update({k: v for k, v in overrides.items() if v is not None})
    return ChatOpenAI(**config)


def get_structured_chat_model(schema: Any, **overrides: Any):
    """Create a structured-output model with centralized config."""
    return get_chat_model(**overrides).with_structured_output(schema)
