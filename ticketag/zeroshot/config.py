"""Runtime settings: which endpoint to call, with which key and limits."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..exceptions import ConfigurationError

ENV_FILE = Path(".env")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TOKEN_ENV_VAR = "OPENROUTER_API_KEY"


def _csv_tuple(value: str) -> tuple[str, ...]:
    """Parse a comma-separated env var into a tuple, dropping blanks."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


# Settings overridable through TICKETAG_<NAME> variables, with their parsers.
ENV_FIELDS: dict[str, Callable[[str], object]] = {
    "model": str,
    "fallback_models": _csv_tuple,
    "base_url": str,
    "timeout": float,
    "max_retries": int,
    "max_tokens": int,
    "requests_per_minute": int,
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration for the chat completion API calls.

    ``base_url`` points at any OpenAI-compatible endpoint. ``fallback_models``
    lets OpenRouter reroute when the primary model is saturated or down.
    """

    api_token: str
    model: str = "nex-agi/nex-n2.5-mini:free"
    fallback_models: tuple[str, ...] = ()
    base_url: str = OPENROUTER_BASE_URL
    timeout: float = 60.0
    max_retries: int = 3
    backoff_seconds: float = 2.0
    temperature: float = 0.0
    # Generous, because reasoning models bill their hidden thinking to this budget.
    max_tokens: int = 900
    requests_per_minute: int = 20

    @classmethod
    def from_env(cls, env_file: Path = ENV_FILE, **overrides: object) -> Self:
        """Build settings from environment variables, loading a local .env first."""
        load_env_file(env_file)
        token = os.getenv(TOKEN_ENV_VAR)
        if not token:
            raise ConfigurationError(
                f"Missing API key: set {TOKEN_ENV_VAR} in the environment or in .env"
            )
        values: dict[str, object] = {"api_token": token}
        for name, cast in ENV_FIELDS.items():
            raw = os.getenv(f"TICKETAG_{name.upper()}")
            if raw is None:
                continue
            try:
                values[name] = cast(raw)
            except ValueError as error:
                raise ConfigurationError(
                    f"Invalid value for TICKETAG_{name.upper()}: {raw!r}"
                ) from error
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)  # type: ignore[arg-type]


def load_env_file(env_file: Path = ENV_FILE) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real env vars."""
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
