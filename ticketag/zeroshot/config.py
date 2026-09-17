"""Runtime settings: which Gemini model to call and how to tune retries/rate limits."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..envfile import ENV_FILE, load_env_file
from ..exceptions import ConfigurationError

DEFAULT_MODEL = "gemini-2.5-flash"

API_KEY_ENV_VAR = "GEMINI_API_KEY"


# Settings overridable through TICKETAG_<NAME> variables, with their parsers.
ENV_FIELDS: dict[str, Callable[[str], object]] = {
    "model": str,
    "timeout": float,
    "max_retries": int,
    "max_tokens": int,
    "requests_per_minute": int,
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration for calling the Gemini Developer API through `google-genai`."""

    api_key: str
    model: str = DEFAULT_MODEL
    timeout: float = 60.0
    max_retries: int = 3
    backoff_seconds: float = 2.0
    temperature: float = 0.0
    max_tokens: int = 900
    requests_per_minute: int = 20

    @classmethod
    def from_env(cls, env_file: Path = ENV_FILE, **overrides: object) -> Self:
        """Build settings from environment variables, loading a local .env first."""
        load_env_file(env_file)
        api_key = os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            raise ConfigurationError(
                f"Missing API key: set {API_KEY_ENV_VAR} in the environment or in .env"
            )
        values: dict[str, object] = {"api_key": api_key}
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
