"""Runtime settings: which Chroma Cloud database to use and how to tune retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..envfile import ENV_FILE, load_env_file
from ..exceptions import ConfigurationError

API_KEY_ENV_VAR = "CHROMA_API_KEY"
DEFAULT_COLLECTION_NAME = "ticketag_tickets"


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration for connecting to Chroma Cloud and tuning the k-NN vote.

    ``chroma_tenant`` and ``chroma_database`` are optional: Chroma Cloud can
    auto-resolve both when the API key is scoped to a single database.
    """

    chroma_api_key: str
    chroma_tenant: str | None = None
    chroma_database: str | None = None
    collection_name: str = DEFAULT_COLLECTION_NAME
    k: int = 5
    auto_add_threshold: float = 0.8

    @classmethod
    def from_env(cls, env_file: Path = ENV_FILE, **overrides: object) -> Self:
        """Build settings from environment variables, loading a local .env first."""
        load_env_file(env_file)
        api_key = os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            raise ConfigurationError(
                f"Missing API key: set {API_KEY_ENV_VAR} in the environment or in .env"
            )
        values: dict[str, object] = {
            "chroma_api_key": api_key,
            "chroma_tenant": os.getenv("CHROMA_TENANT"),
            "chroma_database": os.getenv("CHROMA_DATABASE"),
        }
        collection = os.getenv("TICKETAG_RAG_COLLECTION")
        if collection:
            values["collection_name"] = collection
        raw_k = os.getenv("TICKETAG_RAG_K")
        if raw_k is not None:
            try:
                values["k"] = int(raw_k)
            except ValueError as error:
                raise ConfigurationError(f"Invalid value for TICKETAG_RAG_K: {raw_k!r}") from error
        raw_threshold = os.getenv("TICKETAG_RAG_AUTO_ADD_THRESHOLD")
        if raw_threshold is not None:
            try:
                values["auto_add_threshold"] = float(raw_threshold)
            except ValueError as error:
                raise ConfigurationError(
                    f"Invalid value for TICKETAG_RAG_AUTO_ADD_THRESHOLD: {raw_threshold!r}"
                ) from error
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)  # type: ignore[arg-type]
