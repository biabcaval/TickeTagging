"""Shared ``.env`` file loading, used by every backend's Settings.from_env."""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(".env")


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
