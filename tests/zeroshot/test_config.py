from __future__ import annotations

import pytest

from ticketag.exceptions import ConfigurationError
from ticketag.zeroshot.config import API_KEY_ENV_VAR, DEFAULT_MODEL, Settings


def test_settings_default_to_gemini_flash():
    settings = Settings(api_key="test-key")

    assert settings.model == DEFAULT_MODEL
    assert settings.model.startswith("gemini-")


def test_settings_from_env_raises_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    with pytest.raises(ConfigurationError, match="Missing API key"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_reads_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-abc")
    monkeypatch.setenv("TICKETAG_MAX_RETRIES", "5")

    settings = Settings.from_env(env_file=tmp_path / "absent.env", model="gemini-2.5-pro")

    assert settings.api_key == "test-key-abc"
    assert settings.model == "gemini-2.5-pro"
    assert settings.max_retries == 5


def test_settings_rejects_unparsable_env_value(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-abc")
    monkeypatch.setenv("TICKETAG_MAX_RETRIES", "many")

    with pytest.raises(ConfigurationError, match="TICKETAG_MAX_RETRIES"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_reads_the_requests_per_minute_and_max_tokens_overrides(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-abc")
    monkeypatch.setenv("TICKETAG_REQUESTS_PER_MINUTE", "10")
    monkeypatch.setenv("TICKETAG_MAX_TOKENS", "500")

    settings = Settings.from_env(env_file=tmp_path / "absent.env")

    assert settings.requests_per_minute == 10
    assert settings.max_tokens == 500


def test_settings_applies_keyword_overrides_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key-abc")
    monkeypatch.setenv("TICKETAG_MAX_RETRIES", "5")

    settings = Settings.from_env(env_file=tmp_path / "absent.env", max_retries=1)

    assert settings.max_retries == 1
