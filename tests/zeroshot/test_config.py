from __future__ import annotations

import pytest

from ticketag.exceptions import ConfigurationError
from ticketag.zeroshot.config import OPENROUTER_BASE_URL, TOKEN_ENV_VAR, Settings


def test_settings_default_to_openrouter():
    settings = Settings(api_token="sk-or-test")

    assert settings.base_url == OPENROUTER_BASE_URL
    assert settings.model.endswith(":free")


def test_settings_from_env_raises_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    with pytest.raises(ConfigurationError, match="Missing API key"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_reads_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-abc")
    monkeypatch.setenv("TICKETAG_MAX_RETRIES", "5")

    settings = Settings.from_env(env_file=tmp_path / "absent.env", model="custom/model")

    assert settings.api_token == "sk-or-abc"
    assert settings.model == "custom/model"
    assert settings.max_retries == 5


def test_settings_rejects_unparsable_env_value(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-abc")
    monkeypatch.setenv("TICKETAG_MAX_RETRIES", "many")

    with pytest.raises(ConfigurationError, match="TICKETAG_MAX_RETRIES"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_parses_fallback_models_from_a_comma_separated_list(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, "sk-or-abc")
    monkeypatch.setenv("TICKETAG_FALLBACK_MODELS", "a/one:free, b/two:free ,")

    settings = Settings.from_env(env_file=tmp_path / "absent.env")

    assert settings.fallback_models == ("a/one:free", "b/two:free")


def test_settings_point_at_another_openai_compatible_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, "sk-or-abc")
    monkeypatch.setenv("TICKETAG_BASE_URL", "https://example.test/v1")

    assert Settings.from_env(env_file=tmp_path / "absent.env").base_url == "https://example.test/v1"
