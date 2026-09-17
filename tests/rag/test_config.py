from __future__ import annotations

import pytest

from ticketag.exceptions import ConfigurationError
from ticketag.rag.config import API_KEY_ENV_VAR, Settings


def test_settings_default_collection_and_tuning_values():
    settings = Settings(chroma_api_key="ck-test")

    assert settings.collection_name == "tickets"
    assert settings.k == 5
    assert settings.auto_add_threshold == pytest.approx(0.8)
    assert settings.chroma_tenant is None
    assert settings.chroma_database is None


def test_settings_from_env_raises_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    with pytest.raises(ConfigurationError, match="Missing API key"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_reads_the_chroma_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("CHROMA_TENANT", "tenant-1")
    monkeypatch.setenv("CHROMA_DATABASE", "db-1")

    settings = Settings.from_env(env_file=tmp_path / "absent.env")

    assert settings.chroma_api_key == "ck-abc"
    assert settings.chroma_tenant == "tenant-1"
    assert settings.chroma_database == "db-1"


def test_settings_from_env_reads_rag_tuning_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_COLLECTION", "custom-tickets")
    monkeypatch.setenv("TICKETAG_RAG_K", "3")
    monkeypatch.setenv("TICKETAG_RAG_AUTO_ADD_THRESHOLD", "0.9")

    settings = Settings.from_env(env_file=tmp_path / "absent.env")

    assert settings.collection_name == "custom-tickets"
    assert settings.k == 3
    assert settings.auto_add_threshold == pytest.approx(0.9)


def test_settings_from_env_rejects_an_unparsable_k(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_K", "many")

    with pytest.raises(ConfigurationError, match="TICKETAG_RAG_K"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_rejects_an_unparsable_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_AUTO_ADD_THRESHOLD", "high")

    with pytest.raises(ConfigurationError, match="TICKETAG_RAG_AUTO_ADD_THRESHOLD"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_applies_keyword_overrides_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_K", "3")

    settings = Settings.from_env(env_file=tmp_path / "absent.env", k=7)

    assert settings.k == 7
