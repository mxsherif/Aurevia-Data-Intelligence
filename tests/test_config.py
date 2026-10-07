"""Tests for configuration loading and its no-API-key behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    SUPPORTED_EXTENSIONS,
    Settings,
    configure_logging,
    load_settings,
)

ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "DATAPILOT_MAX_UPLOAD_MB",
    "DATAPILOT_MAX_ROWS",
    "DATAPILOT_PREVIEW_ROWS",
    "DATAPILOT_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_without_any_env(tmp_path: Path):
    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.openai_api_key == ""
    assert settings.has_api_key is False
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.max_upload_mb == 200
    assert settings.max_rows == 500_000
    assert settings.preview_rows == 100


def test_env_file_is_read(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=sk-test-123\n"
        "OPENAI_MODEL=gpt-4.1\n"
        "DATAPILOT_PREVIEW_ROWS=42\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=env, override=True)

    assert settings.has_api_key is True
    assert settings.openai_model == "gpt-4.1"
    assert settings.preview_rows == 42


def test_placeholder_api_key_counts_as_unset(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "your_api_key_here")

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.has_api_key is False


def test_malformed_numeric_values_fall_back_to_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATAPILOT_MAX_ROWS", "not-a-number")
    monkeypatch.setenv("DATAPILOT_PREVIEW_ROWS", "-5")

    settings = load_settings(env_file=tmp_path / "absent.env")

    assert settings.max_rows == 500_000
    assert settings.preview_rows == 100


def test_unreadable_env_file_does_not_raise(tmp_path: Path):
    # A directory where a file is expected must not crash startup.
    directory = tmp_path / "env_dir"
    directory.mkdir()

    settings = load_settings(env_file=directory)

    assert settings.openai_model == "gpt-4.1-mini"


def test_describe_redacts_the_api_key(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")

    described = load_settings(env_file=tmp_path / "absent.env").describe()

    assert "sk-super-secret" not in str(described)
    assert described["api_key_configured"] is True


def test_settings_are_immutable():
    settings = Settings()
    with pytest.raises(Exception):
        settings.openai_model = "something-else"


def test_max_upload_bytes_conversion():
    assert Settings(max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024


def test_supported_extensions_cover_csv_and_excel():
    assert ".csv" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS


def test_configure_logging_is_idempotent():
    configure_logging(Settings(log_level="WARNING"))
    configure_logging(Settings(log_level="DEBUG"))  # must not raise or duplicate


def test_unknown_log_level_falls_back_to_info():
    configure_logging(Settings(log_level="NOT_A_LEVEL"))
