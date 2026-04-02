"""Tests for settings and configuration."""

from __future__ import annotations

from career_intel.config import Settings


def test_settings_loads_from_env(monkeypatch: object) -> None:
    """Settings should load without error when required env vars are present."""
    import os

    os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-not-real")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.openai_model == "gpt-4o"
    assert settings.rate_limit_rpm == 30
    assert settings.max_input_length == 4000


def test_secret_str_not_in_repr() -> None:
    """SecretStr fields should not leak in repr."""
    import os

    os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-not-real")
    settings = Settings()  # type: ignore[call-arg]
    settings_repr = repr(settings)
    assert "sk-test" not in settings_repr
