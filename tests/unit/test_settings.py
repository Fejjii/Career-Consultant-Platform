"""Tests for settings and configuration."""

from __future__ import annotations

from career_intel.config import Settings


def test_settings_loads_from_env(monkeypatch: object) -> None:
    """Settings should load without error when required env vars are present."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.openai_model == "gpt-4o"
    assert settings.rate_limit_rpm == 30
    assert settings.max_input_length == 4000
    assert settings.max_speech_file_bytes > 0
    assert "wav" in settings.speech_allowed_extensions
    assert "mp4" in settings.speech_allowed_extensions
    assert settings.openai_transcription_model
    assert settings.qdrant_timeout_seconds == 120.0
    assert settings.openai_embedding_timeout_seconds > 0
    assert settings.speech_transcription_timeout_seconds > 0
    assert settings.rag_similarity_threshold == 0.55
    assert settings.rag_strong_evidence_threshold == 0.60
    assert settings.rag_rerank_coherence_threshold == 0.48
    assert settings.rag_force_min_chunks == 3


def test_secret_str_not_in_repr(monkeypatch: object) -> None:
    """SecretStr fields should not leak in repr."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    settings = Settings()  # type: ignore[call-arg]
    settings_repr = repr(settings)
    assert "sk-test" not in settings_repr
