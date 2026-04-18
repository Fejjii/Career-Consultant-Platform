"""Tests for Streamlit runtime configuration resolution."""

from __future__ import annotations

from pathlib import Path

import runtime_config


def test_resolve_openai_key_from_dotenv_when_env_missing(tmp_path, monkeypatch) -> None:
    """OpenAI key resolution should read .env candidates when env is empty."""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=sk-test-key\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_config,
        "_dotenv_candidates",
        lambda: (dotenv_path, dotenv_path),
    )
    monkeypatch.setattr(runtime_config, "_DOTENV_CACHE", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = runtime_config.resolve_openai_api_key()

    assert resolved.api_key == "sk-test-key"
    assert resolved.source == "app_managed_env"


def test_resolve_qdrant_uses_local_default_when_unset(monkeypatch) -> None:
    """Retrieval should stay enabled with a local default URL in dev mode."""
    missing_env = Path("__missing__.env")
    monkeypatch.setattr(
        runtime_config,
        "_dotenv_candidates",
        lambda: (missing_env, missing_env),
    )
    monkeypatch.setattr(runtime_config, "_DOTENV_CACHE", None)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    resolved = runtime_config.resolve_qdrant_config()

    assert resolved.retrieval_available is True
    assert resolved.url == "http://localhost:6333"
    assert resolved.message is None
