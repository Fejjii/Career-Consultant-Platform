"""Tests for Qdrant connection resolution."""

from __future__ import annotations

from pydantic import SecretStr

from career_intel.config import Settings
from career_intel.storage.qdrant_store import (
    QdrantConfigurationError,
    get_qdrant_client,
    resolve_qdrant_config,
)


def _settings(**overrides) -> Settings:
    base = {
        "openai_api_key": SecretStr("sk-test-key"),
        "qdrant_url": "https://example.cloud.qdrant.io",
        "qdrant_api_key": SecretStr("qdrant-test-key"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_resolve_qdrant_config_rejects_bind_address() -> None:
    """Client-side Qdrant URLs must not use bind-all addresses."""
    settings = _settings(qdrant_url="http://0.0.0.0:6333", qdrant_api_key=None)

    try:
        resolve_qdrant_config(settings)
    except QdrantConfigurationError as exc:
        assert "bind address" in str(exc).lower()
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected bind-address validation error")


def test_get_qdrant_client_uses_rest_only_and_api_key(monkeypatch) -> None:
    """Qdrant client should be created with explicit REST-safe settings."""
    captured: dict[str, object] = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("career_intel.storage.qdrant_store.QdrantClient", fake_client)

    client = get_qdrant_client(_settings())

    assert client is not None
    assert captured["url"] == "https://example.cloud.qdrant.io"
    assert captured["api_key"] == "qdrant-test-key"
    assert captured["prefer_grpc"] is False
