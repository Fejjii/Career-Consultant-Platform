"""Runtime credential/config resolution helpers for Streamlit execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse


@dataclass(frozen=True)
class OpenAIKeyResolution:
    """Resolved OpenAI key and source metadata."""

    api_key: str | None
    source: str
    source_label: str


@dataclass(frozen=True)
class QdrantConfigResolution:
    """Resolved Qdrant config and retrieval readiness metadata."""

    url: str | None
    api_key: str | None
    requires_api_key: bool
    retrieval_available: bool
    message: str | None


def _secret_value(
    name: str,
    *,
    secrets: Mapping[str, object] | None = None,
) -> str:
    if secrets is not None:
        value = str(secrets.get(name, "")).strip()
        if value:
            return value
    return (os.getenv(name) or "").strip()


def resolve_openai_api_key(
    *,
    user_api_key: str | None = None,
    secrets: Mapping[str, object] | None = None,
) -> OpenAIKeyResolution:
    """Resolve OpenAI API key with deterministic priority."""
    candidate_user = (user_api_key or "").strip()
    if candidate_user:
        return OpenAIKeyResolution(api_key=candidate_user, source="user", source_label="User key")

    secret_key = ""
    if secrets is not None:
        secret_key = str(secrets.get("OPENAI_API_KEY", "")).strip()
        if secret_key:
            return OpenAIKeyResolution(
                api_key=secret_key,
                source="app_managed_secret",
                source_label="App managed key",
            )

    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        return OpenAIKeyResolution(
            api_key=env_key,
            source="app_managed_env",
            source_label="App managed key",
        )

    return OpenAIKeyResolution(api_key=None, source="none", source_label="No key available")


def _qdrant_requires_api_key(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    # Qdrant Cloud endpoints generally require API keys.
    return host.endswith("cloud.qdrant.io") or "qdrant.tech" in host


def resolve_qdrant_config(
    *,
    secrets: Mapping[str, object] | None = None,
) -> QdrantConfigResolution:
    """Resolve Qdrant settings and return an actionable availability message."""
    qdrant_url = _secret_value("QDRANT_URL", secrets=secrets)
    if not qdrant_url:
        return QdrantConfigResolution(
            url=None,
            api_key=None,
            requires_api_key=False,
            retrieval_available=False,
            message="Retrieval unavailable because QDRANT_URL is not configured.",
        )

    qdrant_api_key = _secret_value("QDRANT_API_KEY", secrets=secrets)
    requires_key = _qdrant_requires_api_key(qdrant_url)
    if requires_key and not qdrant_api_key:
        return QdrantConfigResolution(
            url=qdrant_url,
            api_key=None,
            requires_api_key=True,
            retrieval_available=False,
            message="Retrieval unavailable because QDRANT_API_KEY is required for this Qdrant instance.",
        )

    return QdrantConfigResolution(
        url=qdrant_url,
        api_key=qdrant_api_key or None,
        requires_api_key=requires_key,
        retrieval_available=True,
        message=None,
    )

