"""Runtime credential/config resolution helpers for Streamlit execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

_LOCAL_QDRANT_DEFAULT_URL = "http://localhost:6333"
_DOTENV_CACHE: dict[str, str] | None = None


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
    env_value = (os.getenv(name) or "").strip()
    if env_value:
        return env_value
    return _dotenv_value(name)


def _dotenv_value(name: str) -> str:
    dotenv_values = _load_dotenv_values()
    return dotenv_values.get(name, "")


def _load_dotenv_values() -> dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE

    values: dict[str, str] = {}
    for env_path in _dotenv_candidates():
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            raw_key, raw_value = stripped.split("=", 1)
            key = raw_key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            if not key:
                continue
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key not in values and value:
                values[key] = value

    _DOTENV_CACHE = values
    return values


def _dotenv_candidates() -> tuple[Path, Path]:
    project_root_env = Path(__file__).resolve().parents[1] / ".env"
    cwd_env = Path.cwd() / ".env"
    return project_root_env, cwd_env


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

    env_key = _secret_value("OPENAI_API_KEY")
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
    qdrant_url = _secret_value("QDRANT_URL", secrets=secrets) or _LOCAL_QDRANT_DEFAULT_URL

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

