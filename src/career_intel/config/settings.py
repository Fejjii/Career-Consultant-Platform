"""Centralised application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All configurable knobs for the Career Intelligence backend.

    Values are read from a ``.env`` file (if present) then overridden by
    real environment variables.  Secrets use ``SecretStr`` so they never
    appear in repr/logs.
    """

    model_config = SettingsConfigDict(
        env_file=(
            str(_PROJECT_ROOT / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_timeout_seconds: float = 60.0
    openai_embedding_max_attempts: int = 2

    # --- Vector DB ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "career_intel"
    qdrant_timeout_seconds: float = 120.0

    # --- Postgres ---
    postgres_dsn: str = "postgresql+asyncpg://career:career@localhost:5432/career_intel"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Observability ---
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr | None = None
    langchain_project: str = "career-intel-dev"

    # --- Security ---
    admin_secret: SecretStr = SecretStr("change-me-in-production")
    rate_limit_rpm: int = 30
    rate_limit_chat_rpm: int = 30
    rate_limit_speech_rpm: int = 10
    rate_limit_byok_validation_rpm: int = 10
    rate_limit_feedback_rpm: int = 20
    rate_limit_ingest_rpm: int = 5
    max_input_length: int = 4000
    max_cv_file_bytes: int = 5 * 1024 * 1024  # 5 MB
    supported_openai_chat_models: str = "gpt-4.1,gpt-4o,gpt-4o-mini"

    # --- Speech-to-text (upload only; transcript is reviewed before chat) ---
    max_speech_file_bytes: int = 25 * 1024 * 1024  # 25 MB
    speech_allowed_extensions: str = "wav,mp3,m4a,webm,mp4"
    openai_transcription_model: str = "whisper-1"
    speech_transcription_timeout_seconds: float = 120.0

    # --- RAG / ingestion ---
    data_raw_dir: str = "data/raw"
    rag_chunk_size_tokens: int = 400
    rag_chunk_overlap_tokens: int = 80
    rag_initial_top_k: int = 15
    rag_top_k: int = 8
    rag_enable_reranking: bool = True
    rag_similarity_threshold: float = 0.55
    rag_weak_evidence_threshold: float = 0.30
    rag_strong_evidence_threshold: float = 0.60
    rag_rerank_coherence_threshold: float = 0.48
    rag_force_min_chunks: int = 3
    rag_embedding_batch_size: int = 8
    rag_ingest_debug: bool = False

    # --- App ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"
    runtime_default_timezone: str = "UTC"


@lru_cache(maxsize=1)
def _get_base_settings() -> Settings:
    """Return cached app-managed settings loaded from env or .env."""
    return Settings()  # type: ignore[call-arg]


def get_settings() -> Settings:
    """Return settings, honoring any request-scoped BYOK override.

    Direct-mode Streamlit requests can carry a user API key without an app-managed
    `OPENAI_API_KEY`. When a request-scoped override is present, build a fresh
    settings object for that request so any nested `get_settings()` calls inherit
    the active BYOK instead of failing validation.
    """
    from career_intel.llm.request_context import get_request_api_key_override

    override_key = (get_request_api_key_override() or "").strip()
    if override_key:
        return Settings(openai_api_key=SecretStr(override_key))  # type: ignore[call-arg]
    return _get_base_settings()
