"""Centralised application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable knobs for the Career Intelligence backend.

    Values are read from a `.env` file (if present) then overridden by
    real environment variables.  Secrets use ``SecretStr`` so they never
    appear in repr/logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Vector DB ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "career_intel"

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
    max_input_length: int = 4000

    # --- App ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance."""
    return Settings()  # type: ignore[call-arg]
