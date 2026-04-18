"""Embedding generation using centralized OpenAI client with retry."""

from __future__ import annotations

import asyncio
import time

import structlog

from career_intel.config import Settings, get_settings
from career_intel.llm.clients import embed_with_retry, get_embeddings_client

logger = structlog.get_logger()


def get_embeddings(
    texts: list[str],
    settings: Settings | None = None,
    *,
    request_label: str = "embedding_request",
) -> list[list[float]]:
    """Embed a batch of texts using the configured OpenAI embedding model.

    All calls go through ``embed_with_retry`` with explicit attempt logging.
    """
    if settings is None:
        settings = get_settings()

    started = time.perf_counter()
    client = get_embeddings_client(settings)
    try:
        vectors = embed_with_retry(
            client,
            texts,
            settings.openai_embedding_model,
            max_attempts=settings.openai_embedding_max_attempts,
            request_label=request_label,
        )
    except Exception as exc:
        logger.exception(
            "embeddings_request_failed",
            request_label=request_label,
            count=len(texts),
            model=settings.openai_embedding_model,
            error_type=type(exc).__name__,
        )
        raise

    logger.info(
        "embeddings_created",
        request_label=request_label,
        count=len(vectors),
        model=settings.openai_embedding_model,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return vectors


async def aget_embeddings(
    texts: list[str],
    settings: Settings | None = None,
) -> list[list[float]]:
    """Async wrapper — runs the sync embedding call in a thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embeddings, texts, settings)
