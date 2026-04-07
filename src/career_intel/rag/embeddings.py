"""Embedding generation using centralized OpenAI client with retry."""

from __future__ import annotations

import asyncio

import structlog

from career_intel.config import Settings, get_settings
from career_intel.llm.clients import embed_with_retry, get_embeddings_client

logger = structlog.get_logger()


def get_embeddings(
    texts: list[str],
    settings: Settings | None = None,
) -> list[list[float]]:
    """Embed a batch of texts using the configured OpenAI embedding model.

    All calls go through ``embed_with_retry`` which handles 429/5xx with
    exponential backoff + jitter.
    """
    if settings is None:
        settings = get_settings()

    client = get_embeddings_client(settings)
    vectors = embed_with_retry(client, texts, settings.openai_embedding_model)

    logger.info("embeddings_created", count=len(vectors), model=settings.openai_embedding_model)
    return vectors


async def aget_embeddings(
    texts: list[str],
    settings: Settings | None = None,
) -> list[list[float]]:
    """Async wrapper — runs the sync embedding call in a thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embeddings, texts, settings)
