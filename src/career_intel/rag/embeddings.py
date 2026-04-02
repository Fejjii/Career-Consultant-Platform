"""Embedding generation using OpenAI."""

from __future__ import annotations

import structlog
from openai import OpenAI

from career_intel.config import Settings, get_settings

logger = structlog.get_logger()


def get_embeddings(
    texts: list[str],
    settings: Settings | None = None,
) -> list[list[float]]:
    """Embed a batch of texts using the configured OpenAI embedding model.

    Returns a list of float vectors (one per input text).
    """
    if settings is None:
        settings = get_settings()

    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

    response = client.embeddings.create(
        input=texts,
        model=settings.openai_embedding_model,
    )

    vectors = [item.embedding for item in response.data]
    logger.info("embeddings_created", count=len(vectors), model=settings.openai_embedding_model)
    return vectors


async def aget_embeddings(
    texts: list[str],
    settings: Settings | None = None,
) -> list[list[float]]:
    """Async wrapper around the sync embedding call.

    OpenAI's Python SDK embedding endpoint is synchronous; we run it
    in a thread to avoid blocking the event loop.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embeddings, texts, settings)
