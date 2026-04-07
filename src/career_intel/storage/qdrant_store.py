"""Qdrant vector store helpers — collection management and upsert."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient, models

from career_intel.config import get_settings

logger = structlog.get_logger()

VECTOR_SIZE = 1536  # text-embedding-3-small default


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the collection if it does not already exist."""
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("qdrant_collection_created", name=settings.qdrant_collection)
    else:
        logger.info("qdrant_collection_exists", name=settings.qdrant_collection)


def upsert_vectors(
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    client: QdrantClient | None = None,
) -> None:
    """Batch upsert vectors with payloads into Qdrant."""
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    points = [
        models.PointStruct(id=uid, vector=vec, payload=pay)
        for uid, vec, pay in zip(ids, vectors, payloads, strict=True)
    ]

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    logger.info("qdrant_upsert", count=len(points))


def search_vectors(
    query_vector: list[float],
    top_k: int = 20,
    filters: dict[str, Any] | None = None,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """Search for similar vectors in Qdrant."""
    settings = get_settings()
    if client is None:
        client = get_qdrant_client()

    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        qdrant_filter = models.Filter(must=conditions)

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return results.points
