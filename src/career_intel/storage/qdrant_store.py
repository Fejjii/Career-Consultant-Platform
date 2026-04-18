"""Qdrant vector store helpers — collection management and upsert."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any

import structlog
from qdrant_client import QdrantClient, models

from career_intel.config import Settings, get_settings

logger = structlog.get_logger()

VECTOR_SIZE = 1536  # text-embedding-3-small default
ESCO_DOC_TYPES: tuple[str, ...] = (
    "occupation_summary",
    "skill_summary",
    "relation_detail",
    "taxonomy_mapping",
    "isco_group_summary",
)


@dataclass(frozen=True, slots=True)
class ResolvedQdrantConfig:
    """Cloud-safe Qdrant connection settings."""

    url: str
    api_key: str | None
    timeout_seconds: float
    prefer_grpc: bool


class QdrantConfigurationError(ValueError):
    """Raised when the Qdrant connection settings are invalid."""


_QDRANT_BIND_HOSTS = {"0.0.0.0", "::"}


def resolve_qdrant_config(settings: Settings | None = None) -> ResolvedQdrantConfig:
    """Resolve and validate the canonical Qdrant client configuration."""
    settings = settings or get_settings()
    raw_url = settings.qdrant_url.strip()
    if not raw_url:
        raise QdrantConfigurationError("QDRANT_URL is not configured.")

    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise QdrantConfigurationError(
            "QDRANT_URL must be a full http(s) URL, for example "
            "'https://your-cluster.cloud.qdrant.io'."
        )

    host = (parsed.hostname or "").strip().lower()
    if host in _QDRANT_BIND_HOSTS:
        raise QdrantConfigurationError(
            "QDRANT_URL points to a bind address that clients cannot reach. "
            "Use the actual Qdrant hostname instead of 0.0.0.0/::."
        )

    api_key = settings.qdrant_api_key.get_secret_value().strip() if settings.qdrant_api_key else ""
    if _qdrant_requires_api_key(host) and not api_key:
        raise QdrantConfigurationError(
            "QDRANT_API_KEY is required for this Qdrant host."
        )
    if _qdrant_requires_api_key(host) and parsed.scheme != "https":
        raise QdrantConfigurationError(
            "Qdrant Cloud URLs must use https."
        )

    return ResolvedQdrantConfig(
        url=raw_url,
        api_key=api_key or None,
        timeout_seconds=settings.qdrant_timeout_seconds,
        prefer_grpc=False,
    )


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    resolved = resolve_qdrant_config(settings)
    return QdrantClient(
        url=resolved.url,
        api_key=resolved.api_key,
        timeout=resolved.timeout_seconds,
        prefer_grpc=resolved.prefer_grpc,
    )


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the collection if it does not already exist."""
    settings = get_settings()
    if client is None:
        client = get_qdrant_client(settings)

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
        client = get_qdrant_client(settings)

    points = [
        models.PointStruct(id=uid, vector=vec, payload=pay)
        for uid, vec, pay in zip(ids, vectors, payloads, strict=True)
    ]

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    logger.info("qdrant_upsert", count=len(points))


def delete_vectors_by_metadata(
    metadata_filters: dict[str, Any],
    client: QdrantClient | None = None,
) -> None:
    """Delete points matching metadata filters.

    Used by ingestion to make document re-runs idempotent when a file changes.
    """
    settings = get_settings()
    if client is None:
        client = get_qdrant_client(settings)

    conditions = [
        models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )
        for key, value in metadata_filters.items()
    ]

    if not conditions:
        return

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=conditions),
        ),
    )
    logger.info("qdrant_delete_by_metadata", filters=metadata_filters)


def search_vectors(
    query_vector: list[float],
    top_k: int = 20,
    filters: dict[str, Any] | None = None,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """Search for similar vectors in Qdrant."""
    settings = get_settings()
    if client is None:
        client = get_qdrant_client(settings)

    qdrant_filter = _build_qdrant_filter(filters)

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def count_vectors(
    filters: dict[str, Any] | None = None,
    client: QdrantClient | None = None,
) -> int:
    """Count points matching the given metadata filters."""
    settings = get_settings()
    if client is None:
        client = get_qdrant_client(settings)

    result = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=_build_qdrant_filter(filters),
        exact=True,
    )
    return int(result.count)


def sample_payloads(
    filters: dict[str, Any] | None = None,
    *,
    limit: int = 1,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Return a small payload sample for debugging filter behavior."""
    settings = get_settings()
    if client is None:
        client = get_qdrant_client(settings)

    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=_build_qdrant_filter(filters),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [point.payload or {} for point in points]


def get_esco_vector_diagnostics(
    *,
    doc_types: Sequence[str] = ESCO_DOC_TYPES,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Return grouped ESCO vector diagnostics for live verification."""
    total_esco_vectors = count_vectors(filters={"source": "esco"}, client=client)
    counts_by_doc_type: dict[str, int] = {}
    sample_payloads_by_doc_type: dict[str, dict[str, Any]] = {}

    for doc_type in doc_types:
        filters = {"source": "esco", "esco_doc_type": doc_type}
        count = count_vectors(filters=filters, client=client)
        counts_by_doc_type[doc_type] = count
        if count <= 0:
            continue
        samples = sample_payloads(filters=filters, limit=1, client=client)
        if samples:
            sample_payloads_by_doc_type[doc_type] = _sanitize_payload_for_diagnostics(samples[0])

    return {
        "total_esco_vectors": total_esco_vectors,
        "counts_by_esco_doc_type": counts_by_doc_type,
        "sample_payloads_by_esco_doc_type": sample_payloads_by_doc_type,
    }


def _build_qdrant_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Translate generic metadata filters into a Qdrant filter."""
    if not filters:
        return None

    must_conditions = []
    should_conditions = []
    for key, value in filters.items():
        if key == "source":
            should_conditions.extend(
                [
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=value),
                    ),
                    models.FieldCondition(
                        key="source_name",
                        match=models.MatchValue(value=value),
                    ),
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchValue(value=value),
                    ),
                ]
            )
            continue

        must_conditions.append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
        )

    return models.Filter(
        must=must_conditions or None,
        should=should_conditions or None,
    )


def _sanitize_payload_for_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep verification output compact and focused on grounding metadata."""
    return {
        key: payload.get(key)
        for key in (
            "source",
            "source_name",
            "source_type",
            "file_name",
            "document_title",
            "title",
            "topic",
            "entity_type",
            "esco_doc_type",
            "section_title",
            "occupation_id",
            "occupation_label",
            "occupation_code",
            "skill_id",
            "skill_label",
            "relation_type",
            "isco_group",
            "isco_group_label",
            "uri",
        )
    }


def _qdrant_requires_api_key(host: str) -> bool:
    """Return True when the host looks like a managed Qdrant deployment."""
    return host.endswith("cloud.qdrant.io") or "qdrant.tech" in host
