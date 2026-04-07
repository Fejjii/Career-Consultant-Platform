"""Health and readiness endpoints."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter

from career_intel.config import get_settings
from career_intel.schemas.api import (
    HealthResponse,
    ReadyDetail,
    ReadyResponse,
    SystemStatusResponse,
)

router = APIRouter(tags=["health"])
logger = structlog.get_logger()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness() -> ReadyResponse:
    """Check connectivity to Postgres, Redis, and Qdrant."""
    settings = get_settings()
    details: list[ReadyDetail] = []

    # --- Qdrant ---
    try:
        from qdrant_client import QdrantClient

        t0 = time.monotonic()
        qc = QdrantClient(url=settings.qdrant_url, timeout=3)
        qc.get_collections()
        details.append(ReadyDetail(
            name="qdrant",
            ok=True,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        ))
    except Exception as exc:
        details.append(ReadyDetail(name="qdrant", ok=False, error=str(exc)))

    # --- Redis ---
    try:
        import redis.asyncio as aioredis

        t0 = time.monotonic()
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        await r.ping()
        await r.aclose()
        details.append(ReadyDetail(
            name="redis",
            ok=True,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        ))
    except Exception as exc:
        details.append(ReadyDetail(name="redis", ok=False, error=str(exc)))

    # --- Postgres ---
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        t0 = time.monotonic()
        engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        details.append(ReadyDetail(
            name="postgres",
            ok=True,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        ))
    except Exception as exc:
        details.append(ReadyDetail(name="postgres", ok=False, error=str(exc)))

    all_ok = all(d.ok for d in details)
    return ReadyResponse(ok=all_ok, details=details)


@router.get("/health/system", response_model=SystemStatusResponse)
async def system_status() -> SystemStatusResponse:
    """Return lightweight status for UI smoke tests."""
    settings = get_settings()
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=2)
        collections = {c.name for c in client.get_collections().collections}
        if settings.qdrant_collection not in collections:
            return SystemStatusResponse(
                backend=True,
                qdrant=True,
                indexed_data_present=False,
                collection=settings.qdrant_collection,
                points_count=0,
                error=f"Collection '{settings.qdrant_collection}' not found. Run ingestion.",
            )

        count = client.count(collection_name=settings.qdrant_collection, exact=False).count
        return SystemStatusResponse(
            backend=True,
            qdrant=True,
            indexed_data_present=count > 0,
            collection=settings.qdrant_collection,
            points_count=count,
            error=None if count > 0 else "No indexed data found. Run ingestion.",
        )
    except Exception as exc:
        return SystemStatusResponse(
            backend=True,
            qdrant=False,
            indexed_data_present=False,
            collection=settings.qdrant_collection,
            points_count=0,
            error=str(exc),
        )
