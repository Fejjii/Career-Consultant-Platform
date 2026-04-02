"""Redis client wrapper for caching and rate limiting."""

from __future__ import annotations

import redis.asyncio as aioredis

from career_intel.config import get_settings

_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return a shared async Redis client."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _pool


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
