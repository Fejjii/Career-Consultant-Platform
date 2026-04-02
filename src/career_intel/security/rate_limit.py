"""Redis-backed sliding window rate limiter."""

from __future__ import annotations

import time

import structlog
from fastapi import HTTPException, Request

from career_intel.config import get_settings

logger = structlog.get_logger()


async def check_rate_limit(request: Request) -> None:
    """Enforce per-IP rate limit using Redis sliding window.

    Call as a FastAPI dependency or middleware check.
    Raises HTTPException(429) if the limit is exceeded.
    """
    settings = get_settings()
    rpm = settings.rate_limit_rpm

    client_ip = _get_client_ip(request)
    key = f"ratelimit:{client_ip}"

    try:
        from career_intel.storage.redis_cache import get_redis

        redis = await get_redis()
        now = time.time()
        window_start = now - 60

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, 120)
        results = await pipe.execute()
        request_count = results[2]

        if request_count > rpm:
            logger.warning(
                "rate_limited",
                client_ip=client_ip,
                count=request_count,
                limit=rpm,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {rpm} requests per minute.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        # If Redis is down, log and allow the request (fail open)
        logger.error("rate_limit_redis_error", error=str(exc))


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For if present."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
