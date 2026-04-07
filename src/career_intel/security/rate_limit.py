"""Redis-backed sliding-window rate limiter with session keying and local-dev fallback.

Key design decisions:
  - Primary key: IP address.  Secondary key: session_id (from request body
    or header) when available, to prevent a single session from monopolising
    the shared IP bucket.
  - Redis sorted-set sliding window (O(log n) per check).
  - When Redis is unavailable **and** ENVIRONMENT=development, the limiter
    fails open with a log warning.  In staging/production it fails closed
    (503) to prevent unmetered abuse.
"""

from __future__ import annotations

import time

import structlog
from fastapi import HTTPException, Request

from career_intel.config import get_settings

logger = structlog.get_logger()


async def check_rate_limit(request: Request) -> None:
    """Enforce per-IP and per-session rate limits using Redis sliding window."""
    settings = get_settings()
    rpm = settings.rate_limit_rpm

    client_ip = _get_client_ip(request)
    session_id = _get_session_id(request)

    ip_key = f"ratelimit:ip:{client_ip}"
    keys_to_check = [ip_key]
    if session_id:
        keys_to_check.append(f"ratelimit:session:{session_id}")

    try:
        from career_intel.storage.redis_cache import get_redis

        redis = await get_redis()
        now = time.time()
        window_start = now - 60

        for key in keys_to_check:
            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, "-inf", window_start)
            pipe.zadd(key, {f"{now}:{id(request)}": now})
            pipe.zcard(key)
            pipe.expire(key, 120)
            results = await pipe.execute()
            request_count = results[2]

            if request_count > rpm:
                logger.warning(
                    "rate_limited",
                    client_ip=client_ip,
                    session_id=session_id,
                    key=key,
                    count=request_count,
                    limit=rpm,
                    endpoint=str(request.url.path),
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Maximum {rpm} requests per minute.",
                )

    except HTTPException:
        raise
    except Exception as exc:
        if settings.environment == "development":
            logger.warning(
                "rate_limit_redis_unavailable_dev_failopen",
                error=str(exc)[:200],
            )
            return
        # In staging/production: fail closed to prevent unmetered abuse
        logger.error(
            "rate_limit_redis_unavailable_failclosed",
            error=str(exc)[:200],
            environment=settings.environment,
        )
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable (rate limiter backend).",
        ) from exc


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For if present."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_session_id(request: Request) -> str | None:
    """Try to extract a session ID from headers or query params."""
    return (
        request.headers.get("X-Session-ID")
        or request.query_params.get("session_id")
        or None
    )
