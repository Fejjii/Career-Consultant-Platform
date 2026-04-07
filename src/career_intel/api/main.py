"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from career_intel.api.routers import chat, cv, evaluation, feedback, health, ingest, metrics
from career_intel.config import get_settings
from career_intel.logging import setup_logging

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(log_level=settings.log_level)

    from career_intel.logging.tracing import configure_langsmith

    configure_langsmith()

    logger.info(
        "app_startup",
        environment=settings.environment,
        qdrant_url=settings.qdrant_url,
    )
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Career Intelligence Assistant",
        version="0.1.0",
        description="RAG-powered career guidance with tool calling and citations.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rate_limited_prefixes = ("/chat", "/ingest", "/feedback")

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Apply rate limiting to chat, ingest, and feedback endpoints."""
        if any(request.url.path.startswith(p) for p in rate_limited_prefixes):
            try:
                from career_intel.security.rate_limit import check_rate_limit

                await check_rate_limit(request)
            except Exception as exc:
                if hasattr(exc, "status_code") and exc.status_code == 429:  # type: ignore[attr-defined]
                    return JSONResponse(
                        status_code=429,
                        content={"error": "rate_limited", "detail": str(exc.detail)},  # type: ignore[attr-defined]
                    )
        return await call_next(request)

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        return {"service": "AI Career Intelligence Assistant", "status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.middleware("http")
    async def security_event_logging(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Log security-relevant events for all requests."""
        start = time.monotonic()
        response: Response = await call_next(request)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        if response.status_code in (400, 403, 429):
            logger.warning(
                "security_event",
                path=request.url.path,
                status=response.status_code,
                client_ip=request.client.host if request.client else "unknown",
                elapsed_ms=elapsed_ms,
            )

        return response

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(cv.router)
    app.include_router(ingest.router)
    app.include_router(feedback.router)
    app.include_router(evaluation.router)
    app.include_router(metrics.router)

    return app


app = create_app()
