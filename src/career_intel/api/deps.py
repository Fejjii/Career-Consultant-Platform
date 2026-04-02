"""FastAPI dependency injection helpers."""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import Depends, Header, HTTPException, Request

from career_intel.config import Settings, get_settings

logger = structlog.get_logger()


def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


def request_trace_id(request: Request) -> str:
    """Return or generate a trace ID for the current request."""
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return trace_id


TraceIdDep = Annotated[str, Depends(request_trace_id)]


async def require_admin(
    settings: SettingsDep,
    x_admin_secret: str | None = Header(default=None),
) -> None:
    """Gate admin-only endpoints behind the shared admin secret."""
    expected = settings.admin_secret.get_secret_value()
    if not x_admin_secret or x_admin_secret != expected:
        logger.warning("auth_failed", endpoint="admin")
        raise HTTPException(status_code=403, detail="Invalid or missing admin secret.")


AdminDep = Annotated[None, Depends(require_admin)]
