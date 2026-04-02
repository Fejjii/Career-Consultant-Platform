"""Basic metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["metrics"])

# Simple in-memory counters; replace with prometheus-client if needed.
_counters: dict[str, int] = {
    "chat_requests_total": 0,
    "ingest_runs_total": 0,
    "tool_calls_total": 0,
    "errors_total": 0,
}


def inc(name: str, amount: int = 1) -> None:
    _counters[name] = _counters.get(name, 0) + amount


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus-style text exposition of basic counters."""
    lines = [f"{k} {v}" for k, v in sorted(_counters.items())]
    return "\n".join(lines) + "\n"
