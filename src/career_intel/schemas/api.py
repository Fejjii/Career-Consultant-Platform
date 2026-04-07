"""Pydantic v2 schemas shared across the API, orchestration, and UI layers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    session_id: str | None = None
    use_tools: bool = True
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters passed to retrieval (e.g. year, source_type).",
    )
    cv_text: str | None = Field(
        default=None,
        max_length=50_000,
        description="Pre-processed CV/resume text. Included in context only when the "
        "router determines the query benefits from personalisation.",
    )


class Citation(BaseModel):
    id: int = Field(..., description="Inline reference number [n].")
    source_id: str
    title: str
    section: str | None = None
    page_or_loc: str | None = None
    publish_year: int | None = None
    excerpt: str = Field(..., max_length=500)
    uri: str | None = None


class ToolCallResult(BaseModel):
    tool_name: str
    inputs: dict[str, Any]
    output: dict[str, Any]
    success: bool = True
    error: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    intent: str | None = Field(
        default=None,
        description="Router intent classification (small_talk, direct_answer, retrieval_required, tool_required).",
    )
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyDetail(BaseModel):
    name: str
    ok: bool
    latency_ms: float | None = None
    error: str | None = None


class ReadyResponse(BaseModel):
    ok: bool
    details: list[ReadyDetail]


class SystemStatusResponse(BaseModel):
    backend: bool
    qdrant: bool
    indexed_data_present: bool
    collection: str
    points_count: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    score: int = Field(..., ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    received: bool = True


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    mode: Literal["full", "incremental"] = "full"


class IngestResponse(BaseModel):
    run_id: str
    documents_processed: int
    chunks_created: int
