"""Pydantic v2 schemas shared across the API, orchestration, and UI layers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

AnswerLengthMode = Literal["concise", "balanced", "detailed"]

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
    user_timezone: str | None = Field(
        default=None,
        max_length=64,
        description="Optional user timezone (IANA name, e.g. Europe/London) for runtime/date answers.",
    )
    answer_length: AnswerLengthMode = Field(
        default="balanced",
        description="Target verbosity for the final answer (synthesis only; does not affect retrieval).",
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
    source: str | None = Field(
        default=None,
        description="Corpus namespace from chunk metadata (e.g. wef, esco); supports future web/youtube.",
    )
    file_name: str | None = Field(default=None, description="Original file name when available.")
    esco_doc_type: str | None = Field(default=None, description="ESCO chunk doc type when applicable.")
    entity_type: str | None = Field(default=None, description="Entity classification from chunk metadata.")
    page_number: int | None = Field(default=None, description="PDF page number when available.")


class ToolCallResult(BaseModel):
    tool_name: str
    inputs: dict[str, Any]
    output: dict[str, Any]
    success: bool = True
    error: str | None = None


class TokenUsage(BaseModel):
    """Token counts from the model provider (OpenAI-compatible field names)."""

    prompt_tokens: int = Field(..., ge=0, description="Same meaning as OpenAI ``prompt_tokens``.")
    completion_tokens: int = Field(..., ge=0, description="Same meaning as OpenAI ``completion_tokens``.")
    total_tokens: int = Field(..., ge=0, description="Same meaning as OpenAI ``total_tokens``.")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    answer_source: Literal["rag", "tool", "llm_fallback", "source_inventory", "runtime"] | None = None
    answer_mode: Literal["RAG", "LLM", "RUNTIME", "TOOL", "SOURCE_INVENTORY"] | None = None
    runtime_utility_used: str | None = None
    intent: str | None = Field(
        default=None,
        description=(
            "Router intent classification "
            "(small_talk, domain_specific, general_knowledge, dynamic_runtime, tool_required)."
        ),
    )
    answer_length: AnswerLengthMode = "balanced"
    trace_id: str | None = None
    usage: TokenUsage | None = Field(
        default=None,
        description="Aggregated provider token usage for this turn (router + answer generation).",
    )
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


class ProviderAuthStatusResponse(BaseModel):
    provider: Literal["openai"] = "openai"
    ok: bool
    model: str
    message: str | None = None
    supported_models: list[str] = Field(default_factory=list)
    accessible_models: list[str] = Field(default_factory=list)
    normalized_accessible_models: list[str] = Field(default_factory=list)
    selectable_models: list[str] = Field(default_factory=list)
    supported_but_unavailable_models: list[str] = Field(default_factory=list)
    accessible_but_unsupported_models: list[str] = Field(default_factory=list)
    ignored_accessible_models: list[str] = Field(default_factory=list)
    model_unavailability_reasons: dict[str, str] = Field(default_factory=dict)
    validation_stage: str | None = None


class SourceInventoryItemResponse(BaseModel):
    source_name: str
    source_family: str
    description: str
    file_count: int
    ingestion_status: str
    paths: list[str] = Field(default_factory=list)


class SourceInventoryResponse(BaseModel):
    total_source_groups: int
    total_files_present: int
    esco_ingestion_status: str
    esco_status_note: str | None = None
    items: list[SourceInventoryItemResponse] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Speech (transcription only — not part of chat/orchestration)
# ---------------------------------------------------------------------------


class TranscribeResponse(BaseModel):
    """Result of POST /speech/transcribe."""

    text: str = Field(..., min_length=1, description="Normalized transcript text.")
    provider: Literal["openai"] = "openai"
    language: str | None = Field(default=None, description="Detected language code, if reported.")
    duration_seconds: float | None = Field(
        default=None,
        description="Audio duration in seconds, if reported by the provider.",
    )
    warnings: list[str] = Field(default_factory=list)
