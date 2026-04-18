"""Domain models for RAG chunks, tools, evaluation, and routing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# RAG / Ingestion
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Payload stored alongside every vector in Qdrant."""

    source_id: str
    source_type: str
    title: str
    document_title: str | None = None
    section: str | None = None
    section_title: str | None = None
    page_or_loc: str | None = None
    page_number: int | None = None
    publish_year: int | None = None
    license: str | None = None
    entity_type: str | None = None
    source_priority: int | None = None
    occupation_id: str | None = None
    occupation_label: str | None = None
    occupation_code: str | None = None
    isco_group: str | None = None
    isco_group_label: str | None = None
    skill_id: str | None = None
    skill_label: str | None = None
    skill_type: str | None = None
    relation_type: str | None = None
    esco_doc_type: str | None = None
    language: str | None = None
    uri: str | None = None
    chunk_index: int
    parent_doc_id: str
    file_name: str | None = None
    topic: str | None = Field(
        default=None,
        description="Logical topic path or file grouping from ingestion (e.g. esco/occupations_en).",
    )
    source: str | None = Field(
        default=None,
        description="Corpus namespace: wef | esco (mirrors ingest metadata).",
    )


class DocumentRecord(BaseModel):
    """Row in the Postgres ``documents`` table."""

    id: str
    uri: str
    checksum: str
    source_type: str
    license: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievedChunk(BaseModel):
    """A single chunk returned by the retriever, enriched with score."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    score: float
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Tool I/O
# ---------------------------------------------------------------------------

class SkillGapInput(BaseModel):
    target_role: str = Field(..., min_length=2, max_length=200)
    current_skills: list[str] = Field(..., min_length=1, max_length=50)
    seniority: str | None = None


class SkillGapOutput(BaseModel):
    target_role: str
    must_have_gaps: list[dict[str, Any]]
    nice_to_have_gaps: list[dict[str, Any]]
    suggested_order: list[str]
    citations: list[dict[str, Any]]


class RoleCompareInput(BaseModel):
    role_a: str = Field(..., min_length=2, max_length=200)
    role_b: str = Field(..., min_length=2, max_length=200)


class RoleCompareOutput(BaseModel):
    role_a: str
    role_b: str
    comparison: dict[str, Any]
    narrative: str
    citations: list[dict[str, Any]]


class LearningPlanInput(BaseModel):
    goal_role: str = Field(..., min_length=2, max_length=200)
    hours_per_week: int = Field(default=10, ge=1, le=80)
    horizon_weeks: int = Field(default=12, ge=1, le=52)
    constraints: str | None = None


class LearningPlanOutput(BaseModel):
    goal_role: str
    total_weeks: int
    milestones: list[dict[str, Any]]
    resources: list[dict[str, Any]]
    citations: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class RouterDecision(BaseModel):
    """Intent-first output of the LLM query router.

    The ``intent`` field is the primary axis — it describes *what the user
    wants*.  ``tool_name`` is secondary and only populated when the intent
    actually requires a tool.
    """

    intent: Literal[
        "small_talk",
        "general_knowledge",
        "domain_specific",
        "dynamic_runtime",
        "tool_required",
        # Legacy aliases kept for compatibility with older tests/callers.
        "direct_answer",
        "retrieval_required",
    ] = Field(
        description="High-level classification of the user's goal.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Router's self-assessed confidence in this classification.",
    )
    tool_name: str | None = Field(
        default=None,
        description="Tool to invoke (only when intent == 'tool_required').",
    )
    params: dict[str, Any] = Field(default_factory=dict)
    use_cv: bool = Field(
        default=False,
        description="Whether the user's CV should be included in context.",
    )
    reason: str = Field(
        default="",
        description="Short explanation of why this routing decision was made.",
    )


# ---------------------------------------------------------------------------
# CV Risk
# ---------------------------------------------------------------------------

class CVRiskScore(BaseModel):
    """Result of heuristic risk assessment on CV content."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = clean, 1.0 = almost certainly adversarial.",
    )
    matched_patterns: list[str] = Field(default_factory=list)
    flagged: bool = Field(
        default=False,
        description="True if score exceeds the blocking threshold.",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class GoldenExample(BaseModel):
    """One row in an evaluation dataset."""

    query: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    expected_behaviour: str | None = Field(
        default=None,
        description="E.g. 'abstain', 'use_tool:skill_gap', 'cite_source'.",
    )
    tags: list[str] = Field(default_factory=list)
