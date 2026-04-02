"""Domain models for RAG chunks, tools, and evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# RAG / Ingestion
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Payload stored alongside every vector in Qdrant."""

    source_id: str
    source_type: str
    title: str
    section: str | None = None
    page_or_loc: str | None = None
    publish_year: int | None = None
    license: str | None = None
    occupation_code: str | None = None
    skill_id: str | None = None
    uri: str | None = None
    chunk_index: int
    parent_doc_id: str


class DocumentRecord(BaseModel):
    """Row in the Postgres ``documents`` table."""

    id: str
    uri: str
    checksum: str
    source_type: str
    license: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    """A single chunk returned by the retriever, enriched with score."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    score: float


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
