"""Ingestion endpoint — admin-only trigger for document processing."""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from career_intel.api.deps import AdminDep, TraceIdDep
from career_intel.schemas.api import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = structlog.get_logger()


@router.post("", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    _admin: AdminDep,
    trace_id: TraceIdDep,
) -> IngestResponse:
    """Ingest documents into the vector store and record lineage."""
    logger.info("ingest_request", paths=body.paths, mode=body.mode, trace_id=trace_id)

    from career_intel.rag.ingest_pipeline import run_ingestion

    result = await run_ingestion(paths=body.paths, mode=body.mode)
    return result
