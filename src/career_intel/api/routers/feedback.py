"""Feedback endpoint — collect user quality signals."""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from career_intel.api.deps import TraceIdDep
from career_intel.schemas.api import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])
logger = structlog.get_logger()


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    trace_id: TraceIdDep,
) -> FeedbackResponse:
    """Record user feedback for a conversation message."""
    logger.info(
        "feedback_received",
        session_id=body.session_id,
        message_id=body.message_id,
        score=body.score,
        tags=body.tags,
        trace_id=trace_id,
    )
    # TODO: persist to Postgres feedback table
    return FeedbackResponse(received=True)
