"""Chat endpoint — main RAG + tool-calling entry point."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter

from career_intel.api.deps import SettingsDep, TraceIdDep
from career_intel.schemas.api import ChatRequest, ChatResponse, ErrorResponse

router = APIRouter(prefix="/chat", tags=["chat"])
logger = structlog.get_logger()


@router.post(
    "",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat(
    body: ChatRequest,
    settings: SettingsDep,
    trace_id: TraceIdDep,
) -> ChatResponse:
    """Run one conversational turn through the RAG + tools pipeline."""
    session_id = body.session_id or str(uuid.uuid4())
    logger.info(
        "chat_request",
        session_id=session_id,
        message_count=len(body.messages),
        use_tools=body.use_tools,
    )

    from career_intel.orchestration.chain import run_turn

    result = await run_turn(
        messages=body.messages,
        session_id=session_id,
        use_tools=body.use_tools,
        filters=body.filters,
        settings=settings,
        trace_id=trace_id,
    )

    return result
