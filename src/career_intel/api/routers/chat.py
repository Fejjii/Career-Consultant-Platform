"""Chat endpoints — non-streaming (POST /chat) and streaming (POST /chat/stream)."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

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
    """Non-streaming turn — returns the full response in one JSON payload.

    Kept as the default for simpler testing, evaluation, and error recovery.
    """
    session_id = body.session_id or str(uuid.uuid4())
    logger.info(
        "chat_request",
        session_id=session_id,
        message_count=len(body.messages),
        use_tools=body.use_tools,
        streaming=False,
    )

    from career_intel.orchestration.chain import run_turn

    result = await run_turn(
        messages=body.messages,
        session_id=session_id,
        use_tools=body.use_tools,
        filters=body.filters,
        settings=settings,
        trace_id=trace_id,
        cv_text=body.cv_text,
    )
    return result


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    settings: SettingsDep,
    trace_id: TraceIdDep,
) -> StreamingResponse:
    """Streaming turn — returns Server-Sent Events with incremental tokens.

    Event payload types: ``token``, ``citations``, ``tool_calls``, ``done``, ``error``.
    """
    session_id = body.session_id or str(uuid.uuid4())
    logger.info(
        "chat_request",
        session_id=session_id,
        message_count=len(body.messages),
        use_tools=body.use_tools,
        streaming=True,
    )

    from career_intel.orchestration.stream import stream_turn

    event_gen = stream_turn(
        messages=body.messages,
        session_id=session_id,
        use_tools=body.use_tools,
        filters=body.filters,
        settings=settings,
        trace_id=trace_id,
        cv_text=body.cv_text,
    )

    return StreamingResponse(event_gen, media_type="text/event-stream")
