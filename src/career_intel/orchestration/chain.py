"""Main orchestration chain: query rewrite -> retrieve -> tools -> synthesize."""

from __future__ import annotations

from typing import Any

import structlog

from career_intel.config import Settings
from career_intel.schemas.api import ChatMessage, ChatResponse, Citation, ToolCallResult

logger = structlog.get_logger()


async def run_turn(
    *,
    messages: list[ChatMessage],
    session_id: str,
    use_tools: bool,
    filters: dict[str, Any] | None,
    settings: Settings,
    trace_id: str,
) -> ChatResponse:
    """Execute a single conversational turn through the full pipeline.

    Flow: input guard -> query rewrite -> retrieve -> (optional) tools -> synthesize -> output guard.
    """
    from career_intel.api.routers.metrics import inc

    inc("chat_requests_total")

    user_query = messages[-1].content
    logger.info("turn_start", session_id=session_id, query_length=len(user_query))

    # --- Step 1: Input guards ---
    from career_intel.security.guards import validate_input

    validate_input(user_query, max_length=settings.max_input_length)

    # --- Step 2: Query rewrite ---
    from career_intel.rag.retriever import rewrite_query

    rewritten = await rewrite_query(user_query, settings=settings)

    # --- Step 3: Retrieve ---
    from career_intel.rag.retriever import retrieve_chunks

    chunks = await retrieve_chunks(
        query=rewritten,
        filters=filters,
        settings=settings,
    )

    # --- Step 4: Tool calling (optional) ---
    tool_results: list[ToolCallResult] = []
    if use_tools:
        from career_intel.tools.registry import maybe_call_tools

        tool_results = await maybe_call_tools(
            query=user_query,
            chunks=chunks,
            settings=settings,
        )
        if tool_results:
            inc("tool_calls_total", len(tool_results))

    # --- Step 5: Synthesize answer with citations ---
    from career_intel.orchestration.synthesize import synthesize_answer

    reply, citations = await synthesize_answer(
        query=user_query,
        rewritten_query=rewritten,
        chunks=chunks,
        tool_results=tool_results,
        settings=settings,
    )

    logger.info(
        "turn_complete",
        session_id=session_id,
        citations_count=len(citations),
        tools_used=[t.tool_name for t in tool_results],
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        citations=citations,
        tool_calls=tool_results,
        trace_id=trace_id,
    )
