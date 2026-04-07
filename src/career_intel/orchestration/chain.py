"""Main orchestration chain: route -> conditional(rewrite -> retrieve -> tools) -> synthesize."""

from __future__ import annotations

import time
from typing import Any

import structlog

from career_intel.config import Settings
from career_intel.schemas.api import ChatMessage, ChatResponse, ToolCallResult

logger = structlog.get_logger()

_SKIP_RETRIEVAL_INTENTS = frozenset({"small_talk", "direct_answer"})


async def run_turn(
    *,
    messages: list[ChatMessage],
    session_id: str,
    use_tools: bool,
    filters: dict[str, Any] | None,
    settings: Settings,
    trace_id: str,
    cv_text: str | None = None,
) -> ChatResponse:
    """Execute a single conversational turn through the full pipeline.

    Flow: input guard -> route -> conditional(rewrite + retrieve + tools) -> synthesize.
    """
    from career_intel.api.routers.metrics import inc

    inc("chat_requests_total")
    t_start = time.monotonic()

    user_query = messages[-1].content
    logger.info(
        "turn_start",
        session_id=session_id,
        query_preview=user_query[:80],
        query_length=len(user_query),
        has_cv=cv_text is not None,
        stream_mode=False,
    )

    # --- Step 1: Input guards ---
    from career_intel.security.guards import validate_input_deep

    await validate_input_deep(user_query, max_length=settings.max_input_length)

    # --- Step 2: Route FIRST (intent classification) ---
    from career_intel.tools.registry import execute_tool, route_query

    decision = await route_query(
        user_query,
        cv_available=cv_text is not None,
        settings=settings,
    )
    intent = decision.intent
    use_cv = decision.use_cv

    # --- Step 3: Fast path for small_talk / direct_answer ---
    if intent in _SKIP_RETRIEVAL_INTENTS:
        from career_intel.orchestration.synthesize import generate_direct_response

        reply = await generate_direct_response(user_query, settings)

        logger.info(
            "turn_complete",
            session_id=session_id,
            intent=intent,
            retrieval_invoked=False,
            tool_invoked=False,
            sources_count=0,
            total_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
        )

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            citations=[],
            tool_calls=[],
            intent=intent,
            trace_id=trace_id,
        )

    # --- Step 4: Rewrite + Retrieve (only for retrieval/tool intents) ---
    from career_intel.rag.retriever import retrieve_chunks, rewrite_query

    rewritten = await rewrite_query(user_query, settings=settings)
    chunks = await retrieve_chunks(query=rewritten, filters=filters, settings=settings)

    # --- Step 5: Tool execution (only if router selected a tool) ---
    tool_results: list[ToolCallResult] = []
    if intent == "tool_required" and decision.tool_name and use_tools:
        try:
            result = await execute_tool(decision, settings)
            tool_results = [result]
            inc("tool_calls_total", 1)
        except Exception as exc:
            logger.error("tool_execution_failed", tool=decision.tool_name, error=str(exc))
            tool_results = [ToolCallResult(
                tool_name=decision.tool_name,
                inputs=decision.params,
                output={"error": str(exc)},
                success=False,
                error=str(exc),
            )]

    # --- Step 6: Synthesize answer with citations ---
    from career_intel.orchestration.synthesize import synthesize_answer

    reply, citations = await synthesize_answer(
        query=user_query,
        rewritten_query=rewritten,
        chunks=chunks,
        tool_results=tool_results,
        settings=settings,
        cv_text=cv_text,
        use_cv=use_cv,
    )

    logger.info(
        "turn_complete",
        session_id=session_id,
        intent=intent,
        retrieval_invoked=True,
        tool_invoked=bool(tool_results),
        sources_count=len(citations),
        total_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        citations=citations,
        tool_calls=tool_results,
        intent=intent,
        trace_id=trace_id,
    )
