"""Streaming orchestration — yields tokens as Server-Sent Events for the chat endpoint.

Design:
  - The router runs FIRST to classify intent.
  - For small_talk / direct_answer: tokens stream immediately (no retrieval).
  - For retrieval / tool intents: retrieval + tool calling happen before
    streaming, then the synthesis is streamed token-by-token.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from career_intel.config import Settings
from career_intel.llm import get_chat_llm
from career_intel.orchestration.context_builder import build_user_prompt
from career_intel.orchestration.prompts.system import SYSTEM_PROMPT
from career_intel.orchestration.synthesize import (
    CONVERSATIONAL_PROMPT,
    WEAK_EVIDENCE_THRESHOLD,
    _build_citations,
)
from career_intel.schemas.api import ChatMessage, ToolCallResult

logger = structlog.get_logger()

_SKIP_RETRIEVAL_INTENTS = frozenset({"small_talk", "direct_answer"})


async def stream_turn(
    *,
    messages: list[ChatMessage],
    session_id: str,
    use_tools: bool,
    filters: dict[str, Any] | None,
    settings: Settings,
    trace_id: str,
    cv_text: str | None = None,
) -> AsyncGenerator[str, None]:
    """Execute a turn and yield SSE-formatted chunks.

    Event types:
      ``data: {"type": "intent", "data": "..."}``
      ``data: {"type": "token", "content": "..."}``
      ``data: {"type": "citations", "data": [...]}``
      ``data: {"type": "tool_calls", "data": [...]}``
      ``data: {"type": "done"}``
      ``data: {"type": "error", "detail": "..."}``
    """
    from career_intel.api.routers.metrics import inc
    from career_intel.security.guards import validate_input_deep
    from career_intel.tools.registry import execute_tool, route_query

    inc("chat_requests_total")
    user_query = messages[-1].content
    t_start = time.monotonic()

    try:
        await validate_input_deep(user_query, max_length=settings.max_input_length)

        # --- Route FIRST ---
        t_route_start = time.monotonic()
        decision = await route_query(
            user_query,
            cv_available=cv_text is not None,
            settings=settings,
        )
        intent = decision.intent
        route_latency_ms = round((time.monotonic() - t_route_start) * 1000, 1)
        yield _sse({"type": "intent", "data": intent})

        # --- Fast path: small_talk / direct_answer ---
        if intent in _SKIP_RETRIEVAL_INTENTS:
            llm = get_chat_llm(settings, temperature=0.7, streaming=True)
            first_token = True
            async for chunk in llm.astream([
                {"role": "system", "content": CONVERSATIONAL_PROMPT},
                {"role": "user", "content": user_query},
            ]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    if first_token:
                        logger.info(
                            "stream_first_token",
                            session_id=session_id,
                            intent=intent,
                            first_token_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
                        )
                        first_token = False
                    yield _sse({"type": "token", "content": token})

            total_ms = round((time.monotonic() - t_start) * 1000, 1)
            logger.info(
                "stream_complete",
                session_id=session_id,
                intent=intent,
                retrieval_invoked=False,
                tool_invoked=False,
                sources_count=0,
                stream_mode=True,
                route_latency_ms=route_latency_ms,
                total_latency_ms=total_ms,
            )
            yield _sse({"type": "debug", "data": {
                "intent": intent,
                "confidence": decision.confidence,
                "retrieval_invoked": False,
                "tool_invoked": False,
                "sources_count": 0,
                "use_cv": decision.use_cv,
                "route_latency_ms": route_latency_ms,
                "total_latency_ms": total_ms,
            }})
            yield _sse({"type": "done"})
            return

        # --- Normal path: retrieval / tool intents ---
        yield _sse({"type": "status", "detail": "Retrieving evidence..."})

        from career_intel.rag.retriever import retrieve_chunks, rewrite_query

        rewritten = await rewrite_query(user_query, settings=settings)
        chunks = await retrieve_chunks(query=rewritten, filters=filters, settings=settings)

        tool_results: list[ToolCallResult] = []
        if intent == "tool_required" and decision.tool_name and use_tools:
            try:
                result = await execute_tool(decision, settings)
                tool_results = [result]
                inc("tool_calls_total", 1)
                yield _sse({"type": "tool_calls", "data": [result.model_dump()]})
            except Exception as exc:
                logger.error("tool_execution_failed", tool=decision.tool_name, error=str(exc))
                tool_results = [ToolCallResult(
                    tool_name=decision.tool_name,
                    inputs=decision.params,
                    output={"error": str(exc)},
                    success=False,
                    error=str(exc),
                )]
                yield _sse({"type": "tool_calls", "data": [tool_results[0].model_dump()]})

        if not chunks or all(c.score < WEAK_EVIDENCE_THRESHOLD for c in chunks):
            yield _sse({
                "type": "token",
                "content": (
                    "I don't have enough evidence in my knowledge base to answer this "
                    "confidently. Could you rephrase or ask about a specific role or skill?"
                ),
            })
            yield _sse({"type": "done"})
            return

        tool_block = ""
        if tool_results:
            parts = [f"### Tool: {tr.tool_name}\n```json\n{tr.output}\n```" for tr in tool_results]
            tool_block = "\n\n".join(parts)

        user_prompt, citation_map = build_user_prompt(
            query=user_query,
            chunks=chunks,
            tool_block=tool_block,
            cv_text=cv_text,
            use_cv=decision.use_cv,
        )

        llm = get_chat_llm(settings, temperature=0.2, streaming=True)
        yield _sse({"type": "status", "detail": "Generating grounded response..."})

        first_token = True
        async for chunk in llm.astream([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                if first_token:
                    logger.info(
                        "stream_first_token",
                        session_id=session_id,
                        intent=intent,
                        first_token_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
                    )
                    first_token = False
                yield _sse({"type": "token", "content": token})

        citations = _build_citations(citation_map, chunks)
        if citations:
            yield _sse({"type": "citations", "data": [c.model_dump() for c in citations]})

        total_ms = round((time.monotonic() - t_start) * 1000, 1)
        logger.info(
            "stream_complete",
            session_id=session_id,
            intent=intent,
            retrieval_invoked=True,
            tool_invoked=bool(tool_results),
            sources_count=len(citations),
            stream_mode=True,
            route_latency_ms=route_latency_ms,
            total_latency_ms=total_ms,
        )
        yield _sse({"type": "debug", "data": {
            "intent": intent,
            "confidence": decision.confidence,
            "retrieval_invoked": True,
            "tool_invoked": bool(tool_results),
            "sources_count": len(citations),
            "use_cv": decision.use_cv,
            "route_latency_ms": route_latency_ms,
            "total_latency_ms": total_ms,
        }})
        yield _sse({"type": "done"})

    except Exception as exc:
        logger.error("stream_error", error=str(exc)[:200], session_id=session_id)
        yield _sse({"type": "error", "detail": str(exc)[:500]})


def _sse(payload: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"
