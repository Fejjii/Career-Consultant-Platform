"""Streaming orchestration — yields tokens as Server-Sent Events for the chat endpoint.

Design:
  - Retrieval + tool calling happen **before** streaming starts (they are
    fast, deterministic, and needed for the context window).
  - Only the final LLM synthesis is streamed token-by-token.
  - A non-streaming fallback exists in ``chain.run_turn`` for simpler
    testing, evaluation, and error recovery.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import structlog

from career_intel.config import Settings
from career_intel.llm import get_chat_llm
from career_intel.orchestration.prompts.system import SYSTEM_PROMPT, build_context_block
from career_intel.orchestration.synthesize import WEAK_EVIDENCE_THRESHOLD, _build_citations
from career_intel.schemas.api import ChatMessage, Citation, ToolCallResult
from career_intel.schemas.domain import RetrievedChunk

logger = structlog.get_logger()


async def stream_turn(
    *,
    messages: list[ChatMessage],
    session_id: str,
    use_tools: bool,
    filters: dict[str, Any] | None,
    settings: Settings,
    trace_id: str,
) -> AsyncGenerator[str, None]:
    """Execute a turn and yield SSE-formatted chunks.

    Event types:
      ``data: {"type": "token", "content": "..."}``
      ``data: {"type": "citations", "data": [...]}``
      ``data: {"type": "tool_calls", "data": [...]}``
      ``data: {"type": "done"}``
      ``data: {"type": "error", "detail": "..."}``
    """
    from career_intel.api.routers.metrics import inc
    from career_intel.rag.retriever import retrieve_chunks, rewrite_query
    from career_intel.security.guards import validate_input_deep
    from career_intel.tools.registry import maybe_call_tools

    inc("chat_requests_total")
    user_query = messages[-1].content

    try:
        yield _sse({"type": "status", "detail": "Validating request and retrieving evidence..."})
        await validate_input_deep(user_query, max_length=settings.max_input_length)

        rewritten = await rewrite_query(user_query, settings=settings)
        chunks = await retrieve_chunks(query=rewritten, filters=filters, settings=settings)

        tool_results: list[ToolCallResult] = []
        if use_tools:
            tool_results = await maybe_call_tools(
                query=user_query, chunks=chunks, settings=settings,
            )
            if tool_results:
                inc("tool_calls_total", len(tool_results))
                yield _sse({"type": "tool_calls", "data": [t.model_dump() for t in tool_results]})

        # Weak evidence check
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

        context_block, citation_map = build_context_block(chunks)

        tool_block = ""
        if tool_results:
            parts = [f"### Tool: {tr.tool_name}\n```json\n{tr.output}\n```" for tr in tool_results]
            tool_block = "\n\n".join(parts)

        user_prompt = f"Question: {user_query}\n\n{context_block}"
        if tool_block:
            user_prompt += f"\n\nTool results:\n{tool_block}"

        llm = get_chat_llm(settings, temperature=0.2, streaming=True)
        yield _sse({"type": "status", "detail": "Generating grounded response..."})

        async for chunk in llm.astream([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield _sse({"type": "token", "content": token})

        citations = _build_citations(citation_map, chunks)
        if citations:
            yield _sse({"type": "citations", "data": [c.model_dump() for c in citations]})

        yield _sse({"type": "done"})

    except Exception as exc:
        logger.error("stream_error", error=str(exc)[:200], session_id=session_id)
        yield _sse({"type": "error", "detail": str(exc)[:500]})


def _sse(payload: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"
