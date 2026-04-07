"""Answer synthesis with citation enforcement."""

from __future__ import annotations

import structlog

from career_intel.config import Settings
from career_intel.llm import get_chat_llm
from career_intel.orchestration.context_builder import build_user_prompt
from career_intel.orchestration.prompts.system import SYSTEM_PROMPT
from career_intel.schemas.api import Citation, ToolCallResult
from career_intel.schemas.domain import RetrievedChunk

logger = structlog.get_logger()

WEAK_EVIDENCE_THRESHOLD = 0.30

CONVERSATIONAL_PROMPT = """\
You are the AI Career Intelligence Assistant. The user is making small talk \
or asking a simple question that does not require evidence retrieval.

Respond naturally and conversationally. Be friendly, brief, and helpful. \
If the user greets you, greet them back and offer to help with career-related \
questions. Do NOT fabricate citations or reference source material. \
Do NOT include disclaimers unless the user asks a career-related question."""


async def generate_direct_response(
    query: str,
    settings: Settings,
) -> str:
    """Generate a direct conversational response without retrieval context.

    Used for small_talk and direct_answer intents where no RAG context is needed.
    """
    llm = get_chat_llm(settings, temperature=0.7)
    response = await llm.ainvoke([
        {"role": "system", "content": CONVERSATIONAL_PROMPT},
        {"role": "user", "content": query},
    ])
    return response.content if hasattr(response, "content") else str(response)


async def synthesize_answer(
    *,
    query: str,
    rewritten_query: str,
    chunks: list[RetrievedChunk],
    tool_results: list[ToolCallResult],
    settings: Settings,
    cv_text: str | None = None,
    use_cv: bool = False,
) -> tuple[str, list[Citation]]:
    """Build the context prompt and call the LLM for a cited answer."""
    if not chunks or all(c.score < WEAK_EVIDENCE_THRESHOLD for c in chunks):
        logger.info("weak_evidence_abstain", query=rewritten_query[:120])
        return (
            "I don't have enough evidence in my knowledge base to answer this "
            "confidently. Could you rephrase or ask about a specific role or skill?",
            [],
        )

    tool_block = ""
    if tool_results:
        parts = []
        for tr in tool_results:
            parts.append(f"### Tool: {tr.tool_name}\n```json\n{tr.output}\n```")
        tool_block = "\n\n".join(parts)

    user_prompt, citation_map = build_user_prompt(
        query=query,
        chunks=chunks,
        tool_block=tool_block,
        cv_text=cv_text,
        use_cv=use_cv,
    )

    llm = get_chat_llm(settings, temperature=0.2)

    response = await llm.ainvoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    reply_text = response.content if hasattr(response, "content") else str(response)

    citations = _build_citations(citation_map, chunks)

    return reply_text, citations


def _build_citations(
    citation_map: dict[int, str],
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """Map citation IDs back to source metadata."""
    chunk_lookup = {c.chunk_id: c for c in chunks}
    citations: list[Citation] = []
    for idx, chunk_id in sorted(citation_map.items()):
        chunk = chunk_lookup.get(chunk_id)
        if not chunk:
            continue
        citations.append(Citation(
            id=idx,
            source_id=chunk.metadata.source_id,
            title=chunk.metadata.title,
            section=chunk.metadata.section,
            page_or_loc=chunk.metadata.page_or_loc,
            publish_year=chunk.metadata.publish_year,
            excerpt=chunk.text[:500],
            uri=chunk.metadata.uri,
        ))
    return citations
