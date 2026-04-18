"""Answer synthesis with explicit grounding-mode enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog

from career_intel.llm import get_chat_llm
from career_intel.llm.token_usage import merge_token_usages, usage_from_langchain_message
from career_intel.orchestration.context_builder import build_user_prompt
from career_intel.orchestration.prompts.system import SYSTEM_PROMPT
from career_intel.rag.citation import extract_cited_ids, validate_citations
from career_intel.schemas.api import AnswerLengthMode, Citation, TokenUsage, ToolCallResult
from career_intel.security.guards import sanitize_model_output
from career_intel.security.hardening import sanitize_public_uri
from career_intel.security.sanitize import wrap_cv_content

if TYPE_CHECKING:
    from career_intel.config import Settings
    from career_intel.schemas.domain import RetrievedChunk

logger = structlog.get_logger()

TOOL_SYSTEM_PROMPT = """\
You are the AI Career Intelligence Assistant. The user request was handled with \
structured analysis tools (JSON below). Summarise the tool outputs clearly and \
actionably. Do NOT invent labour-market statistics or citations [n] — there are \
no knowledge-base chunks for this turn. If a tool returned an error, explain it \
and suggest next steps. End with: "This is guidance based on structured analysis \
— consult a career professional for personalised advice."
"""

FALLBACK_SYSTEM_PROMPT = """\
You are the AI Career Intelligence Assistant. There is no reliable internal RAG \
evidence and no usable tool output for this turn, so you may answer with general \
reasoning only.

Rules:
- Do NOT claim this answer came from the internal knowledge base.
- Do NOT fabricate citations [n].
- Be explicit when you are reasoning from general knowledge rather than retrieved evidence.
- Prefer concise, helpful answers with clear uncertainty when needed.
- End with: "This answer is based on general reasoning, not retrieved internal evidence."
"""

RAG_ONLY_SYSTEM_APPENDIX = """\

Strict grounding mode:
- Use ONLY the retrieved source chunks.
- Do NOT add facts from general world knowledge or unstated assumptions.
- If the evidence does not support a claim, say that directly.
- Every factual bullet or paragraph MUST include at least one inline citation [n].
- Use only citation numbers that appear in the provided source context.
"""

CONVERSATIONAL_PROMPT = """\
You are the AI Career Intelligence Assistant. The user is making small talk \
or asking a simple question that does not require evidence retrieval.

Respond naturally and conversationally. Be friendly, brief, and helpful. \
If the user greets you, greet them back and offer to help with career-related \
questions. Do NOT fabricate citations or reference source material. \
Do NOT include disclaimers unless the user asks a career-related question."""


def answer_length_system_suffix(mode: AnswerLengthMode) -> str:
    """Extra system instructions that make answer length modes visibly distinct."""
    if mode == "balanced":
        return (
            "\n\nAnswer length — balanced:\n"
            "- Keep a moderate level of detail.\n"
            "- Use short paragraphs or a short bullet list, whichever fits the question best.\n"
            "- Give enough explanation to make the guidance actionable, but avoid deep digressions.\n"
            "- Default to a compact but complete answer rather than a terse list or a long explainer.\n"
            "- Still follow every grounding, citation, and safety rule above.\n"
        )
    if mode == "concise":
        return (
            "\n\nAnswer length — concise:\n"
            "- Use bullets only; do not write prose paragraphs.\n"
            "- Use at most 5 to 7 bullets.\n"
            "- Keep each bullet compact and action-oriented.\n"
            "- Do not add explanations unless they are necessary for correctness or safety.\n"
            "- Still follow every grounding, citation, and safety rule above.\n"
        )
    return (
        "\n\nAnswer length — detailed:\n"
        "- Organize the answer into clear sections with short headings.\n"
        "- When relevant, use sections such as Overview, Key skills, Examples, and Practical implications.\n"
        "- Give a deeper explanation and a more educational tone than balanced mode.\n"
        "- Include concrete examples when they clarify the guidance "
        "(ground examples in the evidence when sources are present).\n"
        "- Expand on why the evidence matters instead of only listing conclusions.\n"
        "- Still follow every grounding, citation, and safety rule above.\n"
    )


async def generate_direct_response(
    query: str,
    settings: Settings,
    *,
    answer_length: AnswerLengthMode = "balanced",
) -> tuple[str, TokenUsage | None]:
    """Generate a direct conversational response without retrieval context.

    Used for small_talk and general_knowledge intents where no RAG context is needed.
    """
    system = CONVERSATIONAL_PROMPT + answer_length_system_suffix(answer_length)
    llm = get_chat_llm(settings, temperature=0.7)
    response = await llm.ainvoke([
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)
    return sanitize_model_output(raw_text), usage_from_langchain_message(response)


async def synthesize_answer(
    *,
    query: str,
    rewritten_query: str,
    chunks: list[RetrievedChunk],
    tool_results: list[ToolCallResult],
    answer_source: Literal["rag", "tool", "llm_fallback"],
    settings: Settings,
    cv_text: str | None = None,
    use_cv: bool = False,
    answer_length: AnswerLengthMode = "balanced",
) -> tuple[str, list[Citation], TokenUsage | None]:
    """Build the prompt and call the LLM under an explicit grounding mode."""
    has_evidence = bool(chunks)

    tool_block = ""
    if tool_results:
        parts = []
        for tr in tool_results:
            parts.append(f"### Tool: {tr.tool_name}\n```json\n{tr.output}\n```")
        tool_block = "\n\n".join(parts)

    if answer_source == "rag":
        if not has_evidence:
            raise ValueError("RAG synthesis requested without retrieved chunks")
        user_prompt, citation_map = build_user_prompt(
            query=query,
            chunks=chunks,
            tool_block="",
            cv_text=cv_text,
            use_cv=use_cv,
        )
        system = SYSTEM_PROMPT + RAG_ONLY_SYSTEM_APPENDIX + answer_length_system_suffix(answer_length)
    elif answer_source == "tool":
        if not tool_results:
            raise ValueError("Tool synthesis requested without tool results")
        parts = [f"Question: {query}"]
        if cv_text and use_cv:
            parts.append(wrap_cv_content(cv_text))
        parts.append(f"Tool results:\n{tool_block}")
        user_prompt = "\n\n".join(parts)
        citation_map = {}
        system = TOOL_SYSTEM_PROMPT + answer_length_system_suffix(answer_length)
    else:
        parts = [f"Question: {query}"]
        if cv_text and use_cv:
            parts.append(wrap_cv_content(cv_text))
        if tool_block:
            parts.append(f"Tool results attempted:\n{tool_block}")
        if rewritten_query != query:
            parts.append(f"Normalized retrieval query: {rewritten_query}")
        user_prompt = "\n\n".join(parts)
        citation_map = {}
        system = FALLBACK_SYSTEM_PROMPT + answer_length_system_suffix(answer_length)

    llm = get_chat_llm(settings, temperature=0.2)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    if answer_source == "rag":
        reply_text, cited_ids, synth_usage = await _generate_rag_answer_with_citation_guard(
            llm=llm,
            messages=messages,
            citation_map=citation_map,
        )
        citations = _build_citations(citation_map, chunks, cited_ids=cited_ids)
    else:
        response = await llm.ainvoke(messages)
        reply_text = response.content if hasattr(response, "content") else str(response)
        citations = []
        synth_usage = usage_from_langchain_message(response)

    return sanitize_model_output(reply_text), citations, synth_usage


async def _generate_rag_answer_with_citation_guard(
    *,
    llm: object,
    messages: list[dict[str, str]],
    citation_map: dict[int, str],
    max_attempts: int = 2,
) -> tuple[str, set[int], TokenUsage | None]:
    """Generate a RAG answer and reject drafts without valid citations."""
    conversation = list(messages)
    accumulated_usage: TokenUsage | None = None

    for attempt in range(1, max_attempts + 1):
        response = await llm.ainvoke(conversation)
        accumulated_usage = merge_token_usages(accumulated_usage, usage_from_langchain_message(response))
        reply_text = response.content if hasattr(response, "content") else str(response)
        is_valid, invalid_ids = validate_citations(reply_text, citation_map)
        cited_ids = extract_cited_ids(reply_text)

        logger.info(
            "rag_citation_validation",
            attempt=attempt,
            cited_ids=sorted(cited_ids),
            invalid_ids=sorted(invalid_ids),
            has_citations=bool(cited_ids),
        )

        if is_valid and cited_ids:
            return sanitize_model_output(reply_text), cited_ids, accumulated_usage

        logger.warning(
            "rag_answer_rejected",
            attempt=attempt,
            reason="missing_or_invalid_citations",
            cited_ids=sorted(cited_ids),
            invalid_ids=sorted(invalid_ids),
        )

        conversation.extend([
            {"role": "assistant", "content": reply_text},
            {
                "role": "user",
                "content": (
                    "Rewrite the answer using ONLY the retrieved source chunks. "
                    "Every factual bullet or paragraph must include at least one valid "
                    "inline citation like [1]. Do not use any citation number that is not "
                    "present in the source context."
                ),
            },
        ])

    raise ValueError("RAG answer missing valid citations after regeneration")


def _build_citations(
    citation_map: dict[int, str],
    chunks: list[RetrievedChunk],
    cited_ids: set[int] | None = None,
) -> list[Citation]:
    """Map citation IDs back to source metadata."""
    chunk_lookup = {c.chunk_id: c for c in chunks}
    citations: list[Citation] = []
    selected_ids = cited_ids if cited_ids is not None else set(citation_map.keys())
    for idx, chunk_id in sorted(citation_map.items()):
        if idx not in selected_ids:
            continue
        chunk = chunk_lookup.get(chunk_id)
        if not chunk:
            continue
        citations.append(Citation(
            id=idx,
            source_id=chunk.metadata.source_id,
            title=chunk.metadata.document_title or chunk.metadata.title,
            section=chunk.metadata.section_title or chunk.metadata.section,
            page_or_loc=chunk.metadata.page_or_loc or (
                f"page {chunk.metadata.page_number}" if chunk.metadata.page_number else None
            ),
            publish_year=chunk.metadata.publish_year,
            excerpt=chunk.text[:500],
            uri=sanitize_public_uri(chunk.metadata.uri),
            source=chunk.metadata.source,
            file_name=chunk.metadata.file_name,
            esco_doc_type=chunk.metadata.esco_doc_type,
            entity_type=chunk.metadata.entity_type,
            page_number=chunk.metadata.page_number,
        ))
    return citations
