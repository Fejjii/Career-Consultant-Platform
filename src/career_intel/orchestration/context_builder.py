"""Context builder — assembles the prompt context including optional CV data.

Decides whether to include CV context based on the router's ``use_cv`` signal.
Keeps CV injection behind randomized boundary markers with full sanitization.
"""

from __future__ import annotations

import structlog

from career_intel.orchestration.prompts.system import build_context_block
from career_intel.schemas.domain import RetrievedChunk
from career_intel.security.sanitize import wrap_cv_content

logger = structlog.get_logger()


def build_user_prompt(
    *,
    query: str,
    chunks: list[RetrievedChunk],
    tool_block: str,
    cv_text: str | None,
    use_cv: bool,
) -> tuple[str, dict[int, str]]:
    """Assemble the full user prompt with sources, optional CV, and tool results.

    Parameters
    ----------
    query:
        The user's original question.
    chunks:
        Retrieved evidence chunks from RAG.
    tool_block:
        Pre-formatted tool results (empty string if none).
    cv_text:
        Raw (already processed/truncated) CV text, or None.
    use_cv:
        Whether the router determined this query benefits from CV context.

    Returns
    -------
    user_prompt:
        Fully assembled prompt string.
    citation_map:
        Mapping from ``[n]`` citation numbers to chunk IDs.
    """
    context_block, citation_map = build_context_block(chunks)

    parts: list[str] = [f"Question: {query}", context_block]

    if cv_text and use_cv:
        logger.info("cv_context_included", query_preview=query[:80])
        cv_block = wrap_cv_content(cv_text)
        parts.append(cv_block)

    if tool_block:
        parts.append(f"Tool results:\n{tool_block}")

    user_prompt = "\n\n".join(parts)
    return user_prompt, citation_map
