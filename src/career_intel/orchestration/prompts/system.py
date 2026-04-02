"""System prompts and context-building utilities."""

from __future__ import annotations

from career_intel.schemas.domain import RetrievedChunk

SYSTEM_PROMPT = """\
You are the AI Career Intelligence Assistant — an expert career advisor that provides \
evidence-based guidance grounded in trusted labor-market and skills data.

Rules:
1. Every non-trivial factual claim MUST include an inline citation [n] referencing the \
   provided source chunks.
2. If the retrieved evidence is insufficient, say so honestly. Ask a clarifying question \
   or state what information is missing. NEVER fabricate facts.
3. Do NOT guarantee employment outcomes, salaries, or hiring decisions.
4. When presenting skill gaps or learning plans, base them ONLY on the provided sources.
5. Prefer newer sources when information conflicts.
6. Use structured formatting (tables, bullet points) for comparisons and lists.
7. End with a brief disclaimer: "This is guidance based on available data — consult a \
   career professional for personalised advice."
"""

DISCLAIMER = (
    "\n\n---\n*This is guidance based on available data — consult a "
    "career professional for personalised advice.*"
)


def build_context_block(
    chunks: list[RetrievedChunk],
) -> tuple[str, dict[int, str]]:
    """Build a numbered source context block and a citation_id -> chunk_id map.

    Returns
    -------
    context_block : str
        Formatted text block to inject into the user prompt.
    citation_map : dict[int, str]
        Mapping from citation number ``[n]`` to ``chunk_id``.
    """
    citation_map: dict[int, str] = {}
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        citation_map[idx] = chunk.chunk_id
        header = f"[{idx}] {chunk.metadata.title}"
        if chunk.metadata.section:
            header += f" — {chunk.metadata.section}"
        if chunk.metadata.publish_year:
            header += f" ({chunk.metadata.publish_year})"
        parts.append(f"{header}\n{chunk.text}")

    context_block = "Sources:\n\n" + "\n\n".join(parts)
    return context_block, citation_map
