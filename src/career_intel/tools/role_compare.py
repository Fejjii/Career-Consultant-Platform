"""Role Comparison tool — side-by-side analysis of two roles."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from career_intel.config import Settings, get_settings
from career_intel.rag.retriever import retrieve_chunks
from career_intel.schemas.domain import RoleCompareInput, RoleCompareOutput

logger = structlog.get_logger()

COMPARE_PROMPT = """\
You are a career comparison analyst. Compare two roles using ONLY the provided context.

Context from knowledge base:
{context}

Role A: {role_a}
Role B: {role_b}

Produce a JSON object with:
- "comparison": {{
    "responsibilities": {{"role_a": [list], "role_b": [list]}},
    "skills": {{"role_a": [list], "role_b": [list]}},
    "career_progression": {{"role_a": str, "role_b": str}},
    "overlap": [shared skills/responsibilities]
  }}
- "narrative": A 2-3 paragraph comparison summary with inline [n] citations.

If information for either role is missing from the context, state that clearly.

Respond with ONLY valid JSON, no markdown fences."""


async def run_role_compare(
    input_data: RoleCompareInput,
    settings: Settings | None = None,
) -> RoleCompareOutput:
    """Execute the role comparison tool."""
    if settings is None:
        settings = get_settings()

    query_a = f"{input_data.role_a} role profile responsibilities skills"
    query_b = f"{input_data.role_b} role profile responsibilities skills"

    chunks_a = await retrieve_chunks(query=query_a, settings=settings, top_k=8)
    chunks_b = await retrieve_chunks(query=query_b, settings=settings, top_k=8)

    # Merge and deduplicate
    seen_ids: set[str] = set()
    all_chunks = []
    for c in chunks_a + chunks_b:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            all_chunks.append(c)

    context = "\n\n".join(
        f"[{i+1}] {c.metadata.title} — {c.metadata.section or 'N/A'}\n{c.text}"
        for i, c in enumerate(all_chunks)
    )

    citations = [
        {
            "id": i + 1,
            "source_id": c.metadata.source_id,
            "title": c.metadata.title,
            "section": c.metadata.section,
        }
        for i, c in enumerate(all_chunks)
    ]

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.1,
    )

    prompt = COMPARE_PROMPT.format(
        context=context,
        role_a=input_data.role_a,
        role_b=input_data.role_b,
    )

    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    raw_text = response.content if hasattr(response, "content") else str(response)

    parsed = _parse_json_response(raw_text)

    return RoleCompareOutput(
        role_a=input_data.role_a,
        role_b=input_data.role_b,
        comparison=parsed.get("comparison", {}),
        narrative=parsed.get("narrative", "Comparison could not be generated."),
        citations=citations,
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    import json

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("role_compare_json_parse_failed", raw=text[:200])
        return {"comparison": {}, "narrative": "Failed to parse structured output."}
