"""Skill Gap Analyzer tool — compare current skills against a target role."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from career_intel.config import Settings, get_settings
from career_intel.rag.retriever import retrieve_chunks
from career_intel.schemas.domain import SkillGapInput, SkillGapOutput

logger = structlog.get_logger()

SKILL_GAP_PROMPT = """\
You are a career skills analyst. Given a target role profile and a user's current skills, \
identify the skill gaps.

Target role context (from knowledge base):
{context}

User's current skills: {current_skills}
Target role: {target_role}
Seniority: {seniority}

Produce a JSON object with these fields:
- "must_have_gaps": list of {{"skill": str, "importance": "critical"|"high"|"medium", "reason": str}}
- "nice_to_have_gaps": list of {{"skill": str, "importance": "low"|"medium", "reason": str}}
- "suggested_order": list of skill names in recommended learning order

Only include skills that appear in the provided context. If the knowledge base lacks \
information about this role, say so clearly in a "note" field.

Respond with ONLY valid JSON, no markdown fences."""


async def run_skill_gap(
    input_data: SkillGapInput,
    settings: Settings | None = None,
) -> SkillGapOutput:
    """Execute the skill gap analysis tool."""
    if settings is None:
        settings = get_settings()

    query = f"skills required for {input_data.target_role} role"
    if input_data.seniority:
        query += f" at {input_data.seniority} level"

    chunks = await retrieve_chunks(query=query, settings=settings, top_k=10)

    context = "\n\n".join(
        f"[{i+1}] {c.metadata.title} — {c.metadata.section or 'N/A'}\n{c.text}"
        for i, c in enumerate(chunks)
    )

    citations = [
        {
            "id": i + 1,
            "source_id": c.metadata.source_id,
            "title": c.metadata.title,
            "section": c.metadata.section,
        }
        for i, c in enumerate(chunks)
    ]

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.1,
    )

    prompt = SKILL_GAP_PROMPT.format(
        context=context,
        current_skills=", ".join(input_data.current_skills),
        target_role=input_data.target_role,
        seniority=input_data.seniority or "not specified",
    )

    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    raw_text = response.content if hasattr(response, "content") else str(response)

    parsed = _parse_json_response(raw_text)

    return SkillGapOutput(
        target_role=input_data.target_role,
        must_have_gaps=parsed.get("must_have_gaps", []),
        nice_to_have_gaps=parsed.get("nice_to_have_gaps", []),
        suggested_order=parsed.get("suggested_order", []),
        citations=citations,
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    """Attempt to parse JSON from LLM output, with fallback."""
    import json

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("skill_gap_json_parse_failed", raw=text[:200])
        return {
            "must_have_gaps": [],
            "nice_to_have_gaps": [],
            "suggested_order": [],
            "note": "Failed to parse structured output from the model.",
        }
