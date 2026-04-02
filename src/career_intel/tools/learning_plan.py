"""Learning Plan Generator tool — create a time-boxed study plan from KB resources."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from career_intel.config import Settings, get_settings
from career_intel.rag.retriever import retrieve_chunks
from career_intel.schemas.domain import LearningPlanInput, LearningPlanOutput

logger = structlog.get_logger()

PLAN_PROMPT = """\
You are a career learning planner. Create a structured learning plan using ONLY \
resources and skills mentioned in the provided context.

Context from knowledge base:
{context}

Goal role: {goal_role}
Time budget: {hours_per_week} hours/week for {horizon_weeks} weeks
Constraints: {constraints}

Produce a JSON object with:
- "milestones": list of {{
    "week_range": str (e.g. "1-2"),
    "focus": str,
    "skills": [str],
    "activities": [str]
  }}
- "resources": list of {{"name": str, "type": str, "source_ref": int}} where source_ref \
  is the [n] citation number from context.

CRITICAL: Only reference resources that appear in the provided context. If the knowledge \
base is insufficient for a complete plan, include a "gaps" field listing what's missing.

Respond with ONLY valid JSON, no markdown fences."""


async def run_learning_plan(
    input_data: LearningPlanInput,
    settings: Settings | None = None,
) -> LearningPlanOutput:
    """Execute the learning plan generator tool."""
    if settings is None:
        settings = get_settings()

    query = f"learning path skills required for {input_data.goal_role} role training resources"

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
        temperature=0.2,
    )

    prompt = PLAN_PROMPT.format(
        context=context,
        goal_role=input_data.goal_role,
        hours_per_week=input_data.hours_per_week,
        horizon_weeks=input_data.horizon_weeks,
        constraints=input_data.constraints or "none specified",
    )

    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    raw_text = response.content if hasattr(response, "content") else str(response)

    parsed = _parse_json_response(raw_text)

    return LearningPlanOutput(
        goal_role=input_data.goal_role,
        total_weeks=input_data.horizon_weeks,
        milestones=parsed.get("milestones", []),
        resources=parsed.get("resources", []),
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
        logger.warning("learning_plan_json_parse_failed", raw=text[:200])
        return {"milestones": [], "resources": [], "gaps": ["Failed to parse structured output."]}
