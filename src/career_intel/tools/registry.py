"""Tool registry and dispatch — determines which tools to invoke based on the query."""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from career_intel.config import Settings, get_settings
from career_intel.schemas.api import ToolCallResult
from career_intel.schemas.domain import (
    LearningPlanInput,
    RetrievedChunk,
    RoleCompareInput,
    SkillGapInput,
)

logger = structlog.get_logger()

TOOL_DECISION_PROMPT = """\
You are a tool router for a career intelligence assistant. Based on the user query, \
decide which tool (if any) to call.

Available tools:
1. "skill_gap" — Analyze skill gaps between a user's skills and a target role.
   Required params: {{"target_role": str, "current_skills": [str], "seniority": str|null}}
2. "role_compare" — Compare two roles side by side.
   Required params: {{"role_a": str, "role_b": str}}
3. "learning_plan" — Generate a time-boxed learning plan for a goal role.
   Required params: {{"goal_role": str, "hours_per_week": int, "horizon_weeks": int, "constraints": str|null}}
4. "none" — No tool needed; answer directly from RAG.

User query: {query}

Respond with ONLY a JSON object:
{{"tool": "<tool_name>", "params": {{...}}}}

If no tool is appropriate, respond: {{"tool": "none", "params": {{}}}}"""


async def maybe_call_tools(
    query: str,
    chunks: list[RetrievedChunk],
    settings: Settings | None = None,
) -> list[ToolCallResult]:
    """Decide and execute tools based on the user query."""
    if settings is None:
        settings = get_settings()

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.0,
    )

    prompt = TOOL_DECISION_PROMPT.format(query=query)
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    raw = response.content if hasattr(response, "content") else str(response)

    decision = _parse_tool_decision(raw)
    tool_name = decision.get("tool", "none")
    params = decision.get("params", {})

    if tool_name == "none":
        return []

    logger.info("tool_dispatch", tool=tool_name, params_keys=list(params.keys()))

    try:
        result = await _execute_tool(tool_name, params, settings)
        return [result]
    except Exception as exc:
        logger.error("tool_execution_failed", tool=tool_name, error=str(exc))
        return [ToolCallResult(
            tool_name=tool_name,
            inputs=params,
            output={"error": str(exc)},
            success=False,
            error=str(exc),
        )]


async def _execute_tool(
    tool_name: str,
    params: dict[str, Any],
    settings: Settings,
) -> ToolCallResult:
    """Route to the appropriate tool implementation."""
    if tool_name == "skill_gap":
        from career_intel.tools.skill_gap import run_skill_gap

        input_data = SkillGapInput(**params)
        output = await run_skill_gap(input_data, settings=settings)
        return ToolCallResult(
            tool_name="skill_gap",
            inputs=params,
            output=output.model_dump(),
        )

    elif tool_name == "role_compare":
        from career_intel.tools.role_compare import run_role_compare

        input_data = RoleCompareInput(**params)
        output = await run_role_compare(input_data, settings=settings)
        return ToolCallResult(
            tool_name="role_compare",
            inputs=params,
            output=output.model_dump(),
        )

    elif tool_name == "learning_plan":
        from career_intel.tools.learning_plan import run_learning_plan

        input_data = LearningPlanInput(**params)
        output = await run_learning_plan(input_data, settings=settings)
        return ToolCallResult(
            tool_name="learning_plan",
            inputs=params,
            output=output.model_dump(),
        )

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def _parse_tool_decision(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("tool_decision_parse_failed", raw=text[:200])
        return {"tool": "none", "params": {}}
