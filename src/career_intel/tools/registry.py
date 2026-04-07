"""Tool registry and dispatch — determines which tools to invoke based on the query.

The router is **intent-first**: it classifies *what the user wants* before
deciding *which tool* (if any) to call.  See ``RouterDecision`` in
``schemas.domain`` for the full schema.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from career_intel.config import Settings, get_settings
from career_intel.llm import get_chat_llm
from career_intel.schemas.api import ToolCallResult
from career_intel.schemas.domain import (
    LearningPlanInput,
    RetrievedChunk,
    RoleCompareInput,
    RouterDecision,
    SkillGapInput,
)

logger = structlog.get_logger()

ROUTER_PROMPT = """\
You are the query router for a career intelligence assistant.

## Step 1 — Classify the user's INTENT (pick exactly one):
- "small_talk" — greetings, thanks, off-topic chit-chat.
- "direct_answer" — the user asks a factual question you can answer from \
general knowledge (no retrieval or tools needed).
- "retrieval_required" — the user needs career/skill/role information that \
requires looking up evidence from the knowledge base.
- "tool_required" — the user needs a structured analysis (skill gap, role \
comparison, learning plan) that maps to one of the tools below.

## Step 2 — If intent is "tool_required", pick a tool:
1. "skill_gap" — Analyse skill gaps between a user's skills and a target role.
   Params: {{"target_role": str, "current_skills": [str], "seniority": str|null}}
2. "role_compare" — Compare two roles side by side.
   Params: {{"role_a": str, "role_b": str}}
3. "learning_plan" — Generate a time-boxed learning plan for a goal role.
   Params: {{"goal_role": str, "hours_per_week": int, "horizon_weeks": int, "constraints": str|null}}

## Step 3 — Decide "use_cv":
CV availability: {cv_available}
Set "use_cv" to true ONLY when the query clearly benefits from the user's \
personal background (skill gap analysis, CV improvement, interview prep, \
personalised career advice). False for greetings, generic facts, or when \
no CV is available.

## User query
{query}

Respond with ONLY a JSON object:
{{"intent": "<intent>", "confidence": <0.0-1.0>, "tool_name": "<tool_name_or_null>", \
"params": {{...}}, "use_cv": true/false, "reason": "<one sentence>"}}"""


async def route_query(
    query: str,
    *,
    cv_available: bool = False,
    settings: Settings | None = None,
) -> RouterDecision:
    """Classify the query intent, optional tool, and CV relevance."""
    if settings is None:
        settings = get_settings()

    llm = get_chat_llm(settings, temperature=0.0)
    prompt = ROUTER_PROMPT.format(
        query=query,
        cv_available="The user HAS uploaded a CV." if cv_available else "No CV uploaded.",
    )
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    raw = response.content if hasattr(response, "content") else str(response)

    parsed = _parse_router_response(raw)
    decision = _normalize_decision(parsed, cv_available=cv_available)

    logger.info(
        "router_decision",
        query_preview=query[:80],
        intent=decision.intent,
        confidence=decision.confidence,
        tool=decision.tool_name,
        use_cv=decision.use_cv,
        reason=decision.reason,
    )
    return decision


async def maybe_call_tools(
    query: str,
    chunks: list[RetrievedChunk],
    settings: Settings | None = None,
    *,
    cv_available: bool = False,
) -> tuple[list[ToolCallResult], RouterDecision]:
    """Route the query, then execute the selected tool (if any).

    Returns
    -------
    (tool_results, decision)
        The tool results list and the full ``RouterDecision``.

    .. versionchanged:: 0.2
       Second element changed from ``bool`` (use_cv) to ``RouterDecision``.
    """
    if settings is None:
        settings = get_settings()

    decision = await route_query(query, cv_available=cv_available, settings=settings)

    if decision.tool_name is None:
        return [], decision

    logger.info("tool_dispatch", tool=decision.tool_name, params_keys=list(decision.params.keys()))

    try:
        result = await _execute_tool(decision.tool_name, decision.params, settings)
        return [result], decision
    except Exception as exc:
        logger.error("tool_execution_failed", tool=decision.tool_name, error=str(exc))
        return [ToolCallResult(
            tool_name=decision.tool_name,
            inputs=decision.params,
            output={"error": str(exc)},
            success=False,
            error=str(exc),
        )], decision


async def execute_tool(
    decision: RouterDecision,
    settings: Settings,
) -> ToolCallResult:
    """Execute the tool specified by a RouterDecision.

    Public entry point when routing has already been done externally.
    """
    if not decision.tool_name:
        raise ValueError("RouterDecision has no tool_name")
    return await _execute_tool(decision.tool_name, decision.params, settings)


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


_VALID_INTENTS = {"small_talk", "direct_answer", "retrieval_required", "tool_required"}
_VALID_TOOLS = {"skill_gap", "role_compare", "learning_plan"}


def _parse_router_response(text: str) -> dict[str, Any]:
    """Parse the LLM's JSON response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("router_parse_failed", raw=text[:200])
        return {}


def _normalize_decision(
    parsed: dict[str, Any],
    *,
    cv_available: bool,
) -> RouterDecision:
    """Coerce raw parsed JSON into a validated ``RouterDecision``.

    Handles both the new intent-first format and the legacy ``tool``-only
    format for backward compatibility.
    """
    intent = parsed.get("intent", "").lower()
    tool_name = parsed.get("tool_name") or parsed.get("tool")
    params = parsed.get("params", {})
    confidence = float(parsed.get("confidence", 0.0))
    use_cv = bool(parsed.get("use_cv", False)) and cv_available
    reason = parsed.get("reason", "")

    if tool_name == "none" or tool_name == "null":
        tool_name = None

    if tool_name and tool_name in _VALID_TOOLS and intent != "tool_required":
        intent = "tool_required"

    if intent not in _VALID_INTENTS:
        intent = "retrieval_required" if tool_name is None else "tool_required"

    if tool_name and tool_name not in _VALID_TOOLS:
        logger.warning("unknown_tool_from_router", tool=tool_name)
        tool_name = None
        intent = "retrieval_required"

    return RouterDecision(
        intent=intent,  # type: ignore[arg-type]
        confidence=min(max(confidence, 0.0), 1.0),
        tool_name=tool_name,
        params=params,
        use_cv=use_cv,
        reason=reason,
    )
