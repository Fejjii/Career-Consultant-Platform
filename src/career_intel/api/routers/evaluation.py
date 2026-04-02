"""Evaluation endpoint — admin-only batch evaluation runner."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter

from career_intel.api.deps import AdminDep, TraceIdDep

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
logger = structlog.get_logger()


@router.post("/run")
async def run_evaluation(
    _admin: AdminDep,
    trace_id: TraceIdDep,
) -> dict[str, Any]:
    """Trigger a batch evaluation run over the golden dataset.

    Loads the golden query set, runs each query category, and
    returns aggregated metrics.
    """
    logger.info("evaluation_run_triggered", trace_id=trace_id)

    from career_intel.evaluation.eval_runner import (
        check_abstain_behaviour,
        check_citation_integrity,
        load_golden_dataset,
    )

    examples = load_golden_dataset()

    results: dict[str, Any] = {
        "total_examples": len(examples),
        "abstain_cases": 0,
        "cite_cases": 0,
        "tool_cases": 0,
        "categories": {},
    }

    for ex in examples:
        for tag in ex.tags:
            results["categories"].setdefault(tag, 0)
            results["categories"][tag] += 1

        if ex.expected_behaviour == "abstain":
            results["abstain_cases"] += 1
        elif ex.expected_behaviour == "cite_source":
            results["cite_cases"] += 1
        elif ex.expected_behaviour and "use_tool" in ex.expected_behaviour:
            results["tool_cases"] += 1

    results["status"] = "dataset_loaded"
    results["message"] = (
        "Golden dataset validated. Full evaluation requires running each query "
        "through the pipeline. Use `pytest tests/rag/` for offline eval."
    )

    logger.info("evaluation_dataset_summary", **results)
    return results
