"""Evaluation runner — offline evaluation of routing, retrieval, citations, and safety.

Provides both individual check functions (used by tests) and a batch
``run_evaluation`` entry point that can be called from CLI or notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from career_intel.schemas.domain import GoldenExample

logger = structlog.get_logger()

GOLDEN_DATASET_PATH = Path(__file__).parent / "datasets" / "golden_queries.json"

_INTENT_TAG_MAP = {
    "small_talk": "small_talk",
    "general_knowledge": "general_knowledge",
    "direct_answer": "general_knowledge",
    "domain_specific": "domain_specific",
    "retrieval_required": "domain_specific",
    "dynamic_runtime": "dynamic_runtime",
    "tool_required": "tool_required",
}


def load_golden_dataset(path: Path | None = None) -> list[GoldenExample]:
    """Load the golden query dataset from JSON."""
    if path is None:
        path = GOLDEN_DATASET_PATH
    with open(path) as f:
        data = json.load(f)
    return [GoldenExample(**item) for item in data]


def check_citation_integrity(
    cited_ids: set[int],
    retrieved_ids: set[int],
) -> dict[str, Any]:
    """Verify that all cited IDs are a subset of retrieved IDs."""
    is_valid = cited_ids.issubset(retrieved_ids)
    return {
        "valid": is_valid,
        "cited": sorted(cited_ids),
        "retrieved": sorted(retrieved_ids),
        "orphaned": sorted(cited_ids - retrieved_ids),
    }


def check_abstain_behaviour(response_text: str) -> bool:
    """Check whether the response appropriately abstains / hedges."""
    abstain_indicators = [
        "don't have enough evidence",
        "insufficient",
        "cannot confirm",
        "not in my knowledge base",
        "no reliable data",
        "i'm not able to",
        "consult a",
        "cannot guarantee",
        "not enough information",
    ]
    lower = response_text.lower()
    return any(indicator in lower for indicator in abstain_indicators)


def evaluate_retrieval_hit(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
) -> dict[str, Any]:
    """Compute retrieval hit metrics for a single query."""
    if not expected_chunk_ids:
        return {"skipped": True, "reason": "no_expected_chunks"}

    expected = set(expected_chunk_ids)
    retrieved = set(retrieved_chunk_ids)
    hits = expected & retrieved
    recall = len(hits) / len(expected) if expected else 0.0

    return {
        "recall": recall,
        "hits": len(hits),
        "expected": len(expected),
        "retrieved": len(retrieved),
    }


def check_routing_accuracy(
    example: GoldenExample,
    actual_intent: str,
) -> dict[str, Any]:
    """Check whether the router produced the expected intent for a golden query.

    Uses the first tag that matches a known intent as the expected intent.
    Falls back to ``expected_behaviour`` for tool/abstain cases.
    """
    expected_intent: str | None = None
    for tag in example.tags:
        if tag in _INTENT_TAG_MAP:
            expected_intent = _INTENT_TAG_MAP[tag]
            break

    if expected_intent is None:
        behaviour = example.expected_behaviour or ""
        if behaviour.startswith("use_tool:"):
            expected_intent = "tool_required"
        elif behaviour == "abstain" or behaviour == "cite_source":
            expected_intent = "domain_specific"
        elif behaviour == "small_talk":
            expected_intent = "small_talk"
        elif behaviour == "use_cv":
            expected_intent = "domain_specific"

    return {
        "query": example.query,
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "correct": expected_intent == actual_intent if expected_intent else None,
    }


def run_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluation results into a summary report.

    Parameters
    ----------
    results:
        List of per-query result dicts, each containing at minimum
        ``query``, ``intent``, ``retrieval_invoked``, ``sources_count``.

    Returns
    -------
    Summary with counts and pass rates.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "error": "no results"}

    routing_correct = sum(1 for r in results if r.get("routing_correct") is True)
    routing_checked = sum(1 for r in results if r.get("routing_correct") is not None)

    return {
        "total_queries": total,
        "routing_accuracy": round(routing_correct / routing_checked, 3) if routing_checked else None,
        "routing_checked": routing_checked,
        "avg_latency_ms": round(
            sum(r.get("total_latency_ms", 0) for r in results) / total, 1
        ),
        "retrieval_invoked_count": sum(1 for r in results if r.get("retrieval_invoked")),
        "avg_sources_count": round(
            sum(r.get("sources_count", 0) for r in results) / total, 1
        ),
    }
