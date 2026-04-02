"""Evaluation runner — offline evaluation of retrieval, citations, and safety."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from career_intel.schemas.domain import GoldenExample

logger = structlog.get_logger()

GOLDEN_DATASET_PATH = Path(__file__).parent / "datasets" / "golden_queries.json"


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
