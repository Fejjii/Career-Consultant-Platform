"""Citation mapping and validation utilities."""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_cited_ids(text: str) -> set[int]:
    """Extract all ``[n]`` citation numbers from a response string."""
    return {int(m.group(1)) for m in _CITATION_RE.finditer(text)}


def validate_citations(
    response_text: str,
    citation_map: dict[int, str],
) -> tuple[bool, set[int]]:
    """Check that every ``[n]`` in the response maps to a retrieved chunk.

    Returns (is_valid, invalid_ids).
    """
    cited = extract_cited_ids(response_text)
    valid_ids = set(citation_map.keys())
    invalid = cited - valid_ids
    if invalid:
        logger.warning("invalid_citations", invalid_ids=sorted(invalid))
    return len(invalid) == 0, invalid
