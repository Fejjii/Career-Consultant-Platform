"""Input and output security guards."""

from __future__ import annotations

import re

import structlog
from fastapi import HTTPException

logger = structlog.get_logger()

# Known prompt injection patterns (heuristic, not exhaustive)
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"disregard\s+(all|your)\s+", re.IGNORECASE),
    re.compile(r"override\s+(your|the)\s+", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system|internal)", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(script|iframe|object|embed)", re.IGNORECASE),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def validate_input(text: str, max_length: int = 4000) -> str:
    """Validate and sanitize user input.

    Raises HTTPException(400) if input is rejected.
    Returns the sanitized text on success.
    """
    if not text or not text.strip():
        logger.warning("validation_failed", reason="empty_input")
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    if len(text) > max_length:
        logger.warning(
            "validation_failed",
            reason="length_exceeded",
            length=len(text),
            max_length=max_length,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Message too long ({len(text)} chars). Maximum is {max_length}.",
        )

    # Sanitize HTML
    sanitized = _HTML_TAG_RE.sub("", text)

    # Check for prompt injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            logger.warning(
                "injection_flagged",
                pattern=pattern.pattern[:50],
                input_preview=sanitized[:80],
            )
            raise HTTPException(
                status_code=400,
                detail="Your message was flagged by our safety filter. Please rephrase.",
            )

    return sanitized


def validate_output_citations(
    response_text: str,
    citation_map: dict[int, str],
    require_citations: bool = False,
) -> str:
    """Validate that output citations reference actual retrieved chunks.

    Returns the (possibly unchanged) response text.
    """
    from career_intel.rag.citation import extract_cited_ids

    cited = extract_cited_ids(response_text)
    valid_ids = set(citation_map.keys())
    invalid = cited - valid_ids

    if invalid:
        logger.warning("output_invalid_citations", invalid_ids=sorted(invalid))

    if require_citations and not cited and len(response_text) > 100:
        logger.warning("output_missing_citations", response_length=len(response_text))

    return response_text
