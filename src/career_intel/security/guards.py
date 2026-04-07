"""Multi-layer input and output security guards.

Layer 1: Fast heuristic regex patterns (synchronous, <1ms).
Layer 2: Encoded-attack detection (synchronous, <1ms).
Layer 3: OpenAI moderation API classifier (async, ~100-300ms).

All three layers run in sequence.  If any layer rejects, the request
is blocked.  Layer 3 fails open if the API is unreachable (Layer 1+2
still protect).
"""

from __future__ import annotations

import re

import structlog
from fastapi import HTTPException

logger = structlog.get_logger()

# --- Layer 1: Heuristic patterns ---
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"disregard\s+(all|your)\s+", re.IGNORECASE),
    re.compile(r"override\s+(your|the)\s+(instructions|rules|guidelines|settings|system)", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system|internal)", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(script|iframe|object|embed)", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+(any|the|your)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(to\s+be|you\s+are)\s+", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def validate_input(text: str, max_length: int = 4000) -> str:
    """Validate and sanitize user input through all guard layers.

    Raises HTTPException(400) if any layer rejects the input.
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

    sanitized = _HTML_TAG_RE.sub("", text)

    # Layer 1: Heuristic patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            logger.warning(
                "injection_flagged",
                layer="heuristic",
                pattern=pattern.pattern[:50],
                input_preview=sanitized[:80],
            )
            raise HTTPException(
                status_code=400,
                detail="Your message was flagged by our safety filter. Please rephrase.",
            )

    return sanitized


async def validate_input_deep(text: str, max_length: int = 4000) -> str:
    """Full async validation including classifier layers.

    Call this from async endpoints to get Layer 2+3 protection on top
    of the synchronous Layer 1 checks.
    """
    sanitized = validate_input(text, max_length)

    # Layer 2: Encoded attack detection
    from career_intel.security.injection_classifier import check_encoded_attacks

    is_safe, reason = await check_encoded_attacks(sanitized)
    if not is_safe:
        logger.warning(
            "injection_flagged",
            layer="encoded_attack",
            reason=reason,
            input_preview=sanitized[:80],
        )
        raise HTTPException(
            status_code=400,
            detail="Your message was flagged by our safety filter. Please rephrase.",
        )

    # Layer 3: OpenAI moderation classifier
    from career_intel.security.injection_classifier import check_injection_classifier

    is_safe, reason = await check_injection_classifier(sanitized)
    if not is_safe:
        logger.warning(
            "injection_flagged",
            layer="moderation_classifier",
            reason=reason,
            input_preview=sanitized[:80],
        )
        raise HTTPException(
            status_code=400,
            detail="Your message was flagged by our content safety filter. Please rephrase.",
        )

    return sanitized


def validate_output_citations(
    response_text: str,
    citation_map: dict[int, str],
    require_citations: bool = False,
) -> str:
    """Validate that output citations reference actual retrieved chunks."""
    from career_intel.rag.citation import extract_cited_ids

    cited = extract_cited_ids(response_text)
    valid_ids = set(citation_map.keys())
    invalid = cited - valid_ids

    if invalid:
        logger.warning("output_invalid_citations", invalid_ids=sorted(invalid))

    if require_citations and not cited and len(response_text) > 100:
        logger.warning("output_missing_citations", response_length=len(response_text))

    return response_text
