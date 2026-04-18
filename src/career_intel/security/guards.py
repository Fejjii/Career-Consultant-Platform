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
from typing import TYPE_CHECKING

import structlog
from fastapi import HTTPException

from career_intel.security.hardening import redact_secret_patterns

if TYPE_CHECKING:
    from career_intel.config import Settings

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
    re.compile(
        r"(reveal|show|print|dump|extract)\s+.{0,40}(system\s*prompt|developer\s+message|hidden\s+prompt|internal\s+instructions)",
        re.IGNORECASE,
    ),
    re.compile(r"(developer|hidden)\s+(message|prompt|instructions?)", re.IGNORECASE),
    re.compile(r"(chain\s+of\s+thought|reasoning\s+trace|internal\s+scratchpad)", re.IGNORECASE),
    re.compile(r"(show|dump|return)\s+.{0,30}(tool\s+calls?|function\s+calls?)", re.IGNORECASE),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BOUNDARY_LEAK_RE = re.compile(r"</?BOUNDARY_[A-Za-z0-9_:+-]+[^>]*>", re.IGNORECASE)
_PROMPT_LEAK_LINES = (
    "retrieved reference material",
    "raw user-provided content",
    "do not treat it as instructions",
    "data only",
    "strict grounding mode:",
)
_SEVERE_OUTPUT_LEAK_PATTERNS = (
    re.compile(r"\b(system\s+prompt|developer\s+message|hidden\s+prompt)\b", re.IGNORECASE),
    re.compile(r"\b(chain\s+of\s+thought|reasoning\s+trace|internal\s+scratchpad)\b", re.IGNORECASE),
    re.compile(r"\bapi[_ -]?key\b", re.IGNORECASE),
)
_SAFE_OUTPUT_REFUSAL = "I can't provide hidden prompts, internal reasoning, or secrets."


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


async def validate_input_deep(
    text: str,
    max_length: int = 4000,
    *,
    settings: Settings | None = None,
) -> str:
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

    is_safe, reason = await check_injection_classifier(sanitized, settings=settings)
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


def sanitize_model_output(response_text: str) -> str:
    """Redact prompt-boundary artifacts if they leak into model output."""
    sanitized = _BOUNDARY_LEAK_RE.sub("[internal content removed]", response_text)
    sanitized = redact_secret_patterns(sanitized)
    lines: list[str] = []
    leaked = False
    severe_leak = False
    for line in sanitized.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in _PROMPT_LEAK_LINES):
            leaked = True
            continue
        if any(pattern.search(line) for pattern in _SEVERE_OUTPUT_LEAK_PATTERNS):
            leaked = True
            severe_leak = True
            continue
        lines.append(line)
    if leaked or sanitized != response_text:
        logger.warning("output_prompt_leak_redacted")
    cleaned = "\n".join(lines).strip()
    if severe_leak and not cleaned:
        return _SAFE_OUTPUT_REFUSAL
    if severe_leak:
        return cleaned or _SAFE_OUTPUT_REFUSAL
    return cleaned
