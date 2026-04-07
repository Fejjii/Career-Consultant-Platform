"""Second-layer prompt injection detection using the OpenAI moderation endpoint.

Architecture:
  Layer 1 (guards.py) — fast heuristic regex patterns, catches obvious attacks.
  Layer 2 (this module) — OpenAI moderation API call for higher recall on:
    - Paraphrased injection attempts
    - Multilingual / encoded instruction attacks
    - Indirect prompt injection in pasted documents

Limitations (documented):
  - The moderation endpoint is designed for content policy, not specifically
    for prompt injection.  It catches many but not all injection patterns.
  - Novel attack vectors (e.g. base64-encoded payloads, Unicode homoglyphs)
    may bypass both layers.  Defense-in-depth via delimiter sanitization
    and output guards remains essential.
  - Latency: adds ~100-300ms per check.  Applied only to user-facing
    chat input, not internal retrieval.

Fallback:
  If the moderation API is unreachable, the check logs a warning and
  passes the input through (fail-open for availability, since Layer 1
  heuristics still apply).
"""

from __future__ import annotations

import structlog
from openai import OpenAI

from career_intel.config import Settings, get_settings

logger = structlog.get_logger()

# Moderation categories that suggest prompt injection or manipulation
_SUSPICIOUS_CATEGORIES = {
    "harassment",
    "harassment/threatening",
    "hate",
    "self-harm/instructions",
    "violence",
}


async def check_injection_classifier(
    text: str,
    settings: Settings | None = None,
) -> tuple[bool, str | None]:
    """Run the OpenAI moderation endpoint as a second-layer injection check.

    Returns
    -------
    (is_safe, reason)
        ``is_safe`` is True if the text passes moderation.
        ``reason`` contains the flagged category if blocked.
    """
    if settings is None:
        settings = get_settings()

    try:
        import asyncio

        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            max_retries=2,
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.moderations.create(input=text),
        )

        for moderation_result in result.results:
            if moderation_result.flagged:
                flagged_cats = [
                    cat for cat, flagged in moderation_result.categories.model_dump().items()
                    if flagged
                ]
                logger.warning(
                    "moderation_flagged",
                    categories=flagged_cats,
                    input_preview=text[:80],
                )
                return False, f"Content flagged: {', '.join(flagged_cats)}"

        return True, None

    except Exception as exc:
        logger.warning(
            "moderation_api_unavailable",
            error=str(exc)[:200],
        )
        return True, None  # Fail open — Layer 1 heuristics still apply


async def check_encoded_attacks(text: str) -> tuple[bool, str | None]:
    """Detect common encoded instruction injection patterns.

    Catches base64-encoded payloads, Unicode direction overrides,
    and zero-width character sequences.
    """
    import base64
    import re

    # Check for base64-encoded blocks that decode to injection-like content
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
    for match in b64_pattern.finditer(text):
        try:
            decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore").lower()
            injection_keywords = ["ignore", "instructions", "system prompt", "override", "disregard"]
            if any(kw in decoded for kw in injection_keywords):
                logger.warning("encoded_injection_detected", encoding="base64", preview=decoded[:80])
                return False, "Encoded injection attempt detected."
        except Exception:
            continue

    # Check for Unicode direction override characters
    bidi_chars = {"\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069"}
    if any(c in text for c in bidi_chars):
        logger.warning("bidi_injection_detected", input_preview=text[:80])
        return False, "Suspicious Unicode control characters detected."

    # Check for zero-width characters used to hide content
    zwc_pattern = re.compile(r"[\u200b\u200c\u200d\ufeff]{3,}")
    if zwc_pattern.search(text):
        logger.warning("zero_width_injection_detected", input_preview=text[:80])
        return False, "Suspicious zero-width character sequence detected."

    return True, None
