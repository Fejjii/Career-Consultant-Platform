"""Normalize retrieval queries for the English-indexed knowledge base."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

import structlog

from career_intel.config import Settings, get_settings
from career_intel.llm import get_chat_llm

logger = structlog.get_logger()

_QUERY_NORMALIZATION_PROMPT = """\
You normalize user queries for retrieval against an English-indexed career knowledge base.

Return only valid JSON with this exact schema:
{{
  "detected_language": "<iso-639-1 lowercase code or unknown>",
  "requires_translation": true,
  "english_query": "<query translated to English for retrieval>"
}}

Rules:
- Detect the language of the user's query.
- If the query is already English, set "requires_translation" to false and copy the query into
  "english_query" without paraphrasing it.
- If the query is not English, translate it into natural English for retrieval.
- Preserve ESCO, ISCO, job titles, acronyms, quoted text, and intent.
- Do not answer the question.
- Do not add commentary or markdown.

User query:
{query}
"""


@dataclass(frozen=True)
class RetrievalQueryNormalization:
    """Result of language detection and translation for retrieval."""

    detected_language: str
    retrieval_query: str
    translated_query: str | None
    translation_applied: bool


async def normalize_query_for_retrieval(
    query: str,
    *,
    settings: Settings | None = None,
) -> RetrievalQueryNormalization:
    """Detect query language and translate non-English queries for retrieval only."""
    if settings is None:
        settings = get_settings()

    original_query = query.strip()
    if not original_query:
        return RetrievalQueryNormalization(
            detected_language="unknown",
            retrieval_query=query,
            translated_query=None,
            translation_applied=False,
        )

    try:
        llm = get_chat_llm(settings, temperature=0.0)
        response = await llm.ainvoke([
            {"role": "user", "content": _QUERY_NORMALIZATION_PROMPT.format(query=original_query)},
        ])
        raw_content = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_normalization_payload(raw_content)

        detected_language = _normalize_language_code(parsed.get("detected_language"))
        english_query = _normalize_translated_query(parsed.get("english_query"), fallback=original_query)
        requires_translation = _coerce_bool(parsed.get("requires_translation"))
        translation_applied = requires_translation or detected_language not in {"en", "unknown"}

        retrieval_query = english_query if translation_applied else original_query
        translated_query = english_query if translation_applied else None

        logger.info(
            "retrieval_query_normalized",
            query_preview=original_query[:80],
            detected_language=detected_language,
            translation_applied=translation_applied,
            translated_query=translated_query,
            retrieval_query=retrieval_query[:120],
        )

        return RetrievalQueryNormalization(
            detected_language=detected_language,
            retrieval_query=retrieval_query,
            translated_query=translated_query,
            translation_applied=translation_applied,
        )
    except Exception as exc:  # pragma: no cover - defensive live-system fallback
        logger.warning(
            "retrieval_query_normalization_failed",
            query_preview=original_query[:80],
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        return RetrievalQueryNormalization(
            detected_language="unknown",
            retrieval_query=original_query,
            translated_query=None,
            translation_applied=False,
        )


def _parse_normalization_payload(raw_content: str) -> dict[str, object]:
    """Extract and parse the JSON payload returned by the normalization model."""
    candidate = raw_content.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        json_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Normalization payload must be a JSON object.")
    return parsed


def _normalize_language_code(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    if not normalized:
        return "unknown"
    return normalized[:8]


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _normalize_translated_query(value: object, *, fallback: str) -> str:
    if isinstance(value, str):
        normalized = " ".join(value.split()).strip()
        if normalized:
            return normalized
    return fallback
