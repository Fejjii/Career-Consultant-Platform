"""Shared security hardening helpers.

These utilities keep security-sensitive formatting rules centralized so
logging, uploads, citations, and secret handling stay consistent.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:bearer|token)\s+[A-Za-z0-9._\-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bOPENAI_API_KEY\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{8,}['\"]?", re.IGNORECASE),
)

_SAFE_FILENAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._ -]+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_LOG_REDACTED_TEXT_FIELDS = frozenset(
    {
        "input_preview",
        "normalized",
        "normalized_query",
        "original",
        "preview",
        "query",
        "query_preview",
        "raw",
        "retrieval_query",
        "rewritten",
        "translated_query",
        "user_query",
    }
)
_LOG_REDACTED_FIELD_HINTS = ("api_key", "authorization", "password", "secret", "token")


def mask_secret(value: str | None, *, keep_start: int = 4, keep_end: int = 4) -> str:
    """Return a masked representation safe for logs or UI notices."""

    raw = (value or "").strip()
    if not raw:
        return ""
    if len(raw) <= keep_start + keep_end:
        return "*" * len(raw)
    return f"{raw[:keep_start]}{'*' * (len(raw) - keep_start - keep_end)}{raw[-keep_end:]}"


def redact_secret_patterns(text: str) -> str:
    """Redact common secret-like values from arbitrary text."""

    redacted = text
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[redacted-secret]", redacted)
    return redacted


def summarize_text_for_logs(text: str | None) -> str:
    """Return a content-free summary for user-controlled text."""

    raw = text or ""
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"[redacted-text len={len(raw)} sha256={digest}]"


def redact_log_event(logger: object, method_name: str, event_dict: dict[str, object]) -> dict[str, object]:
    """Structlog processor that removes secrets and raw user text."""

    redacted: dict[str, object] = {}
    for key, value in event_dict.items():
        lowered = key.lower()
        if isinstance(value, str):
            if key in _LOG_REDACTED_TEXT_FIELDS:
                redacted[key] = summarize_text_for_logs(value)
                continue
            if any(hint in lowered for hint in _LOG_REDACTED_FIELD_HINTS):
                redacted[key] = mask_secret(value)
                continue
            redacted[key] = redact_secret_patterns(value)
            continue
        if isinstance(value, dict):
            redacted[key] = redact_log_event(logger, method_name, value)
            continue
        if isinstance(value, list):
            redacted[key] = [
                redact_secret_patterns(item) if isinstance(item, str) else item
                for item in value
            ]
            continue
        redacted[key] = value
    return redacted


def sanitize_upload_filename(filename: str | None, *, default_name: str = "upload.bin") -> str:
    """Return a safe basename for uploaded files.

    Keeps the original extension when possible, strips path segments and
    control characters, and removes characters that are awkward in logs or
    parsers.
    """

    candidate = Path((filename or "").strip()).name
    candidate = _CONTROL_CHARS_RE.sub("", candidate)
    candidate = _SAFE_FILENAME_CHARS_RE.sub("_", candidate).strip(" ._")
    if not candidate:
        return default_name
    return candidate[:120]


def sanitize_public_uri(uri: str | None) -> str | None:
    """Only expose public HTTP(S) URIs to clients."""

    raw = (uri or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return raw
