"""Frontend validation and abuse-control helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PROMPT_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.IGNORECASE), "Prompt override attempt"),
    (re.compile(r"reveal\s+(your|the)\s+(system|internal)\s+(prompt|instructions)", re.IGNORECASE), "System prompt exfiltration attempt"),
    (re.compile(r"override\s+(your|the)\s+(instructions|rules|guidelines|system)", re.IGNORECASE), "Instruction override attempt"),
    (re.compile(r"do\s+not\s+follow\s+(any|the|your)\s+(rules|instructions)", re.IGNORECASE), "Safety bypass attempt"),
]


@dataclass(frozen=True)
class PromptRisk:
    """Prompt-injection screening result."""

    flagged: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SessionRateLimitResult:
    """Frontend session limiter result."""

    allowed: bool
    retry_after_seconds: int = 0
    reason: str | None = None
    recent_timestamps: tuple[float, ...] = ()


def inspect_prompt(text: str) -> PromptRisk:
    """Flag obvious prompt-injection phrasing before sending to the backend."""
    reasons = tuple(label for pattern, label in _PROMPT_INJECTION_PATTERNS if pattern.search(text))
    return PromptRisk(flagged=bool(reasons), reasons=reasons)


def validate_uploaded_file(
    *,
    filename: str,
    size_bytes: int,
    allowed_extensions: set[str],
    max_bytes: int,
) -> str | None:
    """Validate a client-side upload before sending it to the backend."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        return f"Unsupported file type. Allowed: {allowed}."
    if size_bytes > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return f"File is too large. Limit is {limit_mb:.1f} MB."
    return None


def apply_session_rate_limit(
    timestamps: list[float],
    *,
    now: float,
    max_requests: int,
    window_seconds: int,
    min_interval_seconds: float,
) -> SessionRateLimitResult:
    """Apply a simple local session rate limit for faster client feedback."""
    recent = [stamp for stamp in timestamps if now - stamp < window_seconds]
    if recent and now - recent[-1] < min_interval_seconds:
        retry_after = max(1, int(min_interval_seconds - (now - recent[-1])) + 1)
        return SessionRateLimitResult(
            allowed=False,
            retry_after_seconds=retry_after,
            reason="Please wait a moment before sending another message.",
            recent_timestamps=tuple(recent),
        )
    if len(recent) >= max_requests:
        retry_after = max(1, int(window_seconds - (now - recent[0])) + 1)
        return SessionRateLimitResult(
            allowed=False,
            retry_after_seconds=retry_after,
            reason="Session rate limit reached. Please try again shortly.",
            recent_timestamps=tuple(recent),
        )
    recent.append(now)
    return SessionRateLimitResult(allowed=True, recent_timestamps=tuple(recent))
