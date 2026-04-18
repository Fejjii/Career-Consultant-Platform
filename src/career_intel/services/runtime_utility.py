"""Deterministic runtime utilities for dynamic date/time style queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DAY_OFFSET_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bday after tomorrow\b", re.IGNORECASE), 2),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), 1),
    (re.compile(r"\byesterday\b", re.IGNORECASE), -1),
)
_IN_DAYS_PATTERN = re.compile(
    r"\b(?:in|after)\s+(?P<qty>\d{1,3})\s+(?P<unit>day|days|week|weeks)\b",
    re.IGNORECASE,
)
_FROM_NOW_PATTERN = re.compile(
    r"\b(?P<qty>\d{1,3})\s+(?P<unit>day|days|week|weeks)\s+from\s+now\b",
    re.IGNORECASE,
)

_TIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btime\b", re.IGNORECASE),
    re.compile(r"\bclock\b", re.IGNORECASE),
)
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdate\b", re.IGNORECASE),
    re.compile(r"\bday\b", re.IGNORECASE),
    re.compile(r"\bweekday\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
    re.compile(r"\bnext week\b", re.IGNORECASE),
    re.compile(r"\blast week\b", re.IGNORECASE),
    re.compile(r"\bthis week\b", re.IGNORECASE),
    re.compile(r"\bfrom now\b", re.IGNORECASE),
)
_DOMAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bskills?\b", re.IGNORECASE),
    re.compile(r"\bcareer\b", re.IGNORECASE),
    re.compile(r"\bjobs?\b", re.IGNORECASE),
    re.compile(r"\broles?\b", re.IGNORECASE),
    re.compile(r"\binterview\b", re.IGNORECASE),
    re.compile(r"\bresume\b", re.IGNORECASE),
    re.compile(r"\bcv\b", re.IGNORECASE),
    re.compile(r"\bsalary\b", re.IGNORECASE),
    re.compile(r"\bindustry\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RuntimeIntentAssessment:
    """Assessment of whether a query should use runtime utilities."""

    is_dynamic_runtime: bool
    confidence: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeUtilityResult:
    """Structured deterministic output for runtime queries."""

    utility_name: str
    timezone: str
    answer_text: str
    resolved_at_iso: str
    target_date_iso: str | None = None
    target_time_iso: str | None = None


def resolve_preferred_timezone(
    *,
    user_timezone: str | None,
    configured_timezone: str | None,
) -> str:
    """Resolve timezone precedence: user -> configured app -> UTC."""
    for candidate in (user_timezone, configured_timezone, "UTC"):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate).key
        except ZoneInfoNotFoundError:
            continue
    return "UTC"


def assess_dynamic_runtime_query(query: str) -> RuntimeIntentAssessment:
    """Return a confidence-scored dynamic-runtime assessment for a query."""
    lowered = f" {query.strip().lower()} "
    signals: list[str] = []
    score = 0.0

    if any(pattern.search(lowered) for pattern in _TIME_PATTERNS):
        signals.append("time_reference")
        score += 0.55

    if any(pattern.search(lowered) for pattern in _DATE_PATTERNS):
        signals.append("calendar_reference")
        score += 0.45

    if _extract_relative_day_offset(lowered) is not None:
        signals.append("relative_date_reference")
        score += 0.25

    # Avoid misclassifying domain questions containing words like "current skills".
    if any(pattern.search(lowered) for pattern in _DOMAIN_PATTERNS):
        signals.append("domain_overlap")
        score -= 0.35

    if "?" in query:
        score += 0.05
    if query.strip().lower().startswith(("what", "which", "tell", "give")):
        score += 0.05

    bounded = min(max(score, 0.0), 1.0)
    return RuntimeIntentAssessment(
        is_dynamic_runtime=bounded >= 0.5,
        confidence=bounded,
        signals=tuple(signals),
    )


def resolve_runtime_query(
    query: str,
    *,
    timezone_name: str = "UTC",
    now_utc: datetime | None = None,
) -> RuntimeUtilityResult | None:
    """Resolve a query using deterministic runtime data if supported."""
    assessment = assess_dynamic_runtime_query(query)
    if not assessment.is_dynamic_runtime:
        return None

    tz = _resolve_timezone(timezone_name)
    utc_now = now_utc or datetime.now(UTC)
    now_local = utc_now.astimezone(tz)
    lowered = f" {query.strip().lower()} "

    relative_offset = _extract_relative_day_offset(lowered)
    asks_time = any(pattern.search(lowered) for pattern in _TIME_PATTERNS)
    asks_date_like = any(pattern.search(lowered) for pattern in _DATE_PATTERNS) or relative_offset is not None

    if asks_time and relative_offset is None:
        text = (
            f"The current time is {now_local:%H:%M:%S} {tz.key} "
            f"on {_format_date(now_local.date())}."
        )
        return RuntimeUtilityResult(
            utility_name="current_time",
            timezone=tz.key,
            answer_text=text,
            resolved_at_iso=utc_now.isoformat(),
            target_time_iso=now_local.time().isoformat(),
            target_date_iso=now_local.date().isoformat(),
        )

    if asks_date_like:
        target = now_local.date() if relative_offset is None else now_local.date() + timedelta(days=relative_offset)
        descriptor = _relative_descriptor(relative_offset)
        text = f"{descriptor} is {_format_date(target)} ({tz.key})."
        return RuntimeUtilityResult(
            utility_name="relative_date" if relative_offset is not None else "current_date",
            timezone=tz.key,
            answer_text=text,
            resolved_at_iso=utc_now.isoformat(),
            target_date_iso=target.isoformat(),
            target_time_iso=None,
        )

    return None


def _resolve_timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _extract_relative_day_offset(text: str) -> int | None:
    for pattern, offset in _DAY_OFFSET_PATTERNS:
        if pattern.search(text):
            return offset

    for expr in (_IN_DAYS_PATTERN, _FROM_NOW_PATTERN):
        match = expr.search(text)
        if match:
            qty = int(match.group("qty"))
            unit = match.group("unit").lower()
            multiplier = 7 if unit.startswith("week") else 1
            return qty * multiplier

    if " next week " in text:
        return 7
    if " last week " in text:
        return -7
    return None


def _relative_descriptor(offset_days: int | None) -> str:
    if offset_days is None or offset_days == 0:
        return "Today"
    if offset_days == 1:
        return "Tomorrow"
    if offset_days == -1:
        return "Yesterday"
    if offset_days > 0:
        return f"In {offset_days} days"
    return f"{abs(offset_days)} days ago"


def _format_date(value: date) -> str:
    return f"{value:%A, %B} {value.day}, {value:%Y}"
