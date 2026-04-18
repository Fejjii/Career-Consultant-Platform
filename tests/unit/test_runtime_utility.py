"""Tests for deterministic runtime date/time utilities."""

from __future__ import annotations

from datetime import UTC, datetime

from career_intel.services.runtime_utility import (
    assess_dynamic_runtime_query,
    resolve_preferred_timezone,
    resolve_runtime_query,
)


def _fixed_now() -> datetime:
    return datetime(2026, 4, 15, 14, 30, 0, tzinfo=UTC)


def test_assess_dynamic_runtime_detects_date_query() -> None:
    assessment = assess_dynamic_runtime_query("What day is today?")
    assert assessment.is_dynamic_runtime is True
    assert assessment.confidence >= 0.55


def test_assess_dynamic_runtime_avoids_domain_overlap_false_positive() -> None:
    assessment = assess_dynamic_runtime_query("What are the current skills required for data engineers?")
    assert assessment.is_dynamic_runtime is False


def test_resolve_current_time() -> None:
    result = resolve_runtime_query(
        "What time is it right now?",
        timezone_name="UTC",
        now_utc=_fixed_now(),
    )
    assert result is not None
    assert result.utility_name == "current_time"
    assert "14:30:00 UTC" in result.answer_text


def test_resolve_today_date() -> None:
    result = resolve_runtime_query(
        "Tell me today's date.",
        timezone_name="UTC",
        now_utc=_fixed_now(),
    )
    assert result is not None
    assert result.utility_name == "current_date"
    assert "Wednesday, April 15, 2026" in result.answer_text


def test_resolve_relative_tomorrow_day() -> None:
    result = resolve_runtime_query(
        "What day will it be tomorrow?",
        timezone_name="UTC",
        now_utc=_fixed_now(),
    )
    assert result is not None
    assert result.utility_name == "relative_date"
    assert result.target_date_iso == "2026-04-16"
    assert "Thursday, April 16, 2026" in result.answer_text


def test_resolve_non_runtime_query_returns_none() -> None:
    result = resolve_runtime_query(
        "What is the capital of France?",
        timezone_name="UTC",
        now_utc=_fixed_now(),
    )
    assert result is None


def test_resolve_preferred_timezone_uses_user_first() -> None:
    resolved = resolve_preferred_timezone(
        user_timezone="Europe/London",
        configured_timezone="UTC",
    )
    assert resolved == "Europe/London"


def test_resolve_preferred_timezone_falls_back_to_config_then_utc() -> None:
    resolved_from_config = resolve_preferred_timezone(
        user_timezone="Invalid/Timezone",
        configured_timezone="Asia/Tokyo",
    )
    assert resolved_from_config == "Asia/Tokyo"

    resolved_to_utc = resolve_preferred_timezone(
        user_timezone="Invalid/Timezone",
        configured_timezone="Also/Invalid",
    )
    assert resolved_to_utc == "UTC"
