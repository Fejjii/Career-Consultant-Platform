"""Regression tests for BYOK model discovery normalization and filtering."""

from __future__ import annotations

from career_intel.api.routers.health import (
    _normalize_accessible_chat_models,
    _normalize_model_id,
    _normalize_unknown_chat_family,
)


def test_normalize_supported_aliases_and_dated_snapshots() -> None:
    assert _normalize_model_id("gpt-4.1-2025-04-14") == "gpt-4.1"
    assert _normalize_model_id("chatgpt-4o-latest") == "gpt-4o"
    assert _normalize_model_id("gpt-4o-mini-2024-09-01") == "gpt-4o-mini"


def test_filter_noisy_or_irrelevant_models() -> None:
    assert _normalize_model_id("gpt-4o-realtime-preview") is None
    assert _normalize_model_id("whisper-1") is None
    assert _normalize_model_id("gpt-4o-audio-preview") is None
    assert _normalize_model_id("text-embedding-3-large") is None


def test_normalize_accessible_models_with_many_variants() -> None:
    normalized, ignored = _normalize_accessible_chat_models(
        [
            "gpt-4.1-2025-04-14",
            "gpt-4.1-2025-02-01",
            "chatgpt-4o-latest",
            "gpt-4o-mini-2024-10-01",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-realtime-preview",
            "whisper-1",
        ]
    )
    assert normalized == ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]
    assert ignored == ["gpt-4o-realtime-preview", "whisper-1"]


def test_unavailable_newer_family_exposed_as_unsupported_chat_model() -> None:
    assert _normalize_unknown_chat_family("gpt-5-2026-01-01") == "gpt-5"
