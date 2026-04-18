"""Tests for frontend model availability helpers."""

from __future__ import annotations

from model_config import (
    explain_model_unavailability,
    format_unavailability_for_ui,
    get_available_model_ids,
    get_default_model_id,
    get_model_label,
    get_supported_model_ids,
    resolve_selected_model,
    summarize_model_availability,
)


def test_supported_models_follow_allowlist_env(monkeypatch) -> None:
    monkeypatch.setenv("CAREER_INTEL_OPENAI_SUPPORTED_MODELS", "gpt-4o-mini,gpt-4o,gpt-4.1")
    assert get_supported_model_ids() == ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]


def test_available_models_use_byok_selectable_intersection() -> None:
    auth_status = {
        "ok": True,
        "normalized_accessible_models": ["gpt-4o-mini", "gpt-4.1", "gpt-5"],
        "selectable_models": ["gpt-4o-mini", "gpt-4.1", "gpt-5"],
    }
    assert get_available_model_ids(auth_status) == ["gpt-4.1", "gpt-4o-mini"]


def test_default_model_prefers_first_available_model_when_override_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("CAREER_INTEL_DEFAULT_MODEL", "gpt-4.5-preview")
    assert get_default_model_id(["gpt-4o-mini"]) == "gpt-4o-mini"


def test_default_model_uses_recommended_rank_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("CAREER_INTEL_DEFAULT_MODEL", raising=False)
    assert get_default_model_id(["gpt-4o-mini", "gpt-4o", "gpt-4.1"]) == "gpt-4.1"


def test_model_availability_summary_handles_large_raw_list() -> None:
    auth_status = {
        "ok": True,
        "accessible_models": [
            "gpt-4.1-2025-03-01",
            "gpt-4o",
            "gpt-5-2026-01-01",
            "gpt-4o-realtime-preview",
            "whisper-1",
        ],
        "selectable_models": ["gpt-4.1", "gpt-4o"],
        "supported_but_unavailable_models": ["gpt-4o-mini"],
        "accessible_but_unsupported_models": ["gpt-5"],
        "ignored_accessible_models": ["gpt-4o-realtime-preview", "whisper-1"],
    }
    summary = summarize_model_availability(auth_status)
    assert summary.raw_accessible_count == 5
    assert summary.selectable_models == ["gpt-4.1", "gpt-4o"]
    assert summary.supported_but_unavailable_models == ["gpt-4o-mini"]
    assert summary.accessible_but_unsupported_models == ["gpt-5"]
    assert summary.ignored_raw_models_count == 2


def test_model_availability_summary_exposes_unsupported_models_without_selecting_them() -> None:
    auth_status = {
        "ok": True,
        "normalized_accessible_models": ["gpt-4o", "gpt-5", "gpt-5-codex"],
        "selectable_models": ["gpt-4o", "gpt-5", "gpt-5-codex"],
    }
    summary = summarize_model_availability(auth_status)

    assert summary.selectable_models == ["gpt-4o"]
    assert summary.supported_but_unavailable_models == ["gpt-4.1", "gpt-4o-mini"]
    assert summary.accessible_but_unsupported_models == ["gpt-5", "gpt-5-codex"]


def test_model_unavailability_reason_lookup() -> None:
    auth_status = {
        "model_unavailability_reasons": {
            "gpt-4.1": "not_returned_by_provider_or_filtered_as_irrelevant",
        }
    }
    assert explain_model_unavailability(auth_status, "gpt-4.1") == "not_returned_by_provider_or_filtered_as_irrelevant"


def test_model_unavailability_reason_ui_copy() -> None:
    assert format_unavailability_for_ui("not_supported_by_app") == "not supported in this app"
    assert (
        format_unavailability_for_ui("not_returned_by_provider_or_filtered_as_irrelevant")
        == "not available to this key for selectable chat usage"
    )


def test_resolve_selected_model_falls_back_when_unavailable() -> None:
    resolution = resolve_selected_model(current_model="gpt-4o-mini", available_models=["gpt-4.1", "gpt-4o"])
    assert resolution.changed is True
    assert resolution.selected_model == "gpt-4.1"
    assert resolution.reason_code == "selected_model_not_available_for_credential_source"


def test_resolve_selected_model_keeps_existing_when_valid() -> None:
    resolution = resolve_selected_model(current_model="gpt-4o", available_models=["gpt-4.1", "gpt-4o"])
    assert resolution.changed is False
    assert resolution.selected_model == "gpt-4o"


def test_model_label_is_friendly_for_unknown_family() -> None:
    assert get_model_label("gpt-5") == "GPT-5"
