"""Tests for BYOK credential state transitions."""

from __future__ import annotations

from credential_state import (
    APP_MANAGED_SOURCE,
    USER_BYOK_SOURCE,
    resolve_credential_source,
    transition_after_clear,
    transition_after_validation,
)


def test_validate_success_switches_to_user_byok() -> None:
    transition = transition_after_validation(
        credential_source_before=APP_MANAGED_SOURCE,
        validated_byok_api_key_before="",
        user_provider_auth_status_before=None,
        byok_input_key="sk-user",
        auth_result={"ok": True, "selectable_models": ["gpt-4.1"]},
    )
    assert transition.credential_source_after == USER_BYOK_SOURCE
    assert transition.validated_byok_api_key == "sk-user"
    assert transition.byok_validated is True
    assert transition.byok_cleared is False


def test_validate_failure_does_not_adopt_unvalidated_key() -> None:
    transition = transition_after_validation(
        credential_source_before=APP_MANAGED_SOURCE,
        validated_byok_api_key_before="",
        user_provider_auth_status_before=None,
        byok_input_key="sk-typed-only",
        auth_result={"ok": False, "message": "bad key"},
    )
    assert transition.credential_source_after == APP_MANAGED_SOURCE
    assert transition.validated_byok_api_key == ""
    assert transition.byok_validated is False
    assert transition.byok_last_validation_error == "bad key"


def test_failed_validation_preserves_existing_valid_auth_state() -> None:
    previous_status = {"ok": True, "selectable_models": ["gpt-4.1"]}
    transition = transition_after_validation(
        credential_source_before=USER_BYOK_SOURCE,
        validated_byok_api_key_before="sk-valid",
        user_provider_auth_status_before=previous_status,
        byok_input_key="sk-invalid-attempt",
        auth_result={"ok": False, "message": "bad key"},
    )
    assert transition.credential_source_after == USER_BYOK_SOURCE
    assert transition.validated_byok_api_key == "sk-valid"
    assert transition.user_provider_auth_status == previous_status


def test_clear_key_switches_back_to_app_managed() -> None:
    transition = transition_after_clear()
    assert transition.credential_source_after == APP_MANAGED_SOURCE
    assert transition.validated_byok_api_key == ""
    assert transition.user_provider_auth_status is None
    assert transition.byok_cleared is True


def test_resolve_credential_source_requires_validated_key() -> None:
    assert (
        resolve_credential_source(active_source=USER_BYOK_SOURCE, validated_byok_api_key="")
        == APP_MANAGED_SOURCE
    )
    assert (
        resolve_credential_source(active_source=USER_BYOK_SOURCE, validated_byok_api_key="sk-valid")
        == USER_BYOK_SOURCE
    )
