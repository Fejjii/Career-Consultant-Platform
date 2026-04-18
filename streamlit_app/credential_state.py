"""Credential-source transition helpers for Streamlit session state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

APP_MANAGED_SOURCE = "app_managed"
USER_BYOK_SOURCE = "user_byok"


@dataclass(frozen=True)
class CredentialTransition:
    """State transition outcome for BYOK validation/clear actions."""

    credential_source_after: str
    validated_byok_api_key: str
    user_provider_auth_status: dict[str, Any] | None
    byok_last_validation_error: str | None
    byok_status_notice: str | None
    byok_validated: bool
    byok_cleared: bool


def resolve_credential_source(*, active_source: str, validated_byok_api_key: str) -> str:
    """Return a safe credential source based on validated key presence."""

    if active_source == USER_BYOK_SOURCE and validated_byok_api_key.strip():
        return USER_BYOK_SOURCE
    return APP_MANAGED_SOURCE


def transition_after_validation(
    *,
    credential_source_before: str,
    validated_byok_api_key_before: str,
    user_provider_auth_status_before: dict[str, Any] | None,
    byok_input_key: str,
    auth_result: dict[str, Any],
) -> CredentialTransition:
    """Apply validation result without conflating widget input and active auth."""

    key = byok_input_key.strip()
    if auth_result.get("ok"):
        return CredentialTransition(
            credential_source_after=USER_BYOK_SOURCE,
            validated_byok_api_key=key,
            user_provider_auth_status=auth_result,
            byok_last_validation_error=None,
            byok_status_notice="Using your validated key.",
            byok_validated=True,
            byok_cleared=False,
        )
    return CredentialTransition(
        credential_source_after=resolve_credential_source(
            active_source=credential_source_before,
            validated_byok_api_key=validated_byok_api_key_before,
        ),
        validated_byok_api_key=validated_byok_api_key_before,
        user_provider_auth_status=user_provider_auth_status_before,
        byok_last_validation_error=str(auth_result.get("message") or "Key not accepted."),
        byok_status_notice=None,
        byok_validated=False,
        byok_cleared=False,
    )


def transition_after_clear() -> CredentialTransition:
    """Reset validated BYOK state and switch back to app-managed credentials."""

    return CredentialTransition(
        credential_source_after=APP_MANAGED_SOURCE,
        validated_byok_api_key="",
        user_provider_auth_status=None,
        byok_last_validation_error=None,
        byok_status_notice="Switched back to app managed key.",
        byok_validated=False,
        byok_cleared=True,
    )
