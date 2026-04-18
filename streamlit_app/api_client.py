"""HTTP helpers for the Streamlit frontend."""

from __future__ import annotations

from typing import Any

import httpx
from services.chat_service import discover_provider_models as discover_provider_models_direct
from services.chat_service import is_direct_mode_enabled

OPENAI_API_KEY_HEADER = "X-OpenAI-API-Key"
OPENAI_MODEL_HEADER = "X-OpenAI-Model"
SESSION_ID_HEADER = "X-Session-ID"
USER_TIMEZONE_HEADER = "X-User-Timezone"


def build_request_headers(
    *,
    session_id: str,
    model: str | None = None,
    api_key: str | None = None,
    user_timezone: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build request headers for frontend-to-backend calls."""
    headers = {SESSION_ID_HEADER: session_id}
    if model:
        headers[OPENAI_MODEL_HEADER] = model
    if api_key:
        headers[OPENAI_API_KEY_HEADER] = api_key
    if user_timezone:
        headers[USER_TIMEZONE_HEADER] = user_timezone
    if extra_headers:
        headers.update(extra_headers)
    return headers


def discover_provider_models(
    *,
    api_base: str,
    session_id: str,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Ask the backend to discover models for the active credential source."""
    if is_direct_mode_enabled():
        return discover_provider_models_direct(model=model, api_key=api_key)
    response = httpx.get(
        f"{api_base}/health/provider-auth",
        headers=build_request_headers(session_id=session_id, model=model, api_key=api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def validate_provider_key(
    *,
    api_base: str,
    session_id: str,
    model: str,
    api_key: str,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Validate a BYOK key and discover accessible models."""

    return discover_provider_models(
        api_base=api_base,
        session_id=session_id,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
