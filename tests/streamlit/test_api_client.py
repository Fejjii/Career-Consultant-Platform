"""Tests for Streamlit API client header behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from api_client import discover_provider_models


def test_discover_provider_models_uses_app_managed_source_when_key_missing() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"ok": True, "selectable_models": ["gpt-4.1"]}

    with patch("api_client.httpx.get", return_value=mock_resp) as get_call:
        discover_provider_models(
            api_base="http://test",
            session_id="sid-app",
            model="gpt-4.1",
            api_key=None,
        )

    _args, kwargs = get_call.call_args
    headers = kwargs["headers"]
    assert headers["X-Session-ID"] == "sid-app"
    assert headers["X-OpenAI-Model"] == "gpt-4.1"
    assert "X-OpenAI-API-Key" not in headers


def test_discover_provider_models_uses_user_key_when_present() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"ok": True, "selectable_models": ["gpt-4o"]}

    with patch("api_client.httpx.get", return_value=mock_resp) as get_call:
        discover_provider_models(
            api_base="http://test",
            session_id="sid-user",
            model="gpt-4o",
            api_key="sk-test",
        )

    _args, kwargs = get_call.call_args
    headers = kwargs["headers"]
    assert headers["X-Session-ID"] == "sid-user"
    assert headers["X-OpenAI-Model"] == "gpt-4o"
    assert headers["X-OpenAI-API-Key"] == "sk-test"
