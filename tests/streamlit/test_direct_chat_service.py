"""Direct-mode chat service tests for BYOK request propagation."""

from __future__ import annotations

from career_intel.config import get_settings
from career_intel.schemas.api import ChatResponse
from streamlit_app.services import chat_service


def test_get_settings_uses_request_override_without_base_settings(monkeypatch) -> None:
    """Nested get_settings() calls should inherit the active request BYOK."""
    monkeypatch.setattr(
        "career_intel.config.settings._get_base_settings",
        lambda: (_ for _ in ()).throw(AssertionError("base settings should not be used")),
    )
    api_key_token, model_token = chat_service.set_request_llm_overrides(
        api_key="sk-request-key",
        model=None,
    )
    try:
        settings = get_settings()
    finally:
        chat_service.reset_request_llm_overrides(api_key_token, model_token)

    assert settings.openai_api_key.get_secret_value() == "sk-request-key"


def test_generate_response_uses_current_byok_across_requests(monkeypatch) -> None:
    """Direct-mode submit path should pass the current BYOK into nested settings loads."""
    captured_keys: list[str] = []

    monkeypatch.setattr(chat_service, "_DIRECT_CHAT_SERVICE", None)
    monkeypatch.setattr(
        "career_intel.config.settings._get_base_settings",
        lambda: (_ for _ in ()).throw(AssertionError("base settings should not be used")),
    )

    async def fake_run_turn(**kwargs) -> ChatResponse:
        nested_settings = get_settings()
        captured_keys.append(nested_settings.openai_api_key.get_secret_value())
        session_id = kwargs["session_id"]
        return ChatResponse(
            session_id=session_id,
            reply=f"echo:{kwargs['messages'][-1].content}",
            citations=[],
            tool_calls=[],
            answer_source="llm_fallback",
            answer_mode="LLM",
            intent="small_talk",
            answer_length="balanced",
            trace_id="test-trace",
            usage=None,
        )

    monkeypatch.setattr("career_intel.orchestration.chain.run_turn", fake_run_turn)

    first = chat_service.generate_response(
        body={
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "session-one",
            "use_tools": True,
            "answer_length": "balanced",
        },
        model="gpt-4.1",
        api_key="sk-first-key",
        user_timezone=None,
    )
    second = chat_service.generate_response(
        body={
            "messages": [{"role": "user", "content": "What skills should I build for data engineering? "}],
            "session_id": "session-two",
            "use_tools": True,
            "answer_length": "balanced",
        },
        model="gpt-4.1",
        api_key="sk-second-key",
        user_timezone=None,
    )

    assert first["reply"] == "echo:hello"
    assert second["reply"] == "echo:What skills should I build for data engineering? "
    assert captured_keys == ["sk-first-key", "sk-second-key"]
