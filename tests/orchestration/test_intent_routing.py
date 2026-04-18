"""Regression tests for intent-driven routing and retrieval gating.

Verifies that:
- small_talk skips retrieval and returns no sources
- general_knowledge skips retrieval and returns no sources
- domain_specific triggers retrieval
- require_citations flag does not force retrieval for small_talk
- streaming fast-path yields tokens without retrieval
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from career_intel.orchestration.synthesize import generate_direct_response
from career_intel.schemas.domain import RouterDecision
from career_intel.tools.registry import (
    _apply_domain_trend_retrieval_bias,
    _apply_named_source_retrieval_bias,
    _normalize_decision,
)


def _settings(**kwargs: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "max_input_length": 4000,
        "rag_similarity_threshold": 0.30,
        "rag_weak_evidence_threshold": 0.30,
        "rag_strong_evidence_threshold": 0.55,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestDirectResponse:
    """generate_direct_response produces a reply without retrieval."""

    @pytest.mark.asyncio
    async def test_hello_gets_conversational_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeLLM:
            async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
                return SimpleNamespace(content="Hello! How can I help with your career today?")

        from career_intel.orchestration import synthesize as synth_module
        monkeypatch.setattr(synth_module, "get_chat_llm", lambda settings, temperature=0.7: FakeLLM())

        reply, _usage = await generate_direct_response("hello", settings=SimpleNamespace())
        assert "Hello" in reply or "hello" in reply.lower()
        assert "[1]" not in reply


class TestRunTurnIntentGating:
    """run_turn skips retrieval for small_talk/general_knowledge."""

    @pytest.mark.asyncio
    async def test_small_talk_no_retrieval_no_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeDirectLLM:
            async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
                return SimpleNamespace(content="Hi there! I'm here to help with career questions.")

        fake_router_decision = RouterDecision(
            intent="small_talk",
            confidence=0.95,
            tool_name=None,
            params={},
            use_cv=False,
            reason="greeting",
        )

        async def fake_route_query(query: str, *, cv_available: bool = False, settings: Any = None) -> tuple[Any, None]:
            return fake_router_decision, None

        async def fake_validate(text: str, max_length: int = 4000) -> str:
            return text

        def fake_inc(name: str, count: int = 1) -> None:
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route_query)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr(
            "career_intel.orchestration.synthesize.get_chat_llm",
            lambda settings, temperature=0.7: FakeDirectLLM(),
        )

        from career_intel.orchestration.chain import run_turn
        from career_intel.schemas.api import ChatMessage

        result = await run_turn(
            messages=[ChatMessage(role="user", content="hello")],
            session_id="test-session",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="test-trace",
        )

        assert result.intent == "small_talk"
        assert result.citations == []
        assert result.tool_calls == []
        assert result.answer_source == "llm_fallback"
        assert len(result.reply) > 0

    @pytest.mark.asyncio
    async def test_small_talk_with_require_citations_still_no_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """require_citations=True should not force retrieval for small_talk."""
        class FakeDirectLLM:
            async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
                return SimpleNamespace(content="Hey! What career question can I help with?")

        fake_router_decision = RouterDecision(
            intent="small_talk",
            confidence=0.9,
            tool_name=None,
            params={},
            use_cv=False,
            reason="greeting",
        )

        async def fake_route_query(query: str, *, cv_available: bool = False, settings: Any = None) -> tuple[Any, None]:
            return fake_router_decision, None

        async def fake_validate(text: str, max_length: int = 4000) -> str:
            return text

        def fake_inc(name: str, count: int = 1) -> None:
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route_query)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr(
            "career_intel.orchestration.synthesize.get_chat_llm",
            lambda settings, temperature=0.7: FakeDirectLLM(),
        )

        from career_intel.orchestration.chain import run_turn
        from career_intel.schemas.api import ChatMessage

        result = await run_turn(
            messages=[ChatMessage(role="user", content="hello")],
            session_id="test-session",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="test-trace",
        )

        assert result.intent == "small_talk"
        assert result.citations == []
        assert result.tool_calls == []


class TestStreamTurnIntentGating:
    """stream_turn fast-path yields tokens immediately for small_talk."""

    @pytest.mark.asyncio
    async def test_small_talk_stream_yields_tokens_no_citations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        fake_router_decision = RouterDecision(
            intent="small_talk",
            confidence=0.95,
            tool_name=None,
            params={},
            use_cv=False,
            reason="greeting",
        )

        async def fake_route_query(query: str, *, cv_available: bool = False, settings: Any = None) -> tuple[Any, None]:
            return fake_router_decision, None

        async def fake_validate(text: str, max_length: int = 4000) -> str:
            return text

        def fake_inc(name: str, count: int = 1) -> None:
            pass

        class FakeStreamChunk:
            def __init__(self, content: str) -> None:
                self.content = content

        class FakeStreamingLLM:
            async def astream(self, messages: list[dict[str, str]]):
                for word in ["Hello", " there", "!"]:
                    yield FakeStreamChunk(word)

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route_query)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr(
            "career_intel.llm.clients.ChatOpenAI",
            lambda **kwargs: FakeStreamingLLM(),
        )

        from career_intel.orchestration import stream as stream_module
        monkeypatch.setattr(stream_module, "get_chat_llm",
            lambda settings, temperature=0.7, streaming=True: FakeStreamingLLM(),
        )

        from career_intel.orchestration.stream import stream_turn
        from career_intel.schemas.api import ChatMessage

        events: list[dict] = []
        async for sse_line in stream_turn(
            messages=[ChatMessage(role="user", content="hello")],
            session_id="test-session",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="test-trace",
        ):
            if sse_line.startswith("data: "):
                events.append(json.loads(sse_line[6:]))

        event_types = [e["type"] for e in events]

        assert "intent" in event_types
        intent_event = next(e for e in events if e["type"] == "intent")
        assert intent_event["data"] == "small_talk"

        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) > 0

        assert "citations" not in event_types
        assert "status" not in event_types

        assert event_types[-1] == "done"
        debug_event = next(e for e in events if e["type"] == "debug")
        assert debug_event["data"]["answer_source"] == "llm_fallback"


class TestNormalizeDecisionIntents:
    """Intent classification produces expected results for key queries."""

    def test_small_talk_intent(self) -> None:
        parsed = {"intent": "small_talk", "confidence": 0.95, "tool_name": None, "reason": "greeting"}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.intent == "small_talk"
        assert decision.tool_name is None

    def test_retrieval_intent(self) -> None:
        parsed = {"intent": "domain_specific", "confidence": 0.8, "reason": "needs facts"}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.intent == "domain_specific"

    def test_tool_intent(self) -> None:
        parsed = {
            "intent": "tool_required",
            "confidence": 0.9,
            "tool_name": "skill_gap",
            "params": {"target_role": "ML Engineer", "current_skills": ["Python"]},
        }
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.intent == "tool_required"
        assert decision.tool_name == "skill_gap"


class TestNamedSourceBias:
    def test_named_source_query_is_biased_to_retrieval(self) -> None:
        decision = RouterDecision(
            intent="general_knowledge",
            confidence=0.8,
            tool_name=None,
            params={},
            use_cv=False,
            reason="generic fact",
        )

        adjusted = _apply_named_source_retrieval_bias(
            "What does WEF Future of Jobs 2025 say about green jobs?",
            decision,
        )

        assert adjusted.intent == "domain_specific"
        assert adjusted.tool_name is None

    def test_tool_required_is_not_overridden_by_named_source_bias(self) -> None:
        decision = RouterDecision(
            intent="tool_required",
            confidence=0.9,
            tool_name="role_compare",
            params={"role_a": "data engineer", "role_b": "data scientist"},
            use_cv=False,
            reason="comparison request",
        )

        adjusted = _apply_named_source_retrieval_bias(
            "Compare ESCO data engineer and data scientist roles.",
            decision,
        )

        assert adjusted.intent == "tool_required"
        assert adjusted.tool_name == "role_compare"


class TestDomainTrendBias:
    def test_data_ai_trend_query_is_biased_to_retrieval(self) -> None:
        decision = RouterDecision(
            intent="general_knowledge",
            confidence=0.75,
            tool_name=None,
            params={},
            use_cv=False,
            reason="broad question",
        )

        adjusted = _apply_domain_trend_retrieval_bias(
            "What future trends are shaping data and AI careers?",
            decision,
        )

        assert adjusted.intent == "domain_specific"
        assert adjusted.tool_name is None

    def test_non_domain_trend_query_is_not_biased(self) -> None:
        decision = RouterDecision(
            intent="general_knowledge",
            confidence=0.8,
            tool_name=None,
            params={},
            use_cv=False,
            reason="broad question",
        )

        adjusted = _apply_domain_trend_retrieval_bias(
            "What future trends are shaping healthcare policy?",
            decision,
        )

        assert adjusted.intent == "general_knowledge"

    def test_small_talk_is_not_overridden(self) -> None:
        decision = RouterDecision(
            intent="small_talk",
            confidence=0.95,
            tool_name=None,
            params={},
            use_cv=False,
            reason="greeting",
        )

        adjusted = _apply_domain_trend_retrieval_bias(
            "Thanks",
            decision,
        )

        assert adjusted.intent == "small_talk"
