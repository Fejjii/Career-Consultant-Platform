"""Tests for dynamic runtime intent routing and guardrails."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from career_intel.schemas.domain import RouterDecision


def _settings(**kwargs: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "max_input_length": 4000,
        "runtime_default_timezone": "UTC",
        "rag_similarity_threshold": 0.30,
        "rag_weak_evidence_threshold": 0.30,
        "rag_strong_evidence_threshold": 0.55,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _runtime_decision() -> RouterDecision:
    return RouterDecision(
        intent="dynamic_runtime",
        confidence=0.91,
        tool_name=None,
        params={},
        use_cv=False,
        reason="runtime date/time lookup required",
    )


@pytest.mark.asyncio
async def test_run_turn_dynamic_runtime_uses_deterministic_utility(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_route(query: str, *, cv_available: bool = False, settings: Any = None) -> tuple[Any, None]:
        return _runtime_decision(), None

    async def fake_validate(text: str, max_length: int = 4000) -> str:
        return text

    def fake_inc(name: str, count: int = 1) -> None:
        pass

    from career_intel.services import runtime_utility as runtime_module

    monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
    monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
    monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
    original_resolver = runtime_module.resolve_runtime_query
    monkeypatch.setattr(
        runtime_module,
        "resolve_runtime_query",
        lambda *args, **kwargs: original_resolver(
            "What day will it be tomorrow?",
            timezone_name="UTC",
            now_utc=datetime(2026, 4, 15, 14, 0, 0, tzinfo=UTC),
        ),
    )

    from career_intel.orchestration.chain import run_turn
    from career_intel.schemas.api import ChatMessage

    result = await run_turn(
        messages=[ChatMessage(role="user", content="What day will it be tomorrow?")],
        session_id="runtime-session",
        use_tools=True,
        filters=None,
        settings=_settings(),
        trace_id="runtime-trace",
    )

    assert result.intent == "dynamic_runtime"
    assert result.answer_source == "runtime"
    assert result.answer_mode == "RUNTIME"
    assert result.runtime_utility_used in {"relative_date", "current_date"}
    assert "April 16, 2026" in result.reply
    assert result.citations == []


@pytest.mark.asyncio
async def test_run_turn_dynamic_runtime_guard_blocks_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_route(query: str, *, cv_available: bool = False, settings: Any = None) -> tuple[Any, None]:
        return _runtime_decision(), None

    async def fake_validate(text: str, max_length: int = 4000) -> str:
        return text

    def fake_inc(name: str, count: int = 1) -> None:
        pass

    monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
    monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
    monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
    monkeypatch.setattr("career_intel.services.runtime_utility.resolve_runtime_query", lambda *args, **kwargs: None)

    from career_intel.orchestration.chain import run_turn
    from career_intel.schemas.api import ChatMessage

    result = await run_turn(
        messages=[ChatMessage(role="user", content="What day is today?")],
        session_id="runtime-block-session",
        use_tools=True,
        filters=None,
        settings=_settings(),
        trace_id="runtime-block-trace",
    )

    assert result.intent == "dynamic_runtime"
    assert result.answer_source == "runtime"
    assert result.answer_mode == "RUNTIME"
    assert result.runtime_utility_used == "blocked_no_runtime_resolution"
    assert "deterministically" in result.reply.lower()


@pytest.mark.asyncio
async def test_stream_turn_dynamic_runtime_emits_runtime_mode_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_route(query: str, *, cv_available: bool = False, settings: Any = None) -> tuple[Any, None]:
        return _runtime_decision(), None

    async def fake_validate(text: str, max_length: int = 4000) -> str:
        return text

    def fake_inc(name: str, count: int = 1) -> None:
        pass

    monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
    monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
    monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)

    from career_intel.orchestration.stream import stream_turn
    from career_intel.schemas.api import ChatMessage

    events: list[dict[str, Any]] = []
    async for sse_line in stream_turn(
        messages=[ChatMessage(role="user", content="What time is it now?")],
        session_id="runtime-stream-session",
        use_tools=True,
        filters=None,
        settings=_settings(),
        trace_id="runtime-stream-trace",
    ):
        if sse_line.startswith("data: "):
            events.append(json.loads(sse_line[6:]))

    debug_event = next(evt for evt in events if evt.get("type") == "debug")
    assert debug_event["data"]["intent"] == "dynamic_runtime"
    assert debug_event["data"]["answer_source"] == "runtime"
    assert debug_event["data"]["answer_mode"] == "RUNTIME"
    assert debug_event["data"]["runtime_utility_used"] in {"current_time", "current_date", "relative_date"}
