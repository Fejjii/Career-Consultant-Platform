"""Tests for the dedicated source inventory response path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from career_intel.schemas.api import ChatMessage


@pytest.mark.asyncio
async def test_run_turn_short_circuits_source_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_validate(text: str, max_length: int = 4000) -> str:
        return text

    def fake_inc(name: str, count: int = 1) -> None:
        return None

    async def fail_route(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("route_query should not run for source inventory questions")

    monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
    monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
    monkeypatch.setattr("career_intel.tools.registry.route_query", fail_route)

    from career_intel.orchestration.chain import run_turn

    result = await run_turn(
        messages=[ChatMessage(role="user", content="What sources are you using?")],
        session_id="sid-1",
        use_tools=True,
        filters=None,
        settings=SimpleNamespace(max_input_length=4000),
        trace_id="trace-1",
    )

    assert result.answer_source == "source_inventory"
    assert result.intent == "source_inventory"
    assert "WEF Future of Jobs 2018" in result.reply
    assert "ESCO occupations" in result.reply


@pytest.mark.asyncio
async def test_stream_turn_short_circuits_source_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_validate(text: str, max_length: int = 4000) -> str:
        return text

    def fake_inc(name: str, count: int = 1) -> None:
        return None

    async def fail_route(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("route_query should not run for source inventory questions")

    monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
    monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
    monkeypatch.setattr("career_intel.tools.registry.route_query", fail_route)

    from career_intel.orchestration.stream import stream_turn

    events = []
    async for line in stream_turn(
        messages=[ChatMessage(role="user", content="How many sources are in the knowledge base?")],
        session_id="sid-2",
        use_tools=True,
        filters=None,
        settings=SimpleNamespace(max_input_length=4000),
        trace_id="trace-2",
    ):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    assert events[0] == {"type": "intent", "data": "source_inventory"}
    assert any(
        evt.get("type") == "token"
        and "groups" in evt.get("content", "")
        and "files found" in evt.get("content", "")
        for evt in events
    )
    assert any(
        evt.get("type") == "debug" and evt.get("data", {}).get("answer_source") == "source_inventory"
        for evt in events
    )
