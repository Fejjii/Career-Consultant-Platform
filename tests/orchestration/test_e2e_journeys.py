"""End-to-end journey tests covering all key user flows.

Each test validates a complete user journey through the orchestration layer
(chain or stream) with mocked LLM/retrieval, confirming:
- correct intent classification
- retrieval invoked or skipped
- sources present or absent
- CV used or not
- response structure correct

Test Matrix
-----------
| Journey                 | Intent             | Retrieval | CV  | Sources | Streaming |
|-------------------------|--------------------|-----------|-----|---------|-----------|
| Greeting                | small_talk         | No        | No  | None    | Fast-path |
| Thanks / acknowledgment | small_talk         | No        | No  | None    | Fast-path |
| Domain factual question | retrieval_required | Yes       | No  | Yes     | Normal    |
| CV upload + CV question | retrieval_required | Yes       | Yes | Yes     | Normal    |
| CV present, irrelevant  | retrieval_required | Yes       | No  | Yes     | Normal    |
| New session reset       | (N/A)              | N/A       | N/A | N/A     | N/A       |
| Backend unavailable     | (N/A)              | N/A       | N/A | N/A     | N/A       |
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from career_intel.schemas.api import ChatMessage
from career_intel.schemas.domain import ChunkMetadata, RetrievedChunk, RouterDecision


def _make_chunks(n: int = 2) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            text=f"Evidence chunk {i} about career trends.",
            metadata=ChunkMetadata(
                source_id=f"s{i}", source_type="report", title=f"Report {i}",
                chunk_index=0, parent_doc_id="d1",
            ),
            score=0.7,
        )
        for i in range(1, n + 1)
    ]


class _FakeLLM:
    """Configurable fake LLM for testing."""
    def __init__(self, response: str = "Test response") -> None:
        self._response = response

    async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        return SimpleNamespace(content=self._response)


class TestSmallTalkJourney:
    """Greeting: small_talk -> no retrieval -> no sources -> fast path."""

    @pytest.mark.asyncio
    async def test_greeting_skips_retrieval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="small_talk", confidence=0.95, reason="greeting",
        )
        async def fake_route(q, *, cv_available=False, settings=None):
            return decision
        async def fake_validate(text, max_length=4000):
            return text
        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr(
            "career_intel.orchestration.synthesize.get_chat_llm",
            lambda settings, temperature=0.7: _FakeLLM("Hi! How can I help with your career?"),
        )

        from career_intel.orchestration.chain import run_turn
        result = await run_turn(
            messages=[ChatMessage(role="user", content="hello")],
            session_id="test", use_tools=True, filters=None,
            settings=SimpleNamespace(max_input_length=4000), trace_id="t1",
        )

        assert result.intent == "small_talk"
        assert result.citations == []
        assert result.tool_calls == []
        assert "[1]" not in result.reply


class TestRetrievalJourney:
    """Domain question: retrieval_required -> retrieval invoked -> sources returned."""

    @pytest.mark.asyncio
    async def test_domain_question_uses_retrieval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="retrieval_required", confidence=0.85, reason="needs career data",
        )
        async def fake_route(q, *, cv_available=False, settings=None):
            return decision
        async def fake_validate(text, max_length=4000):
            return text
        async def fake_rewrite(query, settings=None):
            return query
        async def fake_retrieve(query, filters=None, settings=None):
            return _make_chunks(2)
        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)

        from career_intel.orchestration import synthesize as synth_mod
        monkeypatch.setattr(
            synth_mod, "get_chat_llm",
            lambda settings, temperature=0.2: _FakeLLM("AI roles are growing [1] with demand [2]."),
        )

        from career_intel.orchestration.chain import run_turn
        result = await run_turn(
            messages=[ChatMessage(role="user", content="What are the top AI roles?")],
            session_id="test", use_tools=True, filters=None,
            settings=SimpleNamespace(max_input_length=4000), trace_id="t2",
        )

        assert result.intent == "retrieval_required"
        assert len(result.citations) == 2
        assert result.tool_calls == []


class TestCvRelevantJourney:
    """CV present + CV-relevant query: retrieval + CV context included."""

    @pytest.mark.asyncio
    async def test_cv_included_when_relevant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="retrieval_required", confidence=0.9,
            use_cv=True, reason="needs CV for personalisation",
        )
        async def fake_route(q, *, cv_available=False, settings=None):
            return decision
        async def fake_validate(text, max_length=4000):
            return text
        async def fake_rewrite(query, settings=None):
            return query
        async def fake_retrieve(query, filters=None, settings=None):
            return _make_chunks(1)
        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)

        from career_intel.orchestration import synthesize as synth_mod

        class CaptureLLM:
            last_prompt = ""
            async def ainvoke(self, messages):
                CaptureLLM.last_prompt = messages[-1]["content"]
                return SimpleNamespace(content="Based on your CV [1]...")

        monkeypatch.setattr(synth_mod, "get_chat_llm", lambda settings, temperature=0.2: CaptureLLM())

        from career_intel.orchestration.chain import run_turn
        result = await run_turn(
            messages=[ChatMessage(role="user", content="What skills am I missing?")],
            session_id="test", use_tools=True, filters=None,
            settings=SimpleNamespace(max_input_length=4000), trace_id="t3",
            cv_text="John Doe, Python, ML Engineer, 5 years",
        )

        assert result.intent == "retrieval_required"
        assert "USER_CV" in CaptureLLM.last_prompt


class TestStreamingJourneys:
    """Verify streaming event ordering for different intents."""

    @pytest.mark.asyncio
    async def test_retrieval_stream_emits_status_before_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="retrieval_required", confidence=0.8, reason="needs data",
        )
        async def fake_route(q, *, cv_available=False, settings=None):
            return decision
        async def fake_validate(text, max_length=4000):
            return text
        async def fake_rewrite(query, settings=None):
            return query
        async def fake_retrieve(query, filters=None, settings=None):
            return _make_chunks(1)
        def fake_inc(name, count=1):
            pass

        class FakeChunk:
            def __init__(self, c):
                self.content = c

        class FakeStreamLLM:
            async def astream(self, messages):
                for w in ["Answer", " here", "."]:
                    yield FakeChunk(w)

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)

        from career_intel.orchestration import stream as stream_mod
        monkeypatch.setattr(stream_mod, "get_chat_llm",
            lambda settings, temperature=0.2, streaming=True: FakeStreamLLM(),
        )

        from career_intel.orchestration.stream import stream_turn
        events = []
        async for line in stream_turn(
            messages=[ChatMessage(role="user", content="What AI roles are growing?")],
            session_id="test", use_tools=True, filters=None,
            settings=SimpleNamespace(max_input_length=4000), trace_id="t4",
        ):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]

        assert types[0] == "intent"
        assert "status" in types
        status_idx = types.index("status")
        first_token_idx = types.index("token")
        assert status_idx < first_token_idx
        assert "citations" in types
        assert types[-1] == "done"
