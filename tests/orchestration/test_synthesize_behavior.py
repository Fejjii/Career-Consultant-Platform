"""Behavior tests for grounded answering vs abstention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from career_intel.orchestration.synthesize import synthesize_answer
from career_intel.schemas.api import ToolCallResult
from career_intel.schemas.domain import ChunkMetadata, RetrievedChunk


def _chunk(*, chunk_id: str, score: float, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata=ChunkMetadata(
            source_id=f"src-{chunk_id}",
            source_type="md",
            title="sample_career_data",
            chunk_index=0,
            parent_doc_id="doc-1",
        ),
    )


@pytest.mark.asyncio
async def test_grounded_query_does_not_abstain_and_returns_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        async def ainvoke(self, _: list[dict[str, str]]) -> SimpleNamespace:
            return SimpleNamespace(content="Top roles are AI Engineer [1], Data Analyst [2].")

    from career_intel.orchestration import synthesize as synth_module

    monkeypatch.setattr(synth_module, "get_chat_llm", lambda settings, temperature=0.2: FakeLLM())

    chunks = [
        _chunk(chunk_id="c1", score=0.62, text="AI Engineer is growing quickly."),
        _chunk(chunk_id="c2", score=0.51, text="Data Analyst demand increases through 2030."),
    ]
    reply, citations, _usage = await synthesize_answer(
        query="What are the 3 growing roles (2025-2030)?",
        rewritten_query="What are the three growing roles 2025 to 2030",
        chunks=chunks,
        tool_results=[],
        answer_source="rag",
        settings=SimpleNamespace(rag_weak_evidence_threshold=0.30),
    )

    assert "Top roles" in reply
    assert len(citations) == 2


@pytest.mark.asyncio
async def test_unsupported_query_uses_fallback_without_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        async def ainvoke(self, _: list[dict[str, str]]) -> SimpleNamespace:
            return SimpleNamespace(content="This answer is based on general reasoning, not retrieved internal evidence.")

    from career_intel.orchestration import synthesize as synth_module

    monkeypatch.setattr(synth_module, "get_chat_llm", lambda settings, temperature=0.2: FakeLLM())

    reply, citations, _usage = await synthesize_answer(
        query="What will be the weather on Mars in 2040?",
        rewritten_query="What will be the weather on Mars in 2040?",
        chunks=[_chunk(chunk_id="c1", score=0.12, text="Unrelated labor market text.")],
        tool_results=[],
        answer_source="llm_fallback",
        settings=SimpleNamespace(rag_weak_evidence_threshold=0.30),
    )

    assert "general reasoning" in reply
    assert citations == []


@pytest.mark.asyncio
async def test_tool_only_synthesis_without_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        async def ainvoke(self, _: list[dict[str, str]]) -> SimpleNamespace:
            return SimpleNamespace(content="Here is your structured plan summary.")

    from career_intel.orchestration import synthesize as synth_module

    monkeypatch.setattr(synth_module, "get_chat_llm", lambda settings, temperature=0.2: FakeLLM())

    reply, citations, _usage = await synthesize_answer(
        query="Build my learning plan",
        rewritten_query="Build my learning plan",
        chunks=[],
        tool_results=[
            ToolCallResult(
                tool_name="learning_plan",
                inputs={"goal_role": "Data Engineer"},
                output={"milestones": [{"week": 1, "topic": "SQL"}]},
                success=True,
            ),
        ],
        answer_source="tool",
        settings=SimpleNamespace(rag_weak_evidence_threshold=0.30),
    )

    assert "plan" in reply.lower()
    assert citations == []


@pytest.mark.asyncio
async def test_rag_regenerates_when_first_draft_has_no_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _: list[dict[str, str]]) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content="AI Engineer is growing quickly.")
            return SimpleNamespace(content="AI Engineer is growing quickly [1].")

    from career_intel.orchestration import synthesize as synth_module

    fake_llm = FakeLLM()
    monkeypatch.setattr(synth_module, "get_chat_llm", lambda settings, temperature=0.2: fake_llm)

    reply, citations, _usage = await synthesize_answer(
        query="What are the 3 growing roles (2025-2030)?",
        rewritten_query="What are the three growing roles 2025 to 2030",
        chunks=[_chunk(chunk_id="c1", score=0.62, text="AI Engineer is growing quickly.")],
        tool_results=[],
        answer_source="rag",
        settings=SimpleNamespace(rag_weak_evidence_threshold=0.30),
    )

    assert fake_llm.calls == 2
    assert reply.endswith("[1].")
    assert len(citations) == 1


@pytest.mark.asyncio
async def test_rag_synthesis_includes_length_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, str]] = []

    class FakeLLM:
        async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
            captured.extend(messages)
            return SimpleNamespace(content="Role X grows [1].")

    from career_intel.orchestration import synthesize as synth_module

    monkeypatch.setattr(synth_module, "get_chat_llm", lambda settings, temperature=0.2: FakeLLM())

    chunks = [_chunk(chunk_id="c1", score=0.62, text="Role X grows.")]
    await synthesize_answer(
        query="Trends?",
        rewritten_query="Trends",
        chunks=chunks,
        tool_results=[],
        answer_source="rag",
        settings=SimpleNamespace(rag_weak_evidence_threshold=0.30),
        answer_length="detailed",
    )

    system = next(m["content"] for m in captured if m["role"] == "system")
    assert "Answer length — detailed" in system


@pytest.mark.asyncio
async def test_balanced_synthesis_includes_length_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, str]] = []

    class FakeLLM:
        async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
            captured.extend(messages)
            return SimpleNamespace(content="Role X grows [1].")

    from career_intel.orchestration import synthesize as synth_module

    monkeypatch.setattr(synth_module, "get_chat_llm", lambda settings, temperature=0.2: FakeLLM())

    chunks = [_chunk(chunk_id="c1", score=0.62, text="Role X grows.")]
    await synthesize_answer(
        query="Trends?",
        rewritten_query="Trends",
        chunks=chunks,
        tool_results=[],
        answer_source="rag",
        settings=SimpleNamespace(rag_weak_evidence_threshold=0.30),
        answer_length="balanced",
    )

    system = next(m["content"] for m in captured if m["role"] == "system")
    assert "Answer length — balanced" in system
