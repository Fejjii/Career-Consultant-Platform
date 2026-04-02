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
    reply, citations = await synthesize_answer(
        query="What are the 3 growing roles (2025-2030)?",
        rewritten_query="What are the three growing roles 2025 to 2030",
        chunks=chunks,
        tool_results=[],
        settings=SimpleNamespace(),
    )

    assert "Top roles" in reply
    assert len(citations) == 2


@pytest.mark.asyncio
async def test_unsupported_query_abstains_with_empty_citations() -> None:
    reply, citations = await synthesize_answer(
        query="What will be the weather on Mars in 2040?",
        rewritten_query="What will be the weather on Mars in 2040?",
        chunks=[_chunk(chunk_id="c1", score=0.12, text="Unrelated labor market text.")],
        tool_results=[ToolCallResult(tool_name="noop", inputs={}, output={}, success=True)],
        settings=SimpleNamespace(),
    )

    assert "don't have enough evidence" in reply
    assert citations == []
