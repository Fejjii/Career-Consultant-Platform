"""Retriever consistency tests for query variants and filter fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from career_intel.rag import retriever


def _point(point_id: str, score: float, text: str) -> SimpleNamespace:
    payload = {
        "text": text,
        "source_id": f"src-{point_id}",
        "source_type": "md",
        "title": "sample_career_data",
        "parent_doc_id": "doc-1",
        "chunk_index": 0,
    }
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def test_normalize_query_keeps_semantics_for_date_ranges() -> None:
    q1 = "What are the 3 growing roles (2025-2030)?"
    q2 = "What are the three growing roles 2025 to 2030"
    assert retriever.normalize_query(q1) == "What are the 3 growing roles 2025 to 2030"
    assert retriever.normalize_query(q2) == "What are the three growing roles 2025 to 2030"


@pytest.mark.asyncio
async def test_retrieve_chunks_falls_back_when_filters_return_no_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object] | None] = []

    def fake_get_embeddings(texts: list[str], settings: object | None = None) -> list[list[float]]:
        return [[float(i + 1)] for i, _ in enumerate(texts)]

    def fake_search_vectors(
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        client: object | None = None,
    ) -> list[SimpleNamespace]:
        calls.append(filters)
        if filters:
            return []
        return [_point("a", 0.62, "Top 10 Growing Roles from the sample data.")]

    monkeypatch.setattr(retriever, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retriever, "search_vectors", fake_search_vectors)

    chunks = await retriever.retrieve_chunks(
        query="What are the 3 growing roles (2025-2030)?",
        filters={"publish_year": 2025},
        settings=SimpleNamespace(),
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "a"
    assert calls[0] == {"publish_year": 2025}
    assert calls[-1] is None


@pytest.mark.asyncio
async def test_retrieve_chunks_merges_multi_query_results_by_best_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_embeddings(texts: list[str], settings: object | None = None) -> list[list[float]]:
        # Force two retrieval passes: rewritten + normalized
        return [[1.0], [2.0]]

    def fake_search_vectors(
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        client: object | None = None,
    ) -> list[SimpleNamespace]:
        if query_vector == [1.0]:
            return [
                _point("x", 0.44, "Role growth baseline result."),
                _point("y", 0.58, "Another relevant chunk."),
            ]
        return [
            _point("x", 0.71, "Role growth baseline result."),
        ]

    monkeypatch.setattr(retriever, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retriever, "search_vectors", fake_search_vectors)

    chunks = await retriever.retrieve_chunks(
        query="What are the 3 growing roles (2025-2030)?",
        filters=None,
        settings=SimpleNamespace(),
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "x"
    assert chunks[0].score == pytest.approx(0.71)
    assert {c.chunk_id for c in chunks} == {"x", "y"}
