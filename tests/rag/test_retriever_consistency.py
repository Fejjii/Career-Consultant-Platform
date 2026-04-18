"""Retriever consistency tests for query variants and filter fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from career_intel.rag import retriever


def _settings(**kwargs: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "rag_initial_top_k": 15,
        "rag_top_k": 12,
        "rag_enable_reranking": True,
        "rag_similarity_threshold": 0.55,
        "rag_weak_evidence_threshold": 0.30,
        "rag_strong_evidence_threshold": 0.60,
        "rag_rerank_coherence_threshold": 0.48,
        "rag_force_min_chunks": 3,
        "environment": "production",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _point(point_id: str, score: float, text: str, *, source: str = "md") -> SimpleNamespace:
    payload = {
        "text": text,
        "source_id": f"src-{point_id}",
        "source": source,
        "source_name": source,
        "source_type": source,
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
        settings=_settings(),
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
        settings=_settings(),
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "x"
    assert chunks[0].score == pytest.approx(0.71)
    assert {c.chunk_id for c in chunks} == {"x", "y"}


@pytest.mark.asyncio
async def test_retrieve_chunks_skips_when_best_score_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_embeddings(texts: list[str], settings: object | None = None) -> list[list[float]]:
        return [[1.0]]

    def fake_search_vectors(
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        client: object | None = None,
    ) -> list[SimpleNamespace]:
        return [_point("low", 0.12, "Barely related text.")]

    monkeypatch.setattr(retriever, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retriever, "search_vectors", fake_search_vectors)

    chunks = await retriever.retrieve_chunks(
        query="obscure query xyz",
        filters=None,
        settings=_settings(rag_similarity_threshold=0.30),
    )
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_chunks_can_disable_reranking(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_embeddings(texts: list[str], settings: object | None = None) -> list[list[float]]:
        return [[1.0]]

    def fake_search_vectors(
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        client: object | None = None,
    ) -> list[SimpleNamespace]:
        return [
            _point("high", 0.74, "General labor market text."),
            _point("mid", 0.63, "Somewhat relevant data engineering text."),
            _point("low", 0.57, "Lightly related skills text."),
        ]

    monkeypatch.setattr(retriever, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retriever, "search_vectors", fake_search_vectors)

    chunks = await retriever.retrieve_chunks(
        query="Which occupations in ESCO are closely tied to Python and SQL skills?",
        filters=None,
        settings=_settings(rag_top_k=3, rag_enable_reranking=False),
    )

    assert [chunk.chunk_id for chunk in chunks] == ["high", "mid", "low"]
    assert [chunk.rerank_score for chunk in chunks] == [0.74, 0.63, 0.57]


def test_assess_evidence_strength_uses_similarity_bands() -> None:
    chunks = [
        _point("a", 0.61, "Strongly relevant."),
        _point("b", 0.56, "Also relevant."),
        _point("c", 0.49, "Borderline."),
    ]
    retrieved = [
        retriever.RetrievedChunk(
            chunk_id=str(point.id),
            text=point.payload["text"],
            score=point.score,
            metadata=retriever.ChunkMetadata(
                source_id=point.payload["source_id"],
                source_type=point.payload["source_type"],
                title=point.payload["title"],
                chunk_index=0,
                parent_doc_id=point.payload["parent_doc_id"],
            ),
        )
        for point in chunks
    ]

    strength, reason = retriever.assess_evidence_strength(retrieved, settings=_settings())

    assert strength == "strong"
    assert "strong_similarity_threshold" in reason


def test_should_force_rag_when_enough_reasonable_chunks() -> None:
    retrieved = [
        retriever.RetrievedChunk(
            chunk_id=f"c{i}",
            text=f"Chunk {i}",
            score=0.56 + (i * 0.01),
            metadata=retriever.ChunkMetadata(
                source_id=f"src-{i}",
                source_type="esco",
                title="sample_career_data",
                chunk_index=i,
                parent_doc_id="doc-1",
                source="esco",
            ),
        )
        for i in range(3)
    ]

    forced, reason = retriever.should_force_rag(retrieved, settings=_settings())

    assert forced is True
    assert "chunks_at_or_above_similarity_threshold" in reason


def test_should_not_force_rag_for_only_weak_scores() -> None:
    retrieved = [
        retriever.RetrievedChunk(
            chunk_id=f"c{i}",
            text=f"Chunk {i}",
            score=0.35 + (i * 0.01),
            metadata=retriever.ChunkMetadata(
                source_id=f"src-{i}",
                source_type="esco",
                title="sample_career_data",
                chunk_index=i,
                parent_doc_id="doc-1",
                source="esco",
            ),
        )
        for i in range(3)
    ]

    forced, reason = retriever.should_force_rag(retrieved, settings=_settings())

    assert forced is False
    assert reason == "insufficient_reasonable_chunk_count"


def test_assess_evidence_strength_rejects_mixed_sources_for_named_source_queries() -> None:
    retrieved = [
        retriever.RetrievedChunk(
            chunk_id="wef-1",
            text="WEF report chunk",
            score=0.62,
            metadata=retriever.ChunkMetadata(
                source_id="src-wef",
                source_type="wef",
                title="WEF report",
                chunk_index=0,
                parent_doc_id="doc-wef",
                source="wef",
            ),
        ),
        retriever.RetrievedChunk(
            chunk_id="esco-1",
            text="ESCO skill chunk",
            score=0.61,
            metadata=retriever.ChunkMetadata(
                source_id="src-esco",
                source_type="esco",
                title="ESCO skills",
                chunk_index=1,
                parent_doc_id="doc-esco",
                source="esco",
            ),
        ),
    ]

    strength, reason = retriever.assess_evidence_strength(
        retrieved,
        settings=_settings(),
        detected_source="wef",
    )

    assert strength == "none"
    assert reason.startswith("source_inconsistent_")


def test_assess_source_consistency_requires_named_source_match() -> None:
    retrieved = [
        retriever.RetrievedChunk(
            chunk_id="c1",
            text="Chunk 1",
            score=0.60,
            metadata=retriever.ChunkMetadata(
                source_id="src-1",
                source_type="wef",
                title="WEF report",
                chunk_index=0,
                parent_doc_id="doc-1",
                source="wef",
            ),
        ),
        retriever.RetrievedChunk(
            chunk_id="c2",
            text="Chunk 2",
            score=0.58,
            metadata=retriever.ChunkMetadata(
                source_id="src-2",
                source_type="wef",
                title="WEF report",
                chunk_index=1,
                parent_doc_id="doc-1",
                source="wef",
            ),
        ),
    ]

    consistent, reason, sources = retriever.assess_source_consistency(
        retrieved,
        detected_source="wef",
    )

    assert consistent is True
    assert reason == "all_chunks_match_wef"
    assert sources == ["wef"]


def test_assess_evidence_strength_requires_rerank_coherence() -> None:
    retrieved = [
        retriever.RetrievedChunk(
            chunk_id="c1",
            text="Chunk 1",
            score=0.57,
            rerank_score=0.47,
            metadata=retriever.ChunkMetadata(
                source_id="src-1",
                source_type="esco",
                title="ESCO skills",
                chunk_index=0,
                parent_doc_id="doc-1",
                source="esco",
            ),
        ),
        retriever.RetrievedChunk(
            chunk_id="c2",
            text="Chunk 2",
            score=0.56,
            rerank_score=0.46,
            metadata=retriever.ChunkMetadata(
                source_id="src-2",
                source_type="esco",
                title="ESCO skills",
                chunk_index=1,
                parent_doc_id="doc-1",
                source="esco",
            ),
        ),
    ]

    strength, reason = retriever.assess_evidence_strength(
        retrieved,
        settings=_settings(),
        detected_source="esco",
    )

    assert strength == "none"
    assert reason == "similarity_without_rerank_coherence"


def test_detect_query_source_prefers_wef_for_report_queries() -> None:
    assert retriever.detect_query_source(
        "Which skills are growing fastest in recent WEF future of jobs reports?"
    ) == "wef"


def test_detect_query_source_returns_esco_for_taxonomy_queries() -> None:
    assert retriever.detect_query_source(
        "Which occupations in ESCO are closely tied to Python and SQL skills?"
    ) == "esco"


def test_build_query_profile_extracts_compound_esco_concepts() -> None:
    profile = retriever.build_query_profile(
        "Which occupations in ESCO are closely tied to Python and SQL skills?",
        detected_source="esco",
    )

    assert profile.esco_relation_query is True
    assert "python and sql" in profile.salient_concepts
    assert "python" in profile.salient_concepts
    assert "sql" in profile.salient_concepts


def test_build_query_profile_marks_taxonomy_queries() -> None:
    profile = retriever.build_query_profile(
        "How does ISCO grouping relate to ESCO occupations?",
        detected_source="esco",
    )

    assert profile.esco_relation_query is True
    assert profile.taxonomy_query is True


def test_build_query_profile_detects_related_to_pattern_for_esco() -> None:
    profile = retriever.build_query_profile(
        "What does ESCO define as cloud-related skills or competences?",
        detected_source="esco",
    )

    assert profile.esco_relation_query is True
    assert profile.classification_reason == "pattern_match:x-related skills"


def test_build_query_profile_detects_skills_in_pattern_for_esco() -> None:
    profile = retriever.build_query_profile(
        "Which skills in ESCO are linked to data engineering roles?",
        detected_source="esco",
    )

    assert profile.esco_relation_query is True
    assert profile.classification_reason.startswith("pattern_match:")


def test_merge_query_profiles_preserves_original_esco_relation_signals() -> None:
    original = retriever.build_query_profile(
        "How does ISCO grouping relate to ESCO occupations?",
        detected_source="esco",
    )
    rewritten = retriever.build_query_profile(
        "Explain the grouping relationship.",
        detected_source=None,
    )

    merged = retriever.merge_query_profiles(original, rewritten)

    assert merged.detected_source == "esco"
    assert merged.esco_relation_query is True
    assert merged.taxonomy_query is True
    assert "isco" in merged.salient_concepts


@pytest.mark.asyncio
async def test_retrieve_chunks_applies_detected_source_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object] | None] = []

    def fake_get_embeddings(texts: list[str], settings: object | None = None) -> list[list[float]]:
        return [[1.0]]

    def fake_search_vectors(
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        client: object | None = None,
    ) -> list[SimpleNamespace]:
        calls.append(filters)
        return [_point("w1", 0.62, "WEF report chunk.")]

    monkeypatch.setattr(retriever, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retriever, "search_vectors", fake_search_vectors)

    chunks = await retriever.retrieve_chunks(
        query="Which skills are growing fastest in recent WEF future of jobs reports?",
        filters=None,
        settings=_settings(),
    )

    assert len(chunks) == 1
    assert calls == [{"source": "wef"}]


@pytest.mark.asyncio
async def test_retrieve_chunks_ignores_count_diagnostics_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object] | None] = []

    def fake_get_embeddings(texts: list[str], settings: object | None = None) -> list[list[float]]:
        return [[1.0]]

    def fake_search_vectors(
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        client: object | None = None,
    ) -> list[SimpleNamespace]:
        calls.append(filters)
        return [_point("e1", 0.66, "ESCO relation bundle for data engineer.", source="esco")]

    monkeypatch.setattr(retriever, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retriever, "search_vectors", fake_search_vectors)
    monkeypatch.setattr(retriever, "count_vectors", lambda filters: (_ for _ in ()).throw(TimeoutError("slow count")))

    chunks = await retriever.retrieve_chunks(
        query="Which occupations in ESCO are closely tied to Python and SQL skills?",
        filters=None,
        settings=_settings(),
    )

    assert len(chunks) == 1
    assert calls == [{"source": "esco"}]


@pytest.mark.asyncio
async def test_retrieve_chunks_falls_back_to_full_corpus_when_source_filter_returns_no_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object] | None] = []

    def fake_get_embeddings(texts: list[str], settings: object | None = None) -> list[list[float]]:
        return [[1.0]]

    def fake_search_vectors(
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        client: object | None = None,
    ) -> list[SimpleNamespace]:
        calls.append(filters)
        if filters == {"source": "wef"}:
            return []
        return [_point("x", 0.62, "Fallback chunk.")]

    monkeypatch.setattr(retriever, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(retriever, "search_vectors", fake_search_vectors)

    chunks = await retriever.retrieve_chunks(
        query="Which skills are growing fastest in recent WEF future of jobs reports?",
        filters=None,
        settings=_settings(),
    )

    assert len(chunks) == 1
    assert calls == [{"source": "wef"}, None]
