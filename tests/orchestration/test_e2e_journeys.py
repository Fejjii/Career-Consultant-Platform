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
| Domain factual question | domain_specific    | Yes       | No  | Yes     | Normal    |
| CV upload + CV question | domain_specific    | Yes       | Yes | Yes     | Normal    |
| CV present, irrelevant  | domain_specific    | Yes       | No  | Yes     | Normal    |
| New session reset       | (N/A)              | N/A       | N/A | N/A     | N/A       |
| Backend unavailable     | (N/A)              | N/A       | N/A | N/A     | N/A       |
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from career_intel.rag.query_preprocessor import RetrievalQueryNormalization
from career_intel.schemas.api import ChatMessage
from career_intel.schemas.api import Citation
from career_intel.schemas.domain import ChunkMetadata, RetrievedChunk, RouterDecision


def _settings(**kwargs: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "max_input_length": 4000,
        "rag_similarity_threshold": 0.55,
        "rag_weak_evidence_threshold": 0.30,
        "rag_strong_evidence_threshold": 0.60,
        "rag_rerank_coherence_threshold": 0.48,
        "rag_force_min_chunks": 3,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


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
            return decision, None
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
            settings=_settings(),
            trace_id="t1",
        )

        assert result.intent == "small_talk"
        assert result.citations == []
        assert result.tool_calls == []
        assert "[1]" not in result.reply


class TestRetrievalJourney:
    """Domain question: domain_specific -> retrieval invoked -> sources returned."""

    @pytest.mark.asyncio
    async def test_domain_question_uses_retrieval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="domain_specific", confidence=0.85, reason="needs career data",
        )
        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None
        async def fake_validate(text, max_length=4000):
            return text
        async def fake_rewrite(query, settings=None):
            return query
        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
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
            settings=_settings(),
            trace_id="t2",
        )

        assert result.intent == "domain_specific"
        assert len(result.citations) == 2
        assert result.tool_calls == []
        assert result.answer_source == "rag"

    @pytest.mark.asyncio
    async def test_non_english_domain_query_translates_before_retrieval(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        decision = RouterDecision(
            intent="domain_specific",
            confidence=0.87,
            reason="named ESCO query requires grounding",
        )
        translated_query = "What essential skills are linked to the ESCO occupation data scientist?"
        rewritten_query = "ESCO occupation skill relation data scientist essential skills"
        captured: dict[str, object] = {}

        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None

        async def fake_validate(text, max_length=4000):
            return text

        async def fake_normalize(query, *, settings=None):
            assert query == "Quelles competences essentielles sont liees au data scientist dans ESCO ?"
            return RetrievalQueryNormalization(
                detected_language="fr",
                retrieval_query=translated_query,
                translated_query=translated_query,
                translation_applied=True,
            )

        async def fake_rewrite(query, settings=None):
            captured["rewrite_query"] = query
            return rewritten_query

        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
            captured["retrieve_query"] = query
            return [
                RetrievedChunk(
                    chunk_id="esco-1",
                    text="Data scientist essential skills include statistics and machine learning.",
                    metadata=ChunkMetadata(
                        source_id="src-esco-1",
                        source_type="esco",
                        source="esco",
                        title="ESCO relation detail",
                        chunk_index=0,
                        parent_doc_id="doc-esco-1",
                        esco_doc_type="relation_detail",
                    ),
                    score=0.72,
                    rerank_score=0.63,
                )
            ]

        async def fake_synthesize_answer(**kwargs):
            captured["synthesis_query"] = kwargs["query"]
            captured["synthesis_rewritten_query"] = kwargs["rewritten_query"]
            return (
                "Les competences essentielles incluent l'analyse de donnees [1].",
                [
                    Citation(
                        id=1,
                        source_id="s1",
                        title="Report 1",
                        section=None,
                        page_or_loc=None,
                        publish_year=None,
                        excerpt="Evidence chunk 1 about career trends.",
                        uri=None,
                    )
                ],
                None,
            )

        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr(
            "career_intel.rag.query_preprocessor.normalize_query_for_retrieval",
            fake_normalize,
        )
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)
        monkeypatch.setattr(
            "career_intel.orchestration.synthesize.synthesize_answer",
            fake_synthesize_answer,
        )

        from career_intel.orchestration.chain import run_turn

        result = await run_turn(
            messages=[
                ChatMessage(
                    role="user",
                    content="Quelles competences essentielles sont liees au data scientist dans ESCO ?",
                )
            ],
            session_id="test-fr-rag",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="t-fr-rag",
        )

        assert captured["rewrite_query"] == translated_query
        assert captured["retrieve_query"] == rewritten_query
        assert captured["synthesis_query"] == (
            "Quelles competences essentielles sont liees au data scientist dans ESCO ?"
        )
        assert captured["synthesis_rewritten_query"] == rewritten_query
        assert result.answer_source == "rag"


class TestCvRelevantJourney:
    """CV present + CV-relevant query: retrieval + CV context included."""

    @pytest.mark.asyncio
    async def test_cv_included_when_relevant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="domain_specific", confidence=0.9,
            use_cv=True, reason="needs CV for personalisation",
        )
        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None
        async def fake_validate(text, max_length=4000):
            return text
        async def fake_rewrite(query, settings=None):
            return query
        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
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
            settings=_settings(),
            trace_id="t3",
            cv_text="John Doe, Python, ML Engineer, 5 years",
        )

        assert result.intent == "domain_specific"
        assert "USER_CV" in CaptureLLM.last_prompt


class TestFallbackJourney:
    """Very weak retrieval evidence should still fall back when no usable chunks exist."""

    @pytest.mark.asyncio
    async def test_domain_specific_no_usable_evidence_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="domain_specific",
            confidence=0.75,
            reason="needs evidence but retrieved support is weak",
        )

        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None

        async def fake_validate(text, max_length=4000):
            return text

        async def fake_rewrite(query, settings=None):
            return query

        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
            return [
                RetrievedChunk(
                    chunk_id="weak",
                    text="Loose mention of work trends.",
                    metadata=ChunkMetadata(
                        source_id="src-weak",
                        source_type="report",
                        title="Sample report",
                        chunk_index=0,
                        parent_doc_id="doc-weak",
                    ),
                    score=0.12,
                    rerank_score=0.41,
                ),
            ]

        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)

        from career_intel.orchestration import synthesize as synth_mod
        monkeypatch.setattr(
            synth_mod,
            "get_chat_llm",
            lambda settings, temperature=0.2: _FakeLLM(
                "This answer is based on general reasoning, not retrieved internal evidence.",
            ),
        )

        from career_intel.orchestration.chain import run_turn
        result = await run_turn(
            messages=[ChatMessage(role="user", content="What will work on Mars look like?")],
            session_id="test",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="t-fallback",
        )

        assert result.answer_source == "llm_fallback"
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_esco_relation_query_with_none_evidence_does_not_force_rag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        decision = RouterDecision(
            intent="domain_specific",
            confidence=0.88,
            reason="named ESCO taxonomy question requires evidence",
        )

        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None

        async def fake_validate(text, max_length=4000):
            return text

        async def fake_rewrite(query, settings=None):
            return "Explain this grouping relationship."

        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
            return [
                RetrievedChunk(
                    chunk_id="esco-taxonomy",
                    text="ESCO to ISCO occupation mapping for data engineer.",
                    metadata=ChunkMetadata(
                        source_id="src-esco-taxonomy",
                        source_type="esco",
                        source="esco",
                        title="ESCO mapping",
                        document_title="ESCO occupation to ISCO mapping",
                        chunk_index=0,
                        parent_doc_id="doc-esco-taxonomy",
                        esco_doc_type="taxonomy_mapping",
                    ),
                    score=0.58,
                    rerank_score=0.40,
                ),
            ]

        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)
        monkeypatch.setattr(
            "career_intel.rag.retriever.build_query_profile",
            lambda query, detected_source=None: (
                SimpleNamespace(
                    esco_relation_query=True,
                    taxonomy_query=True,
                    essential_optional_query=False,
                    salient_concepts=("isco", "occupations"),
                    detected_source="esco",
                    classification_reason="no_relation_signal",
                )
                if "ESCO" in query or "ISCO" in query
                else SimpleNamespace(
                    esco_relation_query=False,
                    taxonomy_query=False,
                    essential_optional_query=False,
                    salient_concepts=(),
                    detected_source=detected_source,
                    classification_reason="no_relation_signal",
                )
            ),
        )
        monkeypatch.setattr(
            "career_intel.rag.retriever.assess_evidence_strength",
            lambda chunks, settings=None, detected_source=None: (
                "none",
                "similarity_without_rerank_coherence",
            ),
        )
        monkeypatch.setattr(
            "career_intel.rag.retriever.should_force_rag",
            lambda chunks, settings=None, detected_source=None: (
                True,
                "3_chunks_at_or_above_similarity_threshold",
            ),
        )

        from career_intel.orchestration import synthesize as synth_mod

        monkeypatch.setattr(
            synth_mod,
            "get_chat_llm",
            lambda settings, temperature=0.2: _FakeLLM(
                "This answer is based on general reasoning, not retrieved ESCO evidence.",
            ),
        )

        from career_intel.orchestration.chain import run_turn

        result = await run_turn(
            messages=[ChatMessage(role="user", content="How does ISCO grouping relate to ESCO occupations?")],
            session_id="test-esco-fallback",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="t-esco-fallback",
        )

        assert result.answer_source == "llm_fallback"
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_domain_specific_none_evidence_never_promotes_to_rag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        decision = RouterDecision(
            intent="domain_specific",
            confidence=0.84,
            reason="requires evidence grounding",
        )

        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None

        async def fake_validate(text, max_length=4000):
            return text

        async def fake_rewrite(query, settings=None):
            return query

        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
            return _make_chunks(1)

        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)
        monkeypatch.setattr(
            "career_intel.rag.retriever.assess_evidence_strength",
            lambda chunks, settings=None, detected_source=None: ("none", "no_chunks_at_or_above_similarity_threshold"),
        )
        monkeypatch.setattr(
            "career_intel.rag.retriever.should_force_rag",
            lambda chunks, settings=None, detected_source=None: (True, "forced_by_chunk_count"),
        )

        from career_intel.orchestration import synthesize as synth_mod

        monkeypatch.setattr(
            synth_mod,
            "get_chat_llm",
            lambda settings, temperature=0.2: _FakeLLM(
                "This answer is based on general reasoning, not retrieved internal evidence.",
            ),
        )

        from career_intel.orchestration.chain import run_turn

        result = await run_turn(
            messages=[ChatMessage(role="user", content="What data roles are rising this year?")],
            session_id="test-none-evidence-fallback",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="t-none-evidence-fallback",
        )

        assert result.answer_source == "llm_fallback"
        assert result.citations == []


class TestStreamingJourneys:
    """Verify streaming event ordering for different intents."""

    @pytest.mark.asyncio
    async def test_retrieval_stream_emits_status_before_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decision = RouterDecision(
            intent="domain_specific", confidence=0.8, reason="needs data",
        )
        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None
        async def fake_validate(text, max_length=4000):
            return text
        async def fake_rewrite(query, settings=None):
            return query
        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
            return _make_chunks(1)
        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)

        from career_intel.orchestration import stream as stream_mod

        async def fake_synthesize_answer(**kwargs):
            return (
                "Answer here [1].",
                [
                    Citation(
                        id=1,
                        source_id="s1",
                        title="Report 1",
                        section=None,
                        page_or_loc=None,
                        publish_year=None,
                        excerpt="Evidence chunk 1 about career trends.",
                        uri=None,
                    )
                ],
                None,
            )

        monkeypatch.setattr(stream_mod, "synthesize_answer", fake_synthesize_answer)

        from career_intel.orchestration.stream import stream_turn
        events = []
        async for line in stream_turn(
            messages=[ChatMessage(role="user", content="What AI roles are growing?")],
            session_id="test", use_tools=True, filters=None,
            settings=_settings(),
            trace_id="t4",
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

    @pytest.mark.asyncio
    async def test_stream_non_english_domain_query_translates_before_retrieval(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        decision = RouterDecision(
            intent="domain_specific",
            confidence=0.84,
            reason="named ESCO query requires retrieval",
        )
        translated_query = "What essential skills are linked to the ESCO occupation data scientist?"
        rewritten_query = "ESCO occupation skill relation data scientist essential skills"
        captured: dict[str, object] = {}

        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None

        async def fake_validate(text, max_length=4000):
            return text

        async def fake_normalize(query, *, settings=None):
            return RetrievalQueryNormalization(
                detected_language="fr",
                retrieval_query=translated_query,
                translated_query=translated_query,
                translation_applied=True,
            )

        async def fake_rewrite(query, settings=None):
            captured["rewrite_query"] = query
            return rewritten_query

        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
            captured["retrieve_query"] = query
            return [
                RetrievedChunk(
                    chunk_id="esco-1",
                    text="Data scientist essential skills include statistics and machine learning.",
                    metadata=ChunkMetadata(
                        source_id="src-esco-1",
                        source_type="esco",
                        source="esco",
                        title="ESCO relation detail",
                        chunk_index=0,
                        parent_doc_id="doc-esco-1",
                        esco_doc_type="relation_detail",
                    ),
                    score=0.72,
                    rerank_score=0.63,
                )
            ]

        async def fake_synthesize_answer(**kwargs):
            captured["synthesis_query"] = kwargs["query"]
            captured["synthesis_rewritten_query"] = kwargs["rewritten_query"]
            return (
                "Les competences essentielles incluent l'analyse de donnees [1].",
                [
                    Citation(
                        id=1,
                        source_id="s1",
                        title="Report 1",
                        section=None,
                        page_or_loc=None,
                        publish_year=None,
                        excerpt="Evidence chunk 1 about career trends.",
                        uri=None,
                    )
                ],
                None,
            )

        def fake_inc(name, count=1):
            pass

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr(
            "career_intel.rag.query_preprocessor.normalize_query_for_retrieval",
            fake_normalize,
        )
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)

        from career_intel.orchestration import stream as stream_mod

        monkeypatch.setattr(stream_mod, "synthesize_answer", fake_synthesize_answer)

        from career_intel.orchestration.stream import stream_turn

        events = []
        async for line in stream_turn(
            messages=[
                ChatMessage(
                    role="user",
                    content="Quelles competences essentielles sont liees au data scientist dans ESCO ?",
                )
            ],
            session_id="test-fr-stream",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="t-fr-stream",
        ):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert captured["rewrite_query"] == translated_query
        assert captured["retrieve_query"] == rewritten_query
        assert captured["synthesis_query"] == (
            "Quelles competences essentielles sont liees au data scientist dans ESCO ?"
        )
        assert captured["synthesis_rewritten_query"] == rewritten_query
        assert any(event["type"] == "citations" for event in events)

    @pytest.mark.asyncio
    async def test_stream_esco_relation_query_blocks_rag_when_evidence_is_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        decision = RouterDecision(
            intent="domain_specific",
            confidence=0.9,
            reason="needs ESCO grounded evidence",
        )

        async def fake_route(q, *, cv_available=False, settings=None):
            return decision, None

        async def fake_validate(text, max_length=4000):
            return text

        async def fake_rewrite(query, settings=None):
            return "Explain the distinction."

        async def fake_retrieve(query, filters=None, settings=None, **kwargs):
            return [
                RetrievedChunk(
                    chunk_id="esco-rel",
                    text="ESCO occupation-skill relation for data engineer.",
                    metadata=ChunkMetadata(
                        source_id="src-esco-rel",
                        source_type="esco",
                        source="esco",
                        title="ESCO relation detail",
                        document_title="ESCO occupation skill relation",
                        chunk_index=0,
                        parent_doc_id="doc-esco-rel",
                        esco_doc_type="relation_detail",
                    ),
                    score=0.57,
                    rerank_score=0.42,
                ),
            ]

        def fake_inc(name, count=1):
            pass

        class FakeStreamChunk:
            def __init__(self, content: str) -> None:
                self.content = content

        class FakeStreamingLLM:
            async def astream(self, messages: list[dict[str, str]]):
                for token in ["Fallback", " response"]:
                    yield FakeStreamChunk(token)

        monkeypatch.setattr("career_intel.security.guards.validate_input_deep", fake_validate)
        monkeypatch.setattr("career_intel.tools.registry.route_query", fake_route)
        monkeypatch.setattr("career_intel.api.routers.metrics.inc", fake_inc)
        monkeypatch.setattr("career_intel.rag.retriever.rewrite_query", fake_rewrite)
        monkeypatch.setattr("career_intel.rag.retriever.retrieve_chunks", fake_retrieve)
        monkeypatch.setattr(
            "career_intel.rag.retriever.build_query_profile",
            lambda query, detected_source=None: (
                SimpleNamespace(
                    esco_relation_query=True,
                    taxonomy_query=False,
                    essential_optional_query=True,
                    salient_concepts=("essential", "optional"),
                    detected_source="esco",
                    classification_reason="no_relation_signal",
                )
                if "ESCO" in query or "essential" in query.lower() or "optional" in query.lower()
                else SimpleNamespace(
                    esco_relation_query=False,
                    taxonomy_query=False,
                    essential_optional_query=False,
                    salient_concepts=(),
                    detected_source=detected_source,
                    classification_reason="no_relation_signal",
                )
            ),
        )
        monkeypatch.setattr(
            "career_intel.rag.retriever.assess_evidence_strength",
            lambda chunks, settings=None, detected_source=None: (
                "none",
                "similarity_without_rerank_coherence",
            ),
        )
        monkeypatch.setattr(
            "career_intel.rag.retriever.should_force_rag",
            lambda chunks, settings=None, detected_source=None: (
                True,
                "3_chunks_at_or_above_similarity_threshold",
            ),
        )

        from career_intel.orchestration import stream as stream_mod

        monkeypatch.setattr(
            stream_mod,
            "get_chat_llm",
            lambda settings, temperature=0.2, streaming=True: FakeStreamingLLM(),
        )

        from career_intel.orchestration.stream import stream_turn

        events = []
        async for line in stream_turn(
            messages=[
                ChatMessage(
                    role="user",
                    content="What is the difference between essential and optional skills in ESCO relations?",
                )
            ],
            session_id="test-esco-stream-fallback",
            use_tools=True,
            filters=None,
            settings=_settings(),
            trace_id="t-esco-stream-fallback",
        ):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        debug_event = next(e for e in events if e["type"] == "debug")
        assert debug_event["data"]["answer_source"] == "llm_fallback"
        assert debug_event["data"]["path_reason"] == "blocked_none_evidence_for_esco_relation_query"
