"""Tests for retrieval-side language detection and translation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from career_intel.rag.query_preprocessor import normalize_query_for_retrieval


@pytest.mark.asyncio
async def test_normalize_query_for_retrieval_translates_non_english_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        async def ainvoke(self, _: list[dict[str, str]]) -> SimpleNamespace:
            return SimpleNamespace(
                content=(
                    '{"detected_language":"fr","requires_translation":true,'
                    '"english_query":"What essential skills are linked to the ESCO occupation data scientist?"}'
                )
            )

    from career_intel.rag import query_preprocessor

    monkeypatch.setattr(query_preprocessor, "get_chat_llm", lambda settings, temperature=0.0: FakeLLM())

    result = await normalize_query_for_retrieval(
        "Quelles competences essentielles sont liees au data scientist dans ESCO ?",
        settings=SimpleNamespace(),
    )

    assert result.detected_language == "fr"
    assert result.translation_applied is True
    assert result.translated_query == (
        "What essential skills are linked to the ESCO occupation data scientist?"
    )
    assert result.retrieval_query == result.translated_query


@pytest.mark.asyncio
async def test_normalize_query_for_retrieval_preserves_english_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        async def ainvoke(self, _: list[dict[str, str]]) -> SimpleNamespace:
            return SimpleNamespace(
                content=(
                    '{"detected_language":"en","requires_translation":false,'
                    '"english_query":"What essential skills are linked to the ESCO occupation data scientist?"}'
                )
            )

    from career_intel.rag import query_preprocessor

    monkeypatch.setattr(query_preprocessor, "get_chat_llm", lambda settings, temperature=0.0: FakeLLM())

    query = "What essential skills are linked to the ESCO occupation data scientist?"
    result = await normalize_query_for_retrieval(query, settings=SimpleNamespace())

    assert result.detected_language == "en"
    assert result.translation_applied is False
    assert result.translated_query is None
    assert result.retrieval_query == query
