"""Tests for the context builder module."""

from __future__ import annotations

from career_intel.orchestration.context_builder import build_user_prompt
from career_intel.schemas.domain import ChunkMetadata, RetrievedChunk


def _make_chunk(chunk_id: str = "c1", text: str = "Sample evidence.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            source_id="s1",
            source_type="report",
            title="Test Report",
            chunk_index=0,
            parent_doc_id="d1",
        ),
        score=0.85,
    )


class TestBuildUserPrompt:
    def test_basic_prompt_without_cv(self) -> None:
        chunks = [_make_chunk()]
        prompt, citation_map = build_user_prompt(
            query="What skills are needed?",
            chunks=chunks,
            tool_block="",
            cv_text=None,
            use_cv=False,
        )
        assert "What skills are needed?" in prompt
        assert "SOURCES" in prompt
        assert "USER_CV" not in prompt
        assert 1 in citation_map

    def test_cv_included_when_relevant(self) -> None:
        chunks = [_make_chunk()]
        prompt, _ = build_user_prompt(
            query="What skills am I missing?",
            chunks=chunks,
            tool_block="",
            cv_text="John Doe, Python, ML",
            use_cv=True,
        )
        assert "USER_CV" in prompt
        assert "John Doe" in prompt

    def test_cv_excluded_when_not_relevant(self) -> None:
        chunks = [_make_chunk()]
        prompt, _ = build_user_prompt(
            query="Hello",
            chunks=chunks,
            tool_block="",
            cv_text="John Doe, Python, ML",
            use_cv=False,
        )
        assert "USER_CV" not in prompt
        assert "John Doe" not in prompt

    def test_cv_excluded_when_none(self) -> None:
        chunks = [_make_chunk()]
        prompt, _ = build_user_prompt(
            query="What skills am I missing?",
            chunks=chunks,
            tool_block="",
            cv_text=None,
            use_cv=True,
        )
        assert "USER_CV" not in prompt

    def test_tool_block_included(self) -> None:
        chunks = [_make_chunk()]
        prompt, _ = build_user_prompt(
            query="Compare roles",
            chunks=chunks,
            tool_block="### Tool: role_compare\n```json\n{}\n```",
            cv_text=None,
            use_cv=False,
        )
        assert "Tool results:" in prompt
        assert "role_compare" in prompt
