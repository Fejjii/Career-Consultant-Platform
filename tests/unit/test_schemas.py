"""Tests for API and domain Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from career_intel.schemas.api import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    ErrorResponse,
    FeedbackRequest,
    IngestRequest,
)


class TestChatRequest:
    def test_valid_request(self) -> None:
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            session_id="abc",
        )
        assert len(req.messages) == 1
        assert req.answer_length == "balanced"

    def test_answer_length_accepts_valid_modes(self) -> None:
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            answer_length="detailed",
        )
        assert req.answer_length == "detailed"

    def test_invalid_answer_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(
                messages=[ChatMessage(role="user", content="Hello")],
                answer_length="verbose",  # type: ignore[arg-type]
            )

    def test_empty_messages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(messages=[ChatMessage(role="user", content="")])


class TestChatResponse:
    def test_response_with_citations(self) -> None:
        resp = ChatResponse(
            session_id="s1",
            reply="Based on [1], data analysts need SQL.",
            citations=[
                Citation(
                    id=1,
                    source_id="doc1",
                    title="Career Guide",
                    excerpt="Data analysts use SQL...",
                )
            ],
        )
        assert resp.citations[0].id == 1


class TestFeedbackRequest:
    def test_valid_feedback(self) -> None:
        fb = FeedbackRequest(
            session_id="s1",
            message_id="m1",
            score=4,
            tags=["helpful"],
        )
        assert fb.score == 4

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            FeedbackRequest(session_id="s1", message_id="m1", score=0)

        with pytest.raises(ValidationError):
            FeedbackRequest(session_id="s1", message_id="m1", score=6)


class TestIngestRequest:
    def test_valid_request(self) -> None:
        req = IngestRequest(paths=["data/raw/test.md"])
        assert req.mode == "full"

    def test_empty_paths_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(paths=[])


class TestErrorResponse:
    def test_error_shape(self) -> None:
        err = ErrorResponse(error="not_found", detail="Resource missing")
        assert err.error == "not_found"
