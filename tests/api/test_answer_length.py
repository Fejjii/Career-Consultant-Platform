"""Chat API forwards answer_length into orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest
from httpx import AsyncClient

from career_intel.schemas.api import AnswerLengthMode, ChatResponse


@pytest.mark.asyncio
async def test_post_chat_passes_answer_length_to_run_turn(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_run_turn(**kwargs: object) -> ChatResponse:
        seen.update(kwargs)
        al = cast(AnswerLengthMode, kwargs["answer_length"])
        return ChatResponse(
            session_id="test-session",
            reply="OK",
            citations=[],
            tool_calls=[],
            intent="small_talk",
            trace_id=None,
            answer_length=al,
            created_at=datetime.now(UTC),
        )

    monkeypatch.setattr(
        "career_intel.orchestration.chain.run_turn",
        _fake_run_turn,
    )

    resp = await client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "answer_length": "concise",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["answer_length"] == "concise"
    assert seen.get("answer_length") == "concise"
