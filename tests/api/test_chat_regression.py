"""Light regression checks: chat path still reachable after unrelated API changes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import AsyncClient

from career_intel.schemas.api import ChatResponse


@pytest.mark.asyncio
async def test_post_chat_invokes_run_turn(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run_turn(**kwargs: object) -> ChatResponse:
        return ChatResponse(
            session_id="test-session",
            reply="OK",
            citations=[],
            tool_calls=[],
            intent="small_talk",
            trace_id=None,
            created_at=datetime.now(UTC),
        )

    monkeypatch.setattr(
        "career_intel.orchestration.chain.run_turn",
        _fake_run_turn,
    )

    resp = await client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "OK"
    assert data["session_id"] == "test-session"


@pytest.mark.asyncio
async def test_cv_process_still_accepts_txt(client: AsyncClient) -> None:
    resp = await client.post(
        "/cv/process",
        files={"file": ("cv.txt", b"Name\nRole", "text/plain")},
    )
    assert resp.status_code == 200
    assert "Name" in resp.json()["cv_text"]
