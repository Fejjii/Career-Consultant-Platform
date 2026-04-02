"""Tests for admin endpoint authentication."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ingest_without_secret_returns_403(client: AsyncClient) -> None:
    resp = await client.post("/ingest", json={"paths": ["test.md"]})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_ingest_with_wrong_secret_returns_403(client: AsyncClient) -> None:
    resp = await client.post(
        "/ingest",
        json={"paths": ["test.md"]},
        headers={"X-Admin-Secret": "wrong-secret"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_evaluation_without_secret_returns_403(client: AsyncClient) -> None:
    resp = await client.post("/evaluation/run")
    assert resp.status_code == 403
