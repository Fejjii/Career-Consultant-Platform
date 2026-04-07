"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject minimal env vars so Settings can load in tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-not-real")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async test client for the FastAPI app."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-not-real")

    from career_intel.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
