"""Tests for health endpoints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness_returns_structure(client: AsyncClient) -> None:
    resp = await client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ok" in data
    assert "details" in data
    assert isinstance(data["details"], list)


@pytest.mark.asyncio
async def test_provider_auth_returns_dynamic_model_availability(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeModels:
        async def list(self) -> SimpleNamespace:
            return SimpleNamespace(
                data=[
                    SimpleNamespace(id="gpt-4.1-2025-04-14"),
                    SimpleNamespace(id="chatgpt-4o-latest"),
                    SimpleNamespace(id="gpt-5-2026-01-01"),
                    SimpleNamespace(id="gpt-4o-realtime-preview"),
                    SimpleNamespace(id="whisper-1"),
                ]
            )

    class _FakeClient:
        def __init__(self) -> None:
            self.models = _FakeModels()

    monkeypatch.setenv("CAREER_INTEL_OPENAI_SUPPORTED_MODELS", "gpt-4.1,gpt-4o,gpt-4o-mini")
    monkeypatch.setattr(
        "career_intel.llm.clients.get_async_openai_client",
        lambda settings, timeout_seconds=8.0: _FakeClient(),
    )

    resp = await client.get("/health/provider-auth")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["selectable_models"] == ["gpt-4.1", "gpt-4o", "gpt-5"]
    assert data["supported_but_unavailable_models"] == ["gpt-4o-mini"]
    assert data["accessible_but_unsupported_models"] == ["gpt-5"]
    assert data["normalized_accessible_models"] == ["gpt-4.1", "gpt-4o", "gpt-5"]
    assert "gpt-4o-realtime-preview" in data["ignored_accessible_models"]
    assert "whisper-1" in data["ignored_accessible_models"]
    assert data["model_unavailability_reasons"]["gpt-4o-mini"] == "not_returned_by_provider_or_filtered_as_irrelevant"


@pytest.mark.asyncio
async def test_source_inventory_returns_expected_structure(client: AsyncClient) -> None:
    resp = await client.get("/health/source-inventory")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_source_groups"] >= 8
    assert isinstance(data["items"], list)
    assert any(item["source_name"] == "WEF Future of Jobs 2018" for item in data["items"])
    assert all("/" not in path and "\\" not in path for item in data["items"] for path in item["paths"])


@pytest.mark.asyncio
async def test_provider_auth_rejects_unsupported_model_override(client: AsyncClient) -> None:
    resp = await client.get(
        "/health/provider-auth",
        headers={"X-OpenAI-Model": "gpt-unknown"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_model_override"
