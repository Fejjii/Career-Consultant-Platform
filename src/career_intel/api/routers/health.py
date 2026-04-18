"""Health, provider validation, and source coverage endpoints."""

from __future__ import annotations

import os
import re
import time

import structlog
from fastapi import APIRouter

from career_intel.config import get_settings
from career_intel.storage.qdrant_store import get_qdrant_client
from career_intel.schemas.api import (
    HealthResponse,
    ProviderAuthStatusResponse,
    ReadyDetail,
    ReadyResponse,
    SourceInventoryResponse,
    SystemStatusResponse,
)
from career_intel.llm.clients import get_supported_chat_models
from career_intel.services.source_inventory import build_source_inventory_summary

router = APIRouter(tags=["health"])
logger = structlog.get_logger()

_SUPPORTED_CHAT_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "gpt-4.1": ("gpt-4.1",),
    "gpt-4o-mini": ("gpt-4o-mini",),
    "gpt-4o": ("gpt-4o", "chatgpt-4o-latest"),
}
_IRRELEVANT_MODEL_HINTS = (
    "audio",
    "realtime",
    "transcribe",
    "tts",
    "embedding",
    "image",
    "whisper",
    "moderation",
    "omni-moderation",
    "search",
    "instruct",
)
_PREVIEW_HINT = "preview"
_MODEL_DATE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-\d{4}-\d{2}-\d{2}$")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness() -> ReadyResponse:
    """Check connectivity to Postgres, Redis, and Qdrant."""
    settings = get_settings()
    details: list[ReadyDetail] = []

    # --- Qdrant ---
    try:
        t0 = time.monotonic()
        qc = get_qdrant_client(settings)
        qc.get_collections()
        details.append(ReadyDetail(
            name="qdrant",
            ok=True,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        ))
    except Exception as exc:
        details.append(ReadyDetail(name="qdrant", ok=False, error=_health_error_detail(exc, settings)))

    # --- Redis ---
    try:
        import redis.asyncio as aioredis

        t0 = time.monotonic()
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        await r.ping()
        await r.aclose()
        details.append(ReadyDetail(
            name="redis",
            ok=True,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        ))
    except Exception as exc:
        details.append(ReadyDetail(name="redis", ok=False, error=_health_error_detail(exc, settings)))

    # --- Postgres ---
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        t0 = time.monotonic()
        engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        details.append(ReadyDetail(
            name="postgres",
            ok=True,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        ))
    except Exception as exc:
        details.append(ReadyDetail(name="postgres", ok=False, error=_health_error_detail(exc, settings)))

    all_ok = all(d.ok for d in details)
    return ReadyResponse(ok=all_ok, details=details)


@router.get("/health/system", response_model=SystemStatusResponse)
async def system_status() -> SystemStatusResponse:
    """Return lightweight status for UI smoke tests."""
    settings = get_settings()
    try:
        client = get_qdrant_client(settings)
        collections = {c.name for c in client.get_collections().collections}
        if settings.qdrant_collection not in collections:
            return SystemStatusResponse(
                backend=True,
                qdrant=True,
                indexed_data_present=False,
                collection=settings.qdrant_collection,
                points_count=0,
                error=(
                    f"Collection '{settings.qdrant_collection}' not found. Run ingestion."
                    if settings.environment == "development"
                    else "indexed data unavailable"
                ),
            )

        count = client.count(collection_name=settings.qdrant_collection, exact=False).count
        return SystemStatusResponse(
            backend=True,
            qdrant=True,
            indexed_data_present=count > 0,
            collection=settings.qdrant_collection,
            points_count=count,
            error=(
                None
                if count > 0
                else (
                    "No indexed data found. Run ingestion."
                    if settings.environment == "development"
                    else "indexed data unavailable"
                )
            ),
        )
    except Exception as exc:
        return SystemStatusResponse(
            backend=True,
            qdrant=False,
            indexed_data_present=False,
            collection=settings.qdrant_collection,
            points_count=0,
            error=_health_error_detail(exc, settings),
        )


@router.get("/health/provider-auth", response_model=ProviderAuthStatusResponse)
async def provider_auth_status() -> ProviderAuthStatusResponse:
    """Validate BYOK credentials and resolve accessible chat models."""
    settings = get_settings()
    from career_intel.llm.clients import get_async_openai_client, resolve_chat_model

    model = resolve_chat_model(settings)
    supported_models = get_supported_chat_models(settings)
    try:
        client = get_async_openai_client(settings, timeout_seconds=8.0)
        page = await client.models.list()
        accessible_models = sorted({item.id for item in page.data if getattr(item, "id", None)})
        normalized_accessible_models, ignored_accessible_models = _normalize_accessible_chat_models(accessible_models)
        selectable_models = list(normalized_accessible_models)
        supported_but_unavailable = [item for item in supported_models if item not in normalized_accessible_models]
        accessible_but_unsupported = [item for item in normalized_accessible_models if item not in supported_models]
        model_unavailability_reasons = {
            model_id: _availability_reason(
                model_id=model_id,
                normalized_accessible_models=normalized_accessible_models,
                supported_models=supported_models,
                selectable_models=selectable_models,
            )
            for model_id in sorted(set(supported_models + normalized_accessible_models))
        }
        logger.info(
            "provider_model_discovery",
            raw_accessible_count=len(accessible_models),
            normalized_accessible_count=len(normalized_accessible_models),
            selectable_count=len(selectable_models),
            ignored_count=len(ignored_accessible_models),
        )
        if selectable_models:
            message = "Credentials are valid and supported models were loaded for this key."
        else:
            message = "Credentials are valid, but this key cannot access any models currently supported by the app."
        return ProviderAuthStatusResponse(
            ok=True,
            model=model,
            message=message,
            supported_models=supported_models,
            accessible_models=accessible_models,
            normalized_accessible_models=normalized_accessible_models,
            selectable_models=selectable_models,
            supported_but_unavailable_models=supported_but_unavailable,
            accessible_but_unsupported_models=accessible_but_unsupported,
            ignored_accessible_models=ignored_accessible_models,
            model_unavailability_reasons=model_unavailability_reasons,
            validation_stage="model_catalog_loaded",
        )
    except Exception:
        return ProviderAuthStatusResponse(
            ok=False,
            model=model,
            message="Provider authentication failed. Check the selected key and model.",
            supported_models=supported_models,
            validation_stage="provider_authentication",
        )


@router.get("/health/source-inventory", response_model=SourceInventoryResponse)
async def source_inventory() -> SourceInventoryResponse:
    """Return corpus source coverage metadata for UI cards and source questions."""

    summary = build_source_inventory_summary()
    return SourceInventoryResponse.model_validate(
        {
            "total_source_groups": summary.total_source_groups,
            "total_files_present": summary.total_files_present,
            "esco_ingestion_status": summary.esco_ingestion_status,
            "esco_status_note": summary.esco_status_note,
            "items": summary.items,
        }
    )


def _health_error_detail(exc: Exception, settings: object) -> str:
    """Return safe health-check errors without leaking infrastructure details."""

    environment = getattr(settings, "environment", "development")
    if environment == "development":
        return str(exc)
    return "dependency unavailable"


def _normalize_accessible_chat_models(raw_models: list[str]) -> tuple[list[str], list[str]]:
    """Normalize raw provider model ids into supported chat families."""

    normalized: list[str] = []
    ignored: list[str] = []
    for model_id in raw_models:
        canonical = _normalize_model_id(model_id)
        if canonical:
            normalized.append(canonical)
            continue
        fallback_chat_family = _normalize_unknown_chat_family(model_id)
        if fallback_chat_family:
            normalized.append(fallback_chat_family)
        else:
            ignored.append(model_id)
    return sorted(set(normalized)), sorted(set(ignored))


def _normalize_model_id(model_id: str) -> str | None:
    """Return a canonical chat model id for app-supported families."""

    value = model_id.strip().lower()
    if not value:
        return None
    if any(hint in value for hint in _IRRELEVANT_MODEL_HINTS):
        return None
    if _PREVIEW_HINT in value:
        return None
    dated = _MODEL_DATE_SUFFIX_RE.match(value)
    if dated:
        value = dated.group("base")
    alias_pairs = [
        (alias, canonical)
        for canonical, aliases in _SUPPORTED_CHAT_MODEL_ALIASES.items()
        for alias in aliases
    ]
    for alias, canonical in sorted(alias_pairs, key=lambda item: len(item[0]), reverse=True):
        if value == alias or value.startswith(f"{alias}-"):
            return canonical
    return None


def _normalize_unknown_chat_family(model_id: str) -> str | None:
    """Return generic chat family ids for unsupported-but-chat-capable models."""

    value = model_id.strip().lower()
    if not value:
        return None
    if any(hint in value for hint in _IRRELEVANT_MODEL_HINTS):
        return None
    if _PREVIEW_HINT in value:
        return None
    dated = _MODEL_DATE_SUFFIX_RE.match(value)
    if dated:
        value = dated.group("base")
    if value.startswith("gpt-"):
        return value
    return None


def _availability_reason(
    *,
    model_id: str,
    normalized_accessible_models: list[str],
    supported_models: list[str],
    selectable_models: list[str],
) -> str:
    if model_id in selectable_models:
        return "selectable"
    if model_id not in supported_models:
        return "not_supported_by_app"
    if model_id in normalized_accessible_models:
        return "available_but_not_selected"
    return "not_returned_by_provider_or_filtered_as_irrelevant"
