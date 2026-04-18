"""Frontend model catalog and pricing helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ModelPricing:
    """UI-facing metadata for a selectable chat model."""

    model_id: str
    label: str
    description: str
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float


@dataclass(frozen=True)
class ModelCatalogEntry:
    """App-facing model metadata for filtering and presentation."""

    model_id: str
    label: str
    description: str
    recommended_rank: int
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ModelAvailabilitySummary:
    """Normalized model availability for compact UI rendering."""

    raw_accessible_count: int
    supported_count: int
    selectable_count: int
    selectable_models: list[str]
    supported_but_unavailable_models: list[str]
    accessible_but_unsupported_models: list[str]
    ignored_raw_models_count: int


@dataclass(frozen=True)
class ModelSelectionResolution:
    """Outcome of reconciling selected model against current options."""

    selected_model: str
    changed: bool
    reason_code: str | None = None


MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-4.1": ModelPricing(
        model_id="gpt-4.1",
        label="GPT-4.1",
        description="Recommended for highest quality grounded answers and reasoning depth.",
        input_cost_per_million_tokens=2.0,
        output_cost_per_million_tokens=8.0,
    ),
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        label="GPT-4o",
        description="Best answer quality for complex grounded and tool-backed responses.",
        input_cost_per_million_tokens=5.0,
        output_cost_per_million_tokens=15.0,
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        label="GPT-4o mini",
        description="Lower-cost option for faster exploratory sessions.",
        input_cost_per_million_tokens=0.15,
        output_cost_per_million_tokens=0.60,
    ),
}

MODEL_CATALOG: dict[str, ModelCatalogEntry] = {
    "gpt-4.1": ModelCatalogEntry(
        model_id="gpt-4.1",
        label="GPT-4.1",
        description="Recommended for best quality in this app.",
        recommended_rank=1,
        aliases=("gpt-4.1",),
    ),
    "gpt-4o": ModelCatalogEntry(
        model_id="gpt-4o",
        label="GPT-4o",
        description="Strong balanced quality and speed.",
        recommended_rank=2,
        aliases=("gpt-4o", "chatgpt-4o-latest"),
    ),
    "gpt-4o-mini": ModelCatalogEntry(
        model_id="gpt-4o-mini",
        label="GPT-4o mini",
        description="Lower-cost option for lightweight iterations.",
        recommended_rank=3,
        aliases=("gpt-4o-mini",),
    ),
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
_DATE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-\d{4}-\d{2}-\d{2}$")


def get_available_model_ids(auth_status: dict[str, Any] | None = None) -> list[str]:
    """Return selectable model IDs for the active credential source."""

    if auth_status and auth_status.get("ok"):
        supported = get_supported_model_ids()
        available_to_key = _get_models_available_to_key(auth_status)
        return [model_id for model_id in supported if model_id in set(available_to_key)]
    return list(get_supported_model_ids())


def get_supported_model_ids() -> list[str]:
    """Return the app allowlist of supported chat models."""

    raw = os.getenv("CAREER_INTEL_OPENAI_SUPPORTED_MODELS", "gpt-4.1,gpt-4o,gpt-4o-mini")
    configured = [item.strip() for item in raw.split(",") if item.strip()]
    filtered = [_normalize_model_id(item) for item in configured]
    filtered = [item for item in filtered if item and item in MODEL_CATALOG]
    if filtered:
        return list(dict.fromkeys(filtered))
    return list(MODEL_CATALOG.keys())


def get_default_model_id(preferred_model_ids: Sequence[str] | None = None) -> str:
    """Return the frontend default model, preferring the environment override."""

    configured = _normalize_model_id(os.getenv("CAREER_INTEL_DEFAULT_MODEL", "gpt-4.1"))
    allowed = list(preferred_model_ids) if preferred_model_ids else get_supported_model_ids()
    if configured and configured in allowed:
        return configured
    ranked = sorted(
        (model_id for model_id in allowed if model_id in MODEL_CATALOG),
        key=lambda model_id: MODEL_CATALOG[model_id].recommended_rank,
    )
    if ranked:
        return ranked[0]
    unknown = [_normalize_unknown_chat_family(model_id) for model_id in allowed]
    unknown = [item for item in unknown if item]
    if unknown:
        return sorted(set(unknown))[0]
    return "gpt-4o"


def get_model_pricing(model_id: str) -> ModelPricing:
    """Return pricing metadata for a model."""
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    return MODEL_PRICING[get_default_model_id()]


def estimate_cost_usd(*, model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate request cost from token counts and the pricing table."""
    pricing = get_model_pricing(model_id)
    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_million_tokens
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_million_tokens
    return round(input_cost + output_cost, 6)


def get_recommended_model_ids() -> list[str]:
    """Return supported models sorted by recommendation rank."""

    supported = set(get_supported_model_ids())
    return [
        model_id
        for model_id, _entry in sorted(
            MODEL_CATALOG.items(),
            key=lambda item: item[1].recommended_rank,
        )
        if model_id in supported
    ]


def get_model_label(model_id: str) -> str:
    """Return a compact friendly label for selector and summaries."""

    if model_id in MODEL_CATALOG:
        return MODEL_CATALOG[model_id].label
    value = model_id.strip().lower()
    if value.startswith("gpt-"):
        return value.replace("gpt-", "GPT-").replace("-mini", " mini")
    return model_id


def get_model_description(model_id: str) -> str:
    """Return a short user-facing model description."""

    if model_id in MODEL_CATALOG:
        return MODEL_CATALOG[model_id].description
    return "Chat-capable model available for the active credential source."


def summarize_model_availability(auth_status: dict[str, Any] | None) -> ModelAvailabilitySummary:
    """Build a compact BYOK summary that separates raw/provider/app/selectable sets."""

    supported = get_supported_model_ids()
    if auth_status and auth_status.get("ok"):
        available_to_key = _get_models_available_to_key(auth_status)
        available_lookup = set(available_to_key)
        selectable = [model_id for model_id in supported if model_id in available_lookup]
        unavailable = [model_id for model_id in supported if model_id not in available_lookup]
        unsupported = [model_id for model_id in available_to_key if model_id not in supported]
    else:
        selectable = list(supported)
        unavailable = []
        unsupported = []
    raw_accessible = list(auth_status.get("accessible_models") or []) if auth_status else []
    ignored = list(auth_status.get("ignored_accessible_models") or []) if auth_status else []
    return ModelAvailabilitySummary(
        raw_accessible_count=len(raw_accessible),
        supported_count=len(supported),
        selectable_count=len(selectable),
        selectable_models=selectable,
        supported_but_unavailable_models=unavailable,
        accessible_but_unsupported_models=unsupported,
        ignored_raw_models_count=len(ignored),
    )


def explain_model_unavailability(auth_status: dict[str, Any] | None, model_id: str) -> str | None:
    """Return a concise reason string for why a supported model is not selectable."""

    if not auth_status:
        return None
    reasons = auth_status.get("model_unavailability_reasons")
    if isinstance(reasons, dict):
        value = reasons.get(model_id)
        if isinstance(value, str):
            return value
    return None


def format_unavailability_for_ui(reason_code: str | None) -> str:
    """Return polished user-facing copy for model unavailability reasons."""

    if not reason_code:
        return "not available with this key"
    if reason_code == "selectable":
        return "available"
    if reason_code == "not_supported_by_app":
        return "not supported in this app"
    if reason_code == "available_but_not_selected":
        return "available but not currently selectable here"
    if reason_code == "not_returned_by_provider_or_filtered_as_irrelevant":
        return "not available to this key for selectable chat usage"
    return "not available with this key"


def resolve_selected_model(
    *,
    current_model: str,
    available_models: Sequence[str],
) -> ModelSelectionResolution:
    """Ensure selected model remains valid after credential-source changes."""

    options = [item for item in available_models if item]
    if not options:
        return ModelSelectionResolution(selected_model="gpt-4o", changed=current_model != "gpt-4o", reason_code="empty_options")
    if current_model in options:
        return ModelSelectionResolution(selected_model=current_model, changed=False)
    fallback = get_default_model_id(options)
    return ModelSelectionResolution(
        selected_model=fallback,
        changed=True,
        reason_code="selected_model_not_available_for_credential_source",
    )


def _normalize_model_id(model_id: str) -> str | None:
    """Normalize raw model IDs and aliases to the app catalog."""

    value = model_id.strip().lower()
    if not value:
        return None
    if any(hint in value for hint in _IRRELEVANT_MODEL_HINTS):
        return None
    if "preview" in value:
        return None
    dated = _DATE_SUFFIX_RE.match(value)
    if dated:
        value = dated.group("base")
    alias_pairs = [
        (alias, canonical)
        for canonical, entry in MODEL_CATALOG.items()
        for alias in entry.aliases
    ]
    for alias, canonical in sorted(alias_pairs, key=lambda item: len(item[0]), reverse=True):
        if value == alias or value.startswith(f"{alias}-"):
            return canonical
    return None


def _normalize_unknown_chat_family(model_id: str) -> str | None:
    """Normalize unknown-but-chat-capable models for dynamic selector support."""

    value = model_id.strip().lower()
    if not value:
        return None
    if any(hint in value for hint in _IRRELEVANT_MODEL_HINTS):
        return None
    if "preview" in value:
        return None
    dated = _DATE_SUFFIX_RE.match(value)
    if dated:
        value = dated.group("base")
    if value.startswith("gpt-"):
        return value
    return None


def _get_models_available_to_key(auth_status: dict[str, Any]) -> list[str]:
    """Return all chat-capable models the active key can access."""

    raw_models = (
        list(auth_status.get("normalized_accessible_models") or [])
        or list(auth_status.get("accessible_models") or [])
        or list(auth_status.get("selectable_models") or [])
    )
    normalized = [_normalize_model_id(item) or _normalize_unknown_chat_family(item) for item in raw_models]
    return list(dict.fromkeys(item for item in normalized if item))
