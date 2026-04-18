"""Usage estimation and session rollups for the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tiktoken
from model_config import estimate_cost_usd
from token_cost import compute_request_cost_usd, cost_label_for_session


@dataclass(frozen=True)
class UsageEstimate:
    """Estimated token and cost data for one request (tiktoken-based)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


def estimate_text_tokens(text: str, model: str) -> int:
    """Estimate tokens for free-form text using the best available encoding."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text or ""))


def estimate_chat_input_tokens(
    *,
    messages: list[dict[str, str]],
    model: str,
    cv_text: str | None = None,
) -> int:
    """Estimate prompt tokens from chat history and optional CV context."""
    total = 0
    for message in messages:
        total += 4  # simple role/message envelope approximation
        total += estimate_text_tokens(message.get("content", ""), model)
    if cv_text:
        total += estimate_text_tokens(cv_text, model)
    return total + 2


def estimate_request_usage(
    *,
    messages: list[dict[str, str]],
    reply_text: str,
    model: str,
    cv_text: str | None = None,
) -> UsageEstimate:
    """Estimate per-request tokens and cost for display in the UI."""
    input_tokens = estimate_chat_input_tokens(messages=messages, model=model, cv_text=cv_text)
    output_tokens = estimate_text_tokens(reply_text, model)
    total_tokens = input_tokens + output_tokens
    return UsageEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(
            model_id=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _coerce_provider_usage(raw: dict[str, Any] | None) -> tuple[int, int, int] | None:
    if not raw:
        return None
    try:
        prompt = int(raw.get("prompt_tokens", 0))
        completion = int(raw.get("completion_tokens", 0))
    except (TypeError, ValueError):
        return None
    if prompt < 0 or completion < 0:
        return None
    total_raw = raw.get("total_tokens")
    try:
        total = int(total_raw) if total_raw is not None else prompt + completion
    except (TypeError, ValueError):
        total = prompt + completion
    if prompt == 0 and completion == 0 and total == 0:
        return None
    return prompt, completion, max(total, prompt + completion)


def build_message_usage_fields(
    *,
    model: str,
    estimate: UsageEstimate,
    provider_usage: dict[str, Any] | None,
    byok: bool,
) -> dict[str, Any]:
    """Merge provider-reported usage (when present) with tiktoken estimates for UI meta."""

    parsed = _coerce_provider_usage(provider_usage)
    if parsed is not None:
        prompt_tokens, completion_tokens, total_tokens = parsed
        counts_from_provider = True
        cost_usd = compute_request_cost_usd(
            model_id=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    else:
        prompt_tokens = estimate.input_tokens
        completion_tokens = estimate.output_tokens
        total_tokens = estimate.total_tokens
        counts_from_provider = False
        cost_usd = estimate.estimated_cost_usd

    label = cost_label_for_session(byok=byok, counts_from_provider=counts_from_provider)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "display_cost_usd": cost_usd,
        "estimated_cost_usd": cost_usd,
        "usage_counts_from_provider": counts_from_provider,
        "cost_label_short": label,
    }


def update_usage_summary(
    summary: dict[str, Any],
    *,
    model: str,
    usage: UsageEstimate,
    provider_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update aggregate session usage totals."""
    parsed = _coerce_provider_usage(provider_usage)
    if parsed is not None:
        prompt_delta, completion_delta, total_delta = parsed
        cost_delta = compute_request_cost_usd(
            model_id=model,
            prompt_tokens=prompt_delta,
            completion_tokens=completion_delta,
        )
        provider_requests = 1
    else:
        prompt_delta = usage.input_tokens
        completion_delta = usage.output_tokens
        total_delta = usage.total_tokens
        cost_delta = usage.estimated_cost_usd
        provider_requests = 0

    by_model = dict(summary.get("by_model") or {})
    model_summary = dict(
        by_model.get(model)
        or {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "provider_requests": 0,
        }
    )
    model_summary["requests"] += 1
    model_summary["prompt_tokens"] += prompt_delta
    model_summary["completion_tokens"] += completion_delta
    model_summary["total_tokens"] += total_delta
    model_summary["estimated_cost_usd"] = round(
        float(model_summary.get("estimated_cost_usd", 0.0)) + cost_delta,
        6,
    )
    model_summary["provider_requests"] = int(model_summary.get("provider_requests", 0)) + provider_requests
    by_model[model] = model_summary

    base_provider = int(summary.get("provider_requests", 0))
    return {
        "requests": int(summary.get("requests", 0)) + 1,
        "prompt_tokens": int(summary.get("prompt_tokens", 0)) + prompt_delta,
        "completion_tokens": int(summary.get("completion_tokens", 0)) + completion_delta,
        "total_tokens": int(summary.get("total_tokens", 0)) + total_delta,
        "estimated_cost_usd": round(float(summary.get("estimated_cost_usd", 0.0)) + cost_delta, 6),
        "provider_requests": base_provider + provider_requests,
        "by_model": by_model,
    }
