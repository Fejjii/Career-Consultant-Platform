"""Extract OpenAI-style token usage from LangChain chat responses."""

from __future__ import annotations

from typing import Any

from career_intel.schemas.api import TokenUsage


def usage_from_langchain_message(message: Any) -> TokenUsage | None:
    """Parse ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` from a message or chunk."""

    if message is None:
        return None

    prompt: int | None = None
    completion: int | None = None
    total: int | None = None

    usage_meta = getattr(message, "usage_metadata", None)
    if isinstance(usage_meta, dict) and usage_meta:
        prompt = _coerce_int(
            usage_meta.get("input_tokens")
            or usage_meta.get("prompt_tokens")
            or usage_meta.get("input_token_count")
        )
        completion = _coerce_int(
            usage_meta.get("output_tokens")
            or usage_meta.get("completion_tokens")
            or usage_meta.get("output_token_count")
        )
        total = _coerce_int(usage_meta.get("total_tokens") or usage_meta.get("total_token_count"))

    response_meta = getattr(message, "response_metadata", None)
    if isinstance(response_meta, dict):
        token_usage = response_meta.get("token_usage")
        if isinstance(token_usage, dict):
            if prompt is None:
                prompt = _coerce_int(token_usage.get("prompt_tokens"))
            if completion is None:
                completion = _coerce_int(token_usage.get("completion_tokens"))
            if total is None:
                total = _coerce_int(token_usage.get("total_tokens"))

    if prompt is None and completion is None and total is None:
        return None

    prompt = int(prompt or 0)
    completion = int(completion or 0)
    if total is None:
        total = prompt + completion
    total = int(total)
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def merge_token_usages(*parts: TokenUsage | None) -> TokenUsage | None:
    """Sum usage across multiple completions in one logical turn (e.g. router + synthesis)."""

    valid = [p for p in parts if p is not None]
    if not valid:
        return None
    return TokenUsage(
        prompt_tokens=sum(p.prompt_tokens for p in valid),
        completion_tokens=sum(p.completion_tokens for p in valid),
        total_tokens=sum(p.total_tokens for p in valid),
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
