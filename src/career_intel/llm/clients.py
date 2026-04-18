"""Centralized LLM and embedding client factory with retry/backoff.

Every OpenAI call in the codebase MUST go through these factories so that
retry behavior, structured logging, and cost tracking are consistent.

Retry strategy:
  - Retryable status codes: 429 (rate limit), 500, 502, 503 (transient).
  - Exponential backoff with jitter: base 1s, max 60s, 4 attempts.
  - The OpenAI SDK's ``max_retries`` provides a clean first layer.
  - We add a ``tenacity`` wrapper for embedding calls (sync SDK) so
    non-SDK transient failures (network, DNS) are also handled.
  - All retries and terminal failures emit structured log events.
"""

from __future__ import annotations

import random
import time
import os

import structlog
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI, RateLimitError

from career_intel.config import Settings, get_settings
from career_intel.llm.request_context import (
    get_request_api_key_override,
    get_request_model_override,
)
from career_intel.security.hardening import mask_secret

logger = structlog.get_logger()

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
_MAX_RETRIES_SDK = 3
_BACKOFF_BASE = 1
_BACKOFF_MAX = 60


def get_supported_chat_models(settings: Settings) -> list[str]:
    """Return the canonical backend allowlist for chat models."""

    legacy_env = os.getenv("CAREER_INTEL_OPENAI_SUPPORTED_MODELS", "").strip()
    raw_value = legacy_env or settings.supported_openai_chat_models
    supported = [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]
    if supported:
        return list(dict.fromkeys(supported))
    return [settings.openai_model]


def validate_chat_model_override(model_override: str | None, settings: Settings) -> str | None:
    """Validate a request-scoped chat model override against the server allowlist."""

    if model_override is None:
        return None
    candidate = model_override.strip()
    if not candidate:
        return None
    allowed = set(get_supported_chat_models(settings))
    if candidate not in allowed:
        raise ValueError(
            "Unsupported model override. Choose one of: "
            + ", ".join(sorted(allowed))
        )
    return candidate


def resolve_openai_api_key(settings: Settings) -> str:
    """Resolve the OpenAI API key for the current request."""
    return get_request_api_key_override() or settings.openai_api_key.get_secret_value()


def resolve_chat_model(
    settings: Settings,
    *,
    model_override: str | None = None,
) -> str:
    """Resolve the chat model for the current request."""
    return (
        validate_chat_model_override(model_override, settings)
        or validate_chat_model_override(get_request_model_override(), settings)
        or settings.openai_model
    )


def get_chat_llm(
    settings: Settings | None = None,
    *,
    temperature: float = 0.2,
    streaming: bool = False,
    model_override: str | None = None,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance with SDK-native retry built in.

    The ``max_retries`` parameter on ChatOpenAI uses the OpenAI SDK's
    internal retry machinery (exponential backoff on 429/5xx).
    """
    if settings is None:
        settings = get_settings()

    return ChatOpenAI(
        model=resolve_chat_model(settings, model_override=model_override),
        api_key=resolve_openai_api_key(settings),
        temperature=temperature,
        max_retries=_MAX_RETRIES_SDK,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Embedding client with tenacity retry
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an OpenAI exception is worth retrying."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code in _RETRYABLE_STATUS_CODES


def get_embeddings_client(settings: Settings | None = None) -> OpenAI:
    """Return a raw OpenAI client for embedding calls."""
    if settings is None:
        settings = get_settings()
    return OpenAI(
        api_key=resolve_openai_api_key(settings),
        max_retries=0,
        timeout=settings.openai_embedding_timeout_seconds,
    )


def get_async_openai_client(
    settings: Settings | None = None,
    *,
    timeout_seconds: float | None = None,
) -> AsyncOpenAI:
    """Return an async OpenAI client for non-chat APIs (e.g. speech transcription).

    Chat and embeddings continue to use ``get_chat_llm`` / ``get_embeddings_client``.
    """
    if settings is None:
        settings = get_settings()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.speech_transcription_timeout_seconds
    )
    return AsyncOpenAI(
        api_key=resolve_openai_api_key(settings),
        max_retries=_MAX_RETRIES_SDK,
        timeout=timeout,
    )


def embed_with_retry(
    client: OpenAI,
    texts: list[str],
    model: str,
    *,
    max_attempts: int,
    request_label: str = "embedding_request",
) -> list[list[float]]:
    """Embed texts with explicit retry logging and fail-fast behavior."""
    attempt_limit = max(1, max_attempts)
    last_error: BaseException | None = None

    for attempt in range(1, attempt_limit + 1):
        started = time.perf_counter()
        logger.info(
            "embedding_request_start",
            request_label=request_label,
            attempt=attempt,
            max_attempts=attempt_limit,
            batch_size=len(texts),
            model=model,
        )
        try:
            response = client.embeddings.create(input=texts, model=model)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            vectors = [item.embedding for item in response.data]
            logger.info(
                "embedding_request_success",
                request_label=request_label,
                attempt=attempt,
                batch_size=len(texts),
                model=model,
                elapsed_ms=elapsed_ms,
            )
            return vectors
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            last_error = exc
            retryable = _is_retryable(exc)
            logger.warning(
                "embedding_request_failure",
                request_label=request_label,
                attempt=attempt,
                max_attempts=attempt_limit,
                batch_size=len(texts),
                model=model,
                elapsed_ms=elapsed_ms,
                retryable=retryable,
                error_type=type(exc).__name__,
                error=mask_secret(str(exc)[:300]),
            )
            if not retryable or attempt >= attempt_limit:
                logger.exception(
                    "embedding_terminal_failure",
                    request_label=request_label,
                    attempts=attempt,
                    batch_size=len(texts),
                    model=model,
                )
                raise
            wait_seconds = min(_BACKOFF_MAX, _BACKOFF_BASE * (2 ** (attempt - 1))) + random.uniform(
                0,
                1,
            )
            logger.warning(
                "embedding_retry_sleep",
                request_label=request_label,
                attempt=attempt,
                next_attempt=attempt + 1,
                wait_seconds=round(wait_seconds, 2),
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedding request failed without raising a concrete exception.")
