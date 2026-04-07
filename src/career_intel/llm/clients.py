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

import structlog
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from career_intel.config import Settings, get_settings

logger = structlog.get_logger()

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
_MAX_RETRIES_SDK = 3
_MAX_RETRIES_TENACITY = 4
_BACKOFF_BASE = 1
_BACKOFF_MAX = 60


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
        model=model_override or settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
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


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.warning(
        "embedding_retry",
        attempt=state.attempt_number,
        error=str(exc)[:200] if exc else None,
        wait_seconds=round(state.next_action.sleep if state.next_action else 0, 2),  # type: ignore[union-attr]
    )


def _log_terminal_failure(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.error(
        "embedding_terminal_failure",
        attempts=state.attempt_number,
        error=str(exc)[:200] if exc else None,
    )
    if exc is not None:
        raise exc


def get_embeddings_client(settings: Settings | None = None) -> OpenAI:
    """Return a raw OpenAI client for embedding calls."""
    if settings is None:
        settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        max_retries=_MAX_RETRIES_SDK,
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(_MAX_RETRIES_TENACITY),
    wait=wait_exponential_jitter(initial=_BACKOFF_BASE, max=_BACKOFF_MAX, jitter=2),
    before_sleep=_log_retry,
    retry_error_callback=lambda state: _log_terminal_failure(state),  # type: ignore[arg-type]
    reraise=True,
)
def embed_with_retry(
    client: OpenAI,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Embed texts with tenacity retry around the sync OpenAI SDK call."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]
