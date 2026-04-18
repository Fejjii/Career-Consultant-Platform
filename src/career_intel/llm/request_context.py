"""Request-scoped LLM overrides for model and API key selection."""

from __future__ import annotations

from contextvars import ContextVar, Token

_REQUEST_API_KEY: ContextVar[str | None] = ContextVar("request_openai_api_key", default=None)
_REQUEST_MODEL: ContextVar[str | None] = ContextVar("request_openai_model", default=None)


def set_request_llm_overrides(
    *,
    api_key: str | None,
    model: str | None,
) -> tuple[Token[str | None], Token[str | None]]:
    """Bind request-local OpenAI overrides and return reset tokens."""
    api_key_token = _REQUEST_API_KEY.set(api_key.strip() if api_key else None)
    model_token = _REQUEST_MODEL.set(model.strip() if model else None)
    return api_key_token, model_token


def reset_request_llm_overrides(
    api_key_token: Token[str | None],
    model_token: Token[str | None],
) -> None:
    """Restore the previous request-local override values."""
    _REQUEST_API_KEY.reset(api_key_token)
    _REQUEST_MODEL.reset(model_token)


def get_request_api_key_override() -> str | None:
    """Return the request-local OpenAI API key override, if present."""
    return _REQUEST_API_KEY.get()


def get_request_model_override() -> str | None:
    """Return the request-local chat model override, if present."""
    return _REQUEST_MODEL.get()
