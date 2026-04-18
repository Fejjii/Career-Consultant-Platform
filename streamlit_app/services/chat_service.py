"""Direct in-process backend execution for Streamlit deployments.

This module lets the Streamlit app reuse backend orchestration without HTTP.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    from streamlit_app.runtime_config import resolve_openai_api_key, resolve_qdrant_config
except ImportError:  # pragma: no cover - supports `streamlit run streamlit_app/app.py`
    from runtime_config import resolve_openai_api_key, resolve_qdrant_config

def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()

from career_intel.config import Settings, get_settings
from career_intel.llm.clients import validate_chat_model_override
from career_intel.llm.request_context import reset_request_llm_overrides, set_request_llm_overrides
from career_intel.schemas.api import ChatMessage, ChatResponse
from career_intel.schemas.domain import RetrievedChunk, RouterDecision
from career_intel.security.sanitize import score_cv_risk
from career_intel.services.cv_processor import CVProcessingError, process_cv


class DirectModeError(Exception):
    """Raised when direct in-process execution fails."""


def _extract_missing_setting_names(exc: Exception) -> list[str]:
    if not isinstance(exc, ValidationError):
        return []
    missing: list[str] = []
    for err in exc.errors():
        err_type = str(err.get("type", ""))
        if err_type != "missing":
            continue
        loc = err.get("loc") or ()
        if not loc:
            continue
        name = str(loc[-1]).upper()
        if name not in missing:
            missing.append(name)
    return missing


def _settings_validation_message(exc: Exception) -> str:
    missing = _extract_missing_setting_names(exc)
    if missing:
        return "Missing required secret(s): " + ", ".join(missing)
    return f"Configuration is invalid: {exc}"


def validate_llm_config(*, api_key_override: str | None = None) -> str | None:
    """Validate chat generation configuration and return an actionable message."""
    resolved = resolve_openai_api_key(user_api_key=api_key_override)
    if resolved.api_key:
        return None
    return (
        "Add your OpenAI API key in the sidebar or configure OPENAI_API_KEY in Streamlit secrets."
    )


def validate_retrieval_config() -> str | None:
    """Validate retrieval configuration and return an actionable message."""
    resolved = resolve_qdrant_config()
    if resolved.retrieval_available:
        return None
    return resolved.message


def validate_speech_config(*, api_key_override: str | None = None) -> str | None:
    """Validate speech transcription configuration and return an actionable message."""
    # Speech currently depends on OpenAI transcription APIs.
    return validate_llm_config(api_key_override=api_key_override)


@contextmanager
def _llm_override_context(
    *,
    settings: Settings,
    model: str | None,
    api_key: str | None,
):
    validated_model = validate_chat_model_override(model, settings)
    tokens = set_request_llm_overrides(api_key=api_key, model=validated_model)
    try:
        yield
    finally:
        reset_request_llm_overrides(*tokens)


class DirectChatService:
    """In-process adapter around backend pipeline entry points."""

    def __init__(self) -> None:
        self._settings: Settings | None = None

    def _get_settings(self) -> Settings:
        if self._settings is not None:
            return self._settings
        try:
            self._settings = get_settings()
        except Exception as exc:
            raise DirectModeError(_settings_validation_message(exc)) from exc
        return self._settings

    def _validate_or_raise(
        self,
        *,
        validator: str,
        api_key: str | None = None,
    ) -> None:
        message: str | None
        if validator == "llm":
            message = validate_llm_config(api_key_override=api_key)
        elif validator == "retrieval":
            message = validate_retrieval_config()
        elif validator == "speech":
            message = validate_speech_config(api_key_override=api_key)
        else:
            message = None
        if message:
            raise DirectModeError(message)

    @property
    def settings(self) -> Settings:
        return self._get_settings()

    async def route_query(
        self,
        *,
        query: str,
        cv_available: bool,
        model: str | None,
        api_key: str | None,
    ) -> tuple[RouterDecision, dict[str, Any] | None]:
        from career_intel.tools.registry import route_query as backend_route_query

        self._validate_or_raise(validator="llm", api_key=api_key)
        settings = self._get_settings()
        with _llm_override_context(settings=settings, model=model, api_key=api_key):
            decision, usage = await backend_route_query(
                query,
                cv_available=cv_available,
                settings=settings,
            )
        return decision, usage.model_dump() if usage is not None else None

    async def run_rag(
        self,
        *,
        query: str,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        from career_intel.rag.query_preprocessor import normalize_query_for_retrieval
        from career_intel.rag.retriever import retrieve_chunks, rewrite_query

        self._validate_or_raise(validator="retrieval")
        settings = self._get_settings()
        retrieval_context = await normalize_query_for_retrieval(query, settings=settings)
        rewritten = await rewrite_query(retrieval_context.retrieval_query, settings=settings)
        return await retrieve_chunks(query=rewritten, filters=filters, settings=settings)

    async def run_tool(
        self,
        *,
        decision: RouterDecision,
    ) -> dict[str, Any]:
        from career_intel.tools.registry import execute_tool

        settings = self._get_settings()
        result = await execute_tool(decision, settings)
        return result.model_dump()

    async def run_fallback(
        self,
        *,
        query: str,
        answer_length: str,
        model: str | None,
        api_key: str | None,
    ) -> tuple[str, dict[str, Any] | None]:
        from career_intel.orchestration.synthesize import generate_direct_response

        self._validate_or_raise(validator="llm", api_key=api_key)
        settings = self._get_settings()
        with _llm_override_context(settings=settings, model=model, api_key=api_key):
            reply, usage = await generate_direct_response(
                query,
                settings,
                answer_length=answer_length,  # type: ignore[arg-type]
            )
        return reply, usage.model_dump() if usage is not None else None

    async def generate_response(
        self,
        *,
        body: dict[str, Any],
        model: str | None,
        api_key: str | None,
        user_timezone: str | None,
    ) -> ChatResponse:
        from career_intel.orchestration.chain import run_turn

        try:
            messages = [ChatMessage.model_validate(item) for item in body.get("messages", [])]
        except Exception as exc:  # pragma: no cover - defensive validation
            raise DirectModeError(f"Invalid chat payload: {exc}") from exc

        self._validate_or_raise(validator="llm", api_key=api_key)
        session_id = body.get("session_id") or str(uuid.uuid4())
        settings = self._get_settings()
        with _llm_override_context(settings=settings, model=model, api_key=api_key):
            return await run_turn(
                messages=messages,
                session_id=session_id,
                use_tools=bool(body.get("use_tools", True)),
                filters=body.get("filters"),
                settings=settings,
                trace_id=f"streamlit-direct-{session_id}",
                cv_text=body.get("cv_text"),
                user_timezone=user_timezone,
                answer_length=body.get("answer_length", "balanced"),
            )

    async def discover_provider_models(
        self,
        *,
        model: str | None,
        api_key: str | None,
    ) -> dict[str, Any]:
        from career_intel.api.routers.health import provider_auth_status

        self._validate_or_raise(validator="llm", api_key=api_key)
        settings = self._get_settings()
        with _llm_override_context(settings=settings, model=model, api_key=api_key):
            status = await provider_auth_status()
        return status.model_dump()

    async def get_system_status(self) -> dict[str, Any]:
        retrieval_error = validate_retrieval_config()
        if retrieval_error:
            return {
                "backend": True,
                "qdrant": False,
                "indexed_data_present": False,
                "collection": "unknown",
                "points_count": 0,
                "error": retrieval_error,
            }
        try:
            from career_intel.api.routers.health import system_status

            status = await system_status()
            return status.model_dump()
        except Exception as exc:
            return {
                "backend": True,
                "qdrant": False,
                "indexed_data_present": False,
                "collection": "unknown",
                "points_count": 0,
                "error": f"Qdrant is unavailable right now: {exc}",
            }

    async def get_source_inventory(self) -> dict[str, Any]:
        retrieval_error = validate_retrieval_config()
        if retrieval_error:
            return {
                "total_source_groups": 0,
                "total_files_present": 0,
                "esco_status_note": retrieval_error,
            }
        try:
            from career_intel.api.routers.health import source_inventory

            inventory = await source_inventory()
            return inventory.model_dump()
        except Exception:
            return {
                "total_source_groups": 0,
                "total_files_present": 0,
                "esco_status_note": "Source inventory is temporarily unavailable.",
            }

    async def process_cv_upload(self, *, filename: str, data: bytes) -> dict[str, Any]:
        max_bytes = 5 * 1024 * 1024
        if self._settings is not None:
            max_bytes = self._settings.max_cv_file_bytes
        try:
            cv_text = process_cv(
                data,
                filename,
                max_file_bytes=max_bytes,
            )
        except CVProcessingError as exc:
            raise DirectModeError(str(exc)) from exc

        risk = score_cv_risk(cv_text)
        warnings: list[str] = []
        if risk.flagged:
            warnings.append(
                f"CV scored {risk.score:.2f} risk — matched: {', '.join(risk.matched_patterns)}."
            )
        return {
            "cv_text": cv_text,
            "filename": filename,
            "score": risk.score,
            "matched_patterns": risk.matched_patterns,
            "flagged": risk.flagged,
            "warnings": warnings,
        }

    async def transcribe_audio(
        self,
        *,
        file_name: str,
        content_type: str,
        data: bytes,
        source: str,
        model: str | None,
        api_key: str | None,
    ) -> dict[str, Any]:
        from career_intel.llm.clients import get_async_openai_client
        from career_intel.services.speech_service import (
            normalize_speech_source,
            transcribe_upload_with_logging,
        )

        self._validate_or_raise(validator="speech", api_key=api_key)
        settings = self._get_settings()
        with _llm_override_context(settings=settings, model=model, api_key=api_key):
            client = get_async_openai_client(
                settings,
                timeout_seconds=settings.speech_transcription_timeout_seconds,
            )
            return await transcribe_upload_with_logging(
                data=data,
                filename=file_name,
                content_type=content_type,
                settings=settings,
                client=client,
                speech_source=normalize_speech_source(source),
            )


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from Streamlit's sync execution context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fallback for environments with an existing running loop.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()


_DIRECT_CHAT_SERVICE: DirectChatService | None = None


def _service() -> DirectChatService:
    global _DIRECT_CHAT_SERVICE
    if _DIRECT_CHAT_SERVICE is None:
        _DIRECT_CHAT_SERVICE = DirectChatService()
    return _DIRECT_CHAT_SERVICE


def is_direct_mode_enabled() -> bool:
    raw = os.getenv("STREAMLIT_DIRECT_MODE", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def route_query(
    *,
    query: str,
    cv_available: bool,
    model: str | None,
    api_key: str | None,
) -> tuple[RouterDecision, dict[str, Any] | None]:
    return _run_async(
        _service().route_query(
            query=query,
            cv_available=cv_available,
            model=model,
            api_key=api_key,
        )
    )


def run_rag(
    *,
    query: str,
    filters: dict[str, Any] | None,
) -> list[RetrievedChunk]:
    return _run_async(_service().run_rag(query=query, filters=filters))


def run_tool(
    *,
    decision: RouterDecision,
) -> dict[str, Any]:
    return _run_async(_service().run_tool(decision=decision))


def run_fallback(
    *,
    query: str,
    answer_length: str,
    model: str | None,
    api_key: str | None,
) -> tuple[str, dict[str, Any] | None]:
    return _run_async(
        _service().run_fallback(
            query=query,
            answer_length=answer_length,
            model=model,
            api_key=api_key,
        )
    )


def generate_response(
    *,
    body: dict[str, Any],
    model: str | None,
    api_key: str | None,
    user_timezone: str | None,
) -> dict[str, Any]:
    response = _run_async(
        _service().generate_response(
            body=body,
            model=model,
            api_key=api_key,
            user_timezone=user_timezone,
        )
    )
    return response.model_dump()


def discover_provider_models(
    *,
    model: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    return _run_async(_service().discover_provider_models(model=model, api_key=api_key))


def get_system_status() -> dict[str, Any]:
    return _run_async(_service().get_system_status())


def get_source_inventory() -> dict[str, Any]:
    return _run_async(_service().get_source_inventory())


def process_cv_upload(*, filename: str, data: bytes) -> dict[str, Any]:
    return _run_async(_service().process_cv_upload(filename=filename, data=data))


def transcribe_audio(
    *,
    file_name: str,
    content_type: str,
    data: bytes,
    source: str,
    model: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    return _run_async(
        _service().transcribe_audio(
            file_name=file_name,
            content_type=content_type,
            data=data,
            source=source,
            model=model,
            api_key=api_key,
        )
    )
