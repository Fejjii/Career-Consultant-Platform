"""Speech-to-text HTTP API — media in, transcript out.

Transcripts are not sent through input guards until the user submits them as
normal chat text from the client.
"""

from __future__ import annotations

import time
from typing import Annotated

import structlog
from fastapi import APIRouter, Header, HTTPException, Response, UploadFile

from career_intel.api.deps import SettingsDep, TraceIdDep  # noqa: TC001
from career_intel.llm.clients import get_async_openai_client
from career_intel.schemas.api import ErrorResponse, TranscribeResponse
from career_intel.security.hardening import sanitize_upload_filename
from career_intel.services.speech_service import (
    SpeechTranscriptionError,
    SpeechValidationError,
    normalize_speech_source,
    transcribe_upload_with_logging,
)

router = APIRouter(prefix="/speech", tags=["speech"])
logger = structlog.get_logger()

_STAGE_UPLOAD_VALIDATION = "upload_validation"
_STAGE_PROVIDER_TRANSCRIPTION = "provider_transcription"


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    summary="Transcribe uploaded audio (speech-to-text only)",
    description=(
        "Accepts a single audio file (wav, mp3, m4a, webm). Returns a normalized transcript "
        "for display and editing in the client. Does not invoke chat, routing, or RAG."
    ),
)
async def transcribe_audio(
    file: UploadFile,
    settings: SettingsDep,
    trace_id: TraceIdDep,
    response: Response,
    x_speech_source: Annotated[str | None, Header(alias="X-Speech-Source")] = None,
) -> TranscribeResponse:
    """Validate multipart audio, call the transcription provider, return structured text."""
    filename = sanitize_upload_filename(file.filename, default_name="upload.bin")
    content_type = file.content_type
    data = await file.read()
    speech_src = normalize_speech_source(x_speech_source)

    logger.info(
        "speech_upload_received",
        filename=filename,
        content_type=content_type,
        bytes=len(data),
        source=speech_src,
        trace_id=trace_id,
    )

    client = get_async_openai_client(
        settings,
        timeout_seconds=settings.speech_transcription_timeout_seconds,
    )

    t0 = time.monotonic()
    try:
        payload = await transcribe_upload_with_logging(
            data=data,
            filename=filename,
            content_type=content_type,
            settings=settings,
            client=client,
            speech_source=speech_src,
        )
    except SpeechValidationError as exc:
        detail = f"{_STAGE_UPLOAD_VALIDATION}: {exc.message}"
        logger.warning(
            "speech_transcription_http_error",
            stage=_STAGE_UPLOAD_VALIDATION,
            detail=detail,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    except SpeechTranscriptionError as exc:
        detail = "Transcription service unavailable. Please try again later."
        if exc.error_type == "AuthenticationError":
            detail = "Speech transcription is unavailable because provider credentials are invalid."
        elif exc.error_type == "APIConnectionError":
            detail = "Speech transcription could not reach the provider. Please try again later."
        elif exc.error_type == "APITimeoutError":
            detail = "Speech transcription timed out. Please try again."
        staged_detail = f"{_STAGE_PROVIDER_TRANSCRIPTION}: {detail}"
        logger.error(
            "speech_transcription_http_error",
            stage=_STAGE_PROVIDER_TRANSCRIPTION,
            detail=staged_detail,
            error_type=exc.error_type,
            trace_id=trace_id,
        )
        raise HTTPException(
            status_code=502,
            detail=staged_detail,
        ) from exc

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    body = TranscribeResponse.model_validate(payload)
    logger.info(
        "speech_transcription_response_ready",
        stage="backend_response",
        trace_id=trace_id,
        latency_ms=latency_ms,
        transcript_chars=len(body.text),
        warnings_count=len(body.warnings),
    )

    if settings.environment == "development":
        response.headers["X-Transcription-Latency-Ms"] = str(latency_ms)
    return body
