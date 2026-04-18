"""Speech-to-text upload handling: validation, OpenAI transcription, safe logging.

This module is intentionally isolated from orchestration, routing, retrieval,
and synthesis. It only validates audio bytes and returns a normalized transcript
for the client to review before sending as normal chat input.
"""

from __future__ import annotations

import io
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import structlog
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from career_intel.config import Settings
from career_intel.security.hardening import sanitize_upload_filename

logger = structlog.get_logger()

PROVIDER_OPENAI = "openai"

# Extension -> optional magic-byte probe (False means skip probe; rely on provider).
_EXT_MAGIC: dict[str, Callable[[bytes], bool] | None] = {
    "wav": lambda b: len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE",
    "mp3": lambda b: len(b) >= 3
    and (b[:3] == b"ID3" or (b[0] == 0xFF and (b[1] & 0xE0) == 0xE0)),
    "m4a": lambda b: len(b) >= 12 and b[4:8] == b"ftyp",
    "webm": lambda b: len(b) >= 4 and b[0] == 0x1A and b[1:4] == b"\x45\xdf\xa3",
    "mp4": lambda b: len(b) >= 12 and b[4:8] == b"ftyp",
}

_ALLOWED_MIME_EXACT = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/webm",
    }
)


class SpeechValidationError(Exception):
    """Client-side validation failure (4xx)."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SpeechTranscriptionError(Exception):
    """Upstream transcription failure (5xx)."""

    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


@dataclass(frozen=True, slots=True)
class TranscriptionOutcome:
    """Result of a successful provider call (before empty-text guard)."""

    text: str
    language: str | None
    duration_seconds: float | None
    warnings: list[str]


def parse_allowed_extensions(csv: str) -> frozenset[str]:
    """Parse a comma-separated extension list from settings."""
    parts = [p.strip().lower().lstrip(".") for p in csv.split(",") if p.strip()]
    return frozenset(parts)


def _normalize_mime(mime: str | None) -> str | None:
    if not mime:
        return None
    return mime.split(";")[0].strip().lower()


def _mime_allowed(mime: str | None, ext: str) -> bool:
    """Allow known audio types, octet-stream, or other audio/* when extension is allowed."""
    if mime is None:
        return True
    if mime == "application/octet-stream":
        return ext in _EXT_MAGIC
    if mime == "video/mp4":
        return ext == "mp4"
    return mime in _ALLOWED_MIME_EXACT or (
        mime.startswith("audio/") and ext in _EXT_MAGIC
    )


def validate_audio_bytes(
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
    settings: Settings,
) -> str:
    """Validate upload; return normalized lower-case extension.

    Raises:
        SpeechValidationError: empty file, oversize, bad extension/mime, corrupt header.
    """
    logger.info(
        "speech_validation_start",
        filename=filename,
        content_type=_normalize_mime(content_type) or "unknown",
        bytes=len(data),
    )

    if not data:
        raise SpeechValidationError("Audio file is empty.", status_code=400)

    if len(data) > settings.max_speech_file_bytes:
        raise SpeechValidationError(
            f"Audio exceeds maximum size of {settings.max_speech_file_bytes} bytes.",
            status_code=413,
        )

    ext = Path(filename or "").suffix.lower().lstrip(".")
    allowed = parse_allowed_extensions(settings.speech_allowed_extensions)
    if not ext or ext not in allowed:
        raise SpeechValidationError(
            f"Unsupported audio type. Allowed extensions: {', '.join(sorted(allowed))}.",
            status_code=400,
        )

    mime = _normalize_mime(content_type)
    if mime and not _mime_allowed(mime, ext):
        raise SpeechValidationError(
            f"Content-Type not allowed for audio upload: {mime}",
            status_code=400,
        )

    probe = _EXT_MAGIC.get(ext)
    if probe is not None and not probe(data):
        raise SpeechValidationError(
            "File does not appear to be valid audio for the declared format.",
            status_code=400,
        )

    logger.info(
        "speech_validation_complete",
        filename=filename,
        content_type=_normalize_mime(content_type) or "unknown",
        bytes=len(data),
        extension=ext,
    )
    return ext


def normalize_transcript_text(text: str) -> str:
    """Collapse whitespace; strip ends."""
    return " ".join(text.split()).strip()


async def transcribe_with_openai(
    *,
    data: bytes,
    filename: str,
    settings: Settings,
    client: AsyncOpenAI,
) -> TranscriptionOutcome:
    """Call OpenAI audio transcriptions API (isolated from chat models)."""
    warnings: list[str] = []
    file_tuple = (filename, io.BytesIO(data))
    try:
        logger.info(
            "speech_provider_request_start",
            provider=PROVIDER_OPENAI,
            filename=filename,
            bytes=len(data),
            model=settings.openai_transcription_model,
        )
        response = await client.audio.transcriptions.create(
            file=file_tuple,
            model=settings.openai_transcription_model,
            response_format="verbose_json",
        )
    except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
        et = type(exc).__name__
        raise SpeechTranscriptionError(
            "Transcription provider request failed.",
            error_type=et,
        ) from exc
    except Exception as exc:
        raise SpeechTranscriptionError(
            "Transcription provider request failed.",
            error_type=type(exc).__name__,
        ) from exc

    raw = (response.text or "").strip()
    language = getattr(response, "language", None)
    duration = getattr(response, "duration", None)
    if language is None:
        warnings.append("language_not_reported")
    if duration is None:
        warnings.append("duration_not_reported")

    logger.info(
        "speech_provider_response_received",
        provider=PROVIDER_OPENAI,
        filename=filename,
        transcript_chars=len(raw),
        language=language,
        duration_seconds=duration,
        warnings_count=len(warnings),
    )
    return TranscriptionOutcome(
        text=raw,
        language=language,
        duration_seconds=float(duration) if duration is not None else None,
        warnings=warnings,
    )


async def run_transcription(
    *,
    data: bytes,
    filename: str,
    content_type: str | None,
    settings: Settings,
    client: AsyncOpenAI,
    transcriber: Callable[..., Awaitable[TranscriptionOutcome]] | None = None,
) -> tuple[str, str | None, float | None, list[str]]:
    """Validate audio, transcribe, normalize text.

    Returns:
        Tuple of (normalized_text, language, duration_seconds, warnings).

    Raises:
        SpeechValidationError: invalid upload.
        SpeechTranscriptionError: provider failure.
    """
    ext = Path(filename or "audio").suffix.lower().lstrip(".")
    safe_name = sanitize_upload_filename(
        filename,
        default_name=f"upload.{ext or 'bin'}",
    )

    validate_audio_bytes(
        filename=safe_name,
        content_type=content_type,
        data=data,
        settings=settings,
    )

    fn: Callable[..., Awaitable[TranscriptionOutcome]] = (
        transcriber if transcriber is not None else transcribe_with_openai
    )
    outcome = await fn(data=data, filename=safe_name, settings=settings, client=client)
    normalized = normalize_transcript_text(outcome.text)
    if not normalized:
        raise SpeechValidationError(
            "Transcription returned no usable text.",
            status_code=422,
        )

    return normalized, outcome.language, outcome.duration_seconds, outcome.warnings


def normalize_speech_source(value: str | None) -> str:
    """Normalise client-reported speech input channel for logs (mic, upload, unknown)."""
    if not value:
        return "unknown"
    v = value.strip().lower()
    if v in ("mic", "upload"):
        return v
    return "unknown"


async def transcribe_upload_with_logging(
    *,
    data: bytes,
    filename: str,
    content_type: str | None,
    settings: Settings,
    client: AsyncOpenAI,
    speech_source: str = "unknown",
    transcriber: Callable[..., Awaitable[TranscriptionOutcome]] | None = None,
) -> dict[str, object]:
    """Run transcription with structured observability (no raw audio or transcript in logs)."""
    mime = _normalize_mime(content_type) or "unknown"
    safe_filename = filename or "unknown"
    src = normalize_speech_source(speech_source)
    t0 = time.monotonic()

    logger.info(
        "transcription_start",
        filename=safe_filename,
        mime_type=mime,
        bytes=len(data),
        provider=PROVIDER_OPENAI,
        source=src,
    )

    try:
        text, language, duration, warnings = await run_transcription(
            data=data,
            filename=filename,
            content_type=content_type,
            settings=settings,
            client=client,
            transcriber=transcriber,
        )
    except SpeechValidationError as exc:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        err_kind = "empty_transcript" if exc.status_code == 422 else "validation"
        logger.warning(
            "transcription_failed",
            filename=safe_filename,
            mime_type=mime,
            bytes=len(data),
            provider=PROVIDER_OPENAI,
            source=src,
            latency_ms=latency_ms,
            success=False,
            warnings_count=0,
            error_type=err_kind,
        )
        raise
    except SpeechTranscriptionError as exc:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        logger.error(
            "transcription_failed",
            filename=safe_filename,
            mime_type=mime,
            bytes=len(data),
            provider=PROVIDER_OPENAI,
            source=src,
            latency_ms=latency_ms,
            success=False,
            warnings_count=0,
            error_type=exc.error_type,
        )
        raise
    else:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        logger.info(
            "transcription_complete",
            filename=safe_filename,
            mime_type=mime,
            bytes=len(data),
            provider=PROVIDER_OPENAI,
            source=src,
            latency_ms=latency_ms,
            success=True,
            warnings_count=len(warnings),
            error_type=None,
        )
        return {
            "text": text,
            "provider": PROVIDER_OPENAI,
            "language": language,
            "duration_seconds": duration,
            "warnings": warnings,
        }
