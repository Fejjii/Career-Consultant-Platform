"""HTTP client helpers for speech transcription from the Streamlit app."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from api_client import build_request_headers

logger = logging.getLogger(__name__)


def post_speech_transcribe(
    *,
    api_base: str,
    session_id: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    source: str,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 130.0,
) -> dict[str, Any]:
    """POST multipart audio to ``/speech/transcribe``. Raises ``httpx.HTTPError`` on failure."""
    logger.info(
        "speech frontend stage=upload_request source=%s filename=%s content_type=%s bytes=%s",
        source,
        filename,
        content_type,
        len(audio_bytes),
    )
    resp = httpx.post(
        f"{api_base}/speech/transcribe",
        files={"file": (filename, audio_bytes, content_type)},
        headers=build_request_headers(
            session_id=session_id,
            model=model,
            api_key=api_key,
            extra_headers={"X-Speech-Source": source},
        ),
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    logger.info(
        "speech frontend stage=backend_response source=%s filename=%s transcript_chars=%s",
        source,
        filename,
        len(str(payload.get("text", ""))),
    )
    return payload
