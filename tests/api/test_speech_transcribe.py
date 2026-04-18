"""Tests for POST /speech/transcribe (mocked OpenAI; no real network)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient  # noqa: TC002

from career_intel.config.settings import get_settings
from career_intel.services.speech_service import (
    SpeechTranscriptionError,
    TranscriptionOutcome,
)


def _minimal_wav_header() -> bytes:
    """12+ bytes that satisfy the WAV magic-byte probe."""
    return b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 40


@pytest.fixture
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_transcribe_success(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    clear_settings_cache: None,
) -> None:
    async def _fake_transcribe(
        *,
        data: bytes,
        filename: str,
        settings: object,
        client: object,
    ) -> TranscriptionOutcome:
        return TranscriptionOutcome(
            text="  Plan my next role  ",
            language="en",
            duration_seconds=2.5,
            warnings=[],
        )

    monkeypatch.setattr(
        "career_intel.services.speech_service.transcribe_with_openai",
        _fake_transcribe,
    )

    body = _minimal_wav_header()
    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("note.wav", body, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Plan my next role"
    assert data["provider"] == "openai"
    assert data["language"] == "en"
    assert data["duration_seconds"] == 2.5
    assert data["warnings"] == []


@pytest.mark.asyncio
async def test_transcribe_invalid_extension(client: AsyncClient) -> None:
    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("x.flac", b"abc", "audio/flac")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("upload_validation:")


@pytest.mark.asyncio
async def test_transcribe_empty_file(client: AsyncClient) -> None:
    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_transcribe_oversized(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    clear_settings_cache: None,
) -> None:
    monkeypatch.setenv("MAX_SPEECH_FILE_BYTES", "50")
    get_settings.cache_clear()

    body = _minimal_wav_header() + b"x" * 100
    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("big.wav", body, "audio/wav")},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_transcribe_corrupt_wav(client: AsyncClient) -> None:
    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("bad.wav", b"NOTWAVDATAHERE", "audio/wav")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_transcribe_provider_failure(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    clear_settings_cache: None,
) -> None:
    async def _boom(
        *,
        data: bytes,
        filename: str,
        settings: object,
        client: object,
    ) -> TranscriptionOutcome:
        raise SpeechTranscriptionError(
            "simulated provider failure",
            error_type="ProviderError",
        )

    monkeypatch.setattr(
        "career_intel.services.speech_service.transcribe_with_openai",
        _boom,
    )

    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("note.wav", _minimal_wav_header(), "audio/wav")},
    )
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_transcribe_authentication_failure_returns_specific_detail(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    clear_settings_cache: None,
) -> None:
    async def _boom_auth(
        *,
        data: bytes,
        filename: str,
        settings: object,
        client: object,
    ) -> TranscriptionOutcome:
        raise SpeechTranscriptionError(
            "simulated auth failure",
            error_type="AuthenticationError",
        )

    monkeypatch.setattr(
        "career_intel.services.speech_service.transcribe_with_openai",
        _boom_auth,
    )

    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("note.wav", _minimal_wav_header(), "audio/wav")},
    )
    assert resp.status_code == 502
    assert resp.json()["detail"].startswith("provider_transcription:")
    assert "provider credentials are invalid" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_transcribe_empty_transcript_normalized(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    clear_settings_cache: None,
) -> None:
    async def _empty(
        *,
        data: bytes,
        filename: str,
        settings: object,
        client: object,
    ) -> TranscriptionOutcome:
        return TranscriptionOutcome(
            text="   \n\t  ",
            language=None,
            duration_seconds=None,
            warnings=["duration_not_reported"],
        )

    monkeypatch.setattr(
        "career_intel.services.speech_service.transcribe_with_openai",
        _empty,
    )

    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("note.wav", _minimal_wav_header(), "audio/wav")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_transcribe_accepts_octet_stream_with_valid_extension(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    clear_settings_cache: None,
) -> None:
    async def _ok(
        *,
        data: bytes,
        filename: str,
        settings: object,
        client: object,
    ) -> TranscriptionOutcome:
        return TranscriptionOutcome(text="ok", language=None, duration_seconds=None, warnings=[])

    monkeypatch.setattr(
        "career_intel.services.speech_service.transcribe_with_openai",
        _ok,
    )

    resp = await client.post(
        "/speech/transcribe",
        files={"file": ("note.wav", _minimal_wav_header(), "application/octet-stream")},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "ok"
