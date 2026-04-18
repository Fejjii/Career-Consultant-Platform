"""Tests for Streamlit speech HTTP helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from speech_client import post_speech_transcribe


def test_post_speech_transcribe_sends_source_header() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"text": "hi", "provider": "openai", "language": None, "duration_seconds": None, "warnings": []}

    with patch("speech_client.httpx.post", return_value=mock_resp) as post:
        out = post_speech_transcribe(
            api_base="http://test",
            session_id="sid-1",
            audio_bytes=b"wavdata",
            filename="a.wav",
            content_type="audio/wav",
            source="mic",
        )

    assert out["text"] == "hi"
    _args, kwargs = post.call_args
    assert kwargs["headers"]["X-Speech-Source"] == "mic"
    assert kwargs["headers"]["X-Session-ID"] == "sid-1"


def test_post_speech_transcribe_forwards_selected_model_and_key() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"text": "hi", "provider": "openai", "language": None, "duration_seconds": None, "warnings": []}

    with patch("speech_client.httpx.post", return_value=mock_resp) as post:
        post_speech_transcribe(
            api_base="http://test",
            session_id="sid-2",
            audio_bytes=b"wavdata",
            filename="a.wav",
            content_type="audio/wav",
            source="mic",
            model="gpt-4.1",
            api_key="sk-test",
        )

    _args, kwargs = post.call_args
    assert kwargs["headers"]["X-OpenAI-Model"] == "gpt-4.1"
    assert kwargs["headers"]["X-OpenAI-API-Key"] == "sk-test"
