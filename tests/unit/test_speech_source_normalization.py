"""Speech source header normalisation."""

from __future__ import annotations

from career_intel.services.speech_service import normalize_speech_source


def test_normalize_speech_source() -> None:
    assert normalize_speech_source(None) == "unknown"
    assert normalize_speech_source("") == "unknown"
    assert normalize_speech_source("mic") == "mic"
    assert normalize_speech_source("UPLOAD") == "upload"
    assert normalize_speech_source("  upload  ") == "upload"
    assert normalize_speech_source("evil") == "unknown"
