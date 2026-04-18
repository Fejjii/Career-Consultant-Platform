"""Tests for st_audiorec payload decoding (no browser)."""

from __future__ import annotations

from audiorec_bridge import wav_bytes_from_audiorec_payload


def test_wav_bytes_from_payload_sorts_indices() -> None:
    raw = {"arr": {"2": 67, "0": 65, "1": 66}}
    assert wav_bytes_from_audiorec_payload(raw) == b"ABC"


def test_wav_bytes_single_chunk() -> None:
    raw = {"arr": {"0": 255}}
    assert wav_bytes_from_audiorec_payload(raw) == b"\xff"
