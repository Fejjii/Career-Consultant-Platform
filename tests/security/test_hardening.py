"""Tests for shared security hardening helpers."""

from __future__ import annotations

from career_intel.security.hardening import (
    mask_secret,
    redact_log_event,
    sanitize_public_uri,
)


def test_mask_secret_hides_middle_characters() -> None:
    masked = mask_secret("sk-testsecret1234567890")
    assert masked.startswith("sk-t")
    assert masked.endswith("7890")
    assert "secret1234" not in masked


def test_redact_log_event_masks_secret_fields() -> None:
    event = redact_log_event(
        None,
        "info",
        {
            "api_key": "sk-testsecret1234567890",
            "query_preview": "Tell me about product manager skills",
        },
    )
    assert event["api_key"] != "sk-testsecret1234567890"
    assert "product manager skills" not in str(event["query_preview"])


def test_sanitize_public_uri_blocks_local_paths() -> None:
    assert sanitize_public_uri("C:/private/file.pdf") is None
    assert sanitize_public_uri("file:///tmp/file.pdf") is None
    assert sanitize_public_uri("https://example.com/doc.pdf") == "https://example.com/doc.pdf"
