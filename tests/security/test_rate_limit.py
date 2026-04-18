"""Tests for endpoint-specific rate-limit policies."""

from __future__ import annotations

from types import SimpleNamespace

from career_intel.security.rate_limit import resolve_rate_limit_policy


def _request(path: str) -> SimpleNamespace:
    return SimpleNamespace(url=SimpleNamespace(path=path))


def test_chat_rate_limit_policy(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "career_intel.security.rate_limit.get_settings",
        lambda: SimpleNamespace(
            rate_limit_rpm=30,
            rate_limit_chat_rpm=30,
            rate_limit_speech_rpm=10,
            rate_limit_byok_validation_rpm=10,
            rate_limit_feedback_rpm=20,
            rate_limit_ingest_rpm=5,
        ),
    )
    policy = resolve_rate_limit_policy(_request("/chat"))
    assert policy.scope == "chat"
    assert policy.rpm == 30


def test_speech_rate_limit_policy(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "career_intel.security.rate_limit.get_settings",
        lambda: SimpleNamespace(
            rate_limit_rpm=30,
            rate_limit_chat_rpm=30,
            rate_limit_speech_rpm=10,
            rate_limit_byok_validation_rpm=10,
            rate_limit_feedback_rpm=20,
            rate_limit_ingest_rpm=5,
        ),
    )
    policy = resolve_rate_limit_policy(_request("/speech/transcribe"))
    assert policy.scope == "speech"
    assert policy.rpm == 10


def test_provider_auth_rate_limit_policy(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "career_intel.security.rate_limit.get_settings",
        lambda: SimpleNamespace(
            rate_limit_rpm=30,
            rate_limit_chat_rpm=30,
            rate_limit_speech_rpm=10,
            rate_limit_byok_validation_rpm=10,
            rate_limit_feedback_rpm=20,
            rate_limit_ingest_rpm=5,
        ),
    )
    policy = resolve_rate_limit_policy(_request("/health/provider-auth"))
    assert policy.scope == "provider_auth"
    assert policy.rpm == 10
