"""Unit tests for Streamlit speech session helpers (no Streamlit runtime)."""

from __future__ import annotations

from speech_session import (
    SPEECH_STAGE_RECORDING,
    SPEECH_STAGE_TRANSCRIPT_RECEIVED,
    SPEECH_STATUS_IDLE,
    SPEECH_STATUS_READY,
    apply_deferred_speech_clear,
    clear_speech_draft,
    ensure_speech_session_keys,
    queue_chat_message,
    schedule_clear_speech_draft,
    set_speech_stage,
)


def test_ensure_speech_session_keys_initialises() -> None:
    state: dict = {}
    ensure_speech_session_keys(state)
    assert state["speech_transcript_draft"] == ""
    assert state["speech_status"] == SPEECH_STATUS_IDLE
    assert state["speech_last_error"] is None
    assert state["speech_mic_key"] == 0
    assert state["speech_last_source"] is None
    assert state["speech_stage"] == "idle"
    assert state["speech_stage_history"] == []


def test_clear_speech_draft_resets() -> None:
    state: dict = {
        "speech_transcript_draft": "hello",
        "speech_status": SPEECH_STATUS_READY,
        "speech_last_error": "x",
        "speech_last_upload_name": "a.wav",
        "speech_last_source": "mic",
        "speech_mic_key": 3,
    }
    clear_speech_draft(state)
    assert state["speech_transcript_draft"] == ""
    assert state["speech_status"] == SPEECH_STATUS_IDLE
    assert state["speech_last_error"] is None
    assert state["speech_last_upload_name"] is None
    assert state["speech_last_source"] is None
    assert state["speech_mic_key"] == 4


def test_queue_chat_message_sets_pending() -> None:
    state: dict = {}
    queue_chat_message(state, "  ask this  ")
    assert state["pending_user_message"] == "ask this"


def test_queue_chat_message_ignores_blank() -> None:
    state: dict = {}
    queue_chat_message(state, "   ")
    assert "pending_user_message" not in state


def test_schedule_then_apply_deferred_clear() -> None:
    state: dict = {
        "speech_transcript_draft": "x",
        "speech_status": SPEECH_STATUS_READY,
        "speech_last_error": "e",
        "speech_last_upload_name": "f.wav",
        "speech_last_source": "mic",
        "speech_mic_key": 1,
    }
    schedule_clear_speech_draft(state)
    assert apply_deferred_speech_clear(state) is True
    assert state["speech_transcript_draft"] == ""
    assert state["speech_mic_key"] == 2
    assert apply_deferred_speech_clear(state) is False


def test_set_speech_stage_tracks_order_and_current_stage() -> None:
    state: dict = {}
    ensure_speech_session_keys(state)
    set_speech_stage(state, SPEECH_STAGE_RECORDING)
    set_speech_stage(state, SPEECH_STAGE_TRANSCRIPT_RECEIVED)
    assert state["speech_stage"] == SPEECH_STAGE_TRANSCRIPT_RECEIVED
    assert state["speech_stage_history"] == [SPEECH_STAGE_RECORDING, SPEECH_STAGE_TRANSCRIPT_RECEIVED]
