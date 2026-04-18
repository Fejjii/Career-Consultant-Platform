"""Session-state helpers for speech-to-text in the Streamlit UI.

Kept separate from ``app.py`` for small, testable helpers without importing Streamlit.
"""

from __future__ import annotations

SPEECH_STATUS_IDLE = "idle"
SPEECH_STATUS_TRANSCRIBING = "transcribing"
SPEECH_STATUS_READY = "ready"
SPEECH_STATUS_ERROR = "error"

SPEECH_STAGE_IDLE = "idle"
SPEECH_STAGE_RECORDING = "recording"
SPEECH_STAGE_UPLOADING = "uploading"
SPEECH_STAGE_TRANSCRIBING = "transcribing"
SPEECH_STAGE_TRANSCRIPT_RECEIVED = "transcript_received"
SPEECH_STAGE_TRANSCRIPT_INSERTED = "transcript_inserted"
SPEECH_STAGE_FAILED = "transcription_failed"

SPEECH_SOURCE_MIC = "mic"
SPEECH_SOURCE_UPLOAD = "upload"

# Set to True; next script run clears draft before ``st.text_area(key=...)`` is created.
_DEFER_CLEAR_KEY = "_defer_clear_speech_draft"


def ensure_speech_session_keys(state: dict[str, object]) -> None:
    """Initialise speech-related session keys if missing."""
    if "speech_transcript_draft" not in state:
        state["speech_transcript_draft"] = ""
    if "speech_status" not in state:
        state["speech_status"] = SPEECH_STATUS_IDLE
    if "speech_last_error" not in state:
        state["speech_last_error"] = None
    if "speech_last_upload_name" not in state:
        state["speech_last_upload_name"] = None
    if "speech_last_source" not in state:
        state["speech_last_source"] = None
    if "speech_mic_key" not in state:
        state["speech_mic_key"] = 0
    if "speech_stage" not in state:
        state["speech_stage"] = SPEECH_STAGE_IDLE
    if "speech_failed_stage" not in state:
        state["speech_failed_stage"] = None
    if "speech_stage_history" not in state:
        state["speech_stage_history"] = []
    if "speech_auto_send_pending" not in state:
        state["speech_auto_send_pending"] = False


def set_speech_stage(
    state: dict[str, object],
    stage: str,
    *,
    failed_stage: str | None = None,
) -> None:
    """Persist the current stage and append it to the timeline shown in the UI."""

    state["speech_stage"] = stage
    state["speech_failed_stage"] = failed_stage
    history = [str(item) for item in state.get("speech_stage_history", []) if str(item) != stage]
    history.append(stage)
    state["speech_stage_history"] = history


def schedule_clear_speech_draft(state: dict[str, object]) -> None:
    """Request a full speech draft reset on the next run (before transcript widgets mount)."""
    state[_DEFER_CLEAR_KEY] = True


def apply_deferred_speech_clear(state: dict[str, object]) -> bool:
    """If a clear was scheduled, run ``clear_speech_draft``. Call only before bound transcript widgets."""
    if not state.pop(_DEFER_CLEAR_KEY, False):
        return False
    clear_speech_draft(state)
    return True


def clear_speech_draft(state: dict[str, object]) -> None:
    """Reset speech UI state (does not affect chat or CV).

    Bumps ``speech_mic_key`` so the audiorec iframe remounts without re-sending the last clip.
    """
    state["speech_transcript_draft"] = ""
    state["speech_status"] = SPEECH_STATUS_IDLE
    state["speech_last_error"] = None
    state["speech_last_upload_name"] = None
    state["speech_last_source"] = None
    state["speech_mic_key"] = int(state.get("speech_mic_key", 0)) + 1
    state["speech_stage"] = SPEECH_STAGE_IDLE
    state["speech_failed_stage"] = None
    state["speech_stage_history"] = []
    state["speech_auto_send_pending"] = False


def queue_chat_message(state: dict[str, object], text: str) -> None:
    """Stage a user message to be sent on the next run through the normal chat path."""
    stripped = text.strip()
    if not stripped:
        return
    state["pending_user_message"] = stripped
