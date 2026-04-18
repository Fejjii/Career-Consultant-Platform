"""Streamlit frontend for the AI Career Intelligence Assistant.

Layout: sidebar (secondary) · centered chat column · right sources panel (reserved).
Composer: st.chat_input + native browser mic capture with sidebar file fallback.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import time
import uuid
from datetime import datetime

import httpx
import streamlit.components.v1 as components
from api_client import build_request_headers, discover_provider_models, validate_provider_key
from credential_state import (
    APP_MANAGED_SOURCE,
    USER_BYOK_SOURCE,
    resolve_credential_source,
    transition_after_clear,
    transition_after_validation,
)
from model_config import (
    get_available_model_ids,
    get_default_model_id,
    get_model_description,
    get_model_label,
    get_model_pricing,
    get_recommended_model_ids,
    get_supported_model_ids,
    resolve_selected_model,
    summarize_model_availability,
)
from security_controls import apply_session_rate_limit, inspect_prompt, validate_uploaded_file
from sources_panel import (
    family_icon_html,
    family_section_title,
    format_detail_lines_for_code,
    format_youtube_sources_html,
    group_by_family_ordered,
    prepare_sources_panel_rows,
    render_merged_card_html,
)
from speech_client import post_speech_transcribe
from services.chat_service import (
    DirectModeError,
    generate_response as generate_response_direct,
    get_source_inventory as get_source_inventory_direct,
    get_system_status as get_system_status_direct,
    is_direct_mode_enabled,
    process_cv_upload as process_cv_upload_direct,
)
from speech_session import (
    SPEECH_SOURCE_MIC,
    SPEECH_SOURCE_UPLOAD,
    SPEECH_STAGE_FAILED,
    SPEECH_STAGE_IDLE,
    SPEECH_STAGE_RECORDING,
    SPEECH_STAGE_TRANSCRIBING,
    SPEECH_STAGE_TRANSCRIPT_INSERTED,
    SPEECH_STAGE_TRANSCRIPT_RECEIVED,
    SPEECH_STAGE_UPLOADING,
    SPEECH_STATUS_ERROR,
    SPEECH_STATUS_READY,
    SPEECH_STATUS_TRANSCRIBING,
    apply_deferred_speech_clear,
    clear_speech_draft,
    ensure_speech_session_keys,
    queue_chat_message,
    schedule_clear_speech_draft,
    set_speech_stage,
)
from usage_tracking import build_message_usage_fields, estimate_request_usage, update_usage_summary
from youtube_service import fetch_youtube_suggestions, should_fetch_youtube_support

import streamlit as st

STREAMLIT_DIRECT_MODE = is_direct_mode_enabled()
API_BASE = os.getenv("CAREER_INTEL_API_BASE_URL", "").strip().rstrip("/")
DEV_MODE = os.getenv("CAREER_INTEL_DEV_MODE", "").lower() in ("1", "true", "yes")
logger = logging.getLogger(__name__)
_LOCAL_TZ = datetime.now().astimezone().tzinfo
_USER_TIMEZONE = getattr(_LOCAL_TZ, "key", None) if _LOCAL_TZ is not None else None

_SHELL_MAX_W = "1400px"
_ACCENT = "#4f46e5"  # indigo-600
_SIDEBAR_WIDTH_QUERY_PARAM = "sbw"
_SIDEBAR_DEFAULT_WIDTH_PX = 328
_SIDEBAR_MIN_WIDTH_PX = 272
_SIDEBAR_MAX_WIDTH_PX = 460

# Suggested starters (queued like normal user messages)
_STARTER_PROMPTS = (
    (
        "Prioritize skills for PM roles",
        "What skills should I prioritize for a product manager role?",
    ),
    (
        "Senior engineer vs staff expectations",
        "Compare senior engineer vs staff engineer expectations.",
    ),
    (
        "Explain a career gap in interviews",
        "How do I explain a career gap in an interview?",
    ),
)


def _coerce_sidebar_width(value: object) -> int:
    """Clamp potentially untrusted width values into supported bounds."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = _SIDEBAR_DEFAULT_WIDTH_PX
    return max(_SIDEBAR_MIN_WIDTH_PX, min(_SIDEBAR_MAX_WIDTH_PX, parsed))


def _sync_sidebar_width_state_from_query() -> None:
    """Keep sidebar width sticky across reruns by syncing query params to session state."""
    query_value = st.query_params.get(_SIDEBAR_WIDTH_QUERY_PARAM)
    width_from_query = _coerce_sidebar_width(query_value) if query_value else None

    if "sidebar_width_px" not in st.session_state:
        st.session_state.sidebar_width_px = (
            width_from_query if width_from_query is not None else _SIDEBAR_DEFAULT_WIDTH_PX
        )
    elif width_from_query is not None and width_from_query != st.session_state.sidebar_width_px:
        st.session_state.sidebar_width_px = width_from_query

    st.query_params[_SIDEBAR_WIDTH_QUERY_PARAM] = str(st.session_state.sidebar_width_px)


def _reset_sidebar_width_state() -> None:
    """Restore the sidebar width to the default and persist it immediately."""
    st.session_state.sidebar_width_px = _SIDEBAR_DEFAULT_WIDTH_PX
    st.query_params[_SIDEBAR_WIDTH_QUERY_PARAM] = str(_SIDEBAR_DEFAULT_WIDTH_PX)


def _render_custom_css() -> str:
    """Inject runtime sidebar width values into the static CSS template."""
    return (
        _CUSTOM_CSS.replace("__CI_SIDEBAR_WIDTH_PX__", str(st.session_state.sidebar_width_px))
        .replace("__CI_SIDEBAR_MIN_PX__", str(_SIDEBAR_MIN_WIDTH_PX))
        .replace("__CI_SIDEBAR_MAX_PX__", str(_SIDEBAR_MAX_WIDTH_PX))
    )


def _inject_sidebar_resizer_bridge() -> None:
    """Add a resilient resize handle and persist width through the URL query state."""
    components.html(
        f"""
<script>
(function() {{
  const KEY = "{_SIDEBAR_WIDTH_QUERY_PARAM}";
  const minW = {_SIDEBAR_MIN_WIDTH_PX};
  const maxW = {_SIDEBAR_MAX_WIDTH_PX};
  const initial = {int(st.session_state.sidebar_width_px)};
  const root = window.parent?.document;
  if (!root) return;

  const applyWidth = (sidebar, width) => {{
    const clamped = Math.max(minW, Math.min(maxW, Number(width) || initial));
    sidebar.style.setProperty("width", `${{clamped}}px`, "important");
    sidebar.style.setProperty("min-width", `${{minW}}px`, "important");
    sidebar.style.setProperty("max-width", `${{maxW}}px`, "important");
    sidebar.style.setProperty("flex-basis", `${{clamped}}px`, "important");
    return clamped;
  }};

  const setup = () => {{
    const sidebar = root.querySelector('section[data-testid="stSidebar"]');
    if (!sidebar) return false;

    let handle = sidebar.querySelector(".ci-sidebar-resize-handle");
    if (!handle) {{
      handle = root.createElement("div");
      handle.className = "ci-sidebar-resize-handle";
      handle.setAttribute("role", "separator");
      handle.setAttribute("aria-label", "Resize sidebar");
      handle.setAttribute("aria-orientation", "vertical");
      handle.tabIndex = 0;
      sidebar.appendChild(handle);
    }}
    if (!handle) return false;

    let width = applyWidth(sidebar, sidebar.getBoundingClientRect().width || initial);
    const current = new URL(window.parent.location.href).searchParams.get(KEY);
    width = applyWidth(sidebar, current || width);

    if (handle.dataset.bound !== "1") {{
      let dragging = false;
      let activePointerId = null;

      const persist = () => {{
        const url = new URL(window.parent.location.href);
        url.searchParams.set(KEY, String(width));
        window.parent.history.replaceState(null, "", url.toString());
      }};

      const onMove = (event) => {{
        if (!dragging) return;
        width = applyWidth(sidebar, event.clientX);
      }};

      const onUp = () => {{
        if (!dragging) return;
        dragging = false;
        activePointerId = null;
        root.body.classList.remove("ci-sidebar-resizing");
        persist();
      }};

      const startDrag = (event) => {{
        event.preventDefault();
        dragging = true;
        if (typeof event.pointerId === "number") {{
          activePointerId = event.pointerId;
        }}
        root.body.classList.add("ci-sidebar-resizing");
      }};

      handle.addEventListener("mousedown", startDrag);
      handle.addEventListener("pointerdown", (event) => {{
        startDrag(event);
        if (typeof handle.setPointerCapture === "function" && typeof event.pointerId === "number") {{
          handle.setPointerCapture(event.pointerId);
        }}
      }});
      handle.addEventListener("keydown", (event) => {{
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const delta = event.key === "ArrowLeft" ? -12 : 12;
        width = applyWidth(sidebar, (sidebar.getBoundingClientRect().width || width) + delta);
        persist();
      }});
      window.parent.addEventListener("mousemove", onMove);
      window.parent.addEventListener("pointermove", (event) => {{
        if (activePointerId !== null && event.pointerId !== activePointerId) return;
        onMove(event);
      }});
      window.parent.addEventListener("mouseup", onUp);
      window.parent.addEventListener("pointerup", onUp);
      handle.dataset.bound = "1";
    }}

    return true;
  }};

  // Keep width + handle synchronized even when Streamlit rerenders without remounting this iframe.
  if (!window.parent.__ciSidebarResizerHeartbeat) {{
    window.parent.__ciSidebarResizerHeartbeat = window.setInterval(setup, 450);
  }}

  if (setup()) return;

  // Streamlit can mount the sidebar after this iframe script runs.
  let attempts = 0;
  const maxAttempts = 120;
  const timer = window.setInterval(() => {{
    attempts += 1;
    if (setup() || attempts >= maxAttempts) {{
      window.clearInterval(timer);
    }}
  }}, 100);
}})();
</script>
""",
        height=0,
    )

_CUSTOM_CSS = f"""
<style>
/* Single dark design system (no theme branching). */
.stApp {{
    --ci-bg-page: #0a1120;
    --ci-bg-page-top: #0e1a33;
    --ci-bg-surface: rgba(21, 33, 57, 0.84);
    --ci-bg-surface-strong: rgba(24, 38, 64, 0.96);
    --ci-bg-muted: rgba(29, 45, 73, 0.76);
    --ci-bg-chat: rgba(20, 34, 58, 0.92);
    --ci-bg-sidebar: linear-gradient(180deg, rgba(18, 29, 50, 0.98), rgba(13, 23, 41, 0.96));
    --ci-bg-input: rgba(18, 32, 55, 0.96);
    --ci-bg-hover: rgba(129, 140, 248, 0.2);
    --ci-border: rgba(148, 163, 184, 0.28);
    --ci-border-strong: rgba(148, 163, 184, 0.4);
    --ci-text: #e5ecf8;
    --ci-text-soft: #cdd8ea;
    --ci-text-muted: #a6b6d3;
    --ci-accent: #818cf8;
    --ci-accent-strong: #6366f1;
    --ci-accent-soft: rgba(129, 140, 248, 0.22);
    --ci-accent-soft-strong: rgba(129, 140, 248, 0.3);
    --ci-success: #34d399;
    --ci-warning: #f59e0b;
    --ci-danger: #f87171;
    --ci-alert-bg: rgba(127, 29, 29, 0.3);
    --ci-alert-border: rgba(248, 113, 113, 0.45);
    --ci-alert-text: #fecaca;
    --ci-shadow-rgb: 2, 8, 23;
    --ci-radius-sm: 14px;
    --ci-radius: 18px;
    --ci-radius-lg: 24px;
    --ci-shadow-xs: 0 3px 8px rgba(var(--ci-shadow-rgb), 0.24);
    --ci-shadow-sm: 0 14px 30px rgba(var(--ci-shadow-rgb), 0.34);
    --ci-shadow: 0 26px 60px rgba(var(--ci-shadow-rgb), 0.45);
    --ci-shell-max: {_SHELL_MAX_W};
    --ci-font-chat: 1.03rem;
    --ci-font-input: 1.02rem;
    --ci-sidebar-width: __CI_SIDEBAR_WIDTH_PX__px;
    --ci-sidebar-min-width: __CI_SIDEBAR_MIN_PX__px;
    --ci-sidebar-max-width: __CI_SIDEBAR_MAX_PX__px;
    --ci-sources-sticky-top: clamp(0.9rem, 1.4vh, 1.25rem);
    --ci-sources-sticky-bottom-clearance: 7.2rem;
}}

html,
body,
[data-testid="stAppViewContainer"] {{
    color: var(--ci-text);
    background: transparent !important;
}}

.stApp {{
    background:
        radial-gradient(circle at top, var(--ci-accent-soft-strong), transparent 38%),
        linear-gradient(180deg, var(--ci-bg-page-top) 0%, var(--ci-bg-page) 58%, var(--ci-bg-page) 100%);
}}

[data-testid="stHeader"],
header[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

.main .block-container {{
    max-width: 100% !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    padding-top: 0.75rem !important;
    padding-bottom: 8.5rem !important;
    padding-left: clamp(0.6rem, 1vw, 0.95rem) !important;
    padding-right: clamp(0.6rem, 1vw, 0.95rem) !important;
    min-height: calc(100vh - 1.2rem) !important;
}}

.ci-shell,
.ci-hero,
.ci-alert,
.ci-mic-state {{
    max-width: none;
    margin-left: 0;
    margin-right: 0;
}}

.ci-surface-kicker {{
    margin: 0 0 0.32rem 0;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ci-accent);
}}

.ci-hero {{
    position: relative;
    overflow: hidden;
    margin-bottom: 1.1rem;
    padding: 1.28rem 1.35rem;
    background: linear-gradient(180deg, var(--ci-bg-surface-strong), var(--ci-bg-surface));
    border: 1px solid var(--ci-border);
    border-radius: 26px;
    box-shadow: 0 22px 44px rgba(var(--ci-shadow-rgb), 0.14), 0 2px 8px rgba(var(--ci-shadow-rgb), 0.08);
}}

.ci-hero::after {{
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    top: -92px;
    right: -92px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(79, 70, 229, 0.18) 0%, rgba(79, 70, 229, 0.04) 42%, transparent 72%);
    pointer-events: none;
}}

.ci-hero .hero-row {{
    position: relative;
    z-index: 1;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}}

.ci-hero .hero-mark {{
    width: 54px;
    height: 54px;
    border-radius: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: var(--ci-accent);
    background: linear-gradient(180deg, rgba(79, 70, 229, 0.16), rgba(79, 70, 229, 0.08));
    border: 1px solid rgba(79, 70, 229, 0.14);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.22);
}}

.ci-hero h1 {{
    font-size: 1.95rem;
    font-weight: 760;
    letter-spacing: -0.04em;
    color: var(--ci-text);
    margin: 0 0 0.3rem 0;
    line-height: 1.08;
}}

.ci-hero .sub {{
    font-size: 1rem;
    color: var(--ci-text-soft);
    margin: 0;
    line-height: 1.58;
    max-width: 56rem;
}}

.ci-hero.ci-hero-compact {{
    padding: 0.82rem 1rem;
    margin-bottom: 0.95rem;
    border-radius: 22px;
}}

.ci-hero.ci-hero-compact::after {{
    opacity: 0.6;
    transform: scale(0.82);
}}

.ci-hero.ci-hero-compact .hero-row {{
    align-items: center;
}}

.ci-hero.ci-hero-compact .hero-mark {{
    width: 42px;
    height: 42px;
    border-radius: 14px;
    font-size: 0.86rem;
}}

.ci-hero.ci-hero-compact .ci-surface-kicker,
.ci-hero.ci-hero-compact .sub {{
    display: none;
}}

.ci-hero.ci-hero-compact h1 {{
    font-size: 1.14rem;
    margin: 0;
    letter-spacing: -0.02em;
}}

div[data-testid="stVerticalBlock"]:has(.ci-chip-row) {{
    max-width: none;
    margin-left: 0;
    margin-right: 0;
    margin-bottom: 1.05rem;
}}

.ci-try-label {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0 0 0.58rem 0;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ci-text-muted);
}}

.ci-try-label::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--ci-border-strong), transparent);
}}

div[data-testid="stVerticalBlock"]:has(.ci-chip-row) [data-testid="column"] {{
    display: flex;
}}

div[data-testid="stVerticalBlock"]:has(.ci-chip-row) .stButton,
div[data-testid="stVerticalBlock"]:has(.ci-chip-row) .stButton > button {{
    width: 100%;
}}

div[data-testid="stVerticalBlock"]:has(.ci-chip-row) .stButton > button {{
    height: 100%;
    min-height: 3.1rem !important;
    border-radius: 16px !important;
    border: 1px solid var(--ci-border) !important;
    background: var(--ci-bg-surface) !important;
    color: var(--ci-text-soft) !important;
    box-shadow: var(--ci-shadow-xs) !important;
    padding: 0.7rem 0.95rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
    transition:
        transform 0.14s ease,
        box-shadow 0.14s ease,
        border-color 0.14s ease,
        background 0.14s ease;
}}

div[data-testid="stVerticalBlock"]:has(.ci-chip-row) .stButton > button:hover {{
    transform: translateY(-1px);
    border-color: var(--ci-accent-soft-strong) !important;
    background: var(--ci-bg-surface-strong) !important;
    box-shadow: var(--ci-shadow-sm) !important;
}}

[data-testid="stChatMessage"] {{
    max-width: none !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    margin-bottom: 1.2rem !important;
    gap: 0.95rem !important;
}}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{
    padding: 0.95rem 1.1rem !important;
    border-radius: 24px !important;
    border: 1px solid var(--ci-border) !important;
    box-shadow: var(--ci-shadow-xs) !important;
    backdrop-filter: blur(10px);
    max-width: min(100%, 52rem);
}}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
    background: linear-gradient(180deg, var(--ci-accent-soft-strong), var(--ci-accent-soft)) !important;
    border-color: var(--ci-accent-soft-strong) !important;
    margin-left: auto !important;
    max-width: min(84%, 46rem);
}}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{
    background: var(--ci-bg-surface) !important;
    margin-right: auto !important;
    max-width: min(92%, 52rem);
}}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {{
    border-radius: 16px !important;
    border: 1px solid var(--ci-border) !important;
    background: var(--ci-bg-surface-strong) !important;
    box-shadow: var(--ci-shadow-xs) !important;
}}

[data-testid="stChatMessage"] .stMarkdown {{
    font-size: var(--ci-font-chat) !important;
    line-height: 1.72 !important;
    color: var(--ci-text) !important;
}}

[data-testid="stChatMessage"] .stMarkdown p {{
    margin-bottom: 0.7em !important;
}}

[data-testid="stChatMessage"] .stMarkdown li {{
    margin-bottom: 0.38em !important;
}}

[data-testid="stChatMessage"] .stMarkdown h1,
[data-testid="stChatMessage"] .stMarkdown h2,
[data-testid="stChatMessage"] .stMarkdown h3 {{
    margin-top: 0.9em !important;
    margin-bottom: 0.42em !important;
    font-weight: 680 !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker),
[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-voice-card-marker),
[data-testid="stElementContainer"]:has(.ci-sources-card-marker) {{
    border-radius: 24px !important;
    border: 1px solid var(--ci-border) !important;
    background: var(--ci-bg-surface) !important;
    box-shadow: var(--ci-shadow-sm) !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker),
[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-voice-card-marker) {{
    max-width: none !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    margin-bottom: 1rem !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) {{
    background: linear-gradient(180deg, var(--ci-bg-surface-strong), var(--ci-bg-surface)) !important;
    border-color: var(--ci-border-strong) !important;
    box-shadow: var(--ci-shadow), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
    position: sticky !important;
    bottom: 1rem !important;
    z-index: 20 !important;
    backdrop-filter: blur(18px);
}}

div[data-testid="stColumn"]:has(.ci-sources-card-marker) {{
    position: sticky !important;
    top: var(--ci-sources-sticky-top) !important;
    align-self: flex-start !important;
}}

[data-testid="stElementContainer"]:has(.ci-sources-card-marker) {{
    background: linear-gradient(180deg, var(--ci-bg-surface-strong), var(--ci-bg-surface)) !important;
    max-height: calc(100vh - var(--ci-sources-sticky-top) - var(--ci-sources-sticky-bottom-clearance)) !important;
    overflow: hidden !important;
    z-index: 5 !important;
}}

[data-testid="stElementContainer"]:has(.ci-sources-card-marker) [data-testid="stVerticalBlock"] {{
    max-height: calc(100vh - var(--ci-sources-sticky-top) - var(--ci-sources-sticky-bottom-clearance) - 0.75rem) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-right: 0.35rem !important;
    scrollbar-gutter: stable both-edges;
}}

.ci-composer-head,
.ci-panel-head,
.ci-voice-head {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.95rem;
    margin-bottom: 0.9rem;
}}

.ci-composer-head h3,
.ci-panel-head h3,
.ci-voice-title {{
    margin: 0;
    font-size: 1.03rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ci-text);
}}

.ci-composer-head p,
.ci-panel-head p,
.ci-voice-sub {{
    margin: 0.28rem 0 0 0;
    font-size: 0.88rem;
    line-height: 1.55;
    color: var(--ci-text-muted);
}}

.ci-composer-note,
.ci-voice-source {{
    flex-shrink: 0;
    align-self: flex-start;
    padding: 0.42rem 0.72rem;
    border-radius: 999px;
    border: 1px solid var(--ci-border);
    background: var(--ci-bg-muted);
    color: var(--ci-text-muted);
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}}

.ci-panel-icon {{
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    background: linear-gradient(180deg, rgba(79, 70, 229, 0.14), rgba(79, 70, 229, 0.07));
    border: 1px solid rgba(79, 70, 229, 0.12);
    color: var(--ci-accent);
    font-size: 1rem;
    font-weight: 700;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="column"] {{
    display: flex;
    align-items: stretch;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stPopover"] > button {{
    min-height: 3.35rem !important;
    width: 100% !important;
    border-radius: 18px !important;
    border: 1px solid color-mix(in srgb, var(--ci-accent) 32%, var(--ci-border-strong)) !important;
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--ci-bg-input) 72%, var(--ci-accent-soft) 28%),
        color-mix(in srgb, var(--ci-bg-input) 86%, var(--ci-bg-surface) 14%)
    ) !important;
    box-shadow: 0 8px 16px color-mix(in srgb, var(--ci-accent) 18%, transparent) !important;
    color: var(--ci-accent) !important;
    font-size: 1.12rem !important;
    font-weight: 700 !important;
    padding: 0 !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stPopover"] > button:hover {{
    border-color: color-mix(in srgb, var(--ci-accent) 56%, var(--ci-border-strong)) !important;
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--ci-bg-input) 52%, var(--ci-accent-soft-strong) 48%),
        color-mix(in srgb, var(--ci-bg-input) 76%, var(--ci-accent-soft) 24%)
    ) !important;
    color: var(--ci-text) !important;
    transform: translateY(-1px);
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stPopover"] > button:disabled {{
    border-color: var(--ci-border) !important;
    background: var(--ci-bg-input) !important;
    color: var(--ci-text-muted) !important;
    box-shadow: none !important;
    opacity: 0.62 !important;
}}

/* Popover trigger may be nested; target the real aria-expanded button. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stPopover"] button[aria-expanded] {{
    min-height: 3.35rem !important;
    width: 100% !important;
    border-radius: 18px !important;
    border: 1px solid color-mix(in srgb, var(--ci-accent) 44%, var(--ci-border-strong)) !important;
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--ci-bg-input) 58%, var(--ci-accent-soft) 42%),
        color-mix(in srgb, var(--ci-bg-input) 76%, var(--ci-bg-surface) 24%)
    ) !important;
    box-shadow: 0 10px 18px color-mix(in srgb, var(--ci-accent) 26%, transparent) !important;
    color: var(--ci-accent) !important;
    font-size: 1.12rem !important;
    font-weight: 700 !important;
    padding: 0 !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stPopover"] button[aria-expanded]:hover {{
    border-color: color-mix(in srgb, var(--ci-accent) 56%, var(--ci-border-strong)) !important;
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--ci-bg-input) 42%, var(--ci-accent-soft-strong) 58%),
        color-mix(in srgb, var(--ci-bg-input) 66%, var(--ci-accent-soft) 34%)
    ) !important;
    color: var(--ci-text) !important;
    transform: translateY(-1px);
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stPopover"] button[aria-expanded="true"] {{
    border-color: color-mix(in srgb, var(--ci-accent) 62%, var(--ci-border-strong)) !important;
    color: var(--ci-text) !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) .stChatInput,
[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stChatInput"] {{
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) .stChatInput > div,
[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) [data-testid="stChatInput"] > div {{
    background: var(--ci-bg-input) !important;
    border: 1px solid var(--ci-border-strong) !important;
    border-radius: 20px !important;
    box-shadow: none !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) .stChatInput textarea {{
    font-size: var(--ci-font-input) !important;
    line-height: 1.55 !important;
    min-height: 3.25rem !important;
    color: var(--ci-text) !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) .stChatInput textarea::placeholder {{
    color: var(--ci-text-muted) !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) .stChatInput button {{
    width: 2.65rem !important;
    height: 2.65rem !important;
    border-radius: 14px !important;
    border: 1px solid color-mix(in srgb, var(--ci-accent) 36%, transparent) !important;
    background: linear-gradient(180deg, var(--ci-accent), var(--ci-accent-strong)) !important;
    color: var(--ci-text) !important;
    box-shadow: 0 10px 18px color-mix(in srgb, var(--ci-accent) 26%, transparent) !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) div[data-testid="stAudioInput"] {{
    width: 100%;
    margin: 0 !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-composer-card-marker) div[data-testid="stAudioInput"] button {{
    min-height: 2.8rem !important;
    width: 100% !important;
    border-radius: 14px !important;
    border: 1px solid var(--ci-border-strong) !important;
    background: var(--ci-bg-input) !important;
    box-shadow: none !important;
    color: var(--ci-text-soft) !important;
    font-size: 0.82rem !important;
    justify-content: center !important;
}}

.ci-mic-state {{
    margin-top: 0.55rem;
    font-size: 0.82rem;
    line-height: 1.5;
    color: var(--ci-text-muted);
}}

.ci-mic-state.is-error {{
    color: var(--ci-danger);
}}

.ci-draft-error {{
    margin: 0.1rem 0 0.65rem 0;
    padding: 0.75rem 0.85rem;
    border-radius: 16px;
    border: 1px solid var(--ci-alert-border);
    background: var(--ci-alert-bg);
    color: var(--ci-alert-text);
    font-size: 0.86rem;
    line-height: 1.5;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-voice-card-marker) .stTextArea textarea {{
    min-height: 7rem !important;
    border-radius: 18px !important;
    border: 1px solid var(--ci-border) !important;
    background: var(--ci-bg-input) !important;
    padding: 0.95rem 1rem !important;
    line-height: 1.55 !important;
}}

[data-testid="stVerticalBlockBorderWrapper"]:has(.ci-voice-card-marker) .stButton > button {{
    min-height: 2.8rem !important;
    border-radius: 16px !important;
    font-weight: 600 !important;
}}

.ci-source-empty {{
    min-height: 180px;
    padding: 1.15rem 1rem;
    border-radius: 18px;
    border: 1px dashed var(--ci-border-strong);
    background: linear-gradient(180deg, var(--ci-bg-muted), var(--ci-bg-surface));
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 0.35rem;
}}

.ci-source-empty .empty-icon {{
    width: 2.4rem;
    height: 2.4rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--ci-bg-surface-strong);
    border: 1px solid var(--ci-border);
    color: var(--ci-accent);
    font-weight: 700;
}}

.ci-source-empty .empty-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--ci-text-soft);
}}

.ci-source-empty .empty-copy {{
    font-size: 0.85rem;
    line-height: 1.52;
    color: var(--ci-text-muted);
}}

.ci-source-group-head {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 1.2rem 0 0.65rem 0;
}}

.ci-sources-group-head {{
    margin: 0;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ci-text-muted);
}}

.ci-source-group-head:first-of-type {{
    margin-top: 0.2rem;
}}

.ci-sources-summary {{
    margin: 0.15rem 0 0.95rem 0;
    font-size: 0.82rem;
    line-height: 1.5;
    color: var(--ci-text-muted);
}}

.ci-source-family-icon {{
    width: 1.85rem;
    height: 1.85rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    border: 1px solid var(--ci-border);
    background: var(--ci-bg-muted);
    color: var(--ci-text-soft);
    font-size: 0.8rem;
    font-weight: 800;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
}}

.ci-source-family-wef {{
    background: linear-gradient(180deg, rgba(14, 165, 233, 0.22), rgba(37, 99, 235, 0.12));
    color: #bfdbfe;
}}

.ci-source-family-esco {{
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.18), rgba(5, 150, 105, 0.1));
    color: #bbf7d0;
}}

.ci-source-family-youtube {{
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.24), rgba(220, 38, 38, 0.12));
    color: #fecaca;
}}

.ci-source-family-web {{
    background: linear-gradient(180deg, rgba(168, 85, 247, 0.2), rgba(129, 140, 248, 0.1));
    color: #ddd6fe;
}}

.ci-yt-list {{
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
}}

.ci-yt-row {{
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.75rem 0.85rem;
    background: var(--ci-bg-surface-strong);
    border: 1px solid var(--ci-border);
    border-radius: 16px;
    box-shadow: var(--ci-shadow-xs);
}}

.ci-yt-thumb {{
    flex: 0 0 auto;
    width: 112px;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--ci-border);
    background: var(--ci-bg-muted);
}}

.ci-yt-thumb img {{
    display: block;
    width: 100%;
    height: auto;
    object-fit: cover;
}}

.ci-yt-body {{
    flex: 1;
    min-width: 0;
}}

a.ci-yt-title {{
    display: block;
    font-weight: 650;
    font-size: 0.95rem;
    line-height: 1.38;
    color: var(--ci-text);
    text-decoration: none;
    margin-bottom: 0.28rem;
}}

a.ci-yt-title:hover {{
    color: var(--ci-accent);
    text-decoration: underline;
}}

.ci-yt-channel {{
    font-size: 0.8rem;
    color: var(--ci-text-muted);
    line-height: 1.45;
}}

.ci-source-card {{
    padding: 1rem 1.05rem;
    margin-bottom: 0.8rem;
    background: var(--ci-bg-surface-strong);
    border: 1px solid var(--ci-border);
    border-radius: 18px;
    box-shadow: var(--ci-shadow-xs);
}}

.ci-source-card:last-child {{
    margin-bottom: 0;
}}

.ci-source-card .topline {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.7rem;
    margin-bottom: 0.65rem;
}}

.ci-source-card .brand {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    min-width: 0;
}}

.ci-source-card .badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.22rem 0.52rem;
    border-radius: 999px;
    background: var(--ci-accent-soft);
    color: var(--ci-accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}}

.ci-source-card .label {{
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ci-text-muted);
}}

.ci-source-card .title {{
    font-weight: 680;
    font-size: 1.02rem;
    line-height: 1.4;
    color: var(--ci-text);
    margin-bottom: 0.5rem;
}}

.ci-source-card .meta {{
    font-size: 0.8rem;
    color: var(--ci-text-muted);
    line-height: 1.5;
    margin-bottom: 0.45rem;
}}

.ci-source-card .meta a {{
    color: var(--ci-accent);
    font-weight: 600;
    text-decoration: none;
}}

.ci-source-card .meta a:hover {{
    text-decoration: underline;
}}

.ci-source-card .ex {{
    font-size: 0.84rem;
    color: var(--ci-text-soft);
    line-height: 1.55;
}}

section[data-testid="stSidebar"] {{
    background: var(--ci-bg-sidebar) !important;
    border-right: 1px solid var(--ci-border) !important;
    width: clamp(var(--ci-sidebar-min-width), var(--ci-sidebar-width), var(--ci-sidebar-max-width)) !important;
    min-width: var(--ci-sidebar-min-width) !important;
    max-width: min(var(--ci-sidebar-max-width), calc(100vw - 20rem)) !important;
    flex-basis: clamp(var(--ci-sidebar-min-width), var(--ci-sidebar-width), var(--ci-sidebar-max-width)) !important;
    font-size: 1rem !important;
    overflow: visible !important;
    position: relative !important;
}}

section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {{
    min-width: var(--ci-sidebar-min-width) !important;
    max-width: min(var(--ci-sidebar-max-width), calc(100vw - 20rem)) !important;
}}

.ci-sidebar-resize-handle {{
    position: absolute;
    top: 0;
    right: -2px;
    width: 10px;
    height: 100%;
    cursor: ew-resize;
    z-index: 40;
    background: linear-gradient(180deg, rgba(129, 140, 248, 0), rgba(129, 140, 248, 0.26), rgba(129, 140, 248, 0));
    border-radius: 999px;
    opacity: 0.62;
    will-change: opacity;
    transition: opacity 0.18s ease, background 0.18s ease;
}}

.ci-sidebar-resize-handle::before {{
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 12px;
    height: 44px;
    transform: translate(-50%, -50%);
    border-radius: 999px;
    border: 1px solid rgba(129, 140, 248, 0.42);
    background: linear-gradient(180deg, rgba(129, 140, 248, 0.2), rgba(129, 140, 248, 0.08));
    box-shadow: 0 3px 14px rgba(var(--ci-shadow-rgb), 0.32);
    pointer-events: none;
}}

.ci-sidebar-resize-handle::after {{
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 2px;
    height: 2px;
    transform: translate(-50%, -50%);
    border-radius: 999px;
    background: rgba(207, 218, 245, 0.92);
    box-shadow:
        0 -12px 0 rgba(207, 218, 245, 0.92),
        0 -8px 0 rgba(207, 218, 245, 0.92),
        0 -4px 0 rgba(207, 218, 245, 0.92),
        0 4px 0 rgba(207, 218, 245, 0.92),
        0 8px 0 rgba(207, 218, 245, 0.92),
        0 12px 0 rgba(207, 218, 245, 0.92);
    pointer-events: none;
}}

.ci-sidebar-resize-handle:hover {{
    opacity: 1;
    background: linear-gradient(180deg, rgba(129, 140, 248, 0.2), rgba(129, 140, 248, 0.75), rgba(129, 140, 248, 0.2));
}}

body.ci-sidebar-resizing {{
    cursor: ew-resize !important;
    user-select: none !important;
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 0.95rem !important;
    padding-bottom: 1.1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{
    color: var(--ci-text-soft) !important;
}}

section[data-testid="stSidebar"] .stCaption {{
    font-size: 0.95rem !important;
    line-height: 1.52 !important;
    color: var(--ci-text-muted) !important;
}}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {{
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
    color: var(--ci-text-soft) !important;
}}

section[data-testid="stSidebar"] [data-testid="stMetricValue"],
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
    color: var(--ci-text) !important;
}}

section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: var(--ci-bg-surface-strong) !important;
    border: 1px solid var(--ci-border) !important;
    color: var(--ci-text) !important;
}}

section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.ci-side-card-marker) {{
    border-radius: 22px !important;
    border: 1px solid var(--ci-border) !important;
    background: linear-gradient(180deg, var(--ci-bg-surface-strong), var(--ci-bg-surface)) !important;
    box-shadow: var(--ci-shadow-xs) !important;
    margin-bottom: 0.85rem !important;
}}

.ci-side-section {{
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ci-text-muted) !important;
    margin: 1rem 0 0.55rem 0 !important;
    border-top: 1px solid var(--ci-border);
    padding-top: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}}

.ci-side-section .sec-ic {{
    width: 1.45rem;
    height: 1.45rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: var(--ci-bg-muted);
    color: var(--ci-text-soft);
    font-size: 0.82rem;
    font-weight: 700;
    border: 1px solid var(--ci-border);
}}

.ci-side-section:first-of-type {{
    border-top: none;
    padding-top: 0;
    margin-top: 0;
}}

.ci-sidebar-kicker {{
    margin: 0 0 0.18rem 0;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--ci-text);
}}

.ci-model-meta-block {{
    margin: 0.48rem 0 0.8rem 0;
    padding: 0.62rem 0.72rem;
    border-radius: 14px;
    border: 1px solid var(--ci-border);
    background: linear-gradient(180deg, var(--ci-bg-surface-strong), var(--ci-bg-muted));
}}

.ci-model-meta-text {{
    margin: 0;
    color: var(--ci-text-soft);
    font-size: 0.83rem;
    line-height: 1.5;
}}

.ci-model-meta-row {{
    margin-top: 0.48rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
}}

.ci-model-price-chip {{
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.56rem;
    border-radius: 999px;
    border: 1px solid var(--ci-border-strong);
    background: color-mix(in srgb, var(--ci-accent) 16%, var(--ci-bg-input));
    color: var(--ci-text);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}}

.ci-model-price-text {{
    color: var(--ci-text-muted);
    font-size: 0.78rem;
    line-height: 1.45;
}}

.ci-side-muted {{
    margin: 0.35rem 0 0 0;
    color: var(--ci-text-muted);
    font-size: 0.8rem;
    line-height: 1.5;
}}

.ci-side-key-status {{
    margin: 0.55rem 0 0.2rem 0;
    color: var(--ci-text-muted);
    font-size: 0.78rem;
}}

.ci-sidebar-stats {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
    margin-top: 0.45rem;
}}

.ci-sidebar-stat {{
    padding: 0.72rem 0.75rem;
    border-radius: 16px;
    border: 1px solid var(--ci-border);
    background: color-mix(in srgb, var(--ci-bg-input) 82%, var(--ci-bg-surface) 18%);
}}

.ci-sidebar-stat-label {{
    display: block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ci-text-muted);
    margin-bottom: 0.28rem;
}}

.ci-sidebar-stat-value {{
    display: block;
    font-size: 0.98rem;
    font-weight: 720;
    letter-spacing: -0.02em;
    color: var(--ci-text);
}}

.ci-sidebar-usage-note {{
    margin-top: 0.65rem;
    color: var(--ci-text-muted);
    font-size: 0.78rem;
    line-height: 1.5;
}}

.ci-sidebar-model-breakdown {{
    margin-top: 0.65rem;
    display: flex;
    flex-direction: column;
    gap: 0.42rem;
}}

.ci-sidebar-model-row {{
    padding: 0.55rem 0.65rem;
    border-radius: 14px;
    border: 1px solid var(--ci-border);
    background: rgba(15, 23, 42, 0.26);
}}

.ci-sidebar-model-name {{
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--ci-text-soft);
}}

.ci-sidebar-model-meta {{
    margin-top: 0.18rem;
    font-size: 0.74rem;
    line-height: 1.45;
    color: var(--ci-text-muted);
}}

section[data-testid="stSidebar"] details[data-testid="stExpander"] {{
    background: linear-gradient(180deg, var(--ci-bg-surface-strong), var(--ci-bg-surface));
    border: 1px solid var(--ci-border-strong);
    border-radius: 16px;
    box-shadow: var(--ci-shadow-xs);
    margin-bottom: 0.45rem !important;
}}

section[data-testid="stSidebar"] details[data-testid="stExpander"] > summary,
section[data-testid="stSidebar"] details[data-testid="stExpander"] .streamlit-expanderHeader {{
    background: var(--ci-bg-input) !important;
    border: 1px solid var(--ci-border-strong) !important;
    border-radius: 12px !important;
    color: var(--ci-text) !important;
    font-size: 0.9rem !important;
    font-weight: 680 !important;
}}

section[data-testid="stSidebar"] details[data-testid="stExpander"] > summary:hover,
section[data-testid="stSidebar"] details[data-testid="stExpander"] .streamlit-expanderHeader:hover {{
    background: color-mix(in srgb, var(--ci-bg-input) 82%, var(--ci-accent-soft) 18%) !important;
    border-color: var(--ci-accent-soft-strong) !important;
}}

section[data-testid="stSidebar"] details[data-testid="stExpander"]:not([open]) > summary,
section[data-testid="stSidebar"] details[data-testid="stExpander"]:not([open]) .streamlit-expanderHeader {{
    background: color-mix(in srgb, var(--ci-bg-input) 88%, var(--ci-bg-surface) 12%) !important;
}}

section[data-testid="stSidebar"] details[data-testid="stExpander"] > summary svg,
section[data-testid="stSidebar"] details[data-testid="stExpander"] .streamlit-expanderHeader svg {{
    color: var(--ci-text-soft) !important;
    opacity: 0.96 !important;
}}

section[data-testid="stSidebar"] details[data-testid="stExpander"] > summary:focus-visible,
section[data-testid="stSidebar"] details[data-testid="stExpander"] .streamlit-expanderHeader:focus-visible {{
    outline: 2px solid color-mix(in srgb, var(--ci-accent) 52%, transparent);
    outline-offset: 1px;
}}

/* Streamlit currently renders sidebar expanders as aria-expanded buttons. */
section[data-testid="stSidebar"] [data-testid="stExpander"] button[aria-expanded] {{
    width: 100% !important;
    min-height: 2.5rem !important;
    background: var(--ci-bg-input) !important;
    border: 1px solid var(--ci-border-strong) !important;
    border-radius: 12px !important;
    color: var(--ci-text) !important;
    font-size: 0.9rem !important;
    font-weight: 680 !important;
    box-shadow: inset 0 1px 0 color-mix(in srgb, var(--ci-text) 10%, transparent) !important;
}}

section[data-testid="stSidebar"] [data-testid="stExpander"] button[aria-expanded]:hover {{
    background: color-mix(in srgb, var(--ci-bg-input) 82%, var(--ci-accent-soft) 18%) !important;
    border-color: var(--ci-accent-soft-strong) !important;
}}

section[data-testid="stSidebar"] [data-testid="stExpander"] button[aria-expanded="true"] {{
    background: color-mix(in srgb, var(--ci-bg-input) 78%, var(--ci-accent-soft) 22%) !important;
    border-color: var(--ci-accent-soft-strong) !important;
}}

section[data-testid="stSidebar"] [data-testid="stExpander"] button[aria-expanded] [data-testid="stExpanderToggleIcon"],
section[data-testid="stSidebar"] [data-testid="stExpander"] button[aria-expanded] [aria-hidden="true"] {{
    color: var(--ci-text-soft) !important;
    opacity: 0.98 !important;
}}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    border-radius: 18px;
    border: 1px dashed var(--ci-border-strong);
    background: var(--ci-bg-muted);
    padding: 0.9rem 0.85rem;
}}

section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stDownloadButton > button {{
    min-height: 2.65rem !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    border: 1px solid var(--ci-border) !important;
    background: var(--ci-bg-surface-strong) !important;
    box-shadow: var(--ci-shadow-xs) !important;
}}

section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stDownloadButton > button:hover {{
    border-color: var(--ci-accent-soft-strong) !important;
    background: var(--ci-bg-hover) !important;
}}

section[data-testid="stSidebar"] .stButton > button:disabled,
section[data-testid="stSidebar"] .stDownloadButton > button:disabled,
section[data-testid="stSidebar"] [aria-disabled="true"] {{
    opacity: 0.58 !important;
    color: var(--ci-text-muted) !important;
    border-color: var(--ci-border) !important;
    background: rgba(15, 24, 42, 0.3) !important;
    box-shadow: none !important;
}}

.status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.45rem 0 0.9rem 0;
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.38rem 0.68rem;
    border-radius: 999px;
    border: 1px solid var(--ci-border);
    background: var(--ci-bg-surface);
    color: var(--ci-text-muted);
    font-size: 0.78rem;
    font-weight: 700;
}}

.status-pill .status-icon {{
    width: 1.1rem;
    height: 1.1rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 800;
    color: #ffffff;
}}

.status-pill.ok {{
    color: var(--ci-success);
    border-color: color-mix(in srgb, var(--ci-success) 46%, transparent);
    background: color-mix(in srgb, var(--ci-success) 16%, transparent);
}}

.status-pill.ok .status-icon {{
    background: var(--ci-success);
}}

.status-pill.bad {{
    color: var(--ci-danger);
    border-color: color-mix(in srgb, var(--ci-danger) 45%, transparent);
    background: color-mix(in srgb, var(--ci-danger) 16%, transparent);
}}

.status-pill.bad .status-icon {{
    background: var(--ci-danger);
}}

.status-pill.warn {{
    color: var(--ci-warning);
    border-color: color-mix(in srgb, var(--ci-warning) 45%, transparent);
    background: color-mix(in srgb, var(--ci-warning) 16%, transparent);
}}

.status-pill.warn .status-icon {{
    background: var(--ci-warning);
}}

.ci-sidebar-footer {{
    margin-top: 1.1rem;
    padding: 0.9rem 0.95rem;
    border-radius: 18px;
    border: 1px solid var(--ci-border);
    background: linear-gradient(180deg, var(--ci-bg-surface-strong), var(--ci-bg-muted));
    color: var(--ci-text-muted);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16);
}}

.ci-sidebar-footer strong {{
    display: block;
    margin-bottom: 0.18rem;
    font-size: 0.8rem;
    color: var(--ci-text-soft);
}}

.ci-sidebar-footer p {{
    margin: 0;
    font-size: 0.78rem;
    line-height: 1.5;
}}

.ci-alert {{
    margin: 0.7rem auto;
    padding: 0.8rem 0.95rem;
    border-radius: 18px;
    border: 1px solid var(--ci-alert-border);
    background: var(--ci-alert-bg);
    color: var(--ci-alert-text);
    font-size: 0.88rem;
    line-height: 1.55;
    font-weight: 500;
}}

section.main details[data-testid="stExpander"] {{
    max-width: none;
    margin-left: 0 !important;
    margin-right: 0 !important;
    background: var(--ci-bg-surface);
    border: 1px solid var(--ci-border);
    border-radius: 16px;
    box-shadow: var(--ci-shadow-xs);
    margin-bottom: 0.75rem !important;
}}

section.main .streamlit-expanderHeader {{
    font-size: 0.91rem !important;
    font-weight: 600 !important;
    color: var(--ci-text-soft) !important;
}}

@media (max-width: 1080px) {{
    .ci-composer-head,
    .ci-panel-head,
    .ci-voice-head {{
        flex-direction: column;
    }}

    .ci-composer-note,
    .ci-voice-source {{
        white-space: normal;
    }}

    div[data-testid="stColumn"]:has(.ci-sources-card-marker) {{
        position: static !important;
    }}

    [data-testid="stElementContainer"]:has(.ci-sources-card-marker) {{
        position: static !important;
        max-height: none !important;
    }}

    [data-testid="stElementContainer"]:has(.ci-sources-card-marker) [data-testid="stVerticalBlock"] {{
        max-height: none !important;
        overflow: visible !important;
        padding-right: 0 !important;
    }}

    .ci-sidebar-resize-handle {{
        display: none !important;
    }}

    section[data-testid="stSidebar"] {{
        max-width: min(84vw, 24rem) !important;
    }}

    .ci-sidebar-stats {{
        grid-template-columns: 1fr;
    }}
}}
</style>
"""


CV_ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
AUDIO_ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "webm", "mp4"}
CV_MAX_BYTES = 5 * 1024 * 1024
AUDIO_MAX_BYTES = 25 * 1024 * 1024
SESSION_RATE_LIMIT_COUNT = 12
SESSION_RATE_LIMIT_WINDOW_SECONDS = 60
SESSION_MIN_INTERVAL_SECONDS = 2.0
BYOK_RATE_LIMIT_COUNT = 4
BYOK_RATE_LIMIT_WINDOW_SECONDS = 60
YOUTUBE_RATE_LIMIT_COUNT = 6
YOUTUBE_RATE_LIMIT_WINDOW_SECONDS = 60


def _get_system_status() -> dict:
    if STREAMLIT_DIRECT_MODE:
        try:
            return get_system_status_direct()
        except Exception as exc:
            return {
                "backend": True,
                "qdrant": False,
                "indexed_data_present": False,
                "collection": "unknown",
                "points_count": 0,
                "error": f"Direct mode failed: {exc}",
            }

    if not API_BASE:
        return {
            "backend": False,
            "qdrant": False,
            "indexed_data_present": False,
            "collection": "unknown",
            "points_count": 0,
            "error": "External API mode requires CAREER_INTEL_API_BASE_URL.",
        }
    try:
        resp = httpx.get(
            f"{API_BASE}/health/system",
            headers=build_request_headers(session_id=st.session_state.session_id),
            timeout=3.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {
            "backend": False,
            "qdrant": False,
            "indexed_data_present": False,
            "collection": "unknown",
            "points_count": 0,
            "error": str(exc) if not isinstance(exc, httpx.ConnectError) else f"Cannot reach {API_BASE}",
        }


def _build_request_headers(*, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    return build_request_headers(
        session_id=st.session_state.session_id,
        model=st.session_state.selected_model,
        api_key=_active_request_api_key(),
        user_timezone=_USER_TIMEZONE,
        extra_headers=extra_headers,
    )


def _build_request_body() -> dict:
    body: dict = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        "session_id": st.session_state.session_id,
        "use_tools": st.session_state.get("use_tools", True),
        "answer_length": st.session_state.get("answer_length", "balanced"),
    }
    if st.session_state.cv_text:
        body["cv_text"] = st.session_state.cv_text
    return body


def _friendly_http_error(exc: httpx.HTTPStatusError) -> str:
    status = exc.response.status_code
    detail: str | None = None
    try:
        payload = exc.response.json()
        if isinstance(payload, dict):
            detail = _coerce_error_detail(payload.get("detail") or payload)
        else:
            detail = None
    except Exception:
        detail = exc.response.text or None

    if status == 400:
        return detail or "The request was rejected. Please revise the message and try again."
    if status == 401:
        return "Authentication failed for the selected model provider."
    if status == 403:
        return detail or "This request was blocked by the server."
    if status == 413:
        return detail or "The uploaded file is too large."
    if status == 429:
        return detail or "Too many requests right now. Please wait and try again."
    if status >= 500:
        return "The backend hit an unexpected error. Please try again."
    return detail or f"API error: {status}"


def _format_latency(latency_ms: float | int | None) -> str:
    if latency_ms is None:
        return "n/a"
    return f"{float(latency_ms):.0f} ms"


def _format_model_list(model_ids: list[str]) -> str:
    """Render a compact comma-separated list of model labels for captions."""

    if not model_ids:
        return "none"
    return ", ".join(get_model_label(model_id) for model_id in model_ids)


def _frontend_log(event: str, **fields: object) -> None:
    """Emit concise frontend logs for debugging the Streamlit flows."""

    logger.info("frontend_event=%s fields=%s", event, json.dumps(fields, default=str, sort_keys=True))


def _coerce_error_detail(value: object) -> str:
    """Normalize API error payloads into a readable string."""

    if isinstance(value, dict):
        stage = value.get("stage")
        message = value.get("message") or value.get("detail") or json.dumps(value, sort_keys=True)
        if stage:
            return f"{stage}: {message}"
        return str(message)
    return str(value)


def _parse_speech_error_context(message: str) -> tuple[str | None, str]:
    """Split a speech error string into ``(failed_stage, user_detail)`` when possible."""

    raw = str(message).strip()
    if ":" not in raw:
        return None, raw
    stage, detail = raw.split(":", 1)
    normalized = stage.strip().lower()
    known = {
        "browser_permission",
        "audio_capture",
        "audio_blob_creation",
        "upload_validation",
        "upload_request",
        "provider_transcription",
        "backend_response",
        "transcript_insertion",
        "chat_submission",
    }
    if normalized in known:
        return normalized, detail.strip()
    return None, raw


def _speech_stage_label(stage: str) -> str:
    """Human-readable label for speech pipeline stages."""

    labels = {
        SPEECH_STAGE_RECORDING: "Mic",
        SPEECH_STAGE_UPLOADING: "Upload",
        SPEECH_STAGE_TRANSCRIBING: "Transcribe",
        SPEECH_STAGE_TRANSCRIPT_RECEIVED: "Text ready",
        SPEECH_STAGE_TRANSCRIPT_INSERTED: "Sent to chat",
        SPEECH_STAGE_FAILED: "Failed",
        "browser_permission": "Mic permission",
        "audio_capture": "Recording",
        "audio_blob_creation": "Audio clip",
        "upload_validation": "File check",
        "upload_request": "Send to server",
        "provider_transcription": "Speech-to-text",
        "backend_response": "Server reply",
        "transcript_insertion": "Insert text",
        "chat_submission": "Send message",
    }
    return labels.get(stage, stage.replace("_", " ").title())


def _set_speech_failure(stage: str, message: str) -> None:
    """Store a speech failure consistently for UI rendering and logs."""

    failed_stage, detail = _parse_speech_error_context(message)
    actual_stage = failed_stage or stage
    st.session_state.speech_status = SPEECH_STATUS_ERROR
    st.session_state.speech_last_error = f"{actual_stage}: {detail}"
    set_speech_stage(st.session_state, SPEECH_STAGE_FAILED, failed_stage=actual_stage)
    _frontend_log("speech_failed", stage=actual_stage, detail=detail)


def _activate_transcript(*, transcript: str, source: str, file_name: str) -> None:
    """Inject a transcript into the normal chat flow immediately after success."""

    st.session_state.speech_transcript_draft = transcript
    st.session_state.speech_status = SPEECH_STATUS_READY
    st.session_state.speech_last_upload_name = file_name
    st.session_state.speech_last_source = source
    set_speech_stage(st.session_state, SPEECH_STAGE_TRANSCRIPT_RECEIVED)
    _frontend_log(
        "speech_transcript_received",
        source=source,
        file_name=file_name,
        transcript_chars=len(transcript),
    )
    queue_chat_message(st.session_state, transcript)
    st.session_state.speech_auto_send_pending = True
    set_speech_stage(st.session_state, SPEECH_STAGE_TRANSCRIPT_INSERTED)
    _frontend_log("speech_transcript_inserted", source=source, file_name=file_name)


def _get_source_inventory() -> dict | None:
    """Fetch source coverage metadata for the UI side panel."""
    if STREAMLIT_DIRECT_MODE:
        try:
            return get_source_inventory_direct()
        except Exception:
            return None
    if not API_BASE:
        return None
    try:
        resp = httpx.get(
            f"{API_BASE}/health/source-inventory",
            headers=build_request_headers(session_id=st.session_state.session_id),
            timeout=3.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _active_credential_source() -> str:
    source = str(st.session_state.get("active_credential_source", APP_MANAGED_SOURCE))
    validated = str(st.session_state.get("validated_byok_api_key", ""))
    resolved = resolve_credential_source(active_source=source, validated_byok_api_key=validated)
    if resolved != source:
        st.session_state.active_credential_source = resolved
    return resolved


def _active_request_api_key() -> str | None:
    if _active_credential_source() == USER_BYOK_SOURCE:
        key = str(st.session_state.get("validated_byok_api_key", "")).strip()
        return key or None
    return None


def _current_auth_status() -> dict | None:
    if _active_credential_source() == USER_BYOK_SOURCE:
        return st.session_state.get("user_provider_auth_status")
    return st.session_state.get("app_provider_auth_status")


def _queue_byok_clear() -> None:
    """Signal BYOK clear; processed before BYOK widgets instantiate."""

    st.session_state.byok_clear_requested = True


def _apply_pending_byok_clear() -> None:
    """Apply BYOK clear intent safely before widget-bound session keys are created."""

    if not st.session_state.get("byok_clear_requested"):
        return
    before_source = _active_credential_source()
    selected_before = st.session_state.selected_model
    transition = transition_after_clear()
    st.session_state.byok_clear_requested = False
    st.session_state.byok_api_key_input = ""
    st.session_state.validated_byok_api_key = transition.validated_byok_api_key
    st.session_state.user_provider_auth_status = transition.user_provider_auth_status
    st.session_state.app_provider_auth_status = None
    st.session_state.byok_last_validation_error = transition.byok_last_validation_error
    st.session_state.byok_status_notice = transition.byok_status_notice
    st.session_state.active_credential_source = transition.credential_source_after
    _frontend_log(
        "credential_source_transition",
        credential_source_before=before_source,
        credential_source_after=transition.credential_source_after,
        byok_validated=transition.byok_validated,
        byok_cleared=transition.byok_cleared,
        selected_model_before=selected_before,
        selected_model_after=st.session_state.selected_model,
        selector_reconciled_reason="clear_key",
    )


def _sync_app_key_model_catalog() -> None:
    """Ensure app-managed key discovery is loaded once per session."""

    if st.session_state.get("app_provider_auth_status"):
        return
    try:
        result = discover_provider_models(
            api_base=API_BASE,
            session_id=st.session_state.session_id,
            model=st.session_state.selected_model,
            api_key=None,
        )
        st.session_state.app_provider_auth_status = result
        summary = summarize_model_availability(result)
        _frontend_log(
            "credential_model_discovery",
            credential_source="app_managed_key",
            discovered_raw_models_count=summary.raw_accessible_count,
            normalized_models_count=len(result.get("normalized_accessible_models", [])),
            selectable_models_count=summary.selectable_count,
            selected_model_before=st.session_state.selected_model,
            selected_model_after=st.session_state.selected_model,
        )
    except Exception as exc:
        st.session_state.app_provider_auth_status = {
            "ok": False,
            "message": f"Could not load app-managed model catalog: {exc}",
            "selectable_models": get_supported_model_ids(),
        }


def _selected_model_available_for_byok() -> tuple[bool, str | None]:
    """Check whether the selected model is usable for the active credential source."""

    auth_status = _current_auth_status()
    if not auth_status or not auth_status.get("ok"):
        return True, None

    selectable_models = get_available_model_ids(auth_status)
    if not selectable_models:
        return (
            False,
            "Key is valid, but it cannot use any of the models this app supports. "
            "Try another key or clear BYOK to use the app default.",
        )
    if st.session_state.selected_model not in selectable_models:
        allowed = ", ".join(get_model_label(model_id) for model_id in selectable_models)
        return (
            False,
            f"This key cannot use `{get_model_label(st.session_state.selected_model)}`. "
            f"Pick one of: {allowed}.",
        )
    return True, None


def _reset_conversation() -> None:
    for key in ("messages", "citations", "tool_results", "youtube_suggestions"):
        st.session_state[key] = []
    for key in ("last_intent", "last_debug", "last_response_meta", "ui_error"):
        st.session_state[key] = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.cv_text = None
    st.session_state.cv_filename = None
    st.session_state.cv_file_hash = None
    clear_speech_draft(st.session_state)
    st.session_state.speech_last_audio_hash = None
    st.session_state.pop("pending_user_message", None)


def _response_meta_from_state(latency_ms: float) -> dict:
    debug = st.session_state.get("last_debug") or {}
    previous_meta = st.session_state.get("last_response_meta")
    previous_answer_source = previous_meta.get("answer_source") if isinstance(previous_meta, dict) else None
    answer_source = debug.get("answer_source") or previous_answer_source
    answer_length = debug.get("answer_length") or st.session_state.get("answer_length", "balanced")
    return {
        "answer_source": answer_source or "llm_fallback",
        "answer_length": answer_length,
        "latency_ms": round(latency_ms, 1),
        "model": st.session_state.selected_model,
        "intent": st.session_state.get("last_intent"),
        "sources_count": len(st.session_state.get("citations") or []),
        "tool_count": len(st.session_state.get("tool_results") or []),
    }


def _call_non_streaming(body: dict) -> tuple[str, dict]:
    if STREAMLIT_DIRECT_MODE:
        try:
            data = generate_response_direct(
                body=body,
                model=st.session_state.selected_model,
                api_key=_active_request_api_key(),
                user_timezone=_USER_TIMEZONE,
            )
            st.session_state.citations = data.get("citations", [])
            st.session_state.tool_results = data.get("tool_calls", [])
            st.session_state.last_intent = data.get("intent")
            st.session_state.last_debug = {
                "intent": data.get("intent"),
                "retrieval_invoked": bool(data.get("citations")),
                "sources_count": len(data.get("citations", [])),
                "tool_invoked": bool(data.get("tool_calls")),
                "mode": "direct",
                "answer_source": data.get("answer_source"),
                "answer_length": data.get("answer_length", "balanced"),
            }
            return data["reply"], {
                "answer_source": data.get("answer_source"),
                "provider_usage": data.get("usage"),
            }
        except DirectModeError as exc:
            return str(exc), {"error": True}
        except Exception as exc:
            return f"Direct mode error: {exc}", {"error": True}

    if not API_BASE:
        return "External API mode requires CAREER_INTEL_API_BASE_URL.", {"error": True}
    try:
        resp = httpx.post(
            f"{API_BASE}/chat",
            json=body,
            headers=_build_request_headers(),
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state.citations = data.get("citations", [])
        st.session_state.tool_results = data.get("tool_calls", [])
        st.session_state.last_intent = data.get("intent")
        st.session_state.last_debug = {
            "intent": data.get("intent"),
            "retrieval_invoked": bool(data.get("citations")),
            "sources_count": len(data.get("citations", [])),
            "tool_invoked": bool(data.get("tool_calls")),
            "mode": "non-streaming",
            "answer_source": data.get("answer_source"),
            "answer_length": data.get("answer_length", "balanced"),
        }
        return data["reply"], {
            "answer_source": data.get("answer_source"),
            "provider_usage": data.get("usage"),
        }
    except httpx.HTTPStatusError as exc:
        return _friendly_http_error(exc), {"error": True}
    except httpx.ConnectError:
        return "Cannot connect to the API server. Is it running?", {"error": True}
    except Exception as exc:
        return f"Unexpected error: {exc}", {"error": True}


def _call_streaming(body: dict, placeholder) -> tuple[str, dict]:
    if STREAMLIT_DIRECT_MODE:
        reply, meta = _call_non_streaming(body)
        placeholder.markdown(reply)
        return reply, meta

    full_text = ""
    status_text = ""
    provider_usage: dict | None = None
    st.session_state.citations = []
    st.session_state.tool_results = []
    st.session_state.last_intent = None
    st.session_state.last_debug = None

    if not API_BASE:
        message = "External API mode requires CAREER_INTEL_API_BASE_URL."
        placeholder.markdown(message)
        return message, {"error": True}

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/chat/stream",
            json=body,
            headers=_build_request_headers(),
            timeout=90.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                evt = payload.get("type")

                if evt == "intent":
                    st.session_state.last_intent = payload.get("data")
                    if payload.get("data") in ("domain_specific", "tool_required"):
                        status_text = "Searching knowledge base..."
                        placeholder.markdown(f"*{status_text}*")
                elif evt == "token":
                    full_text += payload["content"]
                    placeholder.markdown(full_text + "▌")
                elif evt == "citations":
                    st.session_state.citations = payload["data"]
                elif evt == "tool_calls":
                    st.session_state.tool_results = payload["data"]
                elif evt == "debug":
                    st.session_state.last_debug = payload.get("data")
                elif evt == "error":
                    full_text += f"\n\nWarning: {_summarize_chat_error(str(payload['detail']))}"
                elif evt == "status":
                    status_text = payload.get("detail", "Working...")
                    if full_text:
                        placeholder.markdown(full_text + "▌")
                    else:
                        placeholder.markdown(f"*{status_text}*")
                elif evt == "usage":
                    raw = payload.get("data")
                    provider_usage = raw if isinstance(raw, dict) else None
                elif evt == "done":
                    break

        placeholder.markdown(full_text or f"*{status_text or 'No response returned.'}*")
        debug = st.session_state.get("last_debug") or {}
        return full_text, {"answer_source": debug.get("answer_source"), "provider_usage": provider_usage}
    except httpx.HTTPStatusError as exc:
        message = _friendly_http_error(exc)
        placeholder.markdown(message)
        return message, {"error": True}
    except httpx.ConnectError:
        message = "Cannot connect to the API server. Is it running?"
        placeholder.markdown(message)
        return message, {"error": True}
    except Exception as exc:
        message = f"Streaming error: {exc}"
        placeholder.markdown(message)
        return message, {"error": True}


def _render_compact_error(message: str, *, details: str | None = None) -> None:
    safe = html.escape(message)
    st.markdown(f'<div class="ci-alert">{safe}</div>', unsafe_allow_html=True)
    if details:
        with st.expander("Technical details"):
            st.code(str(details), language=None)


def _summarize_chat_error(message: str) -> str:
    """Sanitize noisy provider/auth failures for chat UI display."""
    lowered = message.lower()
    if "incorrect api key provided" in lowered or "authenticationerror" in lowered:
        return "Model provider authentication failed. Check the configured API credentials."
    return message


def _summarize_speech_error(message: str) -> str:
    """Return a concise, user-facing summary for speech transcription failures."""
    failed_stage, detail = _parse_speech_error_context(message)
    lowered = detail.lower()
    stage_prefix = f'Stopped at "{_speech_stage_label(failed_stage)}". ' if failed_stage else ""
    if "provider credentials are invalid" in lowered:
        return f"{stage_prefix}Audio was captured, but speech-to-text credentials are invalid."
    if "cannot reach" in lowered or "could not reach the provider" in lowered:
        return f"{stage_prefix}Audio was captured, but speech-to-text could not be reached."
    if "timed out" in lowered:
        return f"{stage_prefix}Audio was captured, but speech-to-text timed out."
    return f"{stage_prefix}{detail or 'Something went wrong with voice input. See details below.'}"


def _render_status_pills(status: dict) -> None:
    api_ok = bool(status.get("backend"))
    db_ok = bool(status.get("qdrant"))
    data_ok = bool(status.get("indexed_data_present"))
    api_cls = "ok" if api_ok else "bad"
    db_cls = "ok" if db_ok else "bad"
    data_cls = "ok" if data_ok else "warn"
    st.markdown(
        '<div class="status-row">'
        f'<span class="status-pill {api_cls}"><span class="status-icon">{"✓" if api_ok else "!"}</span><span>API</span></span>'
        f'<span class="status-pill {db_cls}"><span class="status-icon">{"✓" if db_ok else "!"}</span><span>DB</span></span>'
        f'<span class="status-pill {data_cls}"><span class="status-icon">{"✓" if data_ok else "!"}</span><span>Data</span></span>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_usage_summary_card(usage_summary: dict, *, byok_active: bool) -> None:
    """Render a quieter usage summary than Streamlit metrics for the sidebar."""
    requests = int(usage_summary.get("requests", 0))
    total_tokens = int(usage_summary.get("total_tokens", 0))
    estimated_cost = float(usage_summary.get("estimated_cost_usd", 0.0))
    prov_req = int(usage_summary.get("provider_requests", 0))
    total_req = int(usage_summary.get("requests", 0))
    cost_label = "Your key cost" if byok_active and prov_req == total_req and total_req > 0 else "Est. cost"
    note = "Uses provider counts when available." if byok_active and prov_req == total_req and total_req > 0 else (
        "Estimated from public pricing."
        + (" Some turns used local estimates." if prov_req < total_req else "")
    )

    st.markdown(
        (
            '<div class="ci-sidebar-stats">'
            f'<div class="ci-sidebar-stat"><span class="ci-sidebar-stat-label">Requests</span>'
            f'<span class="ci-sidebar-stat-value">{requests}</span></div>'
            f'<div class="ci-sidebar-stat"><span class="ci-sidebar-stat-label">Tokens</span>'
            f'<span class="ci-sidebar-stat-value">{total_tokens:,}</span></div>'
            f'<div class="ci-sidebar-stat"><span class="ci-sidebar-stat-label">{html.escape(cost_label)}</span>'
            f'<span class="ci-sidebar-stat-value">${estimated_cost:.4f}</span></div>'
            "</div>"
            f'<div class="ci-sidebar-usage-note">{html.escape(note)}</div>'
        ),
        unsafe_allow_html=True,
    )

    if usage_summary.get("by_model"):
        rows = []
        for model_name, model_usage in usage_summary["by_model"].items():
            rows.append(
                '<div class="ci-sidebar-model-row">'
                f'<div class="ci-sidebar-model-name">{html.escape(get_model_label(model_name))}</div>'
                f'<div class="ci-sidebar-model-meta">{int(model_usage["requests"])} req · '
                f'{int(model_usage["total_tokens"]):,} tok · ${float(model_usage["estimated_cost_usd"]):.5f}</div>'
                "</div>"
            )
        st.markdown(
            f'<div class="ci-sidebar-model-breakdown">{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )


def _render_message_meta(meta: dict | None) -> None:
    if not meta:
        return
    answer_source = str(meta.get("answer_source", "unknown")).upper()
    length_raw = str(meta.get("answer_length", "balanced"))
    length_label = html.escape(length_raw.replace("_", " ").title())
    latency = _format_latency(meta.get("latency_ms"))
    model = html.escape(str(meta.get("model", "unknown")))
    prompt_tokens = meta.get("prompt_tokens", meta.get("input_tokens"))
    completion_tokens = meta.get("completion_tokens", meta.get("output_tokens"))
    total_tokens = meta.get("total_tokens")
    cost = meta.get("display_cost_usd", meta.get("estimated_cost_usd"))
    cost_title = html.escape(str(meta.get("cost_label_short", "Estimated cost")))
    usage_note = "provider" if meta.get("usage_counts_from_provider") else "estimate"
    bits = [
        f'<span class="status-pill"><span>{answer_source}</span></span>',
        f'<span class="status-pill"><span>Length: {length_label}</span></span>',
        f'<span class="status-pill"><span>{model}</span></span>',
        f'<span class="status-pill"><span>{latency}</span></span>',
    ]
    if prompt_tokens is not None and completion_tokens is not None:
        total_bit = f" · {int(total_tokens)} total" if total_tokens is not None else ""
        bits.append(
            f'<span class="status-pill"><span>'
            f"{int(prompt_tokens)} prompt / {int(completion_tokens)} completion{total_bit}"
            f" ({usage_note})</span></span>"
        )
    if cost is not None:
        bits.append(
            f'<span class="status-pill"><span title="{cost_title}">'
            f"{cost_title}: ${float(cost):.5f}</span></span>"
        )
    st.markdown(f'<div class="status-row">{"".join(bits)}</div>', unsafe_allow_html=True)


def _youtube_data_api_key() -> str:
    return (os.getenv("YOUTUBE_DATA_API_KEY") or os.getenv("CAREER_INTEL_YOUTUBE_API_KEY") or "").strip()


def _refresh_youtube_suggestions(user_query: str) -> None:
    """Populate supplemental YouTube cards (gated; never mixed into RAG citations)."""
    st.session_state.youtube_suggestions = []
    api_key = _youtube_data_api_key()
    if not api_key:
        return
    rate_limit_result = apply_session_rate_limit(
        list(st.session_state.youtube_request_timestamps),
        now=time.time(),
        max_requests=YOUTUBE_RATE_LIMIT_COUNT,
        window_seconds=YOUTUBE_RATE_LIMIT_WINDOW_SECONDS,
        min_interval_seconds=1.0,
    )
    st.session_state.youtube_request_timestamps = list(rate_limit_result.recent_timestamps)
    if not rate_limit_result.allowed:
        return
    intent = st.session_state.get("last_intent")
    dbg = st.session_state.get("last_debug") or {}
    answer_source = dbg.get("answer_source")
    if not should_fetch_youtube_support(
        intent=intent if isinstance(intent, str) else None,
        answer_source=answer_source if isinstance(answer_source, str) else None,
        user_query=user_query,
    ):
        return
    st.session_state.youtube_suggestions = fetch_youtube_suggestions(
        user_query,
        api_key=api_key,
        max_results=3,
    )


def _render_sources_panel() -> None:
    """Right rail: corpus note, grouped deduplicated citations, optional debug."""
    st.markdown(
        """
<div class="ci-sources-card-marker"></div>
<div class="ci-panel-head">
  <div style="display:flex; align-items:flex-start; gap:0.8rem;">
    <div class="ci-panel-icon" aria-hidden="true">↗</div>
    <div>
      <h3>Sources</h3>
      <p>Structured citations from the latest grounded answer (WEF, ESCO, and more later).</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    source_inventory = _get_source_inventory()
    if source_inventory:
        coverage_copy = (
            f"{source_inventory.get('total_source_groups', 0)} source groups · "
            f"{source_inventory.get('total_files_present', 0)} raw files."
        )
        st.info(
            "**Corpus** · "
            + coverage_copy
            + " Ask *what sources do you use?* for the full list."
        )
        if source_inventory.get("esco_status_note"):
            st.caption(str(source_inventory["esco_status_note"]))
    cites = [c for c in (st.session_state.get("citations") or []) if isinstance(c, dict)]
    yt_raw = st.session_state.get("youtube_suggestions") or []
    yt_videos = [v for v in yt_raw if isinstance(v, dict)]
    yt_html = format_youtube_sources_html(yt_videos) if yt_videos else ""

    if not cites and not yt_html:
        st.markdown(
            """
<div class="ci-source-empty">
  <div class="empty-icon" aria-hidden="true">◇</div>
  <div class="empty-title">No sources available</div>
  <div class="empty-copy">When the assistant uses indexed knowledge, merged citations with excerpts appear here. Related YouTube clips may appear below when enabled.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if DEV_MODE:
            st.caption("Sources debug: 0 citations · unique: 0 · types: —")
        return

    if not cites:
        if yt_html:
            st.markdown(yt_html, unsafe_allow_html=True)
        if DEV_MODE:
            st.caption("Sources debug: 0 RAG citations · YouTube supplemental only")
        return

    cards, dbg = prepare_sources_panel_rows(cites, max_sources=5)
    fams = ", ".join(str(x).upper() for x in dbg["families"]) if dbg["families"] else "—"
    trunc = int(dbg.get("truncated") or 0)
    summary = (
        f"{dbg['unique_sources']} unique source(s) · types: {fams}"
        + (f" · +{trunc} not listed (cap {dbg['displayed']})" if trunc else "")
    )
    st.markdown(f'<div class="ci-sources-summary">{html.escape(summary)}</div>', unsafe_allow_html=True)
    if DEV_MODE:
        st.caption(
            f"Sources debug: {dbg['raw_citation_count']} raw citations · "
            f"listed {dbg['displayed']} / {dbg['unique_sources']} unique after merge"
        )

    grouped = group_by_family_ordered(cards)
    for family, items in grouped:
        title = html.escape(family_section_title(family))
        st.markdown(
            f'<div class="ci-source-group-head">{family_icon_html(family)}<p class="ci-sources-group-head">{title}</p></div>',
            unsafe_allow_html=True,
        )
        for card in items:
            label_bits = [card.type_label, card.title if len(card.title) < 56 else (card.title[:53] + "…")]
            if card.sublabel:
                label_bits.append(card.sublabel)
            if card.reference_count > 1:
                label_bits.append(f"{card.reference_count} refs")
            exp_label = " · ".join(label_bits)
            with st.expander(exp_label, expanded=False):
                st.markdown(render_merged_card_html(card), unsafe_allow_html=True)
                st.code(format_detail_lines_for_code(card), language=None)

    if yt_html:
        st.markdown(yt_html, unsafe_allow_html=True)


def _render_tools_and_debug() -> None:
    has_tools = bool(st.session_state.tool_results)
    has_debug = bool(DEV_MODE and st.session_state.get("last_debug"))
    if not has_tools and not has_debug:
        return

    if has_tools:
        with st.expander(f"Tool runs ({len(st.session_state.tool_results)})", expanded=False):
            for tr in st.session_state.tool_results:
                ok = tr.get("success")
                label = f"{'OK' if ok else 'Err'} · {tr['tool_name']}"
                with st.expander(label, expanded=False):
                    st.json(tr.get("output", {}))
                    if tr.get("error"):
                        _render_compact_error(str(tr["error"]))

    if has_debug:
        with st.expander("Debug", expanded=False):
            st.json(st.session_state.last_debug)


def _render_speech_stage_timeline() -> None:
    """Show the current speech pipeline status in a compact checklist."""

    stages = [
        SPEECH_STAGE_RECORDING,
        SPEECH_STAGE_UPLOADING,
        SPEECH_STAGE_TRANSCRIBING,
        SPEECH_STAGE_TRANSCRIPT_RECEIVED,
        SPEECH_STAGE_TRANSCRIPT_INSERTED,
    ]
    history = set(str(item) for item in st.session_state.get("speech_stage_history", []))
    current_stage = str(st.session_state.get("speech_stage", SPEECH_STAGE_IDLE))
    failed_stage = st.session_state.get("speech_failed_stage")

    lines: list[str] = ["**Steps**"]
    for stage in stages:
        status = "pending"
        if stage in history:
            status = "done"
        if current_stage == stage and stage not in history:
            status = "active"
        if failed_stage == stage:
            status = "failed"
        icon = {"done": "✓", "active": "→", "pending": "○", "failed": "✗"}.get(status, "○")
        lines.append(f"{icon} {_speech_stage_label(stage)}")

    if current_stage == SPEECH_STAGE_FAILED and failed_stage and failed_stage not in stages:
        lines.append(f"✗ {_speech_stage_label(str(failed_stage))}")

    st.markdown("\n".join(lines))


def _process_cv_upload(uploaded_cv) -> None:
    cv_bytes = uploaded_cv.getvalue()
    validation_error = validate_uploaded_file(
        filename=uploaded_cv.name,
        size_bytes=len(cv_bytes),
        allowed_extensions=CV_ALLOWED_EXTENSIONS,
        max_bytes=CV_MAX_BYTES,
    )
    if validation_error:
        st.error(validation_error)
        return

    file_hash = hashlib.sha1(cv_bytes).hexdigest()
    if st.session_state.cv_file_hash == file_hash:
        return

    try:
        with st.spinner("Processing CV..."):
            if STREAMLIT_DIRECT_MODE:
                result = process_cv_upload_direct(filename=uploaded_cv.name, data=cv_bytes)
            else:
                if not API_BASE:
                    raise RuntimeError("External API mode requires CAREER_INTEL_API_BASE_URL.")
                resp = httpx.post(
                    f"{API_BASE}/cv/process",
                    files={"file": (uploaded_cv.name, cv_bytes)},
                    headers=build_request_headers(session_id=st.session_state.session_id),
                    timeout=30.0,
                )
                resp.raise_for_status()
                result = resp.json()
        st.session_state.cv_text = result["cv_text"]
        st.session_state.cv_filename = result["filename"]
        st.session_state.cv_file_hash = file_hash
        if result.get("warnings"):
            for warning in result["warnings"]:
                st.warning(str(warning))
        st.success(f"Loaded {uploaded_cv.name}")
    except httpx.HTTPStatusError as exc:
        st.error(_friendly_http_error(exc))
        st.session_state.cv_text = None
        st.session_state.cv_filename = None
        st.session_state.cv_file_hash = None
    except Exception as exc:
        st.error(str(exc))
        st.session_state.cv_text = None
        st.session_state.cv_filename = None
        st.session_state.cv_file_hash = None


def _transcribe_audio_file(*, file_name: str, content_type: str, data: bytes, source: str) -> None:
    _frontend_log(
        "speech_blob_created",
        source=source,
        file_name=file_name,
        content_type=content_type,
        bytes=len(data),
    )
    validation_error = validate_uploaded_file(
        filename=file_name,
        size_bytes=len(data),
        allowed_extensions=AUDIO_ALLOWED_EXTENSIONS,
        max_bytes=AUDIO_MAX_BYTES,
    )
    if validation_error:
        _set_speech_failure("upload_validation", validation_error)
        return

    byok_allowed, byok_message = _selected_model_available_for_byok()
    if not byok_allowed and byok_message:
        _set_speech_failure("upload_request", byok_message)
        return

    st.session_state.speech_status = SPEECH_STATUS_TRANSCRIBING
    st.session_state.speech_last_error = None
    set_speech_stage(st.session_state, SPEECH_STAGE_UPLOADING)
    _frontend_log("speech_uploading", source=source, file_name=file_name)
    try:
        set_speech_stage(st.session_state, SPEECH_STAGE_TRANSCRIBING)
        with st.spinner("Sending audio for transcription…"):
            response = post_speech_transcribe(
                api_base=API_BASE,
                session_id=st.session_state.session_id,
                audio_bytes=data,
                filename=file_name,
                content_type=content_type,
                source=source,
                model=st.session_state.selected_model,
                api_key=_active_request_api_key(),
            )
        transcript = str(response.get("text", "")).strip()
        if not transcript:
            _set_speech_failure("backend_response", "backend_response: Transcription returned no usable text.")
            return
        _activate_transcript(transcript=transcript, source=source, file_name=file_name)
    except httpx.HTTPStatusError as exc:
        _set_speech_failure("provider_transcription", _friendly_http_error(exc))
    except httpx.ConnectError:
        _set_speech_failure("upload_request", f"upload_request: Cannot reach `{API_BASE}`.")
    except Exception as exc:
        _set_speech_failure("backend_response", f"backend_response: {exc}")


st.set_page_config(
    page_title="Career Intelligence Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
_sync_sidebar_width_state_from_query()
st.markdown(_render_custom_css(), unsafe_allow_html=True)
_inject_sidebar_resizer_bridge()

for key, default in {
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "citations": [],
    "tool_results": [],
    "youtube_suggestions": [],
    "cv_text": None,
    "cv_filename": None,
    "last_intent": None,
    "last_debug": None,
    "speech_last_audio_hash": None,
    "use_tools": True,
    "enable_streaming": True,
    "require_citations": True,
    "selected_model": get_default_model_id(),
    "byok_api_key_input": "",
    "validated_byok_api_key": "",
    "active_credential_source": APP_MANAGED_SOURCE,
    "user_provider_auth_status": None,
    "app_provider_auth_status": None,
    "byok_last_validation_error": None,
    "byok_status_notice": None,
    "byok_clear_requested": False,
    "model_switch_notice": None,
    "last_response_meta": None,
    "answer_length": "balanced",
    "byok_validation_timestamps": [],
    "youtube_request_timestamps": [],
    "usage_summary": {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "provider_requests": 0,
        "by_model": {},
    },
    "request_timestamps": [],
    "ui_error": None,
    "cv_file_hash": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

ensure_speech_session_keys(st.session_state)
apply_deferred_speech_clear(st.session_state)
_apply_pending_byok_clear()
_sync_app_key_model_catalog()

system_status = _get_system_status()
backend_ok = bool(system_status.get("backend"))

with st.sidebar:
    auth_status = _current_auth_status()
    credential_source = _active_credential_source()
    model_summary = summarize_model_availability(auth_status)
    model_ids = model_summary.selectable_models
    selection_resolution = resolve_selected_model(
        current_model=st.session_state.selected_model,
        available_models=model_ids,
    )
    if selection_resolution.changed:
        selected_before = st.session_state.selected_model
        st.session_state.selected_model = selection_resolution.selected_model
        st.session_state.model_switch_notice = (
            f"Selected model not available for this key. Switched to {get_model_label(selection_resolution.selected_model)}."
        )
        _frontend_log(
            "selected_model_fallback",
            credential_source=credential_source,
            discovered_raw_models_count=len((auth_status or {}).get("accessible_models", [])),
            normalized_models_count=len((auth_status or {}).get("normalized_accessible_models", [])),
            selectable_models_count=len(model_ids),
            selected_model_before=selected_before,
            selected_model_after=selection_resolution.selected_model,
            selected_model_fallback_reason=selection_resolution.reason_code,
            selector_reconciled_reason=selection_resolution.reason_code,
        )
    selected_index = model_ids.index(st.session_state.selected_model) if st.session_state.selected_model in model_ids else 0
    disable_model_select = bool(auth_status and auth_status.get("ok") and not model_ids)
    recommended_model_ids = [item for item in get_recommended_model_ids() if item in model_ids]

    with st.container(border=True):
        st.markdown('<div class="ci-side-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="ci-side-section"><span class="sec-ic" aria-hidden="true">◈</span> Model</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<p class="ci-sidebar-kicker">Assistant setup</p>', unsafe_allow_html=True)
        if model_ids:
            st.session_state.selected_model = st.selectbox(
                "Active model",
                options=model_ids,
                index=selected_index,
                format_func=get_model_label,
                disabled=disable_model_select,
            )
        else:
            st.text_input("Active model", value="No supported models available", disabled=True)
        active_model_label = get_model_label(st.session_state.selected_model)
        model_description = html.escape(get_model_description(st.session_state.selected_model))
        pricing_row_html = ""
        if st.session_state.selected_model in get_supported_model_ids():
            pricing = get_model_pricing(st.session_state.selected_model)
            pricing_row_html = (
                '<div class="ci-model-meta-row">'
                '<span class="ci-model-price-chip">Pricing</span>'
                f'<span class="ci-model-price-text">{html.escape(pricing.label)} · '
                f'${pricing.input_cost_per_million_tokens:.2f}/M input · '
                f'${pricing.output_cost_per_million_tokens:.2f}/M output</span>'
                "</div>"
            )
        recommended_row_html = ""
        if recommended_model_ids:
            recommended_labels = ", ".join(get_model_label(item) for item in recommended_model_ids)
            recommended_row_html = (
                '<div class="ci-model-meta-row">'
                '<span class="ci-model-price-chip">Recommended</span>'
                f'<span class="ci-model-price-text">{html.escape(recommended_labels)}</span>'
                "</div>"
            )
        st.markdown(
            (
                '<div class="ci-model-meta-block">'
                f'<p class="ci-model-meta-text">{model_description}</p>'
                f"{pricing_row_html}"
                f"{recommended_row_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.session_state.get("model_switch_notice"):
            st.info(str(st.session_state.model_switch_notice))
            st.session_state.model_switch_notice = None
        if auth_status and auth_status.get("ok"):
            availability_owner = "your key" if credential_source == USER_BYOK_SOURCE else "the app key"
            st.caption("A valid API key does not guarantee all models are supported in this app.")
            st.caption(f"Selectable models: {_format_model_list(model_summary.selectable_models)}")
            if model_summary.supported_but_unavailable_models:
                st.caption(
                    "Supported but not available with "
                    f"{availability_owner}: {_format_model_list(model_summary.supported_but_unavailable_models)}"
                )
            if model_summary.accessible_but_unsupported_models:
                st.caption(
                    "Available with "
                    f"{availability_owner} but not supported in this app: "
                    f"{_format_model_list(model_summary.accessible_but_unsupported_models)}"
                )

        _al_labels = ("Concise", "Balanced", "Detailed")
        _al_values = ("concise", "balanced", "detailed")
        _cur_al = st.session_state.get("answer_length", "balanced")
        _al_index = _al_values.index(_cur_al) if _cur_al in _al_values else 1
        st.markdown(
            '<p class="ci-side-section"><span class="sec-ic" aria-hidden="true">≡</span> Answer length</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="ci-side-muted">Choose the tone-depth tradeoff for each answer.</p>',
            unsafe_allow_html=True,
        )
        _picked = st.radio(
            "Answer length",
            options=_al_labels,
            index=_al_index,
            horizontal=True,
            help="Controls how detailed replies are. Does not change which sources are retrieved.",
        )
        st.session_state.answer_length = _al_values[_al_labels.index(_picked)]

        st.markdown(
            '<p class="ci-side-section"><span class="sec-ic" aria-hidden="true">⌁</span> Access</p>',
            unsafe_allow_html=True,
        )
        st.text_input(
            "Bring your own OpenAI key",
            key="byok_api_key_input",
            type="password",
            placeholder="sk-...",
            help="Type your key, then validate to activate it for requests.",
        )
        auth_cols = st.columns(2)
        with auth_cols[0]:
            if st.button("Validate key", use_container_width=True):
                input_key = str(st.session_state.byok_api_key_input).strip()
                if not input_key:
                    st.session_state.byok_last_validation_error = "Enter an API key before validation."
                    _frontend_log(
                        "credential_source_transition",
                        credential_source_before=_active_credential_source(),
                        credential_source_after=_active_credential_source(),
                        byok_validated=False,
                        byok_cleared=False,
                        selected_model_before=st.session_state.selected_model,
                        selected_model_after=st.session_state.selected_model,
                        selector_reconciled_reason="missing_key",
                    )
                else:
                    byok_rate_limit = apply_session_rate_limit(
                        list(st.session_state.byok_validation_timestamps),
                        now=time.time(),
                        max_requests=BYOK_RATE_LIMIT_COUNT,
                        window_seconds=BYOK_RATE_LIMIT_WINDOW_SECONDS,
                        min_interval_seconds=2.0,
                    )
                    st.session_state.byok_validation_timestamps = list(byok_rate_limit.recent_timestamps)
                    if not byok_rate_limit.allowed:
                        st.session_state.byok_last_validation_error = byok_rate_limit.reason
                        st.rerun()
                    try:
                        with st.spinner("Checking key…"):
                            auth_result = validate_provider_key(
                                api_base=API_BASE,
                                session_id=st.session_state.session_id,
                                model=st.session_state.selected_model,
                                api_key=input_key,
                            )
                        transition = transition_after_validation(
                            credential_source_before=_active_credential_source(),
                            validated_byok_api_key_before=str(st.session_state.validated_byok_api_key),
                            user_provider_auth_status_before=st.session_state.get("user_provider_auth_status"),
                            byok_input_key=input_key,
                            auth_result=auth_result,
                        )
                        source_before = _active_credential_source()
                        selected_before = st.session_state.selected_model
                        st.session_state.active_credential_source = transition.credential_source_after
                        st.session_state.validated_byok_api_key = transition.validated_byok_api_key
                        if transition.byok_validated:
                            st.session_state.byok_api_key_input = ""
                        st.session_state.user_provider_auth_status = transition.user_provider_auth_status
                        st.session_state.byok_last_validation_error = transition.byok_last_validation_error
                        st.session_state.byok_status_notice = transition.byok_status_notice
                        model_summary = summarize_model_availability(
                            auth_result if transition.byok_validated else _current_auth_status()
                        )
                        _frontend_log(
                            "credential_source_transition",
                            credential_source_before=source_before,
                            credential_source_after=transition.credential_source_after,
                            byok_validated=transition.byok_validated,
                            byok_cleared=transition.byok_cleared,
                            discovered_raw_models_count=model_summary.raw_accessible_count,
                            normalized_models_count=len((auth_result or {}).get("normalized_accessible_models", [])),
                            selectable_models_count=model_summary.selectable_count,
                            selected_model_before=selected_before,
                            selected_model_after=st.session_state.selected_model,
                            selector_reconciled_reason="validate_success"
                            if transition.byok_validated
                            else "validate_failed",
                        )
                    except httpx.HTTPStatusError as exc:
                        st.session_state.byok_last_validation_error = _friendly_http_error(exc)
                        _frontend_log(
                            "credential_source_transition",
                            credential_source_before=_active_credential_source(),
                            credential_source_after=_active_credential_source(),
                            byok_validated=False,
                            byok_cleared=False,
                            selected_model_before=st.session_state.selected_model,
                            selected_model_after=st.session_state.selected_model,
                            selector_reconciled_reason="validate_http_error",
                        )
                    except Exception as exc:
                        st.session_state.byok_last_validation_error = str(exc)
                        _frontend_log(
                            "credential_source_transition",
                            credential_source_before=_active_credential_source(),
                            credential_source_after=_active_credential_source(),
                            byok_validated=False,
                            byok_cleared=False,
                            selected_model_before=st.session_state.selected_model,
                            selected_model_after=st.session_state.selected_model,
                            selector_reconciled_reason="validate_exception",
                        )
                st.rerun()
        with auth_cols[1]:
            st.button("Clear key", use_container_width=True, on_click=_queue_byok_clear)

        auth_status = _current_auth_status()
        credential_source = _active_credential_source()
        st.markdown(
            f'<p class="ci-side-key-status">Key source: {"Your validated key" if credential_source == USER_BYOK_SOURCE else "App managed key"}</p>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("byok_status_notice"):
            st.info(str(st.session_state.byok_status_notice))
            st.session_state.byok_status_notice = None
        if st.session_state.get("byok_last_validation_error"):
            st.error(str(st.session_state.byok_last_validation_error))
        if auth_status:
            if auth_status.get("ok"):
                if credential_source == USER_BYOK_SOURCE:
                    st.success(auth_status.get("message") or "Key validated.")
            else:
                if credential_source == USER_BYOK_SOURCE:
                    st.error(auth_status.get("message") or "Key not accepted.")
            st.caption(f"Available to this key: {model_summary.selectable_count}")
            if model_summary.selectable_models:
                st.caption("Selectable here: " + _format_model_list(model_summary.selectable_models))
            else:
                st.caption("Selectable here: none")
            if model_summary.ignored_raw_models_count:
                st.caption(f"Hidden irrelevant variants: {model_summary.ignored_raw_models_count}")
            if DEV_MODE:
                with st.expander("Model discovery debug", expanded=False):
                    st.json(
                        {
                            "credential_source": credential_source,
                            "supported_models": get_supported_model_ids(),
                            "recommended_models": recommended_model_ids,
                            "selectable_models": model_summary.selectable_models,
                            "raw_accessible_models": auth_status.get("accessible_models", []),
                            "normalized_accessible_models": auth_status.get("normalized_accessible_models", []),
                            "ignored_accessible_models": auth_status.get("ignored_accessible_models", []),
                            "model_unavailability_reasons": auth_status.get("model_unavailability_reasons", {}),
                        }
                    )

    with st.container(border=True):
        st.markdown('<div class="ci-side-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="ci-side-section"><span class="sec-ic" aria-hidden="true">◔</span> Usage</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="ci-side-muted">A quiet session-level view of cost and token usage.</p>',
            unsafe_allow_html=True,
        )
        usage_summary = st.session_state.usage_summary
        byok_active = _active_credential_source() == USER_BYOK_SOURCE
        _render_usage_summary_card(usage_summary, byok_active=byok_active)

    with st.container(border=True):
        st.markdown('<div class="ci-side-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="ci-side-section"><span class="sec-ic" aria-hidden="true">⚙</span> Settings</p>',
            unsafe_allow_html=True,
        )
        with st.expander("Preferences", expanded=True):
            st.session_state.use_tools = st.toggle("Tool calling", value=st.session_state.use_tools)
            st.session_state.enable_streaming = st.toggle("Streaming", value=st.session_state.enable_streaming)
            st.session_state.require_citations = st.toggle(
                "Warn if no sources", value=st.session_state.require_citations
            )
            st.caption(
                f"Sidebar width: {int(st.session_state.sidebar_width_px)} px · drag the right-edge grip to resize."
            )
            if st.button("Reset width", key="sidebar_reset_width", use_container_width=False):
                _reset_sidebar_width_state()
                st.rerun()
            st.caption("Chat history, model choice, and uploads persist across reruns in this session.")

    with st.container(border=True):
        st.markdown('<div class="ci-side-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="ci-side-section"><span class="sec-ic" aria-hidden="true">i</span> Workspace</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
Use this assistant for career questions, role comparisons, interview prep, and skills planning.

Ask naturally, upload a CV for more personalized answers, and review the source/type/latency chips after each response.
""".strip()
        )
        _render_status_pills(system_status)

        with st.expander("CV / resume", expanded=False):
            st.caption("Session only. Files are processed server-side after validation.")
            uploaded_cv = st.file_uploader(
                "Upload CV",
                type=sorted(CV_ALLOWED_EXTENSIONS),
                label_visibility="collapsed",
                help="PDF, DOCX, or TXT up to 5 MB",
            )
            if uploaded_cv is not None:
                _process_cv_upload(uploaded_cv)
            elif st.session_state.cv_text:
                st.caption(st.session_state.cv_filename or "CV attached")

            if st.session_state.cv_text:
                prev_row, rm_row = st.columns(2)
                with prev_row, st.expander("Preview", expanded=False):
                    preview = st.session_state.cv_text[:500]
                    if len(st.session_state.cv_text) > 500:
                        preview += "..."
                    st.text(preview)
                with rm_row:
                    if st.button("Remove CV", use_container_width=True):
                        st.session_state.cv_text = None
                        st.session_state.cv_filename = None
                        st.session_state.cv_file_hash = None
                        st.rerun()

        with st.expander("Speech", expanded=False):
            st.caption("Use if the main mic button is blocked or unreliable.")
            speech_file = st.file_uploader(
                "Audio file",
                type=sorted(AUDIO_ALLOWED_EXTENSIONS),
                key="speech_audio_file",
                label_visibility="collapsed",
            )
            transcribe_file = st.button(
                "Transcribe upload",
                use_container_width=True,
                disabled=speech_file is None
                or st.session_state.get("speech_status") == SPEECH_STATUS_TRANSCRIBING,
            )
            if transcribe_file and speech_file is not None:
                _transcribe_audio_file(
                    file_name=speech_file.name,
                    content_type=speech_file.type or "application/octet-stream",
                    data=speech_file.getvalue(),
                    source=SPEECH_SOURCE_UPLOAD,
                )
                st.rerun()

        st.divider()
        col_new, col_export = st.columns(2)
        with col_new:
            if st.button("New chat", use_container_width=True):
                _reset_conversation()
                st.rerun()
        with col_export:
            if st.session_state.messages:
                st.download_button(
                    "Export",
                    use_container_width=True,
                    data=json.dumps(st.session_state.messages, indent=2, default=str),
                    file_name="conversation.json",
                )

        st.markdown(
            """
<div class="ci-sidebar-footer">
  <strong>Trust & safety</strong>
  <p>Custom keys stay in session state only. Uploads are type and size checked before sending.</p>
</div>
""",
            unsafe_allow_html=True,
        )

col_chat, col_sources = st.columns([3.35, 1.35], gap="large")
sources_panel = col_sources.container()

audio_clip = None
user_input: str | None = None

with col_chat:
    if st.session_state.get("ui_error"):
        _render_compact_error(str(st.session_state.ui_error))

    if st.session_state.messages:
        st.markdown(
            """
<div class="ci-hero ci-hero-compact">
  <div class="hero-row">
    <div class="hero-mark" aria-hidden="true">CI</div>
    <div>
      <p class="ci-surface-kicker">Grounded AI assistant</p>
      <h1>Career Intelligence</h1>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="ci-hero">
  <div class="hero-row">
    <div class="hero-mark" aria-hidden="true">CI</div>
    <div>
      <p class="ci-surface-kicker">Grounded AI assistant</p>
      <h1>Career Intelligence</h1>
      <p class="sub">Career guidance with grounded answers, model controls, cost awareness, and secure session-only personalization.</p>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    if not st.session_state.messages:
        with st.container():
            st.markdown(
                '<div class="ci-chip-row"><p class="ci-try-label">Try asking</p></div>',
                unsafe_allow_html=True,
            )
            chip_cols = st.columns(len(_STARTER_PROMPTS), gap="small")
            for idx, col in enumerate(chip_cols):
                with col:
                    label, prompt = _STARTER_PROMPTS[idx]
                    if st.button(label, key=f"starter_{idx}", use_container_width=True):
                        queue_chat_message(st.session_state, prompt)
                        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_message_meta(msg.get("meta"))

    _render_tools_and_debug()

    draft = str(st.session_state.get("speech_transcript_draft", "")).strip()
    speech_err = st.session_state.get("speech_last_error")
    speech_st = st.session_state.get("speech_status")
    speech_stage = str(st.session_state.get("speech_stage", SPEECH_STAGE_IDLE))
    speech_source = st.session_state.get("speech_last_source")
    speech_name = st.session_state.get("speech_last_upload_name")
    auto_send_pending = bool(st.session_state.get("speech_auto_send_pending"))
    show_transcript_card = speech_stage != SPEECH_STAGE_IDLE or bool(draft) or (
        speech_st == SPEECH_STATUS_ERROR and speech_err
    )

    if show_transcript_card:
        if speech_stage == SPEECH_STAGE_RECORDING:
            voice_title = "Mic"
            voice_sub = "Allow the browser mic prompt if you see one, then record."
            voice_source = "Ready"
        elif speech_stage == SPEECH_STAGE_UPLOADING:
            voice_title = "Upload"
            voice_sub = "Checking the file, then sending it for transcription."
            voice_source = "Sending"
        elif speech_stage == SPEECH_STAGE_TRANSCRIBING or speech_st == SPEECH_STATUS_TRANSCRIBING:
            voice_title = "Transcribe"
            voice_sub = "Turning your clip into text."
            voice_source = "Working"
        elif speech_stage == SPEECH_STAGE_TRANSCRIPT_INSERTED and auto_send_pending:
            voice_title = "Sending"
            voice_sub = "Your text is going to the assistant as a normal chat message."
            voice_source = "Chat"
        elif speech_st == SPEECH_STATUS_ERROR and speech_err:
            voice_title = "Voice input issue"
            voice_sub = "See which step failed below, or upload audio in the sidebar."
            voice_source = "Error"
        else:
            voice_title = "Transcript"
            voice_sub = "You can edit and resend, or discard."
            if speech_source == SPEECH_SOURCE_UPLOAD:
                voice_source = f"Upload · {speech_name or 'audio file'}"
            elif speech_source == SPEECH_SOURCE_MIC:
                voice_source = "Mic"
            else:
                voice_source = "Voice"

        with st.container(border=True):
            st.markdown(
                f"""
<div class="ci-voice-card-marker"></div>
<div class="ci-voice-head">
  <div>
    <p class="ci-surface-kicker">Voice draft</p>
    <h3 class="ci-voice-title">{html.escape(voice_title)}</h3>
    <p class="ci-voice-sub">{html.escape(voice_sub)}</p>
  </div>
  <div class="ci-voice-source">{html.escape(voice_source)}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            _render_speech_stage_timeline()
            if speech_st == SPEECH_STATUS_ERROR and speech_err:
                st.markdown(
                    f'<div class="ci-draft-error">{html.escape(_summarize_speech_error(str(speech_err)))}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Technical details"):
                    st.code(str(speech_err), language=None)
            if draft or speech_st == SPEECH_STATUS_ERROR:
                st.text_area(
                    "Edit transcript",
                    key="speech_transcript_draft",
                    height=112,
                    label_visibility="collapsed",
                    placeholder="Transcript appears here…",
                    disabled=auto_send_pending,
                )
                if not auto_send_pending:
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button(
                            "Send transcript",
                            use_container_width=True,
                            type="primary",
                            disabled=not str(st.session_state.get("speech_transcript_draft", "")).strip(),
                        ):
                            st.session_state.ui_error = None
                            queue_chat_message(st.session_state, str(st.session_state.speech_transcript_draft))
                            set_speech_stage(st.session_state, SPEECH_STAGE_TRANSCRIPT_INSERTED)
                            st.session_state.speech_auto_send_pending = True
                            st.rerun()
                    with b2:
                        if st.button("Discard", use_container_width=True):
                            schedule_clear_speech_draft(st.session_state)
                            st.rerun()

    with st.container(border=True):
        st.markdown(
            f"""
<div class="ci-composer-card-marker"></div>
<div class="ci-sticky-composer"></div>
<div class="ci-composer-head">
  <div>
    <p class="ci-surface-kicker">Ask anything</p>
    <h3>Message Career Intelligence</h3>
    <p>Grounded guidance for roles, skills, interviews, and career moves with source-backed answers when available.</p>
  </div>
  <div class="ci-composer-note">Model: {html.escape(active_model_label)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        composer_mic, composer_in = st.columns([0.9, 7.1], gap="small")
        with composer_mic, st.popover("🎙", use_container_width=True):
            st.caption("Record, allow the mic if asked, stop — then we transcribe and send.")
            if st.button("Mic checklist", key="speech_prepare_mic", use_container_width=True):
                st.session_state.speech_last_error = None
                set_speech_stage(st.session_state, SPEECH_STAGE_RECORDING)
                _frontend_log("speech_recording_prepared", source=SPEECH_SOURCE_MIC)
            audio_clip = st.audio_input(
                "Voice input",
                key=f"composer_audio_{st.session_state.speech_mic_key}",
                label_visibility="collapsed",
            )
        with composer_in:
            user_input = st.chat_input("Ask about roles, skills, or your next career move…")

    if speech_stage == SPEECH_STAGE_RECORDING:
        st.markdown(
            '<p class="ci-mic-state">Mic: allow access if the browser asks, then start recording.</p>',
            unsafe_allow_html=True,
        )
    elif speech_stage == SPEECH_STAGE_UPLOADING:
        st.markdown('<p class="ci-mic-state">Uploading your clip…</p>', unsafe_allow_html=True)
    elif speech_st == SPEECH_STATUS_TRANSCRIBING:
        st.markdown('<p class="ci-mic-state">Transcribing…</p>', unsafe_allow_html=True)
    elif speech_st == SPEECH_STATUS_ERROR and speech_err and not show_transcript_card:
        st.markdown(
            f'<p class="ci-mic-state is-error">{html.escape(_summarize_speech_error(str(speech_err)))}</p>',
            unsafe_allow_html=True,
        )

if audio_clip is None:
    st.session_state.speech_last_audio_hash = None
else:
    clip_bytes = audio_clip.getvalue()
    clip_hash = hashlib.sha1(clip_bytes).hexdigest()
    if clip_hash != st.session_state.get("speech_last_audio_hash"):
        set_speech_stage(st.session_state, SPEECH_STAGE_RECORDING)
        st.session_state.speech_status = SPEECH_STATUS_TRANSCRIBING
        st.session_state.speech_last_error = None
        try:
            _transcribe_audio_file(
                file_name=getattr(audio_clip, "name", "recording.wav"),
                content_type=getattr(audio_clip, "type", "audio/wav"),
                data=clip_bytes,
                source=SPEECH_SOURCE_MIC,
            )
        finally:
            st.session_state.speech_last_audio_hash = clip_hash
            st.session_state.speech_mic_key = int(st.session_state.speech_mic_key) + 1
        st.rerun()

pending_msg = st.session_state.pop("pending_user_message", None)
speech_auto_send_now = bool(st.session_state.get("speech_auto_send_pending"))
if pending_msg is not None:
    message_to_send = pending_msg.strip()
elif user_input:
    message_to_send = user_input.strip()
else:
    message_to_send = ""

if message_to_send:
    st.session_state.ui_error = None
    byok_allowed, byok_message = _selected_model_available_for_byok()
    if not byok_allowed and byok_message:
        st.session_state.ui_error = byok_message
        if speech_auto_send_now:
            st.session_state.speech_auto_send_pending = False
            _set_speech_failure("chat_submission", f"chat_submission: {byok_message}")
        st.rerun()

    prompt_risk = inspect_prompt(message_to_send)
    if prompt_risk.flagged:
        st.session_state.ui_error = (
            "This message looks like a prompt-injection or system-exfiltration attempt. "
            "Please rephrase your request."
        )
        if speech_auto_send_now:
            st.session_state.speech_auto_send_pending = False
            _set_speech_failure("chat_submission", f"chat_submission: {st.session_state.ui_error}")
        st.rerun()

    rate_limit_result = apply_session_rate_limit(
        list(st.session_state.request_timestamps),
        now=time.time(),
        max_requests=SESSION_RATE_LIMIT_COUNT,
        window_seconds=SESSION_RATE_LIMIT_WINDOW_SECONDS,
        min_interval_seconds=SESSION_MIN_INTERVAL_SECONDS,
    )
    st.session_state.request_timestamps = list(rate_limit_result.recent_timestamps)
    if not rate_limit_result.allowed:
        st.session_state.ui_error = rate_limit_result.reason
        if speech_auto_send_now:
            st.session_state.speech_auto_send_pending = False
            _set_speech_failure("chat_submission", f"chat_submission: {rate_limit_result.reason}")
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": message_to_send})
    st.session_state.youtube_suggestions = []
    if speech_auto_send_now:
        st.session_state.speech_auto_send_pending = False
    with col_chat:
        with st.chat_message("user"):
            st.markdown(message_to_send)

        with st.chat_message("assistant"):
            body = _build_request_body()
            text_placeholder = st.empty()
            request_started = time.perf_counter()

            if not backend_ok and not STREAMLIT_DIRECT_MODE:
                reply = f"Backend unreachable at `{API_BASE}`."
                _render_compact_error(reply, details=system_status.get("error"))
                st.session_state.citations = []
                st.session_state.tool_results = []
                st.session_state.youtube_suggestions = []
                response_meta = {"answer_source": "llm_fallback", "error": True}
            else:
                st.session_state.last_intent = None
                text_placeholder.markdown("· · ·")

                if st.session_state.enable_streaming:
                    reply, response_meta = _call_streaming(body, text_placeholder)
                else:
                    with st.spinner("Thinking..."):
                        reply, response_meta = _call_non_streaming(body)
                    text_placeholder.markdown(reply)

                intent = st.session_state.get("last_intent")
                retrieval_intent = intent in ("domain_specific", "tool_required", None)
                if st.session_state.require_citations and not st.session_state.citations and retrieval_intent:
                    reply += "\n\n_No citations returned. Try a more specific question if you need grounded evidence._"
                    text_placeholder.markdown(reply)

            elapsed_ms = (time.perf_counter() - request_started) * 1000
            meta = _response_meta_from_state(latency_ms=elapsed_ms)
            if response_meta.get("answer_source"):
                meta["answer_source"] = response_meta["answer_source"]
            usage_estimate = estimate_request_usage(
                messages=st.session_state.messages,
                reply_text=reply,
                model=st.session_state.selected_model,
                cv_text=st.session_state.cv_text,
            )
            provider_usage = response_meta.get("provider_usage")
            if isinstance(provider_usage, dict):
                provider_usage_payload: dict | None = provider_usage
            else:
                provider_usage_payload = None
            usage_fields = build_message_usage_fields(
                model=st.session_state.selected_model,
                estimate=usage_estimate,
                provider_usage=provider_usage_payload,
                byok=_active_credential_source() == USER_BYOK_SOURCE,
            )
            meta.update(usage_fields)
            st.session_state.last_response_meta = meta
            st.session_state.usage_summary = update_usage_summary(
                st.session_state.usage_summary,
                model=st.session_state.selected_model,
                usage=usage_estimate,
                provider_usage=provider_usage_payload,
            )
            _render_message_meta(meta)
            st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
            if response_meta.get("error"):
                st.session_state.youtube_suggestions = []
            else:
                _refresh_youtube_suggestions(message_to_send)
            if speech_auto_send_now:
                if response_meta.get("error"):
                    _set_speech_failure("chat_submission", f"chat_submission: {reply}")
                else:
                    schedule_clear_speech_draft(st.session_state)
                    st.rerun()

with sources_panel, st.container(border=True):
    _render_sources_panel()
