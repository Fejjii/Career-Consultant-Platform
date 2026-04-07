"""Streamlit frontend for the AI Career Intelligence Assistant.

Modern chat UI with intent-first routing, streaming, CV upload,
and tabbed context panels (sources / tools / diagnostics).
"""

from __future__ import annotations

import json
import os
import uuid

import httpx
import streamlit as st

API_BASE = os.getenv("CAREER_INTEL_API_BASE_URL", "http://localhost:8000")
DEV_MODE = os.getenv("CAREER_INTEL_DEV_MODE", "").lower() in ("1", "true", "yes")


# ── Custom CSS ──────────────────────────────────────────────────────────

_CUSTOM_CSS = """
<style>
/* --- Page & background --- */
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}

/* --- Chat container centering --- */
.stChatMessage {
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
}

/* --- Chat input centering --- */
.stChatInput {
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
}

/* --- Header styling --- */
.app-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.app-header p {
    color: #64748b;
    font-size: 0.92rem;
    margin: 0;
}

/* --- Sidebar polish --- */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.88rem;
}

/* --- Tab panel card look --- */
.context-panel {
    max-width: 820px;
    margin: 1rem auto 2rem auto;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.5rem 1rem 1rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* --- Citation card styling --- */
.citation-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.citation-card strong {
    color: #334155;
}

/* --- Status badges --- */
.status-ok { color: #16a34a; }
.status-bad { color: #dc2626; }
.status-warn { color: #d97706; }

/* --- Reduce excessive Streamlit padding --- */
.block-container {
    padding-top: 1rem !important;
}
</style>
"""


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_system_status() -> dict:
    """Fetch backend health for sidebar diagnostics."""
    try:
        resp = httpx.get(f"{API_BASE}/health/system", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {
            "backend": False, "qdrant": False, "indexed_data_present": False,
            "collection": "unknown", "points_count": 0,
            "error": str(exc) if not isinstance(exc, httpx.ConnectError) else f"Cannot reach {API_BASE}",
        }


def _build_request_body() -> dict:
    """Build the JSON payload for chat endpoints."""
    body: dict = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        "session_id": st.session_state.session_id,
        "use_tools": st.session_state.get("use_tools", True),
    }
    if st.session_state.cv_text:
        body["cv_text"] = st.session_state.cv_text
    return body


def _call_non_streaming(body: dict) -> str:
    """Non-streaming POST /chat fallback."""
    try:
        resp = httpx.post(f"{API_BASE}/chat", json=body, timeout=60.0)
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
        }
        return data["reply"]
    except httpx.HTTPStatusError as exc:
        return f"API error: {exc.response.status_code}"
    except httpx.ConnectError:
        return "Cannot connect to the API server. Is it running?"
    except Exception as exc:
        return f"Unexpected error: {exc}"


def _call_streaming(body: dict, placeholder) -> str:
    """Stream tokens from POST /chat/stream, render incrementally."""
    full_text = ""
    st.session_state.citations = []
    st.session_state.tool_results = []
    st.session_state.last_intent = None
    st.session_state.last_debug = None

    try:
        with httpx.stream("POST", f"{API_BASE}/chat/stream", json=body, timeout=90.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                evt = payload.get("type")

                if evt == "intent":
                    st.session_state.last_intent = payload.get("data")
                    if payload.get("data") in ("retrieval_required", "tool_required"):
                        placeholder.markdown("*Searching knowledge base...*")
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
                    full_text += f"\n\n⚠️ {payload['detail']}"
                elif evt == "status":
                    placeholder.markdown(f"*{payload.get('detail', 'Working...')}*")
                elif evt == "done":
                    break

        placeholder.markdown(full_text)
        return full_text
    except httpx.ConnectError:
        return "Cannot connect to the API server. Is it running?"
    except Exception as exc:
        return f"Streaming error: {exc}"


def _render_context_panel() -> None:
    """Render Sources / Tools / Debug as a tabbed card below the chat."""
    has_citations = bool(st.session_state.citations)
    has_tools = bool(st.session_state.tool_results)
    has_debug = DEV_MODE and st.session_state.get("last_debug")

    if not has_citations and not has_tools and not has_debug:
        return

    st.markdown('<div class="context-panel">', unsafe_allow_html=True)

    tab_names = []
    if has_citations:
        tab_names.append(f"📄 Sources ({len(st.session_state.citations)})")
    if has_tools:
        tab_names.append(f"🔧 Tools ({len(st.session_state.tool_results)})")
    if has_debug:
        tab_names.append("🔍 Debug")

    if not tab_names:
        st.markdown('</div>', unsafe_allow_html=True)
        return

    tabs = st.tabs(tab_names)
    tab_idx = 0

    if has_citations:
        with tabs[tab_idx]:
            for cite in st.session_state.citations:
                title = f"[{cite['id']}] {cite['title']}"
                with st.expander(title):
                    parts = []
                    if cite.get("section"):
                        parts.append(f"**Section:** {cite['section']}")
                    if cite.get("publish_year"):
                        parts.append(f"**Year:** {cite['publish_year']}")
                    if cite.get("uri"):
                        parts.append(f"[Link]({cite['uri']})")
                    if parts:
                        st.markdown(" · ".join(parts))
                    if cite.get("excerpt"):
                        st.caption(cite["excerpt"][:300])
        tab_idx += 1

    if has_tools:
        with tabs[tab_idx]:
            for tr in st.session_state.tool_results:
                icon = "✅" if tr.get("success") else "❌"
                with st.expander(f"{icon} {tr['tool_name']}"):
                    st.json(tr.get("output", {}))
                    if tr.get("error"):
                        st.error(tr["error"])
        tab_idx += 1

    if has_debug:
        with tabs[tab_idx]:
            st.json(st.session_state.last_debug)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Page config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Career Intelligence Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────────

for key, default in {
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "citations": [],
    "tool_results": [],
    "cv_text": None,
    "cv_filename": None,
    "last_intent": None,
    "last_debug": None,
    "use_tools": True,
    "enable_streaming": True,
    "require_citations": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.use_tools = st.toggle("Tool calling", value=st.session_state.use_tools)
    st.session_state.enable_streaming = st.toggle("Streaming", value=st.session_state.enable_streaming)
    st.session_state.require_citations = st.toggle("Warn if no sources", value=st.session_state.require_citations)

    st.markdown("---")
    st.markdown("### 📎 CV / Resume")
    uploaded_cv = st.file_uploader(
        "Upload CV", type=["pdf", "docx", "txt"], label_visibility="collapsed",
        help="PDF, DOCX, or TXT. Stays in this session only.",
    )
    if uploaded_cv is not None:
        if st.session_state.cv_filename != uploaded_cv.name:
            cv_bytes = uploaded_cv.getvalue()
            try:
                with st.spinner("Processing..."):
                    resp = httpx.post(
                        f"{API_BASE}/cv/process",
                        files={"file": (uploaded_cv.name, cv_bytes)},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                st.session_state.cv_text = result["cv_text"]
                st.session_state.cv_filename = result["filename"]
                if result.get("warnings"):
                    for w in result["warnings"]:
                        st.warning(w)
                st.success(f"Processed: {uploaded_cv.name}")
            except httpx.HTTPStatusError as exc:
                st.error(exc.response.json().get("detail", str(exc)))
                st.session_state.cv_text = None
                st.session_state.cv_filename = None
            except Exception as exc:
                st.error(str(exc))
                st.session_state.cv_text = None
                st.session_state.cv_filename = None
        elif st.session_state.cv_text:
            st.caption(f"📄 {st.session_state.cv_filename}")

    if st.session_state.cv_text:
        col_a, col_b = st.columns(2)
        with col_a:
            with st.expander("Preview"):
                st.text(st.session_state.cv_text[:500] + ("..." if len(st.session_state.cv_text) > 500 else ""))
        with col_b:
            if st.button("Remove", use_container_width=True):
                st.session_state.cv_text = None
                st.session_state.cv_filename = None
                st.rerun()

    st.markdown("---")
    st.markdown("### 🩺 Status")
    status = _get_system_status()
    backend_ok = bool(status.get("backend"))
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"{'✅' if status.get('backend') else '❌'} API")
    c2.markdown(f"{'✅' if status.get('qdrant') else '❌'} DB")
    c3.markdown(f"{'✅' if status.get('indexed_data_present') else '⚠️'} Data")

    st.markdown("---")
    col_new, col_export = st.columns(2)
    with col_new:
        if st.button("🔄 New", use_container_width=True):
            for k in ("messages", "citations", "tool_results", "last_intent", "last_debug"):
                st.session_state[k] = [] if k in ("messages", "citations", "tool_results") else None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.cv_text = None
            st.session_state.cv_filename = None
            st.rerun()
    with col_export:
        if st.session_state.messages:
            st.download_button(
                "📥 Export", use_container_width=True,
                data=json.dumps(st.session_state.messages, indent=2, default=str),
                file_name="conversation.json",
            )

    st.caption(
        "This assistant provides guidance only. "
        "Consult a career professional for personalised advice."
    )


# ── Header ──────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-header">'
    "<h1>🎯 Career Intelligence Assistant</h1>"
    "<p>Career paths · Skill gaps · Role comparisons · Learning plans</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ── Chat area (full width, no columns) ─────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a career question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        body = _build_request_body()
        text_placeholder = st.empty()

        if not backend_ok:
            reply = f"Backend is unreachable at `{API_BASE}`. Start the API server and try again."
            text_placeholder.error(reply)
            st.session_state.citations = []
            st.session_state.tool_results = []
        else:
            st.session_state.last_intent = None
            text_placeholder.markdown("·&nbsp;·&nbsp;·")

            if st.session_state.enable_streaming:
                reply = _call_streaming(body, text_placeholder)
            else:
                with st.spinner("Thinking..."):
                    reply = _call_non_streaming(body)
                text_placeholder.markdown(reply)

            intent = st.session_state.get("last_intent")
            retrieval_intent = intent in ("retrieval_required", "tool_required", None)
            if st.session_state.require_citations and not st.session_state.citations and retrieval_intent:
                reply += "\n\n⚠️ No citations were returned. Try a more specific question."
                text_placeholder.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


# ── Context panel (tabs: Sources / Tools / Debug) ──────────────────────

_render_context_panel()
