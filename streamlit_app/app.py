"""Streamlit frontend for the AI Career Intelligence Assistant."""

from __future__ import annotations

import json
import os
import uuid

import httpx
import streamlit as st

API_BASE = os.getenv("CAREER_INTEL_API_BASE_URL", "http://localhost:8000")


def _get_system_status() -> dict:
    """Fetch backend + Qdrant + indexed-data status for sidebar diagnostics."""
    try:
        resp = httpx.get(f"{API_BASE}/health/system", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return {
            "backend": False,
            "qdrant": False,
            "indexed_data_present": False,
            "collection": "unknown",
            "points_count": 0,
            "error": f"Cannot reach backend at {API_BASE}.",
        }
    except Exception as exc:
        return {
            "backend": False,
            "qdrant": False,
            "indexed_data_present": False,
            "collection": "unknown",
            "points_count": 0,
            "error": str(exc),
        }

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Career Intelligence Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "citations" not in st.session_state:
    st.session_state.citations = []
if "tool_results" not in st.session_state:
    st.session_state.tool_results = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    use_tools = st.checkbox("Enable tool calling", value=True)
    enable_streaming = st.checkbox("Stream responses", value=True)
    require_citations = st.checkbox("Require citations", value=True)

    st.divider()
    st.subheader("Filters")
    year_filter = st.number_input("Publication year (0 = any)", min_value=0, max_value=2030, value=0)

    st.divider()
    st.subheader("🩺 System Status")
    system_status = _get_system_status()
    backend_ok = bool(system_status.get("backend"))
    qdrant_ok = bool(system_status.get("qdrant"))
    indexed_ok = bool(system_status.get("indexed_data_present"))
    st.markdown(f"**Backend:** {'✅ healthy' if backend_ok else '❌ unreachable'}")
    st.markdown(f"**Qdrant:** {'✅ healthy' if qdrant_ok else '❌ unreachable'}")
    st.markdown(f"**Indexed data:** {'✅ present' if indexed_ok else '⚠️ empty'}")
    if system_status.get("collection"):
        st.caption(
            f"Collection: `{system_status.get('collection')}` | "
            f"Points: {system_status.get('points_count', 0)}"
        )
    if system_status.get("error"):
        st.caption(f"Status detail: {system_status['error']}")

    st.divider()
    if st.button("🔄 New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.citations = []
        st.session_state.tool_results = []
        st.rerun()

    st.divider()
    st.subheader("📥 Export")
    if st.session_state.messages:
        export_data = json.dumps(st.session_state.messages, indent=2, default=str)
        st.download_button("Download conversation", data=export_data, file_name="conversation.json")

    st.divider()
    st.caption(
        "**Disclaimer:** This assistant provides guidance only, based on curated data. "
        "It does not guarantee employment outcomes. Consult a career professional for "
        "personalised advice."
    )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🎯 AI Career Intelligence Assistant")
st.caption("Ask about career paths, skill gaps, role comparisons, and learning plans.")

chat_col, info_col = st.columns([3, 2])


def _build_request_body() -> dict:
    """Build the JSON body shared by streaming and non-streaming calls."""
    filters = {}
    if year_filter > 0:
        filters["publish_year"] = year_filter

    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    return {
        "messages": api_messages,
        "session_id": st.session_state.session_id,
        "use_tools": use_tools,
        "filters": filters or None,
    }


def _call_non_streaming(body: dict) -> str:
    """Non-streaming fallback: single POST /chat."""
    try:
        resp = httpx.post(f"{API_BASE}/chat", json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.citations = data.get("citations", [])
        st.session_state.tool_results = data.get("tool_calls", [])
        return data["reply"]
    except httpx.HTTPStatusError as exc:
        return f"⚠️ API error: {exc.response.status_code} — {exc.response.text}"
    except httpx.ConnectError:
        return "⚠️ Cannot connect to the API server. Is it running?"
    except Exception as exc:
        return f"⚠️ Unexpected error: {exc}"


def _call_streaming(body: dict, placeholder) -> str:
    """Stream tokens from POST /chat/stream and render them incrementally."""
    full_text = ""
    st.session_state.citations = []
    st.session_state.tool_results = []

    try:
        with httpx.stream("POST", f"{API_BASE}/chat/stream", json=body, timeout=90.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                event_type = payload.get("type")

                if event_type == "token":
                    full_text += payload["content"]
                    placeholder.markdown(full_text + "▌")
                elif event_type == "citations":
                    st.session_state.citations = payload["data"]
                elif event_type == "tool_calls":
                    st.session_state.tool_results = payload["data"]
                elif event_type == "error":
                    full_text += f"\n\n⚠️ {payload['detail']}"
                elif event_type == "status":
                    placeholder.info(payload.get("detail", "Working..."))
                elif event_type == "done":
                    break

        placeholder.markdown(full_text)
        return full_text

    except httpx.ConnectError:
        return "⚠️ Cannot connect to the API server. Is it running?"
    except Exception as exc:
        return f"⚠️ Streaming error: {exc}"


with chat_col:
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
                reply = (
                    f"⚠️ Backend is unreachable at `{API_BASE}`. "
                    "Start the API server and try again."
                )
                text_placeholder.error(reply)
                st.session_state.citations = []
                st.session_state.tool_results = []
            elif not qdrant_ok:
                reply = (
                    "⚠️ Qdrant is unreachable. Start Docker services "
                    "(`docker compose up -d`) and try again."
                )
                text_placeholder.error(reply)
                st.session_state.citations = []
                st.session_state.tool_results = []
            elif not indexed_ok:
                reply = (
                    "⚠️ No indexed data found. Run ingestion before asking questions "
                    "so retrieval, sources, and tools can work."
                )
                text_placeholder.warning(reply)
                st.session_state.citations = []
                st.session_state.tool_results = []
            else:
                # Explicit progress text prevents the UI from appearing frozen
                # while rewrite/retrieval/tool steps run before first streamed token.
                text_placeholder.info("Working... retrieving evidence and generating answer.")
                if enable_streaming:
                    reply = _call_streaming(body, text_placeholder)
                else:
                    with st.spinner("Thinking..."):
                        reply = _call_non_streaming(body)
                    text_placeholder.markdown(reply)

                if require_citations and not st.session_state.citations:
                    reply = (
                        f"{reply}\n\n⚠️ No citations were returned for this answer. "
                        "Please ask a more specific grounded question."
                    )
                    text_placeholder.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})

with info_col:
    st.subheader("📄 Sources")
    if st.session_state.citations:
        for cite in st.session_state.citations:
            with st.expander(f"[{cite['id']}] {cite['title']}"):
                if cite.get("section"):
                    st.markdown(f"**Section:** {cite['section']}")
                if cite.get("publish_year"):
                    st.markdown(f"**Year:** {cite['publish_year']}")
                if cite.get("uri"):
                    st.markdown(f"**Link:** {cite['uri']}")
                st.markdown(f"**Excerpt:** {cite.get('excerpt', '')}")
    else:
        st.info("Sources will appear here after you ask a question.")

    st.divider()

    st.subheader("🔧 Tool Results")
    if st.session_state.tool_results:
        for tr in st.session_state.tool_results:
            with st.expander(f"{tr['tool_name']} ({'✅' if tr.get('success') else '❌'})"):
                st.json(tr.get("output", {}))
                if tr.get("error"):
                    st.error(tr["error"])
    else:
        st.info("Tool outputs will appear here when tools are invoked.")
