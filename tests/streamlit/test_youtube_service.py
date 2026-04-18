"""Tests for Streamlit-side YouTube supplemental search."""

from __future__ import annotations

import httpx
import pytest

from sources_panel import format_youtube_sources_html
from youtube_service import (
    VideoSuggestion,
    _CACHE,
    fetch_youtube_suggestions,
    keywords_for_youtube_search,
    should_fetch_youtube_support,
)


@pytest.fixture(autouse=True)
def _clear_youtube_cache() -> None:
    _CACHE._data.clear()
    yield
    _CACHE._data.clear()


def test_keywords_for_youtube_search_strips_stopwords() -> None:
    q = keywords_for_youtube_search("What are the top skills for a data engineer in 2026?")
    assert "what" not in q.split()
    assert "skills" in q
    assert "data" in q


def test_should_fetch_youtube_support_small_talk() -> None:
    assert not should_fetch_youtube_support(
        intent="small_talk",
        answer_source="llm_fallback",
        user_query="hello there friend!!",
    )


def test_should_fetch_youtube_support_tool_only() -> None:
    assert not should_fetch_youtube_support(
        intent="tool_required",
        answer_source="tool",
        user_query="Build a learning plan for a data engineer role",
    )


def test_should_fetch_youtube_support_domain_rag() -> None:
    assert should_fetch_youtube_support(
        intent="domain_specific",
        answer_source="rag",
        user_query="What skills matter for product managers?",
    )


def test_fetch_youtube_suggestions_parses_response() -> None:
    payload = {
        "items": [
            {
                "id": {"videoId": "abc123"},
                "snippet": {
                    "title": "Career Tips",
                    "channelTitle": "Career Channel",
                    "thumbnails": {
                        "medium": {"url": "https://i.ytimg.com/vi/abc123/mqdefault.jpg"},
                    },
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "search" in str(request.url)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        rows = fetch_youtube_suggestions(
            "interview tips for software engineers",
            api_key="fake-key",
            max_results=3,
            client=client,
        )
    assert len(rows) == 1
    assert rows[0]["title"] == "Career Tips"
    assert rows[0]["channel_name"] == "Career Channel"
    assert rows[0]["video_url"] == "https://www.youtube.com/watch?v=abc123"
    assert "ytimg.com" in rows[0]["thumbnail_url"]


def test_fetch_youtube_suggestions_uses_cache_on_repeat() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": {"videoId": "x"},
                        "snippet": {"title": "T", "channelTitle": "C", "thumbnails": {}},
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    query = "unique cache probe query zzyyxx"
    with httpx.Client(transport=transport) as client:
        fetch_youtube_suggestions(query, api_key="k", client=client)
        fetch_youtube_suggestions(query, api_key="k", client=client)
    assert len(calls) == 1


def test_format_youtube_sources_html_escapes_title() -> None:
    html = format_youtube_sources_html(
        [
            {
                "title": 'Learn <script>',
                "channel_name": "A & B",
                "video_url": "https://www.youtube.com/watch?v=1",
                "thumbnail_url": "",
            }
        ]
    )
    assert "<script>" not in html
    assert "&lt;" in html or "Learn" in html


def test_format_youtube_sources_html_rejects_unsafe_links() -> None:
    html = format_youtube_sources_html(
        [
            {
                "title": "Video",
                "channel_name": "Channel",
                "video_url": "javascript:alert(1)",
                "thumbnail_url": "https://evil.example/thumb.png",
            }
        ]
    )
    assert html == ""


def test_video_suggestion_as_dict() -> None:
    v = VideoSuggestion(
        title="t",
        channel_name="c",
        video_url="https://example.com/v",
        thumbnail_url="https://example.com/i.jpg",
    )
    assert v.as_dict()["channel_name"] == "c"


def test_fetch_youtube_suggestions_discards_unsafe_thumbnail_host() -> None:
    payload = {
        "items": [
            {
                "id": {"videoId": "abc123"},
                "snippet": {
                    "title": "Career Tips",
                    "channelTitle": "Career Channel",
                    "thumbnails": {"medium": {"url": "https://evil.example/thumb.jpg"}},
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows = fetch_youtube_suggestions(
            "interview tips for software engineers",
            api_key="fake-key",
            max_results=3,
            client=client,
        )
    assert rows[0]["thumbnail_url"] == ""
