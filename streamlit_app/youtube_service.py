"""Supplemental YouTube search for the Streamlit UI (not part of RAG or citations).

This module is intentionally UI-side so retrieval and citation schemas stay
unchanged. Future supplemental providers (e.g. web search) can follow the same
pattern: keyword shaping → external API → structured cards for the sources panel.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL: Final[str] = "https://www.googleapis.com/youtube/v3/search"
_DEFAULT_MAX_RESULTS: Final[int] = 3
_CACHE_TTL_SECONDS: Final[float] = 3600.0
_CACHE_MAX_KEYS: Final[int] = 128
_YOUTUBE_VIDEO_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
_ALLOWED_THUMBNAIL_HOST_SUFFIXES: Final[tuple[str, ...]] = ("ytimg.com", "googleusercontent.com")

_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "it",
        "me",
        "my",
        "your",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "when",
        "where",
        "why",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "then",
        "once",
        "here",
        "there",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "tell",
        "please",
        "thanks",
        "thank",
    }
)


@dataclass(frozen=True)
class VideoSuggestion:
    """One YouTube result row for the sources panel."""

    title: str
    channel_name: str
    video_url: str
    thumbnail_url: str

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "channel_name": self.channel_name,
            "video_url": self.video_url,
            "thumbnail_url": self.thumbnail_url,
        }


class _QueryCache:
    """Tiny TTL cache keyed by normalized query (avoids repeat Data API calls)."""

    def __init__(self, *, ttl_seconds: float, max_keys: int) -> None:
        self._ttl = ttl_seconds
        self._max = max(8, max_keys)
        self._data: OrderedDict[str, tuple[float, list[VideoSuggestion]]] = OrderedDict()

    def get(self, key: str) -> list[VideoSuggestion] | None:
        now = time.monotonic()
        item = self._data.get(key)
        if not item:
            return None
        ts, videos = item
        if now - ts > self._ttl:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return list(videos)

    def set(self, key: str, videos: list[VideoSuggestion]) -> None:
        self._data[key] = (time.monotonic(), list(videos))
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


_CACHE = _QueryCache(ttl_seconds=_CACHE_TTL_SECONDS, max_keys=_CACHE_MAX_KEYS)


def keywords_for_youtube_search(query: str, *, max_terms: int = 10) -> str:
    """Lightweight keyword extraction for the YouTube `q` parameter."""
    raw = " ".join(str(query).split()).strip()
    if not raw:
        return ""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{1,}", raw.lower())
    picked = [t for t in tokens if t not in _STOPWORDS][:max_terms]
    if not picked:
        return raw[:100]
    return " ".join(picked)


def _cache_key(query: str) -> str:
    normalized = " ".join(str(query).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def should_fetch_youtube_support(
    *,
    intent: str | None,
    answer_source: str | None,
    user_query: str,
    min_query_chars: int = 8,
) -> bool:
    """Gate YouTube enrichment: informational turns only; skip small talk and tool/runtime paths."""
    q = str(user_query).strip()
    if len(q) < min_query_chars:
        return False
    it = (intent or "").strip().lower()
    if it in {"", "small_talk"}:
        return False
    src = (answer_source or "").strip().lower()
    if src in {"tool", "runtime", "source_inventory"}:
        return False
    return True


def fetch_youtube_suggestions(
    user_query: str,
    *,
    api_key: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    client: httpx.Client | None = None,
) -> list[dict[str, str]]:
    """Return 2–3 videos as plain dicts suitable for ``st.session_state``."""
    key = _cache_key(user_query)
    cached = _CACHE.get(key)
    if cached is not None:
        return [v.as_dict() for v in cached]

    q = keywords_for_youtube_search(user_query)
    if not q:
        return []

    own_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
    try:
        resp = http.get(
            YOUTUBE_SEARCH_URL,
            params={
                "part": "snippet",
                "type": "video",
                "maxResults": min(max(2, max_results), 5),
                "q": q[:200],
                "key": api_key,
            },
        )
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("youtube_search_http_error: %s", str(exc)[:200])
        return []
    except Exception as exc:  # noqa: BLE001 — defensive: never break the chat UI
        logger.warning("youtube_search_failed: %s", str(exc)[:200])
        return []
    finally:
        if own_client:
            http.close()

    items = payload.get("items") or []
    out: list[VideoSuggestion] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        vid = (row.get("id") or {}).get("videoId")
        snip = row.get("snippet") or {}
        if not vid or not isinstance(snip, dict) or not _YOUTUBE_VIDEO_ID_RE.fullmatch(str(vid)):
            continue
        title = str(snip.get("title") or "").strip() or "Untitled video"
        channel = str(snip.get("channelTitle") or "").strip() or "Unknown channel"
        thumbs = snip.get("thumbnails") or {}
        medium = thumbs.get("medium") if isinstance(thumbs, dict) else None
        default = thumbs.get("default") if isinstance(thumbs, dict) else None
        thumb_url = ""
        if isinstance(medium, dict) and medium.get("url"):
            thumb_url = _sanitize_thumbnail_url(str(medium["url"]))
        elif isinstance(default, dict) and default.get("url"):
            thumb_url = _sanitize_thumbnail_url(str(default["url"]))
        out.append(
            VideoSuggestion(
                title=title,
                channel_name=channel,
                video_url=f"https://www.youtube.com/watch?v={vid}",
                thumbnail_url=thumb_url,
            )
        )
        if len(out) >= max_results:
            break

    _CACHE.set(key, out)
    return [v.as_dict() for v in out]


def _sanitize_thumbnail_url(url: str) -> str:
    raw = url.strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        return ""
    if not any(host == suffix or host.endswith(f".{suffix}") for suffix in _ALLOWED_THUMBNAIL_HOST_SUFFIXES):
        return ""
    return raw
