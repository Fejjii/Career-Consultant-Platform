"""UI-only supplemental sources (outside RAG and API citations).

The first concrete provider is :mod:`youtube_service` (YouTube Data API v3).
Additional providers (for example web search) can live alongside it and be
wired from ``app.py`` using the same session-state / panel pattern.
"""
