"""Optional cross-encoder reranking (placeholder for bonus task)."""

from __future__ import annotations

import structlog

from career_intel.schemas.domain import RetrievedChunk

logger = structlog.get_logger()


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = 8,
) -> list[RetrievedChunk]:
    """Rerank chunks using a cross-encoder model.

    Currently a pass-through stub. To enable real reranking:
    1. Install ``sentence-transformers``.
    2. Load a cross-encoder (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
    3. Score each (query, chunk.text) pair and re-sort.
    """
    logger.info("rerank_stub", input_count=len(chunks), output_count=min(top_n, len(chunks)))
    return chunks[:top_n]
