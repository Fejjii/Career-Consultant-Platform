"""Retrieval pipeline: query rewrite, vector search, dedup, optional rerank."""

from __future__ import annotations

import re
from typing import Any

import structlog

from career_intel.config import Settings, get_settings
from career_intel.llm import get_chat_llm
from career_intel.rag.embeddings import get_embeddings
from career_intel.schemas.domain import ChunkMetadata, RetrievedChunk
from career_intel.storage.qdrant_store import search_vectors

logger = structlog.get_logger()

REWRITE_PROMPT = """\
You are a query rewriter for a career intelligence knowledge base. \
Your job is to reformulate the user's question so it retrieves the most \
relevant career, skills, and labor-market information.

Rules:
- Expand acronyms (e.g. "ML" -> "Machine Learning").
- Clarify vague terms (e.g. "tech jobs" -> "technology sector roles").
- Keep the rewritten query concise (1-2 sentences).
- Do NOT answer the question — only rewrite it.

User question: {query}

Rewritten query:"""


async def rewrite_query(query: str, settings: Settings | None = None) -> str:
    """Rewrite the user query for better retrieval."""
    if settings is None:
        settings = get_settings()

    llm = get_chat_llm(settings, temperature=0.0)
    normalized_query = normalize_query(query)
    prompt = REWRITE_PROMPT.format(query=normalized_query)
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    rewritten = response.content.strip() if hasattr(response, "content") else normalized_query
    logger.info("query_rewritten", original=query[:100], normalized=normalized_query[:100], rewritten=rewritten[:100])
    return rewritten


async def retrieve_chunks(
    query: str,
    filters: dict[str, Any] | None = None,
    settings: Settings | None = None,
    top_k: int = 20,
    score_threshold: float = 0.0,
) -> list[RetrievedChunk]:
    """Embed query variants and retrieve relevant chunks from Qdrant.

    Strategy:
    - Retrieve using both rewritten and normalized-original variants.
    - Merge by point ID using max score.
    - If filters return zero hits, fall back to unfiltered retrieval.
    """
    if settings is None:
        settings = get_settings()

    normalized_query = normalize_query(query)
    query_variants = [query]
    if normalized_query != query:
        query_variants.append(normalized_query)

    vectors = get_embeddings(query_variants, settings=settings)
    scored_points = _search_merged(vectors=vectors, top_k=top_k, filters=filters)
    filter_fallback_used = False
    if filters and not scored_points:
        scored_points = _search_merged(vectors=vectors, top_k=top_k, filters=None)
        filter_fallback_used = True

    chunks: list[RetrievedChunk] = []
    seen_texts: set[str] = set()

    for point in scored_points:
        score = point.score if hasattr(point, "score") else 0.0
        if score < score_threshold:
            continue

        payload = point.payload or {}
        text = payload.get("text", "")

        # Deduplicate near-identical chunks
        text_fingerprint = text[:200]
        if text_fingerprint in seen_texts:
            continue
        seen_texts.add(text_fingerprint)

        metadata = ChunkMetadata(
            source_id=payload.get("source_id", ""),
            source_type=payload.get("source_type", ""),
            title=payload.get("title", ""),
            section=payload.get("section"),
            page_or_loc=payload.get("page_or_loc"),
            publish_year=payload.get("publish_year"),
            license=payload.get("license"),
            occupation_code=payload.get("occupation_code"),
            skill_id=payload.get("skill_id"),
            uri=payload.get("uri"),
            chunk_index=payload.get("chunk_index", 0),
            parent_doc_id=payload.get("parent_doc_id", ""),
        )

        chunks.append(RetrievedChunk(
            chunk_id=str(point.id),
            text=text,
            metadata=metadata,
            score=score,
        ))

    # Keep top N for context window
    chunks = chunks[:8]

    logger.info(
        "retrieval_complete",
        query=query[:80],
        normalized_query=normalized_query[:80],
        query_variants=len(query_variants),
        filters_applied=bool(filters),
        filter_fallback_used=filter_fallback_used,
        total_hits=len(scored_points),
        returned=len(chunks),
        top_score=chunks[0].score if chunks else 0.0,
        top_chunk_ids=[c.chunk_id for c in chunks[:5]],
        top_scores=[round(c.score, 4) for c in chunks[:5]],
    )
    return chunks


def normalize_query(query: str) -> str:
    """Normalize user query variants to improve retrieval stability."""
    text = query.strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\((\d{4})\s*-\s*(\d{4})\)", r"\1 to \2", text)
    text = re.sub(r"(\d{4})\s*-\s*(\d{4})", r"\1 to \2", text)
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"[!?.,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _search_merged(
    *,
    vectors: list[list[float]],
    top_k: int,
    filters: dict[str, Any] | None,
) -> list[Any]:
    """Search each query vector and merge points by max score."""
    by_id: dict[str, Any] = {}
    for vector in vectors:
        results = search_vectors(query_vector=vector, top_k=top_k, filters=filters)
        for point in results:
            point_id = str(point.id)
            if point_id not in by_id or getattr(point, "score", 0.0) > getattr(by_id[point_id], "score", 0.0):
                by_id[point_id] = point
    merged = list(by_id.values())
    merged.sort(key=lambda p: getattr(p, "score", 0.0), reverse=True)
    return merged[:top_k]
