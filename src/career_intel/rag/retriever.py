"""Retrieval pipeline: query rewrite, vector search, rerank, evidence assessment."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import structlog

from career_intel.config import Settings, get_settings
from career_intel.llm import get_chat_llm
from career_intel.rag.embeddings import get_embeddings
from career_intel.rag.rerank import rerank_chunks, select_rerank_profile
from career_intel.schemas.domain import ChunkMetadata, RetrievedChunk
from career_intel.storage.qdrant_store import (
    QdrantConfigurationError,
    count_vectors,
    get_qdrant_client,
    sample_payloads,
    search_vectors,
)

logger = structlog.get_logger()

_WEF_QUERY_HINTS = (
    "wef",
    "world economic forum",
    "future of jobs",
    "trends",
    "labour market",
)
_ESCO_QUERY_HINTS = (
    "esco",
    "occupation",
    "occupations",
    "skills",
    "isco",
)
_ESCO_RELATION_TERMS = frozenset(
    {
        "essential",
        "optional",
        "relation",
        "relations",
        "linked",
        "ties",
        "tied",
        "mapping",
        "map",
        "taxonomy",
        "isco",
        "occupation",
        "occupations",
        "skill",
        "skills",
        "related",
        "associated",
    }
)
_ESCO_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("related to", re.compile(r"\brelated\s+to\b")),
    ("linked to", re.compile(r"\blinked\s+to\b")),
    ("associated with", re.compile(r"\bassociated\s+with\b")),
    ("skills for", re.compile(r"\bskills?\s+for\b")),
    ("skills in", re.compile(r"\bskills?\s+in\b")),
    ("skills related to", re.compile(r"\bskills?\s+related\s+to\b")),
    ("x-related skills", re.compile(r"\b[a-z0-9+#/.-]+-related\s+skills?\b")),
)
_ESCO_CONCEPT_STOPWORDS = frozenset(
    {
        "what",
        "which",
        "how",
        "does",
        "do",
        "are",
        "is",
        "the",
        "a",
        "an",
        "in",
        "to",
        "of",
        "for",
        "and",
        "or",
        "with",
        "between",
        "difference",
        "closely",
        "tied",
        "linked",
        "linkage",
        "relate",
        "related",
        "associated",
        "strongly",
        "most",
        "style",
        "query",
        "queries",
        "grouping",
        "groups",
        "group",
        "framework",
        "classification",
        "international",
        "standard",
        "european",
        "competences",
        "qualifications",
        "esco",
        "occupation",
        "occupations",
        "skill",
        "skills",
        "relation",
        "relations",
        "occupation-to-skill",
        "map",
    }
)
_SHORT_CONCEPTS = frozenset({"ai", "bi", "ml", "sql", "etl", "ui", "ux"})


class RetrievalBackendUnavailableError(RuntimeError):
    """Raised when retrieval cannot reach the configured vector backend."""


def _safe_preview_for_console(text: str, limit: int = 100) -> str:
    """Return an ASCII-safe preview for development console logging."""
    preview = text[:limit]
    return preview.encode("ascii", errors="replace").decode("ascii")


@dataclass(frozen=True)
class QueryProfile:
    detected_source: str | None
    esco_relation_query: bool
    taxonomy_query: bool
    essential_optional_query: bool
    salient_concepts: tuple[str, ...]
    classification_reason: str = "no_relation_signal"

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
    top_k: int | None = None,
    score_threshold: float | None = None,
    detected_source_override: str | None = None,
    query_profile_override: QueryProfile | None = None,
) -> list[RetrievedChunk]:
    """Embed query variants and retrieve relevant chunks from Qdrant.

    Strategy:
    - Retrieve using both rewritten and normalized-original variants.
    - Merge by point ID using max score.
    - If the best merged score is below ``score_threshold``, return no chunks (weak match).
    - Filter individual hits below ``score_threshold``.
    - Deduplicate by normalized text fingerprint.
    - If filters return zero hits, fall back to unfiltered retrieval.
    """
    if settings is None:
        settings = get_settings()

    eff_top_k = settings.rag_top_k if top_k is None else top_k
    initial_top_k = max(eff_top_k, settings.rag_initial_top_k)
    eff_threshold = (
        settings.rag_weak_evidence_threshold if score_threshold is None else score_threshold
    )
    try:
        client = get_qdrant_client(settings)
    except Exception as exc:
        logger.warning(
            "retrieval_backend_unavailable",
            stage="client_init",
            query_preview=query[:80],
            error=str(exc)[:300],
        )
        raise RetrievalBackendUnavailableError(_friendly_qdrant_error(exc)) from exc

    normalized_query = normalize_query(query)
    detected_source = detected_source_override or detect_query_source(normalized_query)
    query_profile = query_profile_override or build_query_profile(
        normalized_query,
        detected_source=detected_source,
    )
    if query_profile.esco_relation_query and len(query_profile.salient_concepts) >= 2:
        initial_top_k = max(initial_top_k, eff_top_k * 4)
    logger.info(
        "retrieval_top_k_selected",
        query_preview=query[:80],
        effective_top_k=eff_top_k,
        initial_top_k=initial_top_k,
        override_provided=top_k is not None,
    )
    retrieval_filters = _merge_source_filter(filters=filters, detected_source=detected_source)
    matching_filtered_candidates = 0
    if detected_source:
        payload_sample = None
        try:
            matching_filtered_candidates = count_vectors(retrieval_filters, client=client)
            payload_sample = _sanitize_payload_sample(
                sample_payloads(retrieval_filters, limit=1, client=client)[0]
                if matching_filtered_candidates
                else None
            )
        except Exception as exc:  # pragma: no cover - defensive live-system guard
            logger.warning(
                "retrieval_source_filter_diagnostics_failed",
                query_preview=query[:80],
                detected_source=detected_source,
                applied_filter=retrieval_filters,
                error=str(exc)[:300],
            )
        logger.info(
            "retrieval_source_detected",
            query_preview=query[:80],
            detected_source=detected_source,
            applied_filter=retrieval_filters,
            matching_candidates=matching_filtered_candidates,
            payload_sample=payload_sample,
        )
    logger.info(
        "retrieval_query_profile",
        query_preview=query[:80],
        detected_source=detected_source,
        esco_relation_query=query_profile.esco_relation_query,
        taxonomy_query=query_profile.taxonomy_query,
        essential_optional_query=query_profile.essential_optional_query,
        classification_reason=query_profile.classification_reason,
        salient_concepts=list(query_profile.salient_concepts),
        initial_top_k=initial_top_k,
    )
    query_variants = [query]
    if normalized_query != query:
        query_variants.append(normalized_query)
    relation_variant = _build_esco_relation_variant(normalized_query, query_profile)
    if relation_variant and relation_variant not in query_variants:
        query_variants.append(relation_variant)

    vectors = get_embeddings(query_variants, settings=settings)
    try:
        scored_points = _search_merged(
            vectors=vectors,
            top_k=initial_top_k,
            filters=retrieval_filters,
            client=client,
        )
    except Exception as exc:
        logger.warning(
            "retrieval_backend_unavailable",
            stage="vector_search",
            query_preview=query[:80],
            detected_source=detected_source,
            applied_filter=retrieval_filters,
            error=str(exc)[:300],
        )
        raise RetrievalBackendUnavailableError(_friendly_qdrant_error(exc)) from exc
    if detected_source:
        logger.info(
            "retrieval_source_filtered_candidates",
            query_preview=query[:80],
            detected_source=detected_source,
            applied_filter=retrieval_filters,
            matching_candidates=matching_filtered_candidates,
            candidates=len(scored_points),
        )
    filter_fallback_used = False
    if retrieval_filters and not scored_points:
        logger.info(
            "retrieval_source_filter_fallback",
            query_preview=query[:80],
            detected_source=detected_source,
            applied_filter=retrieval_filters,
            reason="no_results_after_filter",
        )
        try:
            scored_points = _search_merged(
                vectors=vectors,
                top_k=initial_top_k,
                filters=None,
                client=client,
            )
        except Exception as exc:
            logger.warning(
                "retrieval_backend_unavailable",
                stage="vector_search_filter_fallback",
                query_preview=query[:80],
                detected_source=detected_source,
                error=str(exc)[:300],
            )
            raise RetrievalBackendUnavailableError(_friendly_qdrant_error(exc)) from exc
        filter_fallback_used = True

    if scored_points:
        best_merged = max(getattr(p, "score", 0.0) for p in scored_points)
        if best_merged < eff_threshold:
            logger.info(
                "retrieval_skipped_weak_similarity",
                query_preview=query[:80],
                best_score=round(best_merged, 4),
                threshold=eff_threshold,
            )
            if settings.environment == "development":
                print(
                    f"[RAG] retrieval_skipped_weak_similarity best_score={best_merged:.4f} "
                    f"threshold={eff_threshold}",
                )
            return []

    logger.info(
        "retrieval_candidates_initial",
        query=query[:80],
        candidates=len(scored_points),
        scores=[round(getattr(point, "score", 0.0), 4) for point in scored_points[:initial_top_k]],
    )

    chunks: list[RetrievedChunk] = []
    seen_fingerprints: set[str] = set()
    skipped_low_score = 0
    skipped_duplicate = 0

    for point in scored_points:
        score = point.score if hasattr(point, "score") else 0.0
        if score < eff_threshold:
            skipped_low_score += 1
            continue

        payload = point.payload or {}
        text = payload.get("text", "")

        fingerprint = _text_fingerprint(text)
        if fingerprint in seen_fingerprints:
            skipped_duplicate += 1
            continue
        seen_fingerprints.add(fingerprint)

        corpus_source = _normalize_payload_source(payload)
        metadata = ChunkMetadata(
            source_id=payload.get("source_id", ""),
            source_type=payload.get("source_type", ""),
            title=payload.get("title", ""),
            document_title=payload.get("document_title"),
            topic=payload.get("topic"),
            section=payload.get("section"),
            section_title=payload.get("section_title"),
            page_or_loc=payload.get("page_or_loc"),
            page_number=payload.get("page_number"),
            publish_year=payload.get("publish_year"),
            license=payload.get("license"),
            entity_type=payload.get("entity_type"),
            source_priority=payload.get("source_priority"),
            occupation_id=payload.get("occupation_id"),
            occupation_label=payload.get("occupation_label"),
            occupation_code=payload.get("occupation_code"),
            skill_id=payload.get("skill_id"),
            skill_label=payload.get("skill_label"),
            relation_type=payload.get("relation_type"),
            isco_group=payload.get("isco_group"),
            isco_group_label=payload.get("isco_group_label"),
            skill_type=payload.get("skill_type"),
            esco_doc_type=payload.get("esco_doc_type"),
            language=payload.get("language"),
            uri=payload.get("uri"),
            chunk_index=int(payload.get("chunk_index", 0)),
            parent_doc_id=payload.get("parent_doc_id", ""),
            file_name=payload.get("file_name"),
            source=corpus_source if isinstance(corpus_source, str) else None,
        )

        chunks.append(RetrievedChunk(
            chunk_id=str(point.id),
            text=text,
            metadata=metadata,
            score=score,
        ))

    rerank_profile = select_rerank_profile(
        detected_source=detected_source,
        esco_relation_query=query_profile.esco_relation_query,
        taxonomy_query=query_profile.taxonomy_query,
    )
    logger.info(
        "retrieval_profile_selected",
        query_preview=query[:80],
        detected_profile=rerank_profile,
        classification_reason=query_profile.classification_reason,
        detected_source=detected_source,
    )
    reranking_enabled = bool(getattr(settings, "rag_enable_reranking", True))
    if reranking_enabled:
        reranked_chunks = rerank_chunks(
            query=query,
            chunks=chunks,
            rerank_profile=rerank_profile,
            detected_source=detected_source,
            esco_relation_query=query_profile.esco_relation_query,
            taxonomy_query=query_profile.taxonomy_query,
        )
    else:
        reranked_chunks = [
            chunk.model_copy(update={"rerank_score": round(chunk.score, 4)})
            for chunk in chunks
        ]
        logger.info(
            "rerank_skipped",
            query=query[:80],
            reason="disabled_by_config",
            candidate_count=len(reranked_chunks),
            detected_query_profile=rerank_profile,
        )

    logger.info(
        "retrieval_candidates_reranked",
        query=query[:80],
        detected_query_profile=rerank_profile,
        reranking_enabled=reranking_enabled,
        input_candidates=len(chunks),
        reranked=len(reranked_chunks),
        initial_scores=[round(chunk.score, 4) for chunk in chunks],
        rerank_scores=[round(chunk.rerank_score or 0.0, 4) for chunk in reranked_chunks],
        reranked_chunk_ids=[chunk.chunk_id for chunk in reranked_chunks],
    )

    final_chunks = reranked_chunks[:eff_top_k]
    selected_esco_doc_types = [
        chunk.metadata.esco_doc_type
        for chunk in final_chunks
        if chunk.metadata.esco_doc_type
    ]
    relation_docs_prioritized = bool(
        query_profile.esco_relation_query
        and final_chunks
        and final_chunks[0].metadata.esco_doc_type in {"relation_summary", "relation_detail", "taxonomy_mapping"}
    )
    logger.info(
        "retrieval_complete",
        query=query[:80],
        normalized_query=normalized_query[:80],
        query_variants=len(query_variants),
        detected_source=detected_source,
        filters_applied=bool(retrieval_filters),
        applied_filters=retrieval_filters,
        filter_fallback_used=filter_fallback_used,
        matching_filtered_candidates=matching_filtered_candidates,
        total_hits=len(scored_points),
        initial_top_k=initial_top_k,
        effective_top_k=eff_top_k,
        skipped_low_score=skipped_low_score,
        skipped_duplicate=skipped_duplicate,
        returned=len(final_chunks),
        top_score=final_chunks[0].score if final_chunks else 0.0,
        top_scores=[round(c.score, 4) for c in final_chunks],
        rerank_scores=[round(c.rerank_score or 0.0, 4) for c in final_chunks],
        final_chunk_ids=[c.chunk_id for c in final_chunks],
        sources_used=[c.metadata.source or c.metadata.source_type for c in final_chunks],
        file_names=[c.metadata.file_name for c in final_chunks],
        selected_esco_doc_types=selected_esco_doc_types,
        relation_docs_prioritized=relation_docs_prioritized,
    )
    if getattr(settings, "environment", None) == "development":
        print(
            "[RAG] initial_candidates="
            f"{len(scored_points)} reranked={len(reranked_chunks)} final={len(final_chunks)}"
        )
        for c in final_chunks:
            src = c.metadata.source or c.metadata.source_type
            fn = c.metadata.file_name or c.metadata.title
            doc_type = c.metadata.esco_doc_type or c.metadata.entity_type
            print(
                "  source="
                f"{src} file={fn} doc_type={doc_type} score={c.score:.4f} rerank={c.rerank_score or 0.0:.4f} "
                f"preview={_safe_preview_for_console(c.text)!r}..."
            )
    return final_chunks


def detect_query_source(query: str) -> str | None:
    """Infer a preferred corpus source from simple query heuristics."""
    lowered = query.lower()
    if any(hint in lowered for hint in _WEF_QUERY_HINTS):
        return "wef"
    if any(hint in lowered for hint in _ESCO_QUERY_HINTS):
        return "esco"
    return None


def _merge_source_filter(
    *,
    filters: dict[str, Any] | None,
    detected_source: str | None,
) -> dict[str, Any] | None:
    """Combine caller filters with a detected source constraint."""
    if not filters and not detected_source:
        return None

    merged = dict(filters or {})
    if detected_source and "source" not in merged:
        merged["source"] = detected_source
    return merged


def assess_evidence_strength(
    chunks: list[RetrievedChunk],
    settings: Settings | None = None,
    detected_source: str | None = None,
) -> tuple[str, str]:
    """Classify evidence as strong, weak, or none for orchestration logic."""
    if settings is None:
        settings = get_settings()
    if not chunks:
        return "none", "no_chunks"

    source_consistent, source_reason, _ = assess_source_consistency(
        chunks,
        detected_source=detected_source,
    )
    if detected_source and not source_consistent:
        return "none", f"source_inconsistent_{source_reason}"

    similarity_scores = sorted((chunk.score for chunk in chunks), reverse=True)
    rerank_scores = sorted(
        ((chunk.rerank_score or chunk.score) for chunk in chunks),
        reverse=True,
    )
    usable_scores = [score for score in similarity_scores if score >= settings.rag_similarity_threshold]
    strong_scores = [
        score for score in similarity_scores if score >= settings.rag_strong_evidence_threshold
    ]
    strong_rerank_scores = [
        score for score in rerank_scores if score >= settings.rag_rerank_coherence_threshold
    ]

    if strong_scores and strong_rerank_scores:
        return "strong", f"{len(strong_scores)}_chunks_at_or_above_strong_similarity_threshold"
    if strong_scores:
        return "none", "strong_similarity_without_rerank_coherence"
    if usable_scores:
        if not strong_rerank_scores:
            return "none", "similarity_without_rerank_coherence"
        return "weak", f"{len(usable_scores)}_chunks_at_or_above_similarity_threshold"
    return "none", "no_chunks_at_or_above_similarity_threshold"


def should_force_rag(
    chunks: list[RetrievedChunk],
    settings: Settings | None = None,
    detected_source: str | None = None,
) -> tuple[bool, str]:
    """Force the RAG path when retrieval returned enough usable context.

    The retrieval stage has already filtered out very low-score chunks using the
    weak evidence threshold, so a minimum chunk count is a practical signal that
    synthesis should prefer grounded answering over generic fallback.
    """
    if settings is None:
        settings = get_settings()

    source_consistent, source_reason, _ = assess_source_consistency(
        chunks,
        detected_source=detected_source,
    )
    if detected_source and not source_consistent:
        return False, f"source_inconsistent_{source_reason}"

    reasonable_chunks = [
        chunk
        for chunk in chunks
        if chunk.score >= settings.rag_similarity_threshold
        and (chunk.rerank_score or chunk.score) >= settings.rag_rerank_coherence_threshold
    ]
    if len(reasonable_chunks) >= settings.rag_force_min_chunks:
        return True, (
            f"{len(reasonable_chunks)}_chunks_at_or_above_similarity_threshold"
        )
    return False, "insufficient_reasonable_chunk_count"


def assess_source_consistency(
    chunks: list[RetrievedChunk],
    *,
    detected_source: str | None,
) -> tuple[bool, str, list[str]]:
    """Check whether retrieved chunks are source-coherent for named-source queries."""
    if not chunks:
        return False, "no_chunks", []

    chunk_sources = sorted(
        {
            _normalize_chunk_source(chunk)
            for chunk in chunks
            if _normalize_chunk_source(chunk)
        }
    )
    if not detected_source:
        return True, "no_named_source_requested", chunk_sources
    if not chunk_sources:
        return False, "missing_chunk_source_metadata", []
    foreign_sources = [source for source in chunk_sources if source != detected_source]
    if foreign_sources:
        return False, f"expected_{detected_source}_got_{'_'.join(chunk_sources)}", chunk_sources
    return True, f"all_chunks_match_{detected_source}", chunk_sources


def _text_fingerprint(text: str) -> str:
    """Create a normalized fingerprint for deduplication.

    Strips whitespace variation so chunks with identical content
    but different formatting are detected as duplicates.
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized[:300]


def _normalize_payload_source(payload: dict[str, Any]) -> str:
    """Resolve a stable source label from the stored payload."""
    for key in ("source", "source_name", "source_type"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _normalize_chunk_source(chunk: RetrievedChunk) -> str:
    payload_like = {
        "source": chunk.metadata.source,
        "source_name": getattr(chunk.metadata, "source_name", None),
        "source_type": chunk.metadata.source_type,
    }
    return _normalize_payload_source(payload_like)


def _sanitize_payload_sample(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep filter-debug payload logs compact and stable."""
    if not payload:
        return None
    return {
        key: payload.get(key)
        for key in (
            "source",
            "source_name",
            "source_type",
            "file_name",
            "document_title",
            "title",
            "topic",
            "entity_type",
            "section_title",
            "page_number",
            "uri",
        )
    }


def normalize_query(query: str) -> str:
    """Normalize user query variants to improve retrieval stability."""
    text = query.strip()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\((\d{4})\s*-\s*(\d{4})\)", r"\1 to \2", text)
    text = re.sub(r"(\d{4})\s*-\s*(\d{4})", r"\1 to \2", text)
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"[!?.,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_query_profile(query: str, *, detected_source: str | None = None) -> QueryProfile:
    lowered = query.lower()
    source = detected_source or detect_query_source(lowered)
    taxonomy_query = "isco" in lowered or "taxonomy" in lowered or "mapping" in lowered
    essential_optional_query = "essential" in lowered or "optional" in lowered
    relation_pattern = next(
        (
            pattern_name
            for pattern_name, pattern in _ESCO_RELATION_PATTERNS
            if pattern.search(lowered)
        ),
        None,
    )
    relation_keyword = next(
        (term for term in _ESCO_RELATION_TERMS if term in lowered),
        None,
    )
    esco_relation_query = bool(
        source == "esco"
        and (
            taxonomy_query
            or essential_optional_query
            or relation_pattern is not None
            or relation_keyword is not None
        )
    )
    classification_reason = "no_relation_signal"
    if source != "esco":
        classification_reason = "keyword:non_esco_source"
    elif taxonomy_query:
        classification_reason = "keyword:taxonomy"
    elif essential_optional_query:
        classification_reason = "keyword:essential_optional"
    elif relation_pattern is not None:
        classification_reason = f"pattern_match:{relation_pattern}"
    elif relation_keyword is not None:
        classification_reason = f"keyword:{relation_keyword}"

    return QueryProfile(
        detected_source=source,
        esco_relation_query=esco_relation_query,
        taxonomy_query=taxonomy_query,
        essential_optional_query=essential_optional_query,
        salient_concepts=_extract_salient_concepts(lowered) if esco_relation_query else (),
        classification_reason=classification_reason,
    )


def merge_query_profiles(*profiles: QueryProfile | None) -> QueryProfile:
    """Combine multiple query profiles, preserving the strongest ESCO intent signals."""
    available_profiles = [profile for profile in profiles if profile is not None]
    detected_source = next(
        (profile.detected_source for profile in available_profiles if profile.detected_source),
        None,
    )
    salient_concepts: list[str] = []
    for profile in available_profiles:
        for concept in profile.salient_concepts:
            if concept not in salient_concepts:
                salient_concepts.append(concept)

    return QueryProfile(
        detected_source=detected_source,
        esco_relation_query=any(profile.esco_relation_query for profile in available_profiles),
        taxonomy_query=any(profile.taxonomy_query for profile in available_profiles),
        essential_optional_query=any(
            profile.essential_optional_query for profile in available_profiles
        ),
        salient_concepts=tuple(salient_concepts[:8]),
        classification_reason=next(
            (
                profile.classification_reason
                for profile in available_profiles
                if profile.classification_reason != "no_relation_signal"
            ),
            "no_relation_signal",
        ),
    )


def _extract_salient_concepts(query: str) -> tuple[str, ...]:
    concepts: list[str] = []

    for pattern in (
        r"\bpython\s+and\s+sql\b",
        r"\bessential\s+and\s+optional\b",
        r"\bdata pipelines?\b",
        r"\bsql server integration services\b",
    ):
        for match in re.finditer(pattern, query):
            phrase = match.group(0).strip()
            if phrase not in concepts:
                concepts.append(phrase)

    tokens = re.findall(r"[a-z0-9+#/.-]{2,}", query)
    for token in tokens:
        if token in _ESCO_CONCEPT_STOPWORDS:
            continue
        if token in _SHORT_CONCEPTS or len(token) >= 3:
            if token not in concepts:
                concepts.append(token)

    return tuple(concepts[:8])


def _build_esco_relation_variant(query: str, profile: QueryProfile) -> str | None:
    if not profile.esco_relation_query or not profile.salient_concepts:
        return None
    concept_text = " ".join(profile.salient_concepts[:4])
    if profile.taxonomy_query:
        return f"ESCO occupation ISCO mapping taxonomy {concept_text}".strip()
    return f"ESCO occupation skill relation {concept_text}".strip()


def _search_merged(
    *,
    vectors: list[list[float]],
    top_k: int,
    filters: dict[str, Any] | None,
    client: Any,
) -> list[Any]:
    """Search each query vector and merge points by max score."""
    by_id: dict[str, Any] = {}
    for vector in vectors:
        results = search_vectors(query_vector=vector, top_k=top_k, filters=filters, client=client)
        for point in results:
            point_id = str(point.id)
            if point_id not in by_id or getattr(point, "score", 0.0) > getattr(by_id[point_id], "score", 0.0):
                by_id[point_id] = point
    merged = list(by_id.values())
    merged.sort(key=lambda p: getattr(p, "score", 0.0), reverse=True)
    return merged[:top_k]


def _friendly_qdrant_error(exc: Exception) -> str:
    """Return a safe retrieval failure message for fallback flows."""
    if isinstance(exc, QdrantConfigurationError):
        return str(exc)
    return "Retrieval is temporarily unavailable because the Qdrant endpoint could not be reached."
