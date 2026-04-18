"""Deterministic lightweight reranking for retrieved chunks."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

logger = structlog.get_logger()

if TYPE_CHECKING:
    from career_intel.schemas.domain import RetrievedChunk


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
        "relate",
        "related",
        "associated",
        "strongly",
        "esco",
        "occupation",
        "occupations",
        "skill",
        "skills",
        "relation",
        "relations",
        "framework",
        "classification",
        "international",
        "standard",
        "european",
        "competences",
        "qualifications",
        "occupation-to-skill",
        "map",
    }
)
_SHORT_CONCEPTS = frozenset({"ai", "bi", "ml", "sql", "etl", "ui", "ux"})
_ESCO_RELATION_HINTS = frozenset(
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
        "closely",
    }
)


@dataclass(frozen=True)
class RerankPolicy:
    """Weight and boost policy for a specific query profile."""

    profile: str
    vector_weight: float
    lexical_weight: float
    title_weight: float
    source_weight: float
    keyword_weight_per_hit: float
    keyword_max_boost: float
    concept_coverage_weight: float
    multi_concept_bonus: float
    shared_occupation_bonus: float
    relation_type_bonus: float
    weak_overlap_penalty: float
    no_keyword_penalty: float
    non_preferred_doc_penalty: float
    preferred_doc_types: frozenset[str]
    doc_type_boosts: dict[str, float]
    taxonomy_relation_bleed_penalty: float
    taxonomy_signal_floor: float


_RERANK_POLICIES: dict[str, RerankPolicy] = {
    "esco_relation": RerankPolicy(
        profile="esco_relation",
        vector_weight=0.51,
        lexical_weight=0.24,
        title_weight=0.10,
        source_weight=0.05,
        keyword_weight_per_hit=0.03,
        keyword_max_boost=0.16,
        concept_coverage_weight=0.14,
        multi_concept_bonus=0.10,
        shared_occupation_bonus=0.08,
        relation_type_bonus=0.08,
        weak_overlap_penalty=0.08,
        no_keyword_penalty=0.04,
        non_preferred_doc_penalty=0.02,
        preferred_doc_types=frozenset({"relation_detail", "relation_summary"}),
        doc_type_boosts={
            "relation_summary": 0.22,
            "relation_detail": 0.20,
            "occupation_summary": 0.10,
            "skill_summary": 0.06,
            "taxonomy_mapping": 0.05,
            "isco_group_summary": 0.03,
        },
        taxonomy_relation_bleed_penalty=0.0,
        taxonomy_signal_floor=0.0,
    ),
    "esco_taxonomy": RerankPolicy(
        profile="esco_taxonomy",
        vector_weight=0.58,
        lexical_weight=0.20,
        title_weight=0.10,
        source_weight=0.06,
        keyword_weight_per_hit=0.015,
        keyword_max_boost=0.08,
        concept_coverage_weight=0.10,
        multi_concept_bonus=0.06,
        shared_occupation_bonus=0.04,
        relation_type_bonus=0.04,
        weak_overlap_penalty=0.08,
        no_keyword_penalty=0.05,
        non_preferred_doc_penalty=0.03,
        preferred_doc_types=frozenset({"taxonomy_mapping", "isco_group_summary"}),
        doc_type_boosts={
            "taxonomy_mapping": 0.28,
            "isco_group_summary": 0.22,
            "occupation_summary": 0.08,
            "relation_detail": 0.01,
            "relation_summary": 0.00,
            "skill_summary": -0.02,
        },
        taxonomy_relation_bleed_penalty=0.12,
        taxonomy_signal_floor=0.22,
    ),
    "esco_general": RerankPolicy(
        profile="esco_general",
        vector_weight=0.60,
        lexical_weight=0.20,
        title_weight=0.10,
        source_weight=0.06,
        keyword_weight_per_hit=0.018,
        keyword_max_boost=0.09,
        concept_coverage_weight=0.10,
        multi_concept_bonus=0.06,
        shared_occupation_bonus=0.05,
        relation_type_bonus=0.05,
        weak_overlap_penalty=0.08,
        no_keyword_penalty=0.04,
        non_preferred_doc_penalty=0.02,
        preferred_doc_types=frozenset(
            {"occupation_summary", "skill_summary", "relation_detail", "taxonomy_mapping"}
        ),
        doc_type_boosts={
            "occupation_summary": 0.11,
            "skill_summary": 0.09,
            "relation_detail": 0.08,
            "relation_summary": 0.08,
            "taxonomy_mapping": 0.08,
            "isco_group_summary": 0.06,
        },
        taxonomy_relation_bleed_penalty=0.0,
        taxonomy_signal_floor=0.0,
    ),
    "wef_general": RerankPolicy(
        profile="wef_general",
        vector_weight=0.94,
        lexical_weight=0.03,
        title_weight=0.02,
        source_weight=0.01,
        keyword_weight_per_hit=0.005,
        keyword_max_boost=0.02,
        concept_coverage_weight=0.0,
        multi_concept_bonus=0.0,
        shared_occupation_bonus=0.0,
        relation_type_bonus=0.0,
        weak_overlap_penalty=0.0,
        no_keyword_penalty=0.0,
        non_preferred_doc_penalty=0.0,
        preferred_doc_types=frozenset(),
        doc_type_boosts={},
        taxonomy_relation_bleed_penalty=0.0,
        taxonomy_signal_floor=0.0,
    ),
}


def select_rerank_profile(
    *,
    detected_source: str | None,
    esco_relation_query: bool,
    taxonomy_query: bool,
) -> str:
    """Return one of the supported reranking profiles."""
    normalized_source = (detected_source or "").strip().lower()
    if normalized_source == "wef":
        return "wef_general"
    if normalized_source == "esco":
        if taxonomy_query:
            return "esco_taxonomy"
        if esco_relation_query:
            return "esco_relation"
        return "esco_general"
    if taxonomy_query:
        return "esco_taxonomy"
    if esco_relation_query:
        return "esco_relation"
    return "wef_general"


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int | None = None,
    *,
    rerank_profile: str | None = None,
    detected_source: str | None = None,
    esco_relation_query: bool | None = None,
    taxonomy_query: bool | None = None,
) -> list[RetrievedChunk]:
    """Rerank chunks with deterministic lexical and metadata features.

    This keeps production behaviour stable without adding a heavy model dependency.
    """
    if not chunks:
        return []

    query_profile = _build_query_profile(query)
    profile_name = rerank_profile or select_rerank_profile(
        detected_source=detected_source or query_profile["detected_source"],
        esco_relation_query=(
            query_profile["esco_relation_query"]
            if esco_relation_query is None
            else esco_relation_query
        ),
        taxonomy_query=(query_profile["taxonomy_query"] if taxonomy_query is None else taxonomy_query),
    )
    policy = _RERANK_POLICIES.get(profile_name, _RERANK_POLICIES["wef_general"])
    query_terms = _tokenize(query)
    keyword_terms = _keyword_terms(query, query_profile["salient_concepts"])
    exact_skill_terms = _exact_skill_terms(query_profile["salient_concepts"])
    concept_hits_by_chunk: dict[str, set[str]] = {}
    occupation_concepts: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        hits = _matched_concepts(chunk, query_profile["salient_concepts"])
        concept_hits_by_chunk[chunk.chunk_id] = hits
        if chunk.metadata.occupation_id and hits:
            occupation_concepts[chunk.metadata.occupation_id].update(hits)

    logger.info(
        "rerank_policy_selected",
        detected_query_profile=profile_name,
        rerank_policy=policy.profile,
        detected_source=detected_source or query_profile["detected_source"],
        esco_relation_query=query_profile["esco_relation_query"],
        taxonomy_query=query_profile["taxonomy_query"],
    )
    top_docs_before = sorted(
        [
            {
                "chunk_id": chunk.chunk_id,
                "vector_score": round(chunk.score, 4),
                "esco_doc_type": chunk.metadata.esco_doc_type,
            }
            for chunk in chunks
        ],
        key=lambda item: float(item["vector_score"]),
        reverse=True,
    )[: min(top_n or len(chunks), 5)]
    logger.info(
        "rerank_input_scores",
        input_count=len(chunks),
        top_docs_before_rerank=top_docs_before,
        initial_chunk_scores=[
            {
                "chunk_id": chunk.chunk_id,
                "vector_score": round(chunk.score, 4),
                "esco_doc_type": chunk.metadata.esco_doc_type,
            }
            for chunk in chunks
        ],
    )

    reranked: list[RetrievedChunk] = []
    top_explanations: list[dict[str, object]] = []
    for chunk in chunks:
        lexical = _lexical_overlap(query_terms, _tokenize(chunk.text))
        title_terms = _tokenize(
            " ".join(
                filter(
                    None,
                    [
                        chunk.metadata.title,
                        chunk.metadata.document_title,
                        chunk.metadata.section_title,
                        chunk.metadata.file_name,
                    ],
                )
            )
        )
        title_overlap = _lexical_overlap(query_terms, title_terms)
        source_boost = min(max((chunk.metadata.source_priority or 0) / 100.0, 0.0), 1.0)
        vector_score = min(max(chunk.score, 0.0), 1.0)
        keyword_boost = _keyword_boost(chunk, keyword_terms, policy)
        irrelevance_penalty = _irrelevance_penalty(
            chunk=chunk,
            lexical=lexical,
            title_overlap=title_overlap,
            policy=policy,
            keyword_terms=keyword_terms,
        )
        reasons: list[str] = []

        rerank_score = (
            policy.vector_weight * vector_score
            + policy.lexical_weight * lexical
            + policy.title_weight * title_overlap
            + policy.source_weight * source_boost
        )
        reasons.append(f"vector={vector_score:.3f}")
        if lexical:
            reasons.append(f"lexical={lexical:.3f}")
        if title_overlap:
            reasons.append(f"title={title_overlap:.3f}")
        if keyword_boost:
            rerank_score += keyword_boost
            reasons.append(f"keyword_match+{keyword_boost:.2f}")
        if policy.profile == "esco_relation":
            exact_skill_boost = _exact_skill_term_boost(chunk, exact_skill_terms)
            if exact_skill_boost:
                rerank_score += exact_skill_boost
                reasons.append(f"exact_skill_match+{exact_skill_boost:.2f}")
        if irrelevance_penalty:
            rerank_score -= irrelevance_penalty
            reasons.append(f"irrelevance_penalty-{irrelevance_penalty:.2f}")

        doc_boost = _doc_type_boost(chunk.metadata.esco_doc_type, policy)
        if doc_boost > 0:
            rerank_score += doc_boost
            reasons.append(f"doc_type={chunk.metadata.esco_doc_type or 'generic'}+{doc_boost:.2f}")
        elif doc_boost < 0:
            rerank_score += doc_boost
            reasons.append(f"doc_type={chunk.metadata.esco_doc_type or 'generic'}{doc_boost:.2f}")

        concept_hits = concept_hits_by_chunk.get(chunk.chunk_id, set())
        salient_concepts = query_profile["salient_concepts"]
        if salient_concepts and policy.concept_coverage_weight > 0:
            coverage = len(concept_hits) / len(salient_concepts)
            coverage_boost = policy.concept_coverage_weight * coverage
            if coverage_boost:
                rerank_score += coverage_boost
                reasons.append(
                    f"concept_coverage={len(concept_hits)}/{len(salient_concepts)}+{coverage_boost:.2f}"
                )
            if len(salient_concepts) >= 2 and len(concept_hits) >= 2 and policy.multi_concept_bonus > 0:
                rerank_score += policy.multi_concept_bonus
                reasons.append(f"multi_concept_match+{policy.multi_concept_bonus:.2f}")
            occupation_hits = occupation_concepts.get(chunk.metadata.occupation_id or "", set())
            if (
                chunk.metadata.occupation_id
                and len(salient_concepts) >= 2
                and len(occupation_hits) >= 2
                and policy.shared_occupation_bonus > 0
            ):
                rerank_score += policy.shared_occupation_bonus
                reasons.append(f"shared_occupation_concepts+{policy.shared_occupation_bonus:.2f}")

        relation_boost = _relation_type_boost(chunk, query_profile, policy)
        if relation_boost:
            rerank_score += relation_boost
            reasons.append(f"relation_type+{relation_boost:.2f}")

        taxonomy_bleed_penalty = _taxonomy_relation_bleed_penalty(
            chunk=chunk,
            profile_name=profile_name,
            policy=policy,
            lexical=lexical,
            title_overlap=title_overlap,
        )
        if taxonomy_bleed_penalty:
            rerank_score -= taxonomy_bleed_penalty
            reasons.append(f"taxonomy_bleed_penalty-{taxonomy_bleed_penalty:.2f}")

        reranked.append(chunk.model_copy(update={"rerank_score": round(rerank_score, 4)}))
        top_explanations.append(
            {
                "chunk_id": chunk.chunk_id,
                "esco_doc_type": chunk.metadata.esco_doc_type,
                "rerank_score": round(rerank_score, 4),
                "reasons": reasons,
            }
        )

    reranked.sort(
        key=lambda chunk: (
            chunk.rerank_score or 0.0,
            chunk.score,
        ),
        reverse=True,
    )
    final_chunks = reranked if top_n is None else reranked[:top_n]
    top_docs_after = [
        {
            "chunk_id": chunk.chunk_id,
            "vector_score": round(chunk.score, 4),
            "rerank_score": round(chunk.rerank_score or 0.0, 4),
            "esco_doc_type": chunk.metadata.esco_doc_type,
        }
        for chunk in final_chunks[: min(top_n or len(final_chunks), 5)]
    ]
    logger.info(
        "rerank_complete",
        detected_query_profile=profile_name,
        rerank_policy=policy.profile,
        input_count=len(chunks),
        output_count=len(final_chunks),
        top_docs_after_rerank=top_docs_after,
        scores=[chunk.rerank_score for chunk in final_chunks],
        reranked_chunk_scores=[
            {
                "chunk_id": chunk.chunk_id,
                "vector_score": round(chunk.score, 4),
                "rerank_score": round(chunk.rerank_score or 0.0, 4),
                "esco_doc_type": chunk.metadata.esco_doc_type,
            }
            for chunk in final_chunks
        ],
        selected_esco_doc_types=[chunk.metadata.esco_doc_type for chunk in final_chunks if chunk.metadata.esco_doc_type],
        relation_docs_prioritized=bool(
            query_profile["esco_relation_query"]
            and final_chunks
            and final_chunks[0].metadata.esco_doc_type in {"relation_summary", "relation_detail", "taxonomy_mapping"}
        ),
        top_explanations=sorted(
            top_explanations,
            key=lambda item: float(item["rerank_score"]),
            reverse=True,
        )[:top_n],
    )
    return final_chunks


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    return {token for token in tokens if not token.isdigit() or len(token) == 4}


def _lexical_overlap(query_terms: set[str], doc_terms: set[str]) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    overlap = len(query_terms & doc_terms)
    denom = math.sqrt(len(query_terms) * len(doc_terms))
    if denom == 0:
        return 0.0
    return overlap / denom


def _keyword_terms(query: str, salient_concepts: tuple[str, ...]) -> set[str]:
    """Extract boosted keywords from query and salient concept profile."""
    keywords = {
        token
        for token in _tokenize(query)
        if token not in _ESCO_CONCEPT_STOPWORDS and (len(token) >= 3 or token in _SHORT_CONCEPTS)
    }
    for concept in salient_concepts:
        for token in _tokenize(concept):
            if token not in _ESCO_CONCEPT_STOPWORDS and (len(token) >= 3 or token in _SHORT_CONCEPTS):
                keywords.add(token)
    return keywords


def _exact_skill_terms(salient_concepts: tuple[str, ...]) -> set[str]:
    terms: set[str] = set()
    for concept in salient_concepts:
        for token in _tokenize(concept):
            if token in _ESCO_CONCEPT_STOPWORDS:
                continue
            if len(token) >= 3 or token in _SHORT_CONCEPTS:
                terms.add(token)
    return terms


def _exact_skill_term_boost(chunk: RetrievedChunk, exact_skill_terms: set[str]) -> float:
    if not exact_skill_terms:
        return 0.0
    haystack = " ".join(
        filter(
            None,
            [
                chunk.text.lower(),
                (chunk.metadata.title or "").lower(),
                (chunk.metadata.document_title or "").lower(),
                (chunk.metadata.section_title or "").lower(),
                (chunk.metadata.skill_label or "").lower(),
            ],
        )
    )
    hits = 0
    for term in exact_skill_terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, haystack):
            hits += 1
    if hits <= 0:
        return 0.0
    return min(0.04 * hits, 0.2)


def _keyword_boost(
    chunk: RetrievedChunk,
    keyword_terms: set[str],
    policy: RerankPolicy,
) -> float:
    if not keyword_terms:
        return 0.0
    chunk_terms = _tokenize(
        " ".join(
            filter(
                None,
                [
                    chunk.text,
                    chunk.metadata.title,
                    chunk.metadata.document_title,
                    chunk.metadata.section_title,
                ],
            )
        )
    )
    hit_count = len(chunk_terms & keyword_terms)
    if hit_count <= 0:
        return 0.0
    return min(policy.keyword_weight_per_hit * hit_count, policy.keyword_max_boost)


def _irrelevance_penalty(
    *,
    chunk: RetrievedChunk,
    lexical: float,
    title_overlap: float,
    policy: RerankPolicy,
    keyword_terms: set[str],
) -> float:
    penalty = 0.0
    if lexical < 0.03 and title_overlap < 0.03:
        penalty += policy.weak_overlap_penalty

    if keyword_terms:
        chunk_terms = _tokenize(
            " ".join(
                filter(
                    None,
                    [
                        chunk.text,
                        chunk.metadata.title,
                        chunk.metadata.document_title,
                        chunk.metadata.section_title,
                    ],
                )
            )
        )
        if not (chunk_terms & keyword_terms):
            penalty += policy.no_keyword_penalty

    if policy.preferred_doc_types and chunk.metadata.esco_doc_type not in policy.preferred_doc_types and lexical < 0.08:
        penalty += policy.non_preferred_doc_penalty

    return penalty


def _build_query_profile(query: str) -> dict[str, object]:
    lowered = query.lower()
    detected_source = "wef" if "wef" in lowered or "future of jobs" in lowered else (
        "esco" if "esco" in lowered or "isco" in lowered else None
    )
    taxonomy_query = "isco" in lowered or "taxonomy" in lowered or "mapping" in lowered
    essential_optional_query = "essential" in lowered or "optional" in lowered
    esco_relation_query = bool(
        detected_source == "esco"
        and (
            taxonomy_query
            or essential_optional_query
            or any(term in lowered for term in _ESCO_RELATION_HINTS)
        )
    )
    return {
        "detected_source": detected_source,
        "esco_relation_query": esco_relation_query,
        "taxonomy_query": taxonomy_query,
        "essential_optional_query": essential_optional_query,
        "salient_concepts": _extract_salient_concepts(lowered) if esco_relation_query else (),
    }


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

    for token in re.findall(r"[a-z0-9+#/.-]{2,}", query):
        if token in _ESCO_CONCEPT_STOPWORDS:
            continue
        if token in _SHORT_CONCEPTS or len(token) >= 3:
            if token not in concepts:
                concepts.append(token)
    return tuple(concepts[:8])


def _matched_concepts(chunk: RetrievedChunk, concepts: tuple[str, ...]) -> set[str]:
    if not concepts:
        return set()
    haystack = " ".join(
        filter(
            None,
            [
                chunk.text.lower(),
                (chunk.metadata.title or "").lower(),
                (chunk.metadata.document_title or "").lower(),
                (chunk.metadata.section_title or "").lower(),
            ],
        )
    )
    matched: set[str] = set()
    for concept in concepts:
        if concept in haystack:
            matched.add(concept)
    return matched


def _doc_type_boost(esco_doc_type: str | None, policy: RerankPolicy) -> float:
    if not esco_doc_type:
        return 0.0
    return policy.doc_type_boosts.get(esco_doc_type, 0.0)


def _relation_type_boost(
    chunk: RetrievedChunk,
    query_profile: dict[str, object],
    policy: RerankPolicy,
) -> float:
    if policy.relation_type_bonus <= 0:
        return 0.0
    boost = 0.0
    text_lower = chunk.text.lower()
    relation_type = (chunk.metadata.relation_type or "").lower()
    if query_profile["essential_optional_query"]:
        if relation_type in {"essential", "optional"}:
            boost += policy.relation_type_bonus
        if "essential" in text_lower and "optional" in text_lower:
            boost += min(policy.relation_type_bonus, 0.05)
    return boost


def _taxonomy_relation_bleed_penalty(
    *,
    chunk: RetrievedChunk,
    profile_name: str,
    policy: RerankPolicy,
    lexical: float,
    title_overlap: float,
) -> float:
    if profile_name != "esco_taxonomy" or policy.taxonomy_relation_bleed_penalty <= 0:
        return 0.0
    doc_type = chunk.metadata.esco_doc_type or ""
    if doc_type not in {"relation_detail", "relation_summary"}:
        return 0.0
    taxonomy_signal = max(lexical, title_overlap)
    if taxonomy_signal >= policy.taxonomy_signal_floor:
        return 0.0
    return policy.taxonomy_relation_bleed_penalty
