from __future__ import annotations

from career_intel.rag.rerank import rerank_chunks
from career_intel.schemas.domain import ChunkMetadata, RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    score: float,
    esco_doc_type: str | None,
    occupation_id: str | None = None,
    relation_type: str | None = None,
    source: str = "esco",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata=ChunkMetadata(
            source_id=f"src-{chunk_id}",
            source_type=source,
            source=source,
            title="ESCO",
            document_title="ESCO relation docs",
            section_title="ESCO relation docs",
            chunk_index=0,
            parent_doc_id=f"doc-{chunk_id}",
            occupation_id=occupation_id,
            relation_type=relation_type,
            esco_doc_type=esco_doc_type,
        ),
    )


def test_rerank_prefers_relation_docs_for_compound_skill_queries() -> None:
    chunks = [
        _chunk(
            "generic-python",
            "Python is a programming language used in many contexts.",
            score=0.68,
            esco_doc_type="skill_summary",
        ),
        _chunk(
            "python-data-engineer",
            "ESCO occupation-skill relation. Occupation: data engineer. Skill: Python. Relation type: essential.",
            score=0.61,
            esco_doc_type="relation_detail",
            occupation_id="occ:data-engineer",
            relation_type="essential",
        ),
        _chunk(
            "sql-data-engineer",
            "ESCO occupation-skill relation. Occupation: data engineer. Skill: SQL. Relation type: optional.",
            score=0.60,
            esco_doc_type="relation_detail",
            occupation_id="occ:data-engineer",
            relation_type="optional",
        ),
    ]

    reranked = rerank_chunks(
        "Which occupations in ESCO are closely tied to Python and SQL skills?",
        chunks,
        top_n=3,
    )

    assert [chunk.chunk_id for chunk in reranked[:2]] == [
        "python-data-engineer",
        "sql-data-engineer",
    ]


def test_rerank_boosts_docs_that_explain_essential_vs_optional_distinction() -> None:
    chunks = [
        _chunk(
            "occupation-summary",
            (
                "ESCO occupation summary for data engineer. "
                "Essential linked skills and knowledge: Python, ETL. "
                "Optional linked skills and knowledge: SQL, PostgreSQL."
            ),
            score=0.56,
            esco_doc_type="occupation_summary",
            occupation_id="occ:data-engineer",
        ),
        _chunk(
            "essential-only",
            "ESCO occupation-skill relation. Occupation: data engineer. Skill: Python. Relation type: essential.",
            score=0.60,
            esco_doc_type="relation_detail",
            occupation_id="occ:data-engineer",
            relation_type="essential",
        ),
        _chunk(
            "generic-skill",
            "Python is a general programming skill.",
            score=0.62,
            esco_doc_type="skill_summary",
        ),
    ]

    reranked = rerank_chunks(
        "What is the difference between essential and optional skills in ESCO relations?",
        chunks,
        top_n=3,
    )

    assert reranked[0].chunk_id == "occupation-summary"


def test_rerank_prefers_taxonomy_mapping_for_taxonomy_queries() -> None:
    chunks = [
        _chunk(
            "relation-doc",
            "ESCO occupation-skill relation for network planner.",
            score=0.70,
            esco_doc_type="relation_detail",
            occupation_id="occ:network-planner",
        ),
        _chunk(
            "taxonomy-doc",
            "ESCO taxonomy mapping between occupation records and ISCO group 2521.",
            score=0.67,
            esco_doc_type="taxonomy_mapping",
            occupation_id="occ:network-planner",
        ),
        _chunk(
            "skill-doc",
            "General skill summary for software architecture.",
            score=0.69,
            esco_doc_type="skill_summary",
        ),
    ]

    reranked = rerank_chunks(
        "How does ISCO grouping relate to ESCO occupations?",
        chunks,
        top_n=3,
    )

    assert reranked[0].chunk_id == "taxonomy-doc"
    assert reranked[0].metadata.esco_doc_type == "taxonomy_mapping"


def test_rerank_is_light_for_wef_profile() -> None:
    chunks = [
        _chunk(
            "wef-high",
            "Future of Jobs report highlights macro labor shifts.",
            score=0.76,
            esco_doc_type=None,
            source="wef",
        ),
        _chunk(
            "wef-mid",
            "WEF discusses AI adoption and job redesign in recent surveys.",
            score=0.74,
            esco_doc_type=None,
            source="wef",
        ),
    ]

    reranked = rerank_chunks(
        "How does WEF describe AI's impact on roles and job redesign?",
        chunks,
        top_n=2,
        rerank_profile="wef_general",
        detected_source="wef",
        esco_relation_query=False,
        taxonomy_query=False,
    )

    assert {chunk.chunk_id for chunk in reranked} == {"wef-high", "wef-mid"}
    assert all(chunk.metadata.source == "wef" for chunk in reranked)
    # WEF profile is intentionally light: scores should stay close to raw vector ordering.
    assert max(abs((chunk.rerank_score or 0.0) - chunk.score) for chunk in reranked) < 0.08
