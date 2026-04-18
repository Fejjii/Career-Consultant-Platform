"""Evaluate routing, retrieval, grounding path, and answers across query categories.

Usage:
    uv run python scripts/evaluate_rag.py
"""

from __future__ import annotations

import asyncio
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuery:
    category: str
    query: str


EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery("wef_trends", "What are the major labour-market shifts highlighted in WEF Future of Jobs 2025?"),
    EvalQuery("wef_trends", "Which skills are growing fastest in recent WEF future of jobs reports?"),
    EvalQuery("wef_trends", "How does WEF describe AI's impact on roles and job redesign?"),
    EvalQuery("wef_trends", "What trends does WEF report about green jobs and sustainability skills?"),
    EvalQuery("wef_trends", "Which role families appear resilient or growing according to WEF?"),
    EvalQuery("esco_occupations", "What does ESCO say about the Data Engineer occupation?"),
    EvalQuery("esco_occupations", "Which occupations in ESCO are closely tied to Python and SQL skills?"),
    EvalQuery("esco_occupations", "How does ISCO grouping relate to ESCO occupations?"),
    EvalQuery("esco_skills", "What essential skills are associated with a data engineer in ESCO?"),
    EvalQuery("esco_skills", "What does ESCO define as cloud-related skills or competences?"),
    EvalQuery("esco_skills", "Which occupations are linked to ETL or data pipeline skills?"),
    EvalQuery("esco_skills", "What is the difference between essential and optional skills in ESCO relations?"),
    EvalQuery("tool_case", "Compare data engineer and machine learning engineer roles."),
    EvalQuery("tool_case", "Build a 12-week learning plan for becoming a data engineer."),
    EvalQuery("tool_case", "Identify the skill gap for a Python analyst moving into data engineering."),
    EvalQuery("no_retrieval", "hello"),
    EvalQuery("no_retrieval", "thanks"),
    EvalQuery("fallback", "What will the job market on Mars look like in 2040?"),
]
ESCO_RELATION_VALIDATION_QUERIES: list[EvalQuery] = [
    EvalQuery("esco_relation_validation", "Which occupations in ESCO are closely tied to Python and SQL skills?"),
    EvalQuery("esco_relation_validation", "What is the difference between essential and optional skills in ESCO relations?"),
    EvalQuery("esco_relation_validation", "How does ISCO grouping relate to ESCO occupations?"),
    EvalQuery("esco_relation_validation", "Which occupations are linked to ETL or data pipeline skills?"),
]
_RELATION_PRIORITY_DOC_TYPES = {"relation_summary", "relation_detail", "taxonomy_mapping"}


def _safe_console_text(text: object) -> str:
    return str(text).encode("ascii", errors="replace").decode("ascii")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate routing, retrieval, and grounding behavior.")
    parser.add_argument(
        "--query-set",
        default="all",
        choices=["all", "esco_relation_validation"],
        help="Run all evaluation queries or only the four ESCO relation validation queries.",
    )
    return parser.parse_args()


def _resolve_queries(query_set: str) -> list[EvalQuery]:
    if query_set == "esco_relation_validation":
        return ESCO_RELATION_VALIDATION_QUERIES
    return EVAL_QUERIES


def _selected_esco_doc_types(chunks: list) -> list[str]:
    return [chunk.metadata.esco_doc_type for chunk in chunks if chunk.metadata.esco_doc_type]


def _relation_docs_prioritized(chunks: list, query_profile: object | None) -> bool:
    if query_profile is None or not getattr(query_profile, "esco_relation_query", False):
        return False
    if not chunks:
        return False
    return (chunks[0].metadata.esco_doc_type or "") in _RELATION_PRIORITY_DOC_TYPES


async def main() -> None:
    args = _parse_args()

    from career_intel.config import get_settings
    from career_intel.orchestration.chain import run_turn
    from career_intel.rag.retriever import (
        assess_evidence_strength,
        build_query_profile,
        detect_query_source,
        merge_query_profiles,
        retrieve_chunks,
        rewrite_query,
    )
    from career_intel.schemas.api import ChatMessage
    from career_intel.tools.registry import route_query

    settings = get_settings()
    summaries: list[dict[str, object]] = []

    for item in _resolve_queries(args.query_set):
        print("\n" + "=" * 88)
        print(f"CATEGORY: {_safe_console_text(item.category)}")
        print(f"QUERY: {_safe_console_text(item.query)}")

        decision, _usage = await route_query(item.query, cv_available=False, settings=settings)
        retrieval_triggered = decision.intent in {"domain_specific", "retrieval_required"}
        print(f"ROUTER_INTENT: {decision.intent}")
        print(f"RETRIEVAL_TRIGGERED: {retrieval_triggered}")
        print(f"RETRIEVER_CALLED: {retrieval_triggered}")

        rewritten = item.query
        chunks = []
        evidence_strength = "none"
        query_profile = None
        detected_source = None
        selected_esco_doc_types: list[str] = []
        relation_docs_prioritized = False
        if retrieval_triggered:
            original_detected_source = detect_query_source(item.query)
            original_query_profile = build_query_profile(
                item.query,
                detected_source=original_detected_source,
            )
            rewritten = await rewrite_query(item.query, settings=settings)
            rewritten_detected_source = detect_query_source(rewritten)
            detected_source = rewritten_detected_source or original_detected_source
            query_profile = merge_query_profiles(
                original_query_profile,
                build_query_profile(rewritten, detected_source=detected_source),
            )
            chunks = await retrieve_chunks(
                query=rewritten,
                filters=None,
                settings=settings,
                detected_source_override=detected_source,
                query_profile_override=query_profile,
            )
            evidence_strength, _ = assess_evidence_strength(
                chunks,
                settings=settings,
                detected_source=detected_source,
            )
            selected_esco_doc_types = _selected_esco_doc_types(chunks)
            relation_docs_prioritized = _relation_docs_prioritized(chunks, query_profile)
            print(f"REWRITTEN_QUERY: {_safe_console_text(rewritten)}")
            print(f"DETECTED_SOURCE: {_safe_console_text(detected_source)}")
            print(f"ORIGINAL_DETECTED_SOURCE: {_safe_console_text(original_detected_source)}")
            print(f"REWRITTEN_DETECTED_SOURCE: {_safe_console_text(rewritten_detected_source)}")
            print(
                "QUERY_PROFILE: "
                f"esco_relation_query={getattr(query_profile, 'esco_relation_query', False)} "
                f"taxonomy_query={getattr(query_profile, 'taxonomy_query', False)} "
                f"essential_optional_query={getattr(query_profile, 'essential_optional_query', False)} "
                f"classification_reason={_safe_console_text(getattr(query_profile, 'classification_reason', ''))} "
                f"salient_concepts={_safe_console_text(repr(list(getattr(query_profile, 'salient_concepts', ())))) }"
            )
            print(f"EVIDENCE_STRENGTH: {evidence_strength}")
            print(f"SELECTED_ESCO_DOC_TYPES: {selected_esco_doc_types}")
            print(
                "SELECTED_ESCO_DOC_TYPES_STATUS: "
                f"{'populated' if selected_esco_doc_types else 'empty'}"
            )
            print(f"RELATION_DOCS_PRIORITIZED: {relation_docs_prioritized}")
            print(f"CHUNKS_RETRIEVED: {len(chunks)}")
            for chunk in chunks:
                print(
                    "  - "
                    f"score={chunk.score:.4f} rerank={chunk.rerank_score or 0.0:.4f} "
                    f"title={_safe_console_text(chunk.metadata.document_title or chunk.metadata.title)} "
                    f"section={_safe_console_text(chunk.metadata.section_title or chunk.metadata.section)}"
                )
        else:
            print("CHUNKS_RETRIEVED: 0")

        response = await run_turn(
            messages=[ChatMessage(role="user", content=item.query)],
            session_id=f"eval-{item.category}",
            use_tools=True,
            filters=None,
            settings=settings,
            trace_id=f"eval-{item.category}",
        )
        print(f"ANSWER_SOURCE: {response.answer_source}")
        print(f"GROUNDED: {'yes' if response.answer_source == 'rag' else 'no'}")
        print(f"FINAL_ANSWER:\n{_safe_console_text(response.reply)}")

        summaries.append(
            {
                "query": item.query,
                "retrieval_triggered": retrieval_triggered,
                "selected_esco_doc_types": selected_esco_doc_types,
                "relation_docs_prioritized": relation_docs_prioritized,
                "evidence_strength": evidence_strength,
                "answer_source": response.answer_source,
                "grounded": response.answer_source == "rag",
            }
        )

    if summaries:
        print("\n" + "=" * 88)
        print("VALIDATION_SUMMARY")
        for summary in summaries:
            print(summary)


if __name__ == "__main__":
    asyncio.run(main())
