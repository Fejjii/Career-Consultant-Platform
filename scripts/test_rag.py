"""Run sample RAG queries against Qdrant (requires OPENAI_API_KEY + Qdrant).

Usage:
    uv run python scripts/test_rag.py
"""

from __future__ import annotations

import asyncio

QUERIES = [
    "What skills are essential for a data engineer according to labour frameworks?",
    "Summarise trends on AI and job disruption from WEF future of jobs reports.",
    "What is an ISCO group and how does it relate to ESCO occupations?",
    "Which digital skills appear most often in European occupation data?",
    "What roles involve Python as an essential skill in ESCO?",
]


async def main() -> None:
    from career_intel.config import get_settings
    from career_intel.orchestration.chain import run_turn
    from career_intel.rag.retriever import retrieve_chunks, rewrite_query
    from career_intel.schemas.api import ChatMessage
    from career_intel.storage.qdrant_store import ensure_collection

    settings = get_settings()
    ensure_collection()

    for query in QUERIES:
        print("\n" + "=" * 72)
        print(f"QUERY: {query}")

        rewritten = await rewrite_query(query, settings=settings)
        print(f"REWRITTEN: {rewritten}")

        chunks = await retrieve_chunks(
            query=rewritten,
            filters=None,
            settings=settings,
        )

        print(f"RETRIEVED ({len(chunks)} chunks):")
        for c in chunks:
            src = c.metadata.source or c.metadata.source_type
            fn = c.metadata.file_name or c.metadata.title
            preview = c.text.replace("\n", " ")[:220]
            print(f"  score={c.score:.4f} source={src} file={fn}")
            print(f"    {preview!r}")

        response = await run_turn(
            messages=[ChatMessage(role="user", content=query)],
            session_id="rag-smoke",
            use_tools=True,
            filters=None,
            settings=settings,
            trace_id="rag-smoke",
        )

        print(f"ANSWER_SOURCE: {response.answer_source}")
        print(f"ANSWER:\n{response.reply}")
        if response.citations:
            print(f"CITATIONS: {len(response.citations)}")


if __name__ == "__main__":
    asyncio.run(main())
