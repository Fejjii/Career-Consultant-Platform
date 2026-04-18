"""Ingest all files under ``data/raw`` (WEF PDFs, ESCO CSV/JSON) into Qdrant.

Usage:
    uv run python scripts/ingest_raw.py
"""

from __future__ import annotations

import asyncio
import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw corpus content into Qdrant.")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "esco_backfill", "esco_only_backfill"],
        help="Use 'esco_backfill' for a safe ESCO-only purge and rebuild.",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent / "data" / "raw"
    print(f"Starting raw corpus ingest from {root} with mode={args.mode} ...")

    from career_intel.rag.raw_corpus_ingest import ingest_raw_corpus

    result = await ingest_raw_corpus(base_dir=root, mode=args.mode)
    print(f"Run ID: {result.run_id}")
    print(f"Documents processed: {result.documents_processed}")
    print(f"Chunks created: {result.chunks_created}")


if __name__ == "__main__":
    asyncio.run(main())
