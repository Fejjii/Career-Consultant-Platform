"""Script to ingest pilot corpus into the vector store.

Usage:
    uv run python scripts/ingest_pilot.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path


async def main() -> None:
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    paths = sorted(str(p) for p in data_dir.glob("*.md"))

    if not paths:
        print(f"No .md files found in {data_dir}")
        return

    print(f"Found {len(paths)} files to ingest:")
    for p in paths:
        print(f"  - {p}")

    from career_intel.rag.ingest_pipeline import run_ingestion

    result = await run_ingestion(paths=paths, mode="full")
    print(f"\nIngestion complete:")
    print(f"  Run ID: {result.run_id}")
    print(f"  Documents: {result.documents_processed}")
    print(f"  Chunks: {result.chunks_created}")


if __name__ == "__main__":
    asyncio.run(main())
