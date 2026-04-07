"""End-to-end ingestion pipeline: file -> chunks -> embed -> Qdrant + Postgres."""

from __future__ import annotations

import csv
import hashlib
import io
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from career_intel.config import get_settings
from career_intel.rag.chunking import chunk_csv_rows, chunk_markdown
from career_intel.rag.embeddings import get_embeddings
from career_intel.schemas.api import IngestResponse
from career_intel.storage.db import Document, IngestionRun, get_session_factory, init_db
from career_intel.storage.qdrant_store import ensure_collection, upsert_vectors

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {".md", ".txt", ".csv"}


async def run_ingestion(
    paths: list[str],
    mode: str = "full",
) -> IngestResponse:
    """Run the full ingestion pipeline for a list of file paths."""
    settings = get_settings()
    run_id = str(uuid.uuid4())

    # Ensure infrastructure
    ensure_collection()
    await init_db()

    session_factory = get_session_factory()
    total_docs = 0
    total_chunks = 0

    async with session_factory() as session:
        ingestion_run = IngestionRun(id=run_id, mode=mode)
        session.add(ingestion_run)

        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning("ingest_file_not_found", path=path_str)
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning("ingest_unsupported_extension", path=path_str, ext=path.suffix)
                continue

            text = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(text.encode()).hexdigest()

            doc_id = str(uuid.uuid4())
            base_metadata: dict[str, Any] = {
                "source_id": doc_id,
                "source_type": path.suffix.lstrip("."),
                "title": path.stem,
                "parent_doc_id": doc_id,
                "uri": str(path.resolve()),
            }

            # Chunk
            if path.suffix.lower() == ".csv":
                reader = csv.DictReader(io.StringIO(text))
                rows = list(reader)
                chunks = chunk_csv_rows(rows, base_metadata)
            else:
                chunks = chunk_markdown(text, base_metadata)

            if not chunks:
                logger.warning("ingest_no_chunks", path=path_str)
                continue

            # Embed
            chunk_texts = [c.text for c in chunks]
            vectors = get_embeddings(chunk_texts, settings=settings)

            # Upsert to Qdrant
            ids = [c.chunk_id for c in chunks]
            payloads = [
                {**c.metadata, "text": c.text}
                for c in chunks
            ]
            upsert_vectors(ids=ids, vectors=vectors, payloads=payloads)

            # Record in Postgres
            doc_record = Document(
                id=doc_id,
                uri=str(path.resolve()),
                checksum=checksum,
                source_type=path.suffix.lstrip("."),
                title=path.stem,
            )
            session.add(doc_record)

            total_docs += 1
            total_chunks += len(chunks)
            logger.info("ingest_file_complete", path=path_str, chunks=len(chunks))

        ingestion_run.documents_processed = total_docs
        ingestion_run.chunks_created = total_chunks
        ingestion_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
        ingestion_run.success = True
        await session.commit()

    logger.info("ingestion_run_complete", run_id=run_id, docs=total_docs, chunks=total_chunks)
    return IngestResponse(
        run_id=run_id,
        documents_processed=total_docs,
        chunks_created=total_chunks,
    )
