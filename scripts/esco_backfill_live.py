"""Safely purge and rebuild only ESCO vectors in the live Qdrant collection."""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from career_intel.config import get_settings
from career_intel.rag.chunking import RawChunk, chunk_text_by_tokens
from career_intel.rag.embeddings import get_embeddings
from career_intel.rag.raw_corpus_ingest import (
    ESCO_ENRICHED_DOC_TYPES,
    _build_esco_documents,
    _print_esco_diagnostics,
)
from career_intel.storage.qdrant_store import (
    count_vectors,
    delete_vectors_by_metadata,
    ensure_collection,
    get_esco_vector_diagnostics,
    upsert_vectors,
)


@dataclass(frozen=True)
class EmbeddedBatch:
    batch_index: int
    docs_processed: int
    chunks: list[RawChunk]
    vectors: list[list[float]]
    embedding_elapsed_ms: float


def _format_elapsed(started_at: float) -> str:
    elapsed_seconds = int(time.perf_counter() - started_at)
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely purge and rebuild only ESCO vectors.")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--occupation-keywords",
        type=str,
        default=None,
        help=(
            "Comma-separated occupation keywords to keep a focused ESCO subset "
            "(e.g. 'data engineer,data analyst,ai engineer')."
        ),
    )
    parser.add_argument("--stop-after-first-flush", action="store_true")
    parser.add_argument("--embedding-concurrency", type=int, default=6)
    parser.add_argument("--qdrant-upsert-batch-size", type=int, default=128)
    return parser.parse_args()


def _embed_batch(
    *,
    batch_index: int,
    docs_processed: int,
    batch: list[RawChunk],
) -> EmbeddedBatch:
    settings = get_settings()
    started = time.perf_counter()
    vectors = get_embeddings(
        [chunk.text for chunk in batch],
        settings=settings,
        request_label=f"esco_live:batch_{batch_index}",
    )
    return EmbeddedBatch(
        batch_index=batch_index,
        docs_processed=docs_processed,
        chunks=batch,
        vectors=vectors,
        embedding_elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


def _iter_chunk_batches(
    docs: list[Any],
    *,
    batch_size: int,
    settings: Any,
    started_at: float,
) -> Iterator[tuple[int, int, list[RawChunk]]]:
    pending_chunks: deque[RawChunk] = deque()
    total_docs = len(docs)
    batch_index = 0

    for index, doc in enumerate(docs, start=1):
        chunks = chunk_text_by_tokens(
            doc.text,
            doc.metadata,
            chunk_size=settings.rag_chunk_size_tokens,
            overlap=settings.rag_chunk_overlap_tokens,
        )
        pending_chunks.extend(chunks)
        if index <= 5 or index % 1000 == 0:
            print(
                "[esco-live] progress "
                f"index={index}/{total_docs} "
                f"progress_pct={round((index / total_docs) * 100, 2) if total_docs else 0.0} "
                f"doc_type={doc.metadata.get('esco_doc_type')} "
                f"pending_chunks={len(pending_chunks)} "
                f"elapsed={_format_elapsed(started_at)}",
                flush=True,
            )
        while len(pending_chunks) >= batch_size:
            batch_index += 1
            yield batch_index, index, [pending_chunks.popleft() for _ in range(batch_size)]

    if pending_chunks:
        batch_index += 1
        yield batch_index, total_docs, list(pending_chunks)


def _parse_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _filter_docs_for_demo_occupations(
    docs: list[Any],
    occupation_keywords: list[str],
) -> list[Any]:
    if not occupation_keywords:
        return docs

    keep_occupation_ids: set[str] = set()
    for doc in docs:
        metadata = doc.metadata
        if metadata.get("esco_doc_type") != "occupation_summary":
            continue
        label = str(metadata.get("section_title") or metadata.get("occupation_label") or "").lower()
        if not label:
            continue
        if any(keyword in label for keyword in occupation_keywords):
            occupation_id = metadata.get("occupation_id")
            if occupation_id:
                keep_occupation_ids.add(str(occupation_id))

    if not keep_occupation_ids:
        return docs

    keep_skill_ids: set[str] = set()
    keep_isco_groups: set[str] = set()
    kept_docs: list[Any] = []

    for doc in docs:
        metadata = doc.metadata
        doc_type = metadata.get("esco_doc_type")
        occupation_id = str(metadata.get("occupation_id")) if metadata.get("occupation_id") else None

        if doc_type == "occupation_summary" and occupation_id in keep_occupation_ids:
            isco_group = metadata.get("isco_group")
            if isco_group:
                keep_isco_groups.add(str(isco_group))
            kept_docs.append(doc)
            continue

        if doc_type == "taxonomy_mapping" and occupation_id in keep_occupation_ids:
            isco_group = metadata.get("isco_group")
            if isco_group:
                keep_isco_groups.add(str(isco_group))
            kept_docs.append(doc)
            continue

        if doc_type == "relation_detail" and occupation_id in keep_occupation_ids:
            skill_id = metadata.get("skill_id")
            if skill_id:
                keep_skill_ids.add(str(skill_id))
            kept_docs.append(doc)
            continue

        if doc_type == "isco_group_summary" and str(metadata.get("isco_group")) in keep_isco_groups:
            kept_docs.append(doc)
            continue

        if doc_type == "skill_summary" and str(metadata.get("skill_id")) in keep_skill_ids:
            kept_docs.append(doc)

    return kept_docs


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    esco_root = Path(settings.data_raw_dir) / "esco"
    batch_size = max(1, int(args.embedding_batch_size))
    embedding_concurrency = max(1, int(args.embedding_concurrency))
    upsert_batch_size = max(batch_size, int(args.qdrant_upsert_batch_size))
    started_at = time.perf_counter()

    print(
        "[esco-live] start "
        f"embedding_batch_size={batch_size} "
        f"embedding_concurrency={embedding_concurrency} "
        f"qdrant_upsert_batch_size={upsert_batch_size} "
        f"embedding_timeout_seconds={settings.openai_embedding_timeout_seconds} "
        f"embedding_max_attempts={settings.openai_embedding_max_attempts}",
        flush=True,
    )

    ensure_collection()
    existing_esco_vectors = count_vectors(filters={"source": "esco"})
    delete_vectors_by_metadata({"source": "esco"})
    print(
        "[esco-live] purge "
        f"existing_esco_vectors={existing_esco_vectors} "
        "reason=avoid_silent_old_new_mix",
        flush=True,
    )

    docs, logical_files = _build_esco_documents(esco_root, settings)
    keywords = _parse_keywords(args.occupation_keywords)
    if keywords:
        pre_filter_count = len(docs)
        docs = _filter_docs_for_demo_occupations(docs, keywords)
        post_filter_counts = Counter(doc.metadata.get("esco_doc_type", "unknown") for doc in docs)
        print(
            "[esco-live] occupation_filter "
            f"keywords={keywords} "
            f"documents_before={pre_filter_count} "
            f"documents_after={len(docs)} "
            f"counts_by_doc_type={dict(post_filter_counts)}",
            flush=True,
        )
    if args.max_docs is not None:
        docs = docs[: max(0, args.max_docs)]
    counts_by_doc_type = Counter(doc.metadata.get("esco_doc_type", "unknown") for doc in docs)
    print(
        "[esco-live] generated "
        f"documents={len(docs)} "
        f"logical_files={logical_files} "
        f"counts_by_doc_type={dict(counts_by_doc_type)}",
        flush=True,
    )

    total_docs = len(docs)
    total_upserted = 0
    completed_embedding_batches = 0
    total_upsert_batches = 0
    latest_docs_processed = 0
    upsert_ids: list[str] = []
    upsert_vectors_buffer: list[list[float]] = []
    upsert_payloads: list[dict[str, Any]] = []

    def flush_pending_upserts(*, force: bool = False) -> None:
        nonlocal total_upserted, total_upsert_batches
        if not upsert_ids:
            return
        if not force and len(upsert_ids) < upsert_batch_size:
            return
        total_upsert_batches += 1
        print(
            "[esco-live] before qdrant upsert "
            f"upsert_batch_index={total_upsert_batches} "
            f"vectors={len(upsert_ids)} "
            f"docs_progress={latest_docs_processed}/{total_docs} "
            f"progress_pct={round((latest_docs_processed / total_docs) * 100, 2) if total_docs else 0.0} "
            f"elapsed={_format_elapsed(started_at)}",
            flush=True,
        )
        upsert_started = time.perf_counter()
        upsert_vectors(
            ids=upsert_ids,
            vectors=upsert_vectors_buffer,
            payloads=upsert_payloads,
        )
        upsert_elapsed_ms = round((time.perf_counter() - upsert_started) * 1000, 2)
        print(
            "[esco-live] after qdrant upsert "
            f"upsert_batch_index={total_upsert_batches} "
            f"vectors={len(upsert_ids)} "
            f"elapsed_ms={upsert_elapsed_ms} "
            f"elapsed={_format_elapsed(started_at)}",
            flush=True,
        )
        total_upserted += len(upsert_ids)
        upsert_ids.clear()
        upsert_vectors_buffer.clear()
        upsert_payloads.clear()

    def handle_completed(future: Future[EmbeddedBatch]) -> None:
        nonlocal completed_embedding_batches, latest_docs_processed
        result = future.result()
        completed_embedding_batches += 1
        latest_docs_processed = max(latest_docs_processed, result.docs_processed)
        upsert_ids.extend(chunk.chunk_id for chunk in result.chunks)
        upsert_vectors_buffer.extend(result.vectors)
        upsert_payloads.extend({**chunk.metadata, "text": chunk.text} for chunk in result.chunks)
        if completed_embedding_batches <= 3 or completed_embedding_batches % 100 == 0:
            print(
                "[esco-live] embedding_batch_complete "
                f"batch_index={result.batch_index} "
                f"batch_size={len(result.chunks)} "
                f"docs_progress={result.docs_processed}/{total_docs} "
                f"progress_pct={round((result.docs_processed / total_docs) * 100, 2) if total_docs else 0.0} "
                f"embedding_elapsed_ms={result.embedding_elapsed_ms} "
                f"completed_embedding_batches={completed_embedding_batches} "
                f"elapsed={_format_elapsed(started_at)}",
                flush=True,
            )
        flush_pending_upserts(force=False)

    inflight_limit = embedding_concurrency * 4
    with ThreadPoolExecutor(max_workers=embedding_concurrency) as executor:
        inflight: dict[Future[EmbeddedBatch], int] = {}
        for batch_index, docs_processed, batch in _iter_chunk_batches(
            docs,
            batch_size=batch_size,
            settings=settings,
            started_at=started_at,
        ):
            future = executor.submit(
                _embed_batch,
                batch_index=batch_index,
                docs_processed=docs_processed,
                batch=batch,
            )
            inflight[future] = batch_index
            latest_docs_processed = docs_processed
            if args.stop_after_first_flush:
                break
            while len(inflight) >= inflight_limit:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for completed in done:
                    inflight.pop(completed, None)
                    handle_completed(completed)

        while inflight:
            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for completed in done:
                inflight.pop(completed, None)
                handle_completed(completed)

    flush_pending_upserts(force=True)

    diagnostics = get_esco_vector_diagnostics(doc_types=ESCO_ENRICHED_DOC_TYPES)
    print(
        "[esco-live] verification "
        f"generated_documents={len(docs)} "
        f"vectors_upserted={total_upserted} "
        f"embedding_batches_completed={completed_embedding_batches} "
        f"qdrant_upsert_batches={total_upsert_batches} "
        f"elapsed={_format_elapsed(started_at)}",
        flush=True,
    )
    _print_esco_diagnostics(diagnostics)


if __name__ == "__main__":
    main()
