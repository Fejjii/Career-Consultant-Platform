"""Document chunking strategies for different source types."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

DEFAULT_CHUNK_SIZE = 1000  # tokens (approximate via chars / 4)
DEFAULT_OVERLAP = 150


def _content_uuid(text: str) -> str:
    """Generate a deterministic UUID string from text content.

    Qdrant point IDs must be either an integer or a UUID-format string.
    We use UUID5 (SHA-1 based, deterministic) so the same chunk text
    always produces the same ID, enabling idempotent re-ingestion.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, text))


@dataclass
class RawChunk:
    """A chunk of text with associated metadata before embedding."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = _content_uuid(self.text)


def chunk_markdown(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[RawChunk]:
    """Split markdown text into heading-aware chunks.

    Attempts to split on headings first, then falls back to
    character-based splitting with overlap.
    """
    sections = _split_on_headings(text)
    chunks: list[RawChunk] = []

    for section_title, section_text in sections:
        section_meta = {**metadata}
        if section_title:
            section_meta["section"] = section_title

        clean_section_text = section_text.strip()
        if not clean_section_text:
            continue

        if _approx_tokens(section_text) <= chunk_size:
            chunks.append(RawChunk(text=clean_section_text, metadata=section_meta))
        else:
            sub_chunks = _split_by_size(section_text, chunk_size, overlap)
            for idx, sc in enumerate(sub_chunks):
                clean_sub_chunk = sc.strip()
                if not clean_sub_chunk:
                    continue
                chunk_meta = {**section_meta, "chunk_index": idx}
                chunks.append(RawChunk(text=clean_sub_chunk, metadata=chunk_meta))

    for i, c in enumerate(chunks):
        c.metadata.setdefault("chunk_index", i)

    logger.info("chunked_document", total_chunks=len(chunks), source=metadata.get("source_id"))
    return chunks


def chunk_csv_rows(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    text_columns: list[str] | None = None,
) -> list[RawChunk]:
    """Convert CSV rows into individual chunks."""
    chunks: list[RawChunk] = []
    for idx, row in enumerate(rows):
        if text_columns:
            text_parts = [str(row.get(col, "")) for col in text_columns if row.get(col)]
        else:
            text_parts = [f"{k}: {v}" for k, v in row.items() if v]

        text = "\n".join(text_parts)
        row_meta = {**metadata, "chunk_index": idx}
        for key in ("occupation_code", "skill_id", "title"):
            if key in row:
                row_meta[key] = row[key]

        chunks.append(RawChunk(text=text, metadata=row_meta))

    logger.info("chunked_csv", total_chunks=len(chunks), source=metadata.get("source_id"))
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _split_on_headings(text: str) -> list[tuple[str | None, str]]:
    """Split text on markdown headings, returning (heading, body) pairs."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[m.end() : end]))

    return sections


def _split_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-based split with overlap (approximating tokens as chars/4)."""
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end])
        start = end - char_overlap
    return chunks


def _approx_tokens(text: str) -> int:
    return len(text) // 4
