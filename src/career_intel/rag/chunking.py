"""Document chunking strategies for different source types."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import structlog
import tiktoken

logger = structlog.get_logger()

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150

RAG_CHUNK_SIZE_TOKENS = 400
RAG_CHUNK_OVERLAP_TOKENS = 80
_TIKTOKEN_ENCODING = "cl100k_base"
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Cache tokenizer instances; repeated initialization is expensive."""
    return tiktoken.get_encoding(encoding_name)


def _content_uuid(text: str, metadata: dict[str, Any] | None = None) -> str:
    """Generate a deterministic UUID string from text plus stable metadata."""
    stable: dict[str, Any] = {}
    if metadata:
        for key in (
            "source_id",
            "parent_doc_id",
            "document_title",
            "file_name",
            "section_title",
            "page_number",
            "entity_type",
            "occupation_id",
            "skill_id",
            "relation_type",
            "chunk_index",
        ):
            value = metadata.get(key)
            if value not in (None, ""):
                stable[key] = value
    fingerprint = json.dumps(
        {"text": text.strip(), "metadata": stable},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return str(uuid.uuid5(uuid.NAMESPACE_OID, fingerprint))


@dataclass
class RawChunk:
    """A chunk of text with associated metadata before embedding."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = _content_uuid(self.text, self.metadata)


def chunk_markdown(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[RawChunk]:
    """Split markdown text into heading-aware chunks."""
    sections = _split_on_headings(text)
    structured_sections = [
        {"title": section_title, "text": section_text}
        for section_title, section_text in sections
    ]
    return chunk_structured_sections(
        structured_sections,
        metadata,
        chunk_size=chunk_size,
        overlap=overlap,
    )


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

        text = "\n".join(text_parts).strip()
        if not text:
            continue

        row_meta = {**metadata, "chunk_index": idx}
        for key in ("occupation_code", "skill_id", "title"):
            value = row.get(key)
            if value:
                row_meta[key] = value

        chunks.append(RawChunk(text=text, metadata=row_meta))

    logger.debug("chunked_csv", total_chunks=len(chunks), source=metadata.get("source_id"))
    return chunks


def split_pdf_like_sections(text: str) -> list[tuple[str | None, str]]:
    """Heuristic section boundaries for report-style PDFs."""
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        detected = detect_section_heading(line)
        if detected:
            if current_lines:
                sections.append(
                    {
                        "title": current_title,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []
            current_title = detected
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "title": current_title,
                "text": "\n".join(current_lines).strip(),
            }
        )

    normalized = [
        (section.get("title"), section.get("text", ""))
        for section in sections
        if section.get("text", "").strip()
    ]
    return normalized or [(None, text)]


def detect_section_heading(line: str) -> str | None:
    """Return a likely section title if the line looks like a heading."""
    compact = re.sub(r"\s+", " ", line.strip())
    if not compact or len(compact) > 140:
        return None
    if re.fullmatch(r"\d+", compact):
        return None

    patterns = (
        r"^(Chapter|Part|Appendix)\s+[A-Z0-9IVX]+[:.\-]?\s+.+$",
        r"^(?:\d+(\.\d+){0,3})\s+.+$",
        r"^[A-Z][A-Za-z0-9/&(),\- ]{3,90}$",
        r"^[A-Z][A-Z0-9/&(),\- ]{3,90}$",
    )
    if any(re.match(pattern, compact) for pattern in patterns):
        return compact
    return None


def chunk_text_by_tokens(
    text: str,
    metadata: dict[str, Any],
    *,
    chunk_size: int = RAG_CHUNK_SIZE_TOKENS,
    overlap: int = RAG_CHUNK_OVERLAP_TOKENS,
    encoding_name: str = _TIKTOKEN_ENCODING,
) -> list[RawChunk]:
    """Split plain text into sentence-aware overlapping chunks."""
    enc = _get_encoding(encoding_name)
    units = _split_into_units(text)
    if not units:
        return []

    unit_tokens = [len(enc.encode(unit)) for unit in units]
    chunks: list[RawChunk] = []
    current_units: list[str] = []
    current_tokens = 0
    idx = 0
    i = 0

    while i < len(units):
        unit = units[i]
        token_count = unit_tokens[i]

        if token_count > chunk_size:
            if current_units:
                idx = _append_chunk(chunks, current_units, metadata, idx)
                current_units, current_tokens = _overlap_units(current_units, enc, overlap)
                continue

            oversize_chunks = _split_oversize_unit(
                unit,
                metadata,
                chunk_size=chunk_size,
                overlap=overlap,
                encoding_name=encoding_name,
                start_index=idx,
            )
            chunks.extend(oversize_chunks)
            idx += len(oversize_chunks)
            i += 1
            current_units = []
            current_tokens = 0
            continue

        if current_units and current_tokens + token_count > chunk_size:
            idx = _append_chunk(chunks, current_units, metadata, idx)
            current_units, current_tokens = _overlap_units(current_units, enc, overlap)
            continue

        current_units.append(unit)
        current_tokens += token_count
        i += 1

    if current_units:
        _append_chunk(chunks, current_units, metadata, idx)

    logger.debug("chunked_by_tokens", total_chunks=len(chunks), source=metadata.get("source_id"))
    return chunks


def chunk_structured_sections(
    sections: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    chunk_size: int = RAG_CHUNK_SIZE_TOKENS,
    overlap: int = RAG_CHUNK_OVERLAP_TOKENS,
) -> list[RawChunk]:
    """Chunk text while preserving higher-level section metadata."""
    chunks: list[RawChunk] = []
    next_index = 0

    for section in sections:
        body = str(section.get("text", "")).strip()
        if not body:
            continue

        section_meta = {**metadata}
        section_title = section.get("title")
        page_number = section.get("page_number")
        if section_title:
            section_meta["section"] = str(section_title)
            section_meta["section_title"] = str(section_title)
        if page_number is not None:
            section_meta["page_number"] = int(page_number)
            section_meta["page_or_loc"] = f"page {page_number}"

        sub_chunks = chunk_text_by_tokens(
            body,
            section_meta,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for sub_chunk in sub_chunks:
            sub_chunk.metadata["chunk_index"] = next_index
            sub_chunk.chunk_id = _content_uuid(sub_chunk.text, sub_chunk.metadata)
            chunks.append(sub_chunk)
            next_index += 1

    logger.debug("chunked_structured_sections", total_chunks=len(chunks), source=metadata.get("source_id"))
    return chunks


def chunk_pdf_text(
    text: str,
    metadata: dict[str, Any],
    *,
    chunk_size: int = RAG_CHUNK_SIZE_TOKENS,
    overlap: int = RAG_CHUNK_OVERLAP_TOKENS,
) -> list[RawChunk]:
    """Chunk extracted PDF text using detected sections and semantic windows."""
    sections = [
        {"title": sec_title, "text": sec_body}
        for sec_title, sec_body in split_pdf_like_sections(text)
    ]
    return chunk_structured_sections(
        sections,
        metadata,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def _split_on_headings(text: str) -> list[tuple[str | None, str]]:
    """Split text on markdown headings, returning (heading, body) pairs."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(2).strip(), text[match.end() : end]))

    return sections


def _split_into_units(text: str) -> list[str]:
    """Split text into paragraphs/sentences while preserving semantic boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized:
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
        if not sentences:
            units.append(normalized)
            continue
        units.extend(sentences)
    return units


def _split_oversize_unit(
    unit: str,
    metadata: dict[str, Any],
    *,
    chunk_size: int,
    overlap: int,
    encoding_name: str,
    start_index: int,
) -> list[RawChunk]:
    """Fallback token-window split for a single sentence/paragraph that is too large."""
    enc = _get_encoding(encoding_name)
    tokens = enc.encode(unit)
    stride = max(1, chunk_size - overlap)
    chunks: list[RawChunk] = []
    idx = start_index
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + chunk_size]
        piece = _trim_to_sentence_boundary(enc.decode(window))
        if not piece:
            continue
        chunk_meta = {**metadata, "chunk_index": idx}
        chunks.append(RawChunk(text=piece, metadata=chunk_meta))
        idx += 1
    return chunks


def _append_chunk(
    chunks: list[RawChunk],
    units: list[str],
    metadata: dict[str, Any],
    idx: int,
) -> int:
    text = " ".join(units).strip()
    if not text:
        return idx
    chunk_meta = {**metadata, "chunk_index": idx}
    chunks.append(RawChunk(text=text, metadata=chunk_meta))
    return idx + 1


def _overlap_units(
    units: list[str],
    enc: tiktoken.Encoding,
    overlap: int,
) -> tuple[list[str], int]:
    if not units or overlap <= 0:
        return [], 0

    kept: list[str] = []
    token_total = 0
    for unit in reversed(units):
        unit_tokens = len(enc.encode(unit))
        if not kept and unit_tokens > overlap:
            break
        if kept and token_total + unit_tokens > overlap:
            break
        kept.insert(0, unit)
        token_total += unit_tokens
    if len(kept) == len(units):
        # Ensure the caller makes forward progress instead of rebuilding the
        # exact same chunk forever when the retained suffix stays under the
        # overlap budget.
        kept = kept[1:]
        token_total = sum(len(enc.encode(unit)) for unit in kept)
    return kept, token_total


def _trim_to_sentence_boundary(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if text[-1] in ".!?":
        return text
    match = re.search(r"^(.+[.!?])\s+\S+$", text)
    if match:
        return match.group(1).strip()
    return text


def _approx_tokens(text: str) -> int:
    return len(text) // 4
