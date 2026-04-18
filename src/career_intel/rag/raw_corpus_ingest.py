"""Ingest ``data/raw`` (WEF PDFs, ESCO CSV/JSON) into Qdrant with enriched metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from pypdf import PdfReader
from sqlalchemy import delete, func, select

from career_intel.config import Settings, get_settings
from career_intel.rag.chunking import (
    RawChunk,
    chunk_structured_sections,
    chunk_text_by_tokens,
    detect_section_heading,
)
from career_intel.rag.embeddings import get_embeddings
from career_intel.schemas.api import IngestResponse
from career_intel.storage.db import Document, IngestionRun, get_session_factory, init_db
from career_intel.storage.qdrant_store import (
    delete_vectors_by_metadata,
    ensure_collection,
    get_esco_vector_diagnostics,
    count_vectors,
    upsert_vectors,
)

logger = structlog.get_logger()

CorpusSource = Literal["wef", "esco"]
RAW_EXTENSIONS = frozenset({".pdf", ".csv", ".json"})
ESCO_AGGREGATED_FILES = frozenset(
    {
        "occupations_en.csv",
        "skills_en.csv",
        "occupationSkillRelations_en.csv",
    }
)
ESCO_ENRICHED_DOC_TYPES = (
    "occupation_summary",
    "skill_summary",
    "relation_detail",
    "taxonomy_mapping",
    "isco_group_summary",
)
ESCO_ONLY_BACKFILL_MODES = frozenset({"esco_backfill", "esco_only_backfill"})


@dataclass
class NormalizedDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    checksum: str
    source: CorpusSource
    logical_file: str


@dataclass
class IngestionStats:
    files_loaded: int = 0
    files_failed: int = 0
    documents_processed: int = 0
    chunks_created: int = 0
    chunks_per_file: dict[str, int] = field(default_factory=dict)
    chunks_per_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _resolve_raw_root(base_dir: Path | None, settings: Settings) -> Path:
    path = Path(settings.data_raw_dir) if base_dir is None else base_dir
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def discover_raw_files(raw_root: Path) -> list[Path]:
    """Recursively list supported files under ``raw_root``."""
    found: list[Path] = []
    if not raw_root.is_dir():
        logger.warning("raw_corpus_dir_missing", path=str(raw_root))
        return found
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS and not path.name.startswith("."):
            found.append(path)
    return found


def infer_corpus_source(path: Path, raw_root: Path) -> CorpusSource | None:
    try:
        rel = path.relative_to(raw_root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    top = rel.parts[0].lower()
    if top in {"wef", "esco"}:
        return top
    return None


def wef_year_from_filename(name: str) -> int | None:
    years = [int(year) for year in re.findall(r"(20\d{2})", name)]
    return max(years) if years else None


def topic_from_relative_path(path: Path, raw_root: Path) -> str:
    try:
        rel = path.relative_to(raw_root)
        return "/".join((*rel.parts[:-1], path.stem))
    except ValueError:
        return path.stem


def make_deterministic_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def source_priority_for(source: CorpusSource, entity_type: str | None) -> int:
    if source == "wef":
        return 100
    if entity_type == "occupation":
        return 95
    if entity_type == "skill":
        return 90
    if entity_type == "relation":
        return 85
    return 80


def build_base_payload(
    *,
    doc_id: str,
    corpus_source: CorpusSource,
    file_name: str,
    document_title: str,
    topic: str,
    year: int | None,
    uri: str | None,
    entity_type: str | None,
) -> dict[str, Any]:
    priority = source_priority_for(corpus_source, entity_type)
    return {
        "source_id": doc_id,
        "parent_doc_id": doc_id,
        "source": corpus_source,
        "source_name": corpus_source,
        "source_type": corpus_source,
        "file_name": file_name,
        "title": document_title,
        "document_title": document_title,
        "topic": topic,
        "uri": uri,
        "publish_year": year,
        "section": None,
        "section_title": None,
        "page_number": None,
        "page_or_loc": None,
        "entity_type": entity_type,
        "source_priority": priority,
        "language": "en",
    }


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _is_esco_only_backfill_mode(mode: str) -> bool:
    return mode in ESCO_ONLY_BACKFILL_MODES


def _debug_print(settings: Settings, message: str) -> None:
    if settings.rag_ingest_debug or settings.environment == "development":
        print(message)


def _sample(text: str, limit: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:limit]


def esco_row_to_sentence(row: dict[str, str], file_stem: str) -> str:
    """Turn one ESCO CSV row into a readable sentence for embedding."""
    stem_l = file_stem.lower().replace("_", "")

    if "occupationskillrelation" in stem_l:
        occ = _clean_text(row.get("occupationLabel", ""))
        skill = _clean_text(row.get("skillLabel", ""))
        rel_t = _clean_text(row.get("relationType", ""))
        st = _clean_text(row.get("skillType", ""))
        skill_def = _clean_text(row.get("skillDefinition", ""))
        sentence = f"Occupation: {occ}. Relation: {rel_t} ({st}). Skill: {skill}."
        if skill_def:
            sentence += f" Definition: {skill_def}"
        return sentence

    if stem_l.startswith("occupations") and "skill" not in stem_l:
        occ = _clean_text(row.get("preferredLabel", ""))
        code = _clean_text(row.get("code", ""))
        desc = _clean_text(row.get("description") or row.get("definition", ""))
        isco = _clean_text(row.get("iscoGroup", ""))
        parts = [f"Occupation: {occ}."]
        if code:
            parts.append(f"Code: {code}.")
        if isco:
            parts.append(f"ISCO group: {isco}.")
        if desc:
            parts.append(f"Description: {desc}")
        return " ".join(parts)

    if stem_l.startswith("skills") and "hierarchy" not in stem_l:
        label = _clean_text(row.get("preferredLabel", ""))
        st = _clean_text(row.get("skillType", ""))
        definition = _clean_text(row.get("definition") or row.get("description", ""))
        parts = [f"Skill: {label}."]
        if st:
            parts.append(f"Type: {st}.")
        if definition:
            parts.append(f"Definition: {definition}")
        return " ".join(parts)

    if "iscogroup" in stem_l:
        label = _clean_text(row.get("preferredLabel", ""))
        code = _clean_text(row.get("code", ""))
        desc = _clean_text(row.get("description", ""))
        parts = [f"ISCO group: {label}."]
        if code:
            parts.append(f"Code: {code}.")
        if desc:
            parts.append(f"Description: {desc}")
        return " ".join(parts)

    if "skillshierarchy" in stem_l:
        terms: list[str] = []
        for key in sorted(row):
            if "preferred term" not in key.lower():
                continue
            value = _clean_text(row.get(key, ""))
            if value:
                terms.append(value)
        desc = _clean_text(row.get("Description", "") or row.get("Scope note", ""))
        sentence = f"Skills hierarchy: {' → '.join(dict.fromkeys(terms))}."
        if desc:
            sentence += f" Notes: {desc}"
        return sentence

    pairs = [f"{key}: {_clean_text(value)}" for key, value in row.items() if _clean_text(value)]
    return "; ".join(pairs[:24])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [{k: v or "" for k, v in row.items()} for row in csv.DictReader(handle)]


def load_json_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("records", "data", "items"):
            inner = raw.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
        return [raw]
    return []


def extract_pdf_pages(path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    raw_pages: list[dict[str, Any]] = []
    page_entries: list[tuple[str | None, str | None]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        raw_pages.append({"page_number": page_number, "raw_text": text})
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_entries.append((lines[0] if lines else None, lines[-1] if lines else None))

    header_counts = Counter(entry[0] for entry in page_entries if entry[0])
    footer_counts = Counter(entry[1] for entry in page_entries if entry[1])
    repeated_headers = {
        line
        for line, count in header_counts.items()
        if count >= 2 and len(line) <= 120 and not re.fullmatch(r"\d+", line)
    }
    repeated_footers = {
        line
        for line, count in footer_counts.items()
        if count >= 2 and len(line) <= 120 and not re.fullmatch(r"\d+", line)
    }

    cleaned_pages: list[dict[str, Any]] = []
    for page in raw_pages:
        cleaned = clean_pdf_page_text(
            page["raw_text"],
            repeated_headers=repeated_headers,
            repeated_footers=repeated_footers,
        )
        if cleaned:
            cleaned_pages.append({"page_number": page["page_number"], "text": cleaned})
    return cleaned_pages


def clean_pdf_page_text(
    text: str,
    *,
    repeated_headers: set[str],
    repeated_footers: set[str],
) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cleaned: list[str] = []
    for line in lines:
        compact = re.sub(r"\s+", " ", line).strip()
        if not compact:
            continue
        if compact in repeated_headers or compact in repeated_footers:
            continue
        if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", compact, flags=re.IGNORECASE):
            continue
        cleaned.append(compact)

    merged: list[str] = []
    idx = 0
    while idx < len(cleaned):
        current = cleaned[idx]
        next_line = cleaned[idx + 1] if idx + 1 < len(cleaned) else None
        if next_line and _should_merge_pdf_lines(current, next_line):
            merged.append(f"{current.rstrip('-')} {next_line}".replace("  ", " "))
            idx += 2
            continue
        merged.append(current)
        idx += 1

    return "\n".join(merged).strip()


def _should_merge_pdf_lines(current: str, next_line: str) -> bool:
    if detect_section_heading(next_line):
        return False
    if current.endswith((".", "!", "?", ":")):
        return False
    if re.match(r"^[-•*]", next_line):
        return False
    if current.endswith("-"):
        return True
    return bool(next_line and next_line[0].islower())


def split_pdf_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    section_start_page: int | None = None

    for page in pages:
        page_number = int(page["page_number"])
        for line in str(page["text"]).splitlines():
            compact = line.strip()
            if not compact:
                continue
            heading = detect_section_heading(compact)
            if heading:
                if current_lines:
                    sections.append(
                        {
                            "title": current_title,
                            "page_number": section_start_page,
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                    current_lines = []
                current_title = heading
                section_start_page = page_number
                continue

            if section_start_page is None:
                section_start_page = page_number
            current_lines.append(compact)

    if current_lines:
        sections.append(
            {
                "title": current_title,
                "page_number": section_start_page,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def _chunk_pdf_document(
    file_path: Path,
    raw_root: Path,
    settings: Settings,
) -> tuple[list[RawChunk], str]:
    pages = extract_pdf_pages(file_path)
    if not pages:
        return [], ""

    cleaned_text = "\n\n".join(page["text"] for page in pages)
    year = wef_year_from_filename(file_path.name)
    document_title = file_path.stem.replace("_", " ").strip()
    doc_id = make_deterministic_id(f"wef::{topic_from_relative_path(file_path, raw_root)}")
    base_meta = build_base_payload(
        doc_id=doc_id,
        corpus_source="wef",
        file_name=file_path.name,
        document_title=document_title,
        topic=topic_from_relative_path(file_path, raw_root),
        year=year,
        uri=str(file_path.resolve()),
        entity_type="report",
    )
    checksum = _checksum(cleaned_text)

    sections = split_pdf_sections(pages)
    if not sections:
        sections = [{"title": None, "page_number": 1, "text": cleaned_text}]

    chunks = chunk_structured_sections(
        sections,
        base_meta,
        chunk_size=settings.rag_chunk_size_tokens,
        overlap=settings.rag_chunk_overlap_tokens,
    )

    _debug_print(settings, f"[ingest][pdf] file={file_path.name} sample_text={_sample(cleaned_text)!r}")
    if chunks:
        _debug_print(
            settings,
            f"[ingest][pdf] sample_chunks={[_sample(chunk.text, 180) for chunk in chunks[:2]]}",
        )
    return chunks, checksum


def _truncate_joined(values: list[str], limit: int) -> str:
    unique = list(dict.fromkeys(value for value in values if value))
    display = unique[:limit]
    return ", ".join(display)


def _batched(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    if size <= 0:
        return [rows]
    return [rows[idx : idx + size] for idx in range(0, len(rows), size)]


def _relation_sort_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        _clean_text(row.get("skillLabel", "")).lower(),
        _clean_text(row.get("relationType", "")).lower(),
        _clean_text(row.get("skillType", "")).lower(),
    )


def _format_relation_items(rows: list[dict[str, str]], *, limit: int | None = None) -> str:
    items: list[str] = []
    selected = rows[:limit] if limit is not None else rows
    for row in selected:
        skill_label = _clean_text(row.get("skillLabel", ""))
        if not skill_label:
            continue
        relation_type = _clean_text(row.get("relationType", "")) or "unspecified"
        skill_type = _clean_text(row.get("skillType", "")) or "unspecified"
        skill_uri = _clean_text(row.get("skillUri", ""))
        suffix = f"; skill_id {skill_uri}" if skill_uri else ""
        items.append(f"{skill_label} [{relation_type}; {skill_type}{suffix}]")
    return ". ".join(items)


def _format_occupation_links(rows: list[dict[str, str]], *, limit: int | None = None) -> str:
    items: list[str] = []
    selected = rows[:limit] if limit is not None else rows
    for row in selected:
        occupation_label = _clean_text(row.get("occupationLabel", ""))
        if not occupation_label:
            continue
        relation_type = _clean_text(row.get("relationType", "")) or "unspecified"
        skill_type = _clean_text(row.get("skillType", "")) or "unspecified"
        occupation_uri = _clean_text(row.get("occupationUri", ""))
        suffix = f"; occupation_id {occupation_uri}" if occupation_uri else ""
        items.append(f"{occupation_label} [{relation_type}; {skill_type}{suffix}]")
    return ". ".join(items)


def _build_esco_documents(esco_root: Path, settings: Settings) -> tuple[list[NormalizedDocument], list[str]]:
    occupations_path = esco_root / "occupations_en.csv"
    skills_path = esco_root / "skills_en.csv"
    relations_path = esco_root / "occupationSkillRelations_en.csv"
    isco_path = esco_root / "ISCOGroups_en.csv"
    hierarchy_path = esco_root / "skillsHierarchy_en.csv"

    required = [occupations_path, skills_path, relations_path]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ESCO files for enrichment: {', '.join(missing)}")

    occupations_rows = read_csv_rows(occupations_path)
    skills_rows = read_csv_rows(skills_path)
    relations_rows = read_csv_rows(relations_path)
    isco_rows = read_csv_rows(isco_path) if isco_path.exists() else []
    hierarchy_rows = read_csv_rows(hierarchy_path) if hierarchy_path.exists() else []

    isco_map = {row.get("code", ""): row for row in isco_rows if row.get("code")}
    hierarchy_map: dict[str, list[str]] = defaultdict(list)
    for row in hierarchy_rows:
        skill_uri = row.get("Level 3 URI") or row.get("Level 2 URI") or row.get("Level 1 URI") or row.get("Level 0 URI")
        path_terms = [
            _clean_text(row.get(key, ""))
            for key in (
                "Level 0 preferred term",
                "Level 1 preferred term",
                "Level 2 preferred term",
                "Level 3 preferred term",
            )
            if _clean_text(row.get(key, ""))
        ]
        if skill_uri and path_terms:
            hierarchy_map[skill_uri].append(" → ".join(dict.fromkeys(path_terms)))

    relations_by_occupation: dict[str, list[dict[str, str]]] = defaultdict(list)
    relations_by_skill: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in relations_rows:
        occupation_uri = row.get("occupationUri", "")
        skill_uri = row.get("skillUri", "")
        if occupation_uri:
            relations_by_occupation[occupation_uri].append(row)
        if skill_uri:
            relations_by_skill[skill_uri].append(row)

    included_occupation_ids = set(relations_by_occupation)
    included_skill_ids = set(relations_by_skill)
    occupations = {
        row.get("conceptUri", ""): row
        for row in occupations_rows
        if row.get("conceptUri") in included_occupation_ids
    }
    skills = {
        row.get("conceptUri", ""): row
        for row in skills_rows
        if row.get("conceptUri") in included_skill_ids
    }
    occupations_by_isco: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in occupations.values():
        isco_code = _clean_text(row.get("iscoGroup", ""))
        if isco_code:
            occupations_by_isco[isco_code].append(row)
    included_isco_codes = set(occupations_by_isco)

    docs: list[NormalizedDocument] = []
    logical_files = [
        occupations_path.name,
        skills_path.name,
        relations_path.name,
    ]
    if isco_rows:
        logical_files.append(isco_path.name)
    if hierarchy_rows:
        logical_files.append(hierarchy_path.name)

    for occupation_uri, occupation_row in occupations.items():
        label = _clean_text(occupation_row.get("preferredLabel", ""))
        if not label:
            continue
        related = sorted(relations_by_occupation.get(occupation_uri, []), key=_relation_sort_key)
        essential_relations = [row for row in related if _clean_text(row.get("relationType", "")).lower() == "essential"]
        optional_relations = [row for row in related if _clean_text(row.get("relationType", "")).lower() != "essential"]
        essential_skills = [_clean_text(row.get("skillLabel", "")) for row in essential_relations]
        optional_skills = [_clean_text(row.get("skillLabel", "")) for row in optional_relations]
        essential_knowledge = [
            _clean_text(row.get("skillLabel", ""))
            for row in essential_relations
            if _clean_text(row.get("skillType", "")).lower() == "knowledge"
        ]
        optional_knowledge = [
            _clean_text(row.get("skillLabel", ""))
            for row in optional_relations
            if _clean_text(row.get("skillType", "")).lower() == "knowledge"
        ]
        isco_code = _clean_text(occupation_row.get("iscoGroup", ""))
        isco_row = isco_map.get(isco_code, {})
        isco_label = _clean_text(isco_row.get("preferredLabel", ""))
        isco_description = _clean_text(isco_row.get("description", ""))
        description = _clean_text(occupation_row.get("description") or occupation_row.get("definition", ""))
        text = (
            f"ESCO occupation summary for {label}. "
            f"Occupation ID: {occupation_uri}. "
            f"Occupation code: {_clean_text(occupation_row.get('code', '')) or 'unknown'}. "
            f"ISCO group: {isco_code or 'unknown'}{f' ({isco_label})' if isco_label else ''}. "
            f"Occupation description: {description or 'No description provided.'} "
            f"Essential linked skills and knowledge: {_truncate_joined(essential_skills, 50) or 'None listed'}. "
            f"Optional linked skills and knowledge: {_truncate_joined(optional_skills, 40) or 'None listed'}. "
            f"Key essential knowledge items: {_truncate_joined(essential_knowledge, 20) or 'None listed'}. "
            f"Key optional knowledge items: {_truncate_joined(optional_knowledge, 20) or 'None listed'}. "
            f"ESCO relation evidence by skill label: {_format_relation_items(related, limit=24) or 'No relations listed.'}."
        )
        doc_id = make_deterministic_id(f"esco::occupation::{occupation_uri}")
        metadata = build_base_payload(
            doc_id=doc_id,
            corpus_source="esco",
            file_name=occupations_path.name,
            document_title=f"ESCO {occupations_path.name}",
            topic="esco/occupations",
            year=None,
            uri=occupation_uri,
            entity_type="occupation",
        )
        metadata.update(
            {
                "section": label,
                "section_title": label,
                "page_or_loc": label,
                "occupation_id": occupation_uri,
                "occupation_code": _clean_text(occupation_row.get("code", "")) or None,
                "isco_group": isco_code or None,
                "isco_group_label": isco_label or None,
                "esco_doc_type": "occupation_summary",
            }
        )
        docs.append(
            NormalizedDocument(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                checksum=_checksum(text),
                source="esco",
                logical_file=occupations_path.name,
            )
        )

        if isco_code:
            mapping_text = (
                f"ESCO to ISCO occupation mapping. "
                f"Occupation: {label}. "
                f"Occupation ID: {occupation_uri}. "
                f"ESCO occupation code: {_clean_text(occupation_row.get('code', '')) or 'unknown'}. "
                f"Maps to ISCO group: {isco_code}{f' ({isco_label})' if isco_label else ''}. "
                f"ISCO description: {isco_description or 'No ISCO description provided.'} "
                f"Occupation summary: {description or 'No occupation description provided.'}"
            )
            mapping_doc_id = make_deterministic_id(f"esco::occupation-isco::{occupation_uri}::{isco_code}")
            mapping_metadata = build_base_payload(
                doc_id=mapping_doc_id,
                corpus_source="esco",
                file_name=occupations_path.name,
                document_title="ESCO occupation to ISCO mapping",
                topic="esco/occupation-isco-mapping",
                year=None,
                uri=occupation_uri,
                entity_type="taxonomy",
            )
            mapping_metadata.update(
                {
                    "section": label,
                    "section_title": label,
                    "page_or_loc": label,
                    "occupation_id": occupation_uri,
                    "occupation_code": _clean_text(occupation_row.get("code", "")) or None,
                    "isco_group": isco_code,
                    "isco_group_label": isco_label or None,
                    "esco_doc_type": "taxonomy_mapping",
                }
            )
            docs.append(
                NormalizedDocument(
                    doc_id=mapping_doc_id,
                    text=mapping_text,
                    metadata=mapping_metadata,
                    checksum=_checksum(mapping_text),
                    source="esco",
                    logical_file=occupations_path.name,
                )
            )

        unique_relations: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for relation_row in related:
            skill_uri = _clean_text(relation_row.get("skillUri", ""))
            skill_label = _clean_text(relation_row.get("skillLabel", ""))
            relation_type = _clean_text(relation_row.get("relationType", "")).lower() or "unspecified"
            skill_type = _clean_text(relation_row.get("skillType", "")).lower() or "unspecified"
            relation_key = (occupation_uri, skill_uri, relation_type, skill_type)
            if relation_key not in unique_relations:
                unique_relations[relation_key] = relation_row

        for relation_row in sorted(
            unique_relations.values(),
            key=lambda row: (
                _clean_text(row.get("skillLabel", "")).lower(),
                _clean_text(row.get("relationType", "")).lower(),
                _clean_text(row.get("skillType", "")).lower(),
            ),
        ):
            skill_uri = _clean_text(relation_row.get("skillUri", ""))
            skill_label = _clean_text(relation_row.get("skillLabel", ""))
            relation_type = _clean_text(relation_row.get("relationType", "")).lower() or "unspecified"
            skill_type = _clean_text(relation_row.get("skillType", "")).lower() or "unspecified"
            skill_row = skills.get(skill_uri, {})
            skill_definition = _clean_text(
                relation_row.get("skillDefinition", "")
                or skill_row.get("definition", "")
                or skill_row.get("description", "")
            )
            relation_text = (
                f"ESCO occupation-skill relation. "
                f"Occupation: {label} (occupation_id: {occupation_uri}). "
                f"Skill: {skill_label or 'unknown'} (skill_id: {skill_uri or 'unknown'}). "
                f"Relation type: {relation_type}. "
                f"Skill type: {skill_type}. "
                f"ISCO group: {isco_code or 'unknown'}{f' ({isco_label})' if isco_label else ''}. "
                f"Skill definition: {skill_definition or 'No definition provided.'}"
            )
            relation_doc_id = make_deterministic_id(
                f"esco::relation::{occupation_uri}::{skill_uri}::{relation_type}::{skill_type}"
            )
            relation_metadata = build_base_payload(
                doc_id=relation_doc_id,
                corpus_source="esco",
                file_name=relations_path.name,
                document_title="ESCO occupation skill relation",
                topic="esco/occupation-skill-relations",
                year=None,
                uri=occupation_uri,
                entity_type="relation",
            )
            relation_metadata.update(
                {
                    "section": label,
                    "section_title": label,
                    "page_or_loc": label,
                    "occupation_id": occupation_uri,
                    "occupation_label": label,
                    "occupation_code": _clean_text(occupation_row.get("code", "")) or None,
                    "skill_id": skill_uri or None,
                    "skill_label": skill_label or None,
                    "relation_type": relation_type,
                    "skill_type": skill_type if skill_type != "unspecified" else None,
                    "isco_group": isco_code or None,
                    "isco_group_label": isco_label or None,
                    "esco_doc_type": "relation_detail",
                }
            )
            docs.append(
                NormalizedDocument(
                    doc_id=relation_doc_id,
                    text=relation_text,
                    metadata=relation_metadata,
                    checksum=_checksum(relation_text),
                    source="esco",
                    logical_file=relations_path.name,
                )
            )

    for skill_uri, skill_row in skills.items():
        label = _clean_text(skill_row.get("preferredLabel", ""))
        if not label:
            continue
        related = sorted(relations_by_skill.get(skill_uri, []), key=lambda row: _clean_text(row.get("occupationLabel", "")).lower())
        essential_for = [
            _clean_text(row.get("occupationLabel", ""))
            for row in related
            if _clean_text(row.get("relationType", "")).lower() == "essential"
        ]
        optional_for = [
            _clean_text(row.get("occupationLabel", ""))
            for row in related
            if _clean_text(row.get("relationType", "")).lower() != "essential"
        ]
        hierarchy_paths = hierarchy_map.get(skill_uri, [])
        definition = _clean_text(skill_row.get("definition") or skill_row.get("description", ""))
        skill_type = _clean_text(skill_row.get("skillType", ""))
        text = (
            f"ESCO skill summary for {label}. "
            f"Skill ID: {skill_uri}. "
            f"Skill type: {skill_type or 'unspecified'}. "
            f"Definition: {definition or 'No definition provided.'} "
            f"Essential for occupations: {_truncate_joined(essential_for, 35) or 'None listed'}. "
            f"Optional for occupations: {_truncate_joined(optional_for, 30) or 'None listed'}. "
            f"Linked occupation relations: {_format_occupation_links(related, limit=24) or 'No linked occupations listed.'}. "
            f"Hierarchy: {_truncate_joined(hierarchy_paths, 5) or 'Not available'}."
        )
        doc_id = make_deterministic_id(f"esco::skill::{skill_uri}")
        metadata = build_base_payload(
            doc_id=doc_id,
            corpus_source="esco",
            file_name=skills_path.name,
            document_title=f"ESCO {skills_path.name}",
            topic="esco/skills",
            year=None,
            uri=skill_uri,
            entity_type="skill",
        )
        metadata.update(
            {
                "section": label,
                "section_title": label,
                "page_or_loc": label,
                "skill_id": skill_uri,
                "skill_type": skill_type or None,
                "esco_doc_type": "skill_summary",
            }
        )
        docs.append(
            NormalizedDocument(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                checksum=_checksum(text),
                source="esco",
                logical_file=skills_path.name,
            )
        )

    for isco_code in sorted(included_isco_codes):
        isco_row = isco_map.get(isco_code, {})
        label = _clean_text(isco_row.get("preferredLabel", ""))
        related_occupations = occupations_by_isco.get(isco_code, [])
        occupation_labels = [
            _clean_text(row.get("preferredLabel", ""))
            for row in sorted(related_occupations, key=lambda row: _clean_text(row.get("preferredLabel", "")).lower())
            if _clean_text(row.get("preferredLabel", ""))
        ]
        text = (
            f"ISCO group summary for {label or isco_code}. "
            f"ISCO group code: {isco_code}. "
            f"ISCO group URI: {_clean_text(isco_row.get('conceptUri', '')) or 'unknown'}. "
            f"ISCO description: {_clean_text(isco_row.get('description', '')) or 'No description provided.'} "
            f"Linked ESCO occupations: {_truncate_joined(occupation_labels, 40) or 'None linked in the current ESCO snapshot'}. "
            f"Linked ESCO occupation count: {len(occupation_labels)}."
        )
        doc_id = make_deterministic_id(f"esco::isco-group::{isco_code}")
        metadata = build_base_payload(
            doc_id=doc_id,
            corpus_source="esco",
            file_name=isco_path.name if isco_rows else occupations_path.name,
            document_title="ESCO ISCO group summary",
            topic="esco/isco-groups",
            year=None,
            uri=_clean_text(isco_row.get("conceptUri", "")) or isco_code,
            entity_type="taxonomy",
        )
        metadata.update(
            {
                "section": label or isco_code,
                "section_title": label or isco_code,
                "page_or_loc": label or isco_code,
                "isco_group": isco_code,
                "isco_group_label": label or None,
                "esco_doc_type": "isco_group_summary",
            }
        )
        docs.append(
            NormalizedDocument(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                checksum=_checksum(text),
                source="esco",
                logical_file=isco_path.name if isco_rows else occupations_path.name,
            )
        )

    return docs, logical_files


async def _upsert_document_record(
    session: Any,
    *,
    doc_id: str,
    uri: str,
    checksum: str,
    source_type: str,
    title: str,
) -> None:
    existing = await session.get(Document, doc_id)
    if existing is None:
        session.add(
            Document(
                id=doc_id,
                uri=uri,
                checksum=checksum,
                source_type=source_type,
                title=title,
            )
        )
        return
    existing.uri = uri
    existing.checksum = checksum
    existing.source_type = source_type
    existing.title = title


def _count_chunks(stats: IngestionStats, *, logical_file: str, source: str, count: int) -> None:
    stats.chunks_per_file[logical_file] = stats.chunks_per_file.get(logical_file, 0) + count
    stats.chunks_per_source[source] += count
    stats.chunks_created += count


async def _count_document_records(session: Any, *, source_type: str) -> int:
    """Count persisted document rows for a given corpus source."""
    return int(
        await session.scalar(
            select(func.count()).select_from(Document).where(Document.source_type == source_type)
        )
        or 0
    )


async def _delete_document_records(session: Any, *, source_type: str) -> int:
    """Delete persisted document rows for a given corpus source."""
    existing = await _count_document_records(session, source_type=source_type)
    if existing <= 0:
        return 0
    await session.execute(delete(Document).where(Document.source_type == source_type))
    return existing


def _print_esco_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Emit a human-readable ESCO verification summary."""
    print(f"[esco] total_vectors={diagnostics['total_esco_vectors']}")
    print(f"[esco] counts_by_doc_type={diagnostics['counts_by_esco_doc_type']}")
    for doc_type in ESCO_ENRICHED_DOC_TYPES:
        sample = diagnostics["sample_payloads_by_esco_doc_type"].get(doc_type)
        if sample:
            print(f"[esco] sample_payload[{doc_type}]={sample}")


def _upsert_chunk_batches(
    chunks: list[RawChunk],
    *,
    settings: Settings,
    batch_size: int,
    stage_label: str = "raw_ingest",
) -> None:
    """Embed and upsert chunks in bounded batches to avoid oversized payloads."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_index = (start // batch_size) + 1
        request_label = f"{stage_label}:batch_{batch_index}"
        logger.info(
            "raw_ingest_batch_before_embedding",
            stage_label=stage_label,
            batch_index=batch_index,
            batch_size=len(batch),
        )
        print(
            f"[ingest] before embedding stage={stage_label} batch_index={batch_index} "
            f"batch_size={len(batch)}"
        )
        embedding_started = time.perf_counter()
        try:
            vectors = get_embeddings(
                [chunk.text for chunk in batch],
                settings=settings,
                request_label=request_label,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - embedding_started) * 1000, 2)
            logger.exception(
                "raw_ingest_batch_embedding_failed",
                stage_label=stage_label,
                batch_index=batch_index,
                batch_size=len(batch),
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
            )
            print(
                f"[ingest] ERROR embedding stage={stage_label} batch_index={batch_index} "
                f"batch_size={len(batch)} elapsed_ms={elapsed_ms} error_type={type(exc).__name__} "
                f"error={str(exc)[:240]}"
            )
            raise
        embedding_elapsed_ms = round((time.perf_counter() - embedding_started) * 1000, 2)
        logger.info(
            "raw_ingest_batch_after_embedding",
            stage_label=stage_label,
            batch_index=batch_index,
            batch_size=len(batch),
            elapsed_ms=embedding_elapsed_ms,
        )
        print(
            f"[ingest] after embedding stage={stage_label} batch_index={batch_index} "
            f"batch_size={len(batch)} elapsed_ms={embedding_elapsed_ms}"
        )
        logger.info(
            "raw_ingest_batch_before_qdrant_upsert",
            stage_label=stage_label,
            batch_index=batch_index,
            batch_size=len(batch),
        )
        print(
            f"[ingest] before qdrant upsert stage={stage_label} batch_index={batch_index} "
            f"batch_size={len(batch)}"
        )
        upsert_started = time.perf_counter()
        try:
            upsert_vectors(
                ids=[chunk.chunk_id for chunk in batch],
                vectors=vectors,
                payloads=[{**chunk.metadata, "text": chunk.text} for chunk in batch],
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - upsert_started) * 1000, 2)
            logger.exception(
                "raw_ingest_batch_qdrant_upsert_failed",
                stage_label=stage_label,
                batch_index=batch_index,
                batch_size=len(batch),
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
            )
            print(
                f"[ingest] ERROR qdrant upsert stage={stage_label} batch_index={batch_index} "
                f"batch_size={len(batch)} elapsed_ms={elapsed_ms} error_type={type(exc).__name__} "
                f"error={str(exc)[:240]}"
            )
            raise
        upsert_elapsed_ms = round((time.perf_counter() - upsert_started) * 1000, 2)
        logger.info(
            "raw_ingest_batch_after_qdrant_upsert",
            stage_label=stage_label,
            batch_index=batch_index,
            batch_size=len(batch),
            elapsed_ms=upsert_elapsed_ms,
        )
        print(
            f"[ingest] after qdrant upsert stage={stage_label} batch_index={batch_index} "
            f"batch_size={len(batch)} elapsed_ms={upsert_elapsed_ms}"
        )


async def ingest_raw_corpus(
    base_dir: Path | None = None,
    *,
    mode: str = "full",
    settings: Settings | None = None,
) -> IngestResponse:
    """Walk ``data/raw`` (or ``base_dir``), chunk, embed, and upsert into Qdrant."""
    if settings is None:
        settings = get_settings()

    raw_root = _resolve_raw_root(base_dir, settings)
    files = discover_raw_files(raw_root)
    run_id = str(uuid.uuid4())
    stats = IngestionStats()
    batch_size = max(1, settings.rag_embedding_batch_size)
    esco_only_backfill = _is_esco_only_backfill_mode(mode)

    logger.info(
        "raw_ingest_start",
        run_id=run_id,
        raw_root=str(raw_root),
        files_found=len(files),
    )
    print(f"[ingest] raw_root={raw_root} files_found={len(files)}")

    ensure_collection()
    delete_vectors_by_metadata({"source_type": "md"})
    logger.info("raw_ingest_legacy_vectors_purged", filters={"source_type": "md"})
    print("[ingest] purged legacy markdown vectors filters={'source_type': 'md'}")
    await init_db()
    session_factory = get_session_factory()

    async with session_factory() as session:
        ingestion_run = IngestionRun(id=run_id, mode=mode)
        session.add(ingestion_run)

        if esco_only_backfill:
            existing_esco_vectors = count_vectors(filters={"source": "esco"})
            existing_esco_documents = await _count_document_records(session, source_type="esco")
            delete_vectors_by_metadata({"source": "esco"})
            deleted_esco_documents = await _delete_document_records(session, source_type="esco")
            logger.info(
                "raw_ingest_esco_backfill_purge",
                existing_esco_vectors=existing_esco_vectors,
                existing_esco_documents=existing_esco_documents,
                deleted_esco_documents=deleted_esco_documents,
                reason="avoid_silent_old_new_mix",
            )
            print(
                "[esco] purge "
                f"existing_vectors={existing_esco_vectors} "
                f"existing_documents={existing_esco_documents} "
                f"deleted_documents={deleted_esco_documents}"
            )

        esco_root = raw_root / "esco"
        esco_processed = False

        for file_path in files:
            corpus_source = infer_corpus_source(file_path, raw_root)
            if corpus_source is None:
                continue
            if esco_only_backfill and corpus_source != "esco":
                continue

            if corpus_source == "esco":
                if esco_processed:
                    continue
                try:
                    esco_docs, logical_files = _build_esco_documents(esco_root, settings)
                    doc_counts_by_type = Counter(
                        doc.metadata.get("esco_doc_type", "unknown") for doc in esco_docs
                    )
                    esco_vectors_upserted = 0
                    pending_esco_chunks: list[RawChunk] = []
                    for doc in esco_docs:
                        chunks = chunk_text_by_tokens(
                            doc.text,
                            doc.metadata,
                            chunk_size=settings.rag_chunk_size_tokens,
                            overlap=settings.rag_chunk_overlap_tokens,
                        )
                        if not chunks:
                            continue
                        pending_esco_chunks.extend(chunks)
                        esco_vectors_upserted += len(chunks)
                        if esco_only_backfill and (
                            stats.documents_processed < 5
                            or stats.documents_processed % 1000 == 0
                        ):
                            logger.info(
                                "raw_ingest_esco_doc_chunked",
                                doc_type=doc.metadata.get("esco_doc_type"),
                                section=doc.metadata.get("section_title"),
                                chunk_count=len(chunks),
                                pending_chunks=len(pending_esco_chunks),
                            )
                        if len(pending_esco_chunks) >= max(128, batch_size * 16):
                            logger.info(
                                "raw_ingest_esco_batch_flush",
                                pending_chunks=len(pending_esco_chunks),
                                batch_size=batch_size,
                            )
                            _upsert_chunk_batches(
                                pending_esco_chunks,
                                settings=settings,
                                batch_size=batch_size,
                                stage_label="esco_backfill",
                            )
                            pending_esco_chunks.clear()
                        if doc.metadata.get("esco_doc_type") != "relation_detail":
                            await _upsert_document_record(
                                session,
                                doc_id=doc.doc_id,
                                uri=doc.metadata.get("uri") or doc.logical_file,
                                checksum=doc.checksum,
                                source_type=doc.source,
                                title=doc.metadata.get("section_title") or doc.metadata["document_title"],
                            )
                        stats.documents_processed += 1
                        _count_chunks(stats, logical_file=doc.logical_file, source=doc.source, count=len(chunks))

                    if pending_esco_chunks:
                        logger.info(
                            "raw_ingest_esco_final_flush",
                            pending_chunks=len(pending_esco_chunks),
                            batch_size=batch_size,
                        )
                        _upsert_chunk_batches(
                            pending_esco_chunks,
                            settings=settings,
                            batch_size=batch_size,
                            stage_label="esco_backfill",
                        )

                    stats.files_loaded += len(logical_files)
                    esco_processed = True
                    logger.info(
                        "raw_ingest_esco_complete",
                        documents=len(esco_docs),
                        logical_files=logical_files,
                        vectors_upserted=esco_vectors_upserted,
                        doc_counts_by_type=dict(doc_counts_by_type),
                    )
                    print(
                        "[ingest] loaded ESCO enriched corpus "
                        f"logical_files={logical_files} documents={len(esco_docs)} "
                        f"vectors_upserted={esco_vectors_upserted}"
                    )
                    print(
                        "[esco] generated "
                        f"documents={len(esco_docs)} "
                        f"vectors_upserted={esco_vectors_upserted} "
                        f"doc_counts_by_type={dict(doc_counts_by_type)}"
                    )
                    if esco_only_backfill:
                        diagnostics = get_esco_vector_diagnostics(doc_types=ESCO_ENRICHED_DOC_TYPES)
                        logger.info(
                            "raw_ingest_esco_backfill_verified",
                            total_esco_vectors=diagnostics["total_esco_vectors"],
                            counts_by_esco_doc_type=diagnostics["counts_by_esco_doc_type"],
                            sample_payloads_by_esco_doc_type=diagnostics["sample_payloads_by_esco_doc_type"],
                        )
                        _print_esco_diagnostics(diagnostics)
                except Exception as exc:
                    stats.files_failed += len(ESCO_AGGREGATED_FILES)
                    logger.error("raw_ingest_esco_failed", error=str(exc)[:300])
                    print(f"[ingest] ERROR esco enriched ingest: {exc}")
                continue

            try:
                if corpus_source == "wef" and file_path.suffix.lower() == ".pdf":
                    logger.info(
                        "raw_ingest_source_inferred",
                        path=str(file_path),
                        inferred_source=corpus_source,
                        topic=topic_from_relative_path(file_path, raw_root),
                    )
                    chunks, checksum = _chunk_pdf_document(file_path, raw_root, settings)
                    if not chunks:
                        logger.warning("raw_ingest_no_chunks", path=str(file_path))
                        stats.files_failed += 1
                        continue

                    _upsert_chunk_batches(
                        chunks,
                        settings=settings,
                        batch_size=batch_size,
                    )
                    logger.info(
                        "raw_ingest_wef_payload_sample",
                        path=str(file_path),
                        inferred_source=corpus_source,
                        payload_sample={key: chunks[0].metadata.get(key) for key in (
                            "source",
                            "source_name",
                            "source_type",
                            "file_name",
                            "document_title",
                            "topic",
                            "entity_type",
                            "section_title",
                            "page_number",
                        )},
                    )

                    doc_id = chunks[0].metadata["parent_doc_id"]
                    await _upsert_document_record(
                        session,
                        doc_id=doc_id,
                        uri=str(file_path.resolve()),
                        checksum=checksum,
                        source_type=corpus_source,
                        title=chunks[0].metadata["document_title"],
                    )
                    stats.files_loaded += 1
                    stats.documents_processed += 1
                    _count_chunks(stats, logical_file=file_path.name, source=corpus_source, count=len(chunks))
                    print(f"[ingest] loaded file={file_path.name} source={corpus_source} chunks={len(chunks)}")
                    continue

                if file_path.suffix.lower() == ".json":
                    records = load_json_records(file_path)
                    if not records:
                        stats.files_failed += 1
                        continue
                    logical_docs: list[NormalizedDocument] = []
                    for idx, record in enumerate(records):
                        body = json.dumps(record, ensure_ascii=False, default=str)
                        doc_id = make_deterministic_id(f"json::{topic_from_relative_path(file_path, raw_root)}::{idx}")
                        metadata = build_base_payload(
                            doc_id=doc_id,
                            corpus_source=corpus_source,
                            file_name=file_path.name,
                            document_title=file_path.name,
                            topic=topic_from_relative_path(file_path, raw_root),
                            year=wef_year_from_filename(file_path.name) if corpus_source == "wef" else None,
                            uri=str(file_path.resolve()),
                            entity_type="record",
                        )
                        metadata.update({"section": f"record {idx}", "section_title": f"record {idx}"})
                        logical_docs.append(
                            NormalizedDocument(
                                doc_id=doc_id,
                                text=body,
                                metadata=metadata,
                                checksum=_checksum(body),
                                source=corpus_source,
                                logical_file=file_path.name,
                            )
                        )
                    for doc in logical_docs:
                        chunks = chunk_text_by_tokens(
                            doc.text,
                            doc.metadata,
                            chunk_size=settings.rag_chunk_size_tokens,
                            overlap=settings.rag_chunk_overlap_tokens,
                        )
                        if not chunks:
                            continue
                        vectors = get_embeddings([chunk.text for chunk in chunks], settings=settings)
                        upsert_vectors(
                            ids=[chunk.chunk_id for chunk in chunks],
                            vectors=vectors,
                            payloads=[{**chunk.metadata, "text": chunk.text} for chunk in chunks],
                        )
                        await _upsert_document_record(
                            session,
                            doc_id=doc.doc_id,
                            uri=str(file_path.resolve()),
                            checksum=doc.checksum,
                            source_type=doc.source,
                            title=doc.metadata["document_title"],
                        )
                        stats.documents_processed += 1
                        _count_chunks(stats, logical_file=file_path.name, source=doc.source, count=len(chunks))
                    stats.files_loaded += 1
                    continue

                if corpus_source == "esco" and file_path.suffix.lower() == ".csv":
                    rows = read_csv_rows(file_path)
                    if not rows:
                        stats.files_failed += 1
                        continue
                    logical_docs = []
                    for idx, row in enumerate(rows):
                        sentence = esco_row_to_sentence(row, file_path.stem)
                        if not sentence:
                            continue
                        uri = row.get("conceptUri") or row.get("occupationUri") or row.get("skillUri") or str(file_path.resolve())
                        doc_id = make_deterministic_id(f"esco::{file_path.name}::{uri}::{idx}")
                        entity_type = "taxonomy"
                        metadata = build_base_payload(
                            doc_id=doc_id,
                            corpus_source="esco",
                            file_name=file_path.name,
                            document_title=f"ESCO {file_path.name}",
                            topic=topic_from_relative_path(file_path, raw_root),
                            year=None,
                            uri=uri,
                            entity_type=entity_type,
                        )
                        metadata.update(
                            {
                                "section": row.get("preferredLabel") or row.get("occupationLabel") or file_path.stem,
                                "section_title": row.get("preferredLabel") or row.get("occupationLabel") or file_path.stem,
                            }
                        )
                        logical_docs.append(
                            NormalizedDocument(
                                doc_id=doc_id,
                                text=sentence,
                                metadata=metadata,
                                checksum=_checksum(sentence),
                                source="esco",
                                logical_file=file_path.name,
                            )
                        )
                    for doc in logical_docs:
                        chunk = RawChunk(text=doc.text, metadata={**doc.metadata, "chunk_index": 0})
                        vectors = get_embeddings([chunk.text], settings=settings)
                        upsert_vectors(
                            ids=[chunk.chunk_id],
                            vectors=vectors,
                            payloads=[{**chunk.metadata, "text": chunk.text}],
                        )
                        await _upsert_document_record(
                            session,
                            doc_id=doc.doc_id,
                            uri=str(file_path.resolve()),
                            checksum=doc.checksum,
                            source_type=doc.source,
                            title=doc.metadata["document_title"],
                        )
                        stats.documents_processed += 1
                        _count_chunks(stats, logical_file=file_path.name, source=doc.source, count=1)
                    stats.files_loaded += 1
            except Exception as exc:
                stats.files_failed += 1
                logger.error("raw_ingest_file_failed", path=str(file_path), error=str(exc)[:300])
                print(f"[ingest] ERROR {file_path}: {exc}")

        if esco_only_backfill and not esco_processed:
            stats.files_failed += 1
            logger.error("raw_ingest_esco_backfill_missing_inputs", esco_root=str(esco_root))
            print(f"[esco] ERROR missing ESCO aggregated inputs under {esco_root}")

        ingestion_run.documents_processed = stats.documents_processed
        ingestion_run.chunks_created = stats.chunks_created
        ingestion_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
        ingestion_run.success = stats.files_failed == 0
        await session.commit()

    logger.info(
        "raw_ingest_complete",
        run_id=run_id,
        documents=stats.documents_processed,
        chunks=stats.chunks_created,
        files_loaded=stats.files_loaded,
        files_failed=stats.files_failed,
        chunks_per_file=dict(stats.chunks_per_file),
        chunks_per_source=dict(stats.chunks_per_source),
    )
    print(
        "[ingest] summary "
        f"files_loaded={stats.files_loaded} files_failed={stats.files_failed} "
        f"chunks_per_source={dict(stats.chunks_per_source)}"
    )

    return IngestResponse(
        run_id=run_id,
        documents_processed=stats.documents_processed,
        chunks_created=stats.chunks_created,
    )
