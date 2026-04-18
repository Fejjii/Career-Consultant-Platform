"""Source inventory metadata for corpus coverage questions.

This module answers "what sources do you use?" style questions without
touching retrieval, embeddings, or vector search. It inspects the raw corpus
layout plus the latest ESCO backfill log to produce a concise grounded
inventory response.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceInventoryItem:
    """One source family or dataset group exposed to the user."""

    source_name: str
    source_family: str
    description: str
    relative_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceInventorySummary:
    """Computed inventory for source coverage answers and UI cards."""

    total_source_groups: int
    total_files_present: int
    esco_ingestion_status: str
    esco_status_note: str | None
    items: list[dict[str, object]]


_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_INVENTORY: tuple[SourceInventoryItem, ...] = (
    SourceInventoryItem(
        source_name="WEF Future of Jobs 2018",
        source_family="WEF report",
        description="Global labour-market trends, roles, and skills highlighted in the 2018 Future of Jobs report.",
        relative_paths=("data/raw/wef/WEF_Future_of_Jobs_2018.pdf",),
    ),
    SourceInventoryItem(
        source_name="WEF Future of Jobs 2020",
        source_family="WEF report",
        description="Pandemic-era labour-market shifts, reskilling priorities, and job transition signals from 2020.",
        relative_paths=("data/raw/wef/WEF_Future_of_Jobs_2020.pdf",),
    ),
    SourceInventoryItem(
        source_name="WEF Future of Jobs 2023",
        source_family="WEF report",
        description="Recent role-growth, disruption, and skills-demand findings from the 2023 report.",
        relative_paths=("data/raw/wef/WEF_Future_of_Jobs_2023.pdf",),
    ),
    SourceInventoryItem(
        source_name="WEF Future of Jobs Report 2025",
        source_family="WEF report",
        description="Latest Future of Jobs outlook used for forward-looking labour-market and skills context.",
        relative_paths=("data/raw/wef/WEF_Future_of_Jobs_Report_2025.pdf",),
    ),
    SourceInventoryItem(
        source_name="ESCO occupations",
        source_family="ESCO taxonomy",
        description="Canonical ESCO occupation records, labels, and occupation metadata.",
        relative_paths=("data/raw/esco/occupations_en.csv",),
    ),
    SourceInventoryItem(
        source_name="ESCO skills",
        source_family="ESCO taxonomy",
        description="Canonical ESCO skills and competences catalog used for skill coverage and terminology.",
        relative_paths=("data/raw/esco/skills_en.csv",),
    ),
    SourceInventoryItem(
        source_name="ESCO occupation skill relations",
        source_family="ESCO relations",
        description="Links between occupations and skills, including essential and optional skill relations.",
        relative_paths=("data/raw/esco/occupationSkillRelations_en.csv",),
    ),
    SourceInventoryItem(
        source_name="ESCO ISCO and taxonomy files",
        source_family="ESCO taxonomy",
        description="Supporting ESCO hierarchy and ISCO mapping files used for taxonomy structure and rollups.",
        relative_paths=(
            "data/raw/esco/ISCOGroups_en.csv",
            "data/raw/esco/skillsHierarchy_en.csv",
        ),
    ),
)

_SOURCE_QUERY_HINTS = (
    "what sources",
    "which sources",
    "what source",
    "source inventory",
    "data sources",
    "knowledge base",
    "what data do you have",
    "what data sources",
    "name the sources",
    "how many sources",
    "how many data sources",
    "what are you using",
    "what sources are you using",
    "what is in your corpus",
    "what's in your corpus",
    "what does your corpus cover",
)


def is_source_inventory_query(query: str) -> bool:
    """Return True when the query asks about corpus/source coverage."""

    lowered = " ".join(query.lower().split())
    return any(hint in lowered for hint in _SOURCE_QUERY_HINTS)


def build_source_inventory_summary(*, project_root: Path | None = None) -> SourceInventorySummary:
    """Inspect raw source files and current ESCO backfill state."""

    root = project_root or _ROOT
    esco_status, esco_note = _detect_esco_ingestion_status(root)
    items: list[dict[str, object]] = []
    total_files_present = 0

    for item in _SOURCE_INVENTORY:
        file_paths = [root / rel_path for rel_path in item.relative_paths]
        existing_files = [path.name for path in file_paths if path.exists()]
        file_count = len(existing_files)
        total_files_present += file_count
        if item.source_family.startswith("ESCO"):
            ingestion_status = esco_status
        else:
            ingestion_status = "available" if file_count else "missing"

        items.append(
            {
                "source_name": item.source_name,
                "source_family": item.source_family,
                "description": item.description,
                "file_count": file_count,
                "ingestion_status": ingestion_status,
                "paths": existing_files,
            }
        )

    return SourceInventorySummary(
        total_source_groups=len(_SOURCE_INVENTORY),
        total_files_present=total_files_present,
        esco_ingestion_status=esco_status,
        esco_status_note=esco_note,
        items=items,
    )


def format_source_inventory_answer(summary: SourceInventorySummary) -> str:
    """Render a concise inventory answer for chat."""

    def _short(text: str, max_len: int = 90) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 1].rstrip() + "…"

    def _status_label(raw: str) -> str:
        return {
            "available": "ready",
            "missing": "missing",
            "indexing_in_progress": "loading",
        }.get(str(raw), str(raw))

    lines = [
        "**Sources (raw files on disk)**",
        f"{summary.total_source_groups} groups · {summary.total_files_present} files found.",
        "",
        "**Catalog**",
    ]
    for item in summary.items:
        file_count = int(item["file_count"])
        count_label = "1 file" if file_count == 1 else f"{file_count} files"
        status = _status_label(str(item["ingestion_status"]))
        lines.append(
            f"- **{item['source_name']}** · {_short(str(item['description']))} · "
            f"{count_label} · **{status}**"
        )

    if summary.esco_status_note:
        lines.extend(["", f"_{summary.esco_status_note}_"])

    return "\n".join(lines)


def _detect_esco_ingestion_status(project_root: Path) -> tuple[str, str | None]:
    """Inspect the newest ESCO backfill log to detect whether indexing is still active."""

    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        return "available", None

    log_paths = sorted(logs_dir.glob("esco_backfill_*.log"))
    if not log_paths:
        return "available", None

    latest_log = log_paths[-1]
    tail_text = _read_log_tail(latest_log)
    lowered = tail_text.lower()

    running_markers = ("[esco-live]", "progress_pct=", "embedding_batch_complete")
    completed_markers = (
        "backfill complete",
        "completed successfully",
        "ingestion complete",
        "all done",
    )
    if any(marker in lowered for marker in running_markers) and not any(
        marker in lowered for marker in completed_markers
    ):
        return (
            "indexing_in_progress",
            "ESCO is still indexing in the background; coverage will grow until it finishes.",
        )

    return "available", None


def _read_log_tail(path: Path, *, max_bytes: int = 8192) -> str:
    """Read the tail of a potentially large log file without loading it all into memory."""

    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(size - max_bytes, 0))
        data = handle.read()
    return data.decode("utf-8", errors="replace")
