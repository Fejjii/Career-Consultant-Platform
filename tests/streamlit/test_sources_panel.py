"""Unit tests for Streamlit sources / citations panel helpers."""

from __future__ import annotations

from sources_panel import (
    build_sublabel,
    group_by_family_ordered,
    infer_source_family,
    merge_citations,
    prepare_sources_panel_rows,
)


def test_infer_source_family_from_metadata() -> None:
    assert infer_source_family({"source": "wef", "source_id": "x"}) == "wef"
    assert infer_source_family({"source": "esco", "source_id": "x"}) == "esco"


def test_infer_source_family_from_source_id_prefix() -> None:
    assert infer_source_family({"source_id": "wef::abc", "source": None}) == "wef"
    assert infer_source_family({"source_id": "esco::def", "source": None}) == "esco"


def test_merge_citations_dedupes_same_chunk_identity() -> None:
    cites = [
        {
            "id": 1,
            "source_id": "wef::same",
            "title": "WEF Future of Jobs 2025",
            "page_number": 32,
            "excerpt": "Short.",
            "source": "wef",
        },
        {
            "id": 2,
            "source_id": "wef::same",
            "title": "WEF Future of Jobs 2025",
            "page_number": 32,
            "excerpt": "Longer excerpt wins for display length logic.",
            "source": "wef",
        },
    ]
    merged = merge_citations(cites)
    assert len(merged) == 1
    assert merged[0].reference_count == 2
    assert merged[0].citation_ids == [1, 2]
    assert "Longer excerpt" in merged[0].snippet


def test_build_sublabel_page_over_doc_type() -> None:
    cite = {"page_number": 12, "esco_doc_type": "relation_detail", "source": "esco"}
    assert build_sublabel(cite, "esco") == "Page 12"


def test_group_by_family_ordered_wef_before_esco() -> None:
    cites = [
        {"id": 1, "source_id": "esco::a", "title": "ESCO A", "excerpt": "e", "source": "esco"},
        {"id": 2, "source_id": "wef::b", "title": "WEF B", "excerpt": "w", "source": "wef"},
    ]
    merged = merge_citations(cites)
    grouped = group_by_family_ordered(merged)
    families = [g[0] for g in grouped]
    assert families == ["wef", "esco"]


def test_prepare_sources_panel_rows_caps() -> None:
    cites = [
        {"id": i, "source_id": f"x::{i}", "title": f"T{i}", "excerpt": "e", "source": "wef"}
        for i in range(1, 9)
    ]
    cards, dbg = prepare_sources_panel_rows(cites, max_sources=5)
    assert len(cards) == 5
    assert dbg["unique_sources"] == 8
    assert dbg["truncated"] == 3
