"""Tests for citation extraction and validation."""

from __future__ import annotations

from career_intel.rag.citation import extract_cited_ids, validate_citations


def test_extract_cited_ids_basic() -> None:
    text = "According to [1], this is true. Also see [3] and [1]."
    ids = extract_cited_ids(text)
    assert ids == {1, 3}


def test_extract_cited_ids_empty() -> None:
    assert extract_cited_ids("No citations here.") == set()


def test_validate_citations_all_valid() -> None:
    citation_map = {1: "chunk_a", 2: "chunk_b", 3: "chunk_c"}
    text = "See [1] and [2]."
    is_valid, invalid = validate_citations(text, citation_map)
    assert is_valid
    assert invalid == set()


def test_validate_citations_invalid_reference() -> None:
    citation_map = {1: "chunk_a", 2: "chunk_b"}
    text = "See [1] and [5]."
    is_valid, invalid = validate_citations(text, citation_map)
    assert not is_valid
    assert 5 in invalid
