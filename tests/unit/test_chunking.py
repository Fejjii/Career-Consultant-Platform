"""Tests for the chunking module."""

from __future__ import annotations

from career_intel.rag.chunking import chunk_csv_rows, chunk_markdown


def test_chunk_markdown_basic() -> None:
    text = "# Title\n\nSome content here.\n\n## Section\n\nMore content."
    metadata = {"source_id": "test", "parent_doc_id": "test"}
    chunks = chunk_markdown(text, metadata)
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)


def test_chunk_markdown_preserves_metadata() -> None:
    text = "# Title\n\nContent"
    metadata = {"source_id": "doc1", "parent_doc_id": "doc1", "source_type": "md"}
    chunks = chunk_markdown(text, metadata)
    assert chunks[0].metadata["source_id"] == "doc1"


def test_chunk_markdown_heading_splitting() -> None:
    text = "# First\n\nContent A\n\n# Second\n\nContent B"
    metadata = {"source_id": "test", "parent_doc_id": "test"}
    chunks = chunk_markdown(text, metadata)
    assert len(chunks) == 2


def test_chunk_csv_rows() -> None:
    rows = [
        {"occupation_code": "15-2051", "title": "Data Scientist", "skills": "Python, ML, Stats"},
        {"occupation_code": "15-1252", "title": "Software Dev", "skills": "Java, Python, SQL"},
    ]
    metadata = {"source_id": "csv1", "parent_doc_id": "csv1", "source_type": "csv"}
    chunks = chunk_csv_rows(rows, metadata)
    assert len(chunks) == 2
    assert chunks[0].metadata["occupation_code"] == "15-2051"


def test_chunk_csv_with_text_columns() -> None:
    rows = [{"name": "ML Engineer", "desc": "Builds ML systems", "salary": "high"}]
    metadata = {"source_id": "csv2", "parent_doc_id": "csv2"}
    chunks = chunk_csv_rows(rows, metadata, text_columns=["name", "desc"])
    assert "ML Engineer" in chunks[0].text
    assert "salary" not in chunks[0].text.lower() or "high" not in chunks[0].text
