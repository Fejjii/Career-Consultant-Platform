"""Tests for evaluation runner utilities."""

from __future__ import annotations

from career_intel.evaluation.eval_runner import (
    check_abstain_behaviour,
    check_citation_integrity,
    evaluate_retrieval_hit,
    load_golden_dataset,
)


def test_load_golden_dataset() -> None:
    examples = load_golden_dataset()
    assert len(examples) >= 5
    assert all(ex.query for ex in examples)


def test_citation_integrity_valid() -> None:
    result = check_citation_integrity(cited_ids={1, 2}, retrieved_ids={1, 2, 3})
    assert result["valid"] is True
    assert result["orphaned"] == []


def test_citation_integrity_orphaned() -> None:
    result = check_citation_integrity(cited_ids={1, 4}, retrieved_ids={1, 2, 3})
    assert result["valid"] is False
    assert 4 in result["orphaned"]


def test_abstain_positive() -> None:
    response = "I don't have enough evidence to answer that confidently."
    assert check_abstain_behaviour(response) is True


def test_abstain_negative() -> None:
    response = "Based on [1], the top growing roles include..."
    assert check_abstain_behaviour(response) is False


def test_retrieval_hit_with_matches() -> None:
    result = evaluate_retrieval_hit(
        expected_chunk_ids=["a", "b", "c"],
        retrieved_chunk_ids=["a", "c", "d", "e"],
    )
    assert result["recall"] > 0.6
    assert result["hits"] == 2


def test_retrieval_hit_no_expected() -> None:
    result = evaluate_retrieval_hit(expected_chunk_ids=[], retrieved_chunk_ids=["a"])
    assert result["skipped"] is True


class TestGoldenDatasetStructure:
    """Verify the golden dataset has the expected shape."""

    def test_all_entries_have_query(self) -> None:
        for ex in load_golden_dataset():
            assert ex.query.strip(), f"Empty query found: {ex}"

    def test_abstain_cases_exist(self) -> None:
        examples = load_golden_dataset()
        abstain_cases = [ex for ex in examples if ex.expected_behaviour == "abstain"]
        assert len(abstain_cases) >= 2, "Need at least 2 abstain test cases"

    def test_cite_source_cases_exist(self) -> None:
        examples = load_golden_dataset()
        cite_cases = [ex for ex in examples if ex.expected_behaviour == "cite_source"]
        assert len(cite_cases) >= 2, "Need at least 2 cite_source test cases"

    def test_tool_cases_exist(self) -> None:
        examples = load_golden_dataset()
        tool_cases = [ex for ex in examples if ex.expected_behaviour and "use_tool" in ex.expected_behaviour]
        assert len(tool_cases) >= 2, "Need at least 2 tool-calling test cases"
