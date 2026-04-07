"""Tests for evaluation runner utilities."""

from __future__ import annotations

from career_intel.evaluation.eval_runner import (
    check_abstain_behaviour,
    check_citation_integrity,
    check_routing_accuracy,
    evaluate_retrieval_hit,
    load_golden_dataset,
    run_evaluation,
)
from career_intel.schemas.domain import GoldenExample


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

    def test_small_talk_cases_exist(self) -> None:
        examples = load_golden_dataset()
        st_cases = [ex for ex in examples if "small_talk" in ex.tags]
        assert len(st_cases) >= 1, "Need at least 1 small_talk test case"

    def test_cv_cases_exist(self) -> None:
        examples = load_golden_dataset()
        cv_cases = [ex for ex in examples if "cv_relevant" in ex.tags]
        assert len(cv_cases) >= 1, "Need at least 1 CV-relevant test case"


class TestRoutingAccuracy:
    def test_correct_routing(self) -> None:
        ex = GoldenExample(query="hello", expected_behaviour="small_talk", tags=["small_talk"])
        result = check_routing_accuracy(ex, actual_intent="small_talk")
        assert result["correct"] is True

    def test_incorrect_routing(self) -> None:
        ex = GoldenExample(query="hello", expected_behaviour="small_talk", tags=["small_talk"])
        result = check_routing_accuracy(ex, actual_intent="retrieval_required")
        assert result["correct"] is False

    def test_tool_routing_from_behaviour(self) -> None:
        ex = GoldenExample(query="compare PM vs TPM", expected_behaviour="use_tool:role_compare", tags=["tool_required"])
        result = check_routing_accuracy(ex, actual_intent="tool_required")
        assert result["correct"] is True


class TestRunEvaluation:
    def test_summary_aggregation(self) -> None:
        results = [
            {"query": "hello", "intent": "small_talk", "routing_correct": True,
             "retrieval_invoked": False, "sources_count": 0, "total_latency_ms": 200},
            {"query": "skills?", "intent": "retrieval_required", "routing_correct": True,
             "retrieval_invoked": True, "sources_count": 3, "total_latency_ms": 1500},
        ]
        summary = run_evaluation(results)
        assert summary["total_queries"] == 2
        assert summary["routing_accuracy"] == 1.0
        assert summary["retrieval_invoked_count"] == 1
