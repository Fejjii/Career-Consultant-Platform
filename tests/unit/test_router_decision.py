"""Tests for the intent-first router decision normalization."""

from __future__ import annotations

from career_intel.schemas.domain import RouterDecision
from career_intel.tools.registry import _normalize_decision, _parse_router_response


class TestParseRouterResponse:
    def test_valid_json(self) -> None:
        raw = '{"intent": "retrieval_required", "confidence": 0.9, "tool_name": null, "params": {}, "use_cv": false, "reason": "fact lookup"}'
        result = _parse_router_response(raw)
        assert result["intent"] == "retrieval_required"

    def test_markdown_fenced_json(self) -> None:
        raw = '```json\n{"intent": "small_talk", "confidence": 0.95}\n```'
        result = _parse_router_response(raw)
        assert result["intent"] == "small_talk"

    def test_invalid_json_returns_empty(self) -> None:
        result = _parse_router_response("not json at all")
        assert result == {}


class TestNormalizeDecision:
    def test_full_intent_format(self) -> None:
        parsed = {
            "intent": "tool_required",
            "confidence": 0.85,
            "tool_name": "skill_gap",
            "params": {"target_role": "ML Engineer", "current_skills": ["Python"]},
            "use_cv": True,
            "reason": "User wants skill gap analysis",
        }
        decision = _normalize_decision(parsed, cv_available=True)
        assert decision.intent == "tool_required"
        assert decision.tool_name == "skill_gap"
        assert decision.use_cv is True
        assert decision.confidence == 0.85
        assert decision.reason == "User wants skill gap analysis"

    def test_legacy_tool_format_backcompat(self) -> None:
        """Legacy format with 'tool' key (no 'intent') is auto-mapped."""
        parsed = {"tool": "role_compare", "params": {"role_a": "PM", "role_b": "TPM"}, "use_cv": False}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.intent == "tool_required"
        assert decision.tool_name == "role_compare"

    def test_legacy_none_tool_maps_to_retrieval(self) -> None:
        parsed = {"tool": "none", "params": {}, "use_cv": False}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.intent == "retrieval_required"
        assert decision.tool_name is None

    def test_unknown_tool_falls_back(self) -> None:
        parsed = {"intent": "tool_required", "tool_name": "nonexistent_tool", "params": {}}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.tool_name is None
        assert decision.intent == "retrieval_required"

    def test_use_cv_false_when_no_cv_available(self) -> None:
        parsed = {"intent": "retrieval_required", "use_cv": True}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.use_cv is False

    def test_confidence_clamped(self) -> None:
        parsed = {"intent": "small_talk", "confidence": 5.0}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.confidence == 1.0

    def test_empty_parsed_defaults(self) -> None:
        decision = _normalize_decision({}, cv_available=False)
        assert decision.intent == "retrieval_required"
        assert decision.tool_name is None
        assert decision.use_cv is False

    def test_small_talk_intent(self) -> None:
        parsed = {"intent": "small_talk", "confidence": 0.95, "reason": "greeting"}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.intent == "small_talk"
        assert decision.tool_name is None

    def test_direct_answer_intent(self) -> None:
        parsed = {"intent": "direct_answer", "confidence": 0.8, "reason": "simple fact"}
        decision = _normalize_decision(parsed, cv_available=False)
        assert decision.intent == "direct_answer"


class TestRouterDecisionModel:
    def test_serialization_roundtrip(self) -> None:
        decision = RouterDecision(
            intent="tool_required",
            confidence=0.9,
            tool_name="skill_gap",
            params={"target_role": "ML Engineer"},
            use_cv=True,
            reason="skill analysis requested",
        )
        data = decision.model_dump()
        restored = RouterDecision.model_validate(data)
        assert restored.intent == "tool_required"
        assert restored.tool_name == "skill_gap"
        assert restored.use_cv is True
