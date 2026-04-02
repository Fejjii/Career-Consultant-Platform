"""Tests for tool input/output Pydantic schemas and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from career_intel.schemas.domain import (
    LearningPlanInput,
    LearningPlanOutput,
    RoleCompareInput,
    RoleCompareOutput,
    SkillGapInput,
    SkillGapOutput,
)


class TestSkillGapSchema:
    def test_valid_input(self) -> None:
        inp = SkillGapInput(
            target_role="ML Engineer",
            current_skills=["Python", "SQL"],
            seniority="mid",
        )
        assert inp.target_role == "ML Engineer"
        assert len(inp.current_skills) == 2

    def test_empty_skills_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SkillGapInput(target_role="ML Engineer", current_skills=[])

    def test_short_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SkillGapInput(target_role="X", current_skills=["Python"])

    def test_output_schema(self) -> None:
        out = SkillGapOutput(
            target_role="ML Engineer",
            must_have_gaps=[{"skill": "PyTorch", "importance": "critical", "reason": "core"}],
            nice_to_have_gaps=[],
            suggested_order=["PyTorch"],
            citations=[],
        )
        assert out.target_role == "ML Engineer"


class TestRoleCompareSchema:
    def test_valid_input(self) -> None:
        inp = RoleCompareInput(role_a="Product Manager", role_b="Program Manager")
        assert inp.role_a == "Product Manager"

    def test_same_roles_allowed(self) -> None:
        inp = RoleCompareInput(role_a="Data Analyst", role_b="Data Analyst")
        assert inp.role_a == inp.role_b

    def test_output_schema(self) -> None:
        out = RoleCompareOutput(
            role_a="PM",
            role_b="PgM",
            comparison={"skills": {}},
            narrative="PM focuses on product strategy...",
            citations=[],
        )
        assert out.narrative.startswith("PM")


class TestLearningPlanSchema:
    def test_valid_input(self) -> None:
        inp = LearningPlanInput(
            goal_role="Data Analyst",
            hours_per_week=10,
            horizon_weeks=12,
        )
        assert inp.horizon_weeks == 12

    def test_hours_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LearningPlanInput(goal_role="Data Analyst", hours_per_week=0)

    def test_output_schema(self) -> None:
        out = LearningPlanOutput(
            goal_role="Data Analyst",
            total_weeks=12,
            milestones=[{"week_range": "1-2", "focus": "SQL", "skills": ["SQL"]}],
            resources=[],
            citations=[],
        )
        assert out.total_weeks == 12
