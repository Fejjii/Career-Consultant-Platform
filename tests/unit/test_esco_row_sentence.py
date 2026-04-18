"""Unit tests for ESCO CSV row → sentence formatting."""

from __future__ import annotations

from career_intel.rag.raw_corpus_ingest import esco_row_to_sentence


def test_occupation_row_sentence() -> None:
    row = {
        "preferredLabel": "Data Engineer",
        "code": "2521.9",
        "iscoGroup": "2521",
        "description": "Designs data pipelines.",
    }
    s = esco_row_to_sentence(row, "occupations_en")
    assert "Occupation: Data Engineer" in s
    assert "Python" not in s
    assert "2521.9" in s


def test_occupation_skill_relation_sentence() -> None:
    row = {
        "occupationLabel": "Data Engineer",
        "relationType": "essential",
        "skillType": "skill/competence",
        "skillLabel": "Python",
    }
    s = esco_row_to_sentence(row, "occupationSkillRelations_en")
    assert "Occupation: Data Engineer" in s
    assert "Skill: Python" in s
    assert "essential" in s


def test_skill_row_sentence() -> None:
    row = {
        "preferredLabel": "manage team",
        "skillType": "skill/competence",
        "definition": "Assign tasks to staff.",
    }
    s = esco_row_to_sentence(row, "skills_en")
    assert "Skill: manage team" in s
    assert "Assign tasks" in s
