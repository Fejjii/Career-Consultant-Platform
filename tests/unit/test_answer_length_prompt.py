"""Answer length prompt instructions stay clearly differentiated."""

from __future__ import annotations

from career_intel.orchestration.synthesize import answer_length_system_suffix


def test_balanced_suffix_mentions_moderate_structure() -> None:
    text = answer_length_system_suffix("balanced")
    assert "balanced" in text.lower()
    assert "moderate" in text.lower()
    assert "short paragraphs" in text.lower()


def test_concise_suffix_mentions_bullets() -> None:
    text = answer_length_system_suffix("concise")
    assert "concise" in text.lower()
    assert "bullet" in text.lower()
    assert "5 to 7" in text.lower()
    assert "do not write prose paragraphs" in text.lower()


def test_detailed_suffix_mentions_examples_and_sections() -> None:
    text = answer_length_system_suffix("detailed")
    assert "example" in text.lower()
    assert "section" in text.lower()
    assert "overview" in text.lower()
    assert "practical implications" in text.lower()
