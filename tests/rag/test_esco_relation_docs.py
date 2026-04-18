from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from career_intel.rag.raw_corpus_ingest import _build_esco_documents


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_esco_documents_emits_relation_level_and_taxonomy_docs(tmp_path: Path) -> None:
    esco_root = tmp_path / "esco"

    _write_csv(
        esco_root / "occupations_en.csv",
        [
            "conceptType",
            "conceptUri",
            "iscoGroup",
            "preferredLabel",
            "description",
            "code",
        ],
        [
            {
                "conceptType": "Occupation",
                "conceptUri": "occ:data-engineer",
                "iscoGroup": "2511",
                "preferredLabel": "data engineer",
                "description": "Designs data platforms and pipelines.",
                "code": "2511.20",
            },
            {
                "conceptType": "Occupation",
                "conceptUri": "occ:unlinked-role",
                "iscoGroup": "2511",
                "preferredLabel": "unlinked role",
                "description": "No relation rows should keep this out of the enriched corpus.",
                "code": "2511.99",
            }
        ],
    )
    _write_csv(
        esco_root / "skills_en.csv",
        [
            "conceptType",
            "conceptUri",
            "skillType",
            "preferredLabel",
            "definition",
            "description",
        ],
        [
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": "skill:python",
                "skillType": "knowledge",
                "preferredLabel": "Python",
                "definition": "Python programming language.",
                "description": "Python programming language.",
            },
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": "skill:sql",
                "skillType": "knowledge",
                "preferredLabel": "SQL",
                "definition": "Structured query language.",
                "description": "Structured query language.",
            },
            {
                "conceptType": "KnowledgeSkillCompetence",
                "conceptUri": "skill:unused",
                "skillType": "knowledge",
                "preferredLabel": "Unused Skill",
                "definition": "Should be excluded from relation-focused docs.",
                "description": "Should be excluded from relation-focused docs.",
            },
        ],
    )
    _write_csv(
        esco_root / "occupationSkillRelations_en.csv",
        [
            "occupationUri",
            "occupationLabel",
            "relationType",
            "skillType",
            "skillUri",
            "skillLabel",
        ],
        [
            {
                "occupationUri": "occ:data-engineer",
                "occupationLabel": "data engineer",
                "relationType": "essential",
                "skillType": "knowledge",
                "skillUri": "skill:python",
                "skillLabel": "Python",
            },
            {
                "occupationUri": "occ:data-engineer",
                "occupationLabel": "data engineer",
                "relationType": "optional",
                "skillType": "knowledge",
                "skillUri": "skill:sql",
                "skillLabel": "SQL",
            },
        ],
    )
    _write_csv(
        esco_root / "ISCOGroups_en.csv",
        ["conceptType", "conceptUri", "code", "preferredLabel", "description"],
        [
            {
                "conceptType": "ISCOGroup",
                "conceptUri": "isco:2511",
                "code": "2511",
                "preferredLabel": "systems analysts",
                "description": "Systems analysts and related occupations.",
            }
        ],
    )
    _write_csv(
        esco_root / "skillsHierarchy_en.csv",
        [
            "Level 0 URI",
            "Level 0 preferred term",
            "Level 1 URI",
            "Level 1 preferred term",
            "Level 2 URI",
            "Level 2 preferred term",
            "Level 3 URI",
            "Level 3 preferred term",
            "Description",
            "Scope note",
            "Level 0 code",
            "Level 1 code",
            "Level 2 code",
            "Level 3 code",
        ],
        [
            {
                "Level 0 URI": "skill:root",
                "Level 0 preferred term": "digital skills",
                "Level 1 URI": "skill:python",
                "Level 1 preferred term": "Python",
                "Level 2 URI": "",
                "Level 2 preferred term": "",
                "Level 3 URI": "",
                "Level 3 preferred term": "",
                "Description": "Programming hierarchy.",
                "Scope note": "",
                "Level 0 code": "S",
                "Level 1 code": "S1",
                "Level 2 code": "",
                "Level 3 code": "",
            }
        ],
    )

    docs, logical_files = _build_esco_documents(esco_root, settings=SimpleNamespace())

    assert set(logical_files) == {
        "occupations_en.csv",
        "skills_en.csv",
        "occupationSkillRelations_en.csv",
        "ISCOGroups_en.csv",
        "skillsHierarchy_en.csv",
    }

    by_doc_type = {doc.metadata.get("esco_doc_type"): doc for doc in docs if doc.metadata.get("esco_doc_type")}
    relation_docs = [doc for doc in docs if doc.metadata.get("esco_doc_type") == "relation_detail"]

    assert "occupation_summary" in by_doc_type
    assert "skill_summary" in by_doc_type
    assert relation_docs
    assert "taxonomy_mapping" in by_doc_type
    assert "isco_group_summary" in by_doc_type

    occupation_summary = by_doc_type["occupation_summary"]
    assert occupation_summary.metadata["isco_group"] == "2511"
    assert occupation_summary.metadata["isco_group_label"] == "systems analysts"
    assert "Python" in occupation_summary.text
    assert "SQL" in occupation_summary.text

    assert len(relation_docs) == 2
    assert sorted(doc.metadata.get("relation_type") for doc in relation_docs) == ["essential", "optional"]
    assert sorted(doc.metadata.get("skill_label") for doc in relation_docs) == ["Python", "SQL"]
    assert all(doc.metadata.get("occupation_id") == "occ:data-engineer" for doc in relation_docs)

    taxonomy_mapping = by_doc_type["taxonomy_mapping"]
    assert "Maps to ISCO group: 2511" in taxonomy_mapping.text
    assert all("unlinked role" not in doc.text.lower() for doc in docs)
    assert all("unused skill" not in doc.text.lower() for doc in docs)
