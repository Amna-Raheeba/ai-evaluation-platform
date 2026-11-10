"""Golden Dataset Engine: loading, validation, versioning, profiles."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_platform.datasets import (DatasetError, dataset_version,
                                    load_dataset, select_cases, validate_case)

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_datasets_load_and_meet_minimum_sizes():
    rag = load_dataset(ROOT / "datasets" / "rag.jsonl")
    research = load_dataset(ROOT / "datasets" / "research.jsonl")
    router = load_dataset(ROOT / "datasets" / "router.jsonl")
    assert len(rag) >= 8 and len(research) >= 6 and len(router) >= 10
    assert all(c.project == "rag" for c in rag)
    assert {c.expected.get("expected_tier") for c in router} <= {"cheap", "balanced", "premium"}
    assert {c.expected["expected_tier"] for c in router} == {"cheap", "balanced", "premium"}


def test_dataset_version_is_content_hash(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text('{"id": "a", "project": "mock", "input": {}}\n')
    v1 = dataset_version(path)
    assert v1.startswith("mini@") and len(v1.split("@")[1]) == 8
    path.write_text('{"id": "b", "project": "mock", "input": {}}\n')
    assert dataset_version(path) != v1  # new content = new version


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "dup.jsonl"
    line = json.dumps({"id": "x", "project": "mock", "input": {}})
    path.write_text(line + "\n" + line + "\n")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(path)


def test_unknown_fields_rejected():
    with pytest.raises(DatasetError, match="unknown fields"):
        validate_case({"id": "x", "project": "mock", "input": {}, "surprise": 1}, "t")
    with pytest.raises(DatasetError, match="unknown expected"):
        validate_case({"id": "x", "project": "mock", "input": {},
                       "expected": {"vibes": "good"}}, "t")


def test_bad_difficulty_and_missing_fields_rejected():
    with pytest.raises(DatasetError):
        validate_case({"id": "x", "project": "mock", "input": {}, "difficulty": "extreme"}, "t")
    with pytest.raises(DatasetError):
        validate_case({"project": "mock", "input": {}}, "t")


def test_profiles_select_subsets():
    cases = load_dataset(ROOT / "datasets" / "router.jsonl")
    quick = select_cases(cases, "quick")
    assert 0 < len(quick) <= 6
    assert all(c.difficulty in ("easy", "medium") for c in quick)
    assert len(select_cases(cases, "standard")) == len(cases)
    with pytest.raises(DatasetError):
        select_cases(cases, "enormous")
