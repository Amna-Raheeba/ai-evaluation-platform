"""Golden Dataset Engine.

A golden dataset is a versioned JSONL file of test cases:

    {"id": "rag-001", "project": "rag",
     "input": {"question": "..."},
     "expected": {"required_facts": ["..."], "must_include_citations": true},
     "category": "factual_qa", "difficulty": "medium", "tags": ["citations"]}

Versioning: a dataset version is ``<name>@<sha8>`` where sha8 is the
first 8 hex chars of the SHA-256 of the file's exact bytes. Once a
dataset version has been used in an evaluation run it is treated as
immutable — edit the file and you have a *new* version, so old runs
stay reproducible. Validation rejects duplicate ids, unknown fields,
and malformed expectations before anything runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CASE_FIELDS = {"id", "project", "input", "expected", "category", "difficulty", "tags"}
EXPECTED_FIELDS = {"required_facts", "must_include_citations",
                   "expected_tier", "should_abstain", "has_contradictions",
                   "allowed_tools", "max_latency_s", "min_sources"}
DIFFICULTIES = ("easy", "medium", "hard")


class DatasetError(Exception):
    pass


@dataclass
class TestCase:
    id: str
    project: str
    input: dict
    expected: dict = field(default_factory=dict)
    category: str = "general"
    difficulty: str = "medium"
    tags: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def dataset_version(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return f"{path.stem}@{digest}"


def validate_case(obj: Any, where: str) -> TestCase:
    if not isinstance(obj, dict):
        raise DatasetError(f"{where}: case must be a JSON object")
    unknown = set(obj) - CASE_FIELDS
    if unknown:
        raise DatasetError(f"{where}: unknown fields {sorted(unknown)}")
    for required in ("id", "project", "input"):
        if required not in obj:
            raise DatasetError(f"{where}: missing required field {required!r}")
    if not isinstance(obj["input"], dict):
        raise DatasetError(f"{where}: input must be an object")
    expected = obj.get("expected", {})
    if not isinstance(expected, dict):
        raise DatasetError(f"{where}: expected must be an object")
    unknown_exp = set(expected) - EXPECTED_FIELDS
    if unknown_exp:
        raise DatasetError(f"{where}: unknown expected fields {sorted(unknown_exp)}")
    if "required_facts" in expected and not (
            isinstance(expected["required_facts"], list)
            and all(isinstance(f, str) for f in expected["required_facts"])):
        raise DatasetError(f"{where}: required_facts must be a list of strings")
    difficulty = obj.get("difficulty", "medium")
    if difficulty not in DIFFICULTIES:
        raise DatasetError(f"{where}: difficulty must be one of {DIFFICULTIES}")
    return TestCase(id=str(obj["id"]), project=str(obj["project"]),
                    input=obj["input"], expected=expected,
                    category=str(obj.get("category", "general")),
                    difficulty=difficulty,
                    tags=list(obj.get("tags", [])), raw=obj)


def load_dataset(path: str | Path) -> list[TestCase]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")
    cases: list[TestCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        case = validate_case(obj, f"{path}:{lineno}")
        if case.id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate test case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError(f"{path}: dataset is empty")
    return cases


def select_cases(cases: list[TestCase], profile: str) -> list[TestCase]:
    """Evaluation profiles. quick = fast smoke subset, standard = all,
    full = all (repeats are the orchestrator's concern)."""
    if profile == "quick":
        subset = [c for c in cases if c.difficulty in ("easy", "medium")]
        return subset[:6] if subset else cases[:6]
    if profile in ("standard", "full", "regression"):
        return list(cases)
    raise DatasetError(f"unknown profile {profile!r}")


def case_to_jsonl(case: TestCase) -> str:
    obj = {"id": case.id, "project": case.project, "input": case.input,
           "expected": case.expected, "category": case.category,
           "difficulty": case.difficulty, "tags": case.tags}
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
