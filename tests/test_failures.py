"""Failure taxonomy, clustering, and failure -> draft golden case."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_platform import failures as F
from eval_platform.adapter import AdapterRegistry
from eval_platform.adapters.mock import MockAdapter
from eval_platform.datasets import load_dataset
from eval_platform.orchestrator import Orchestrator
from eval_platform.store import Store

ROOT = Path(__file__).resolve().parents[1]


def _case(case_id="c1", project="rag", category="factual_qa"):
    cases = {c.id: c for c in load_dataset(ROOT / "datasets" / "mock.jsonl")}
    case = cases["mock-001"]
    case.id, case.project, case.category = case_id, project, category
    return case


def _metrics(**name_to_passed):
    return [{"metric_name": n, "passed": p, "score": 0.0 if p is False else 1.0}
            for n, p in name_to_passed.items()]


def test_routing_failure_classification():
    case = _case(project="router")
    assert F.classify_case(case, _metrics(routing_accuracy=False),
                           {"error_kind": None, "payload": {}}) == "routing_failure"


def test_citation_failure_classification():
    case = _case()
    assert F.classify_case(case, _metrics(citation_presence=False),
                           {"error_kind": None, "payload": {}}) == "citation_failure"


def test_retrieval_vs_hallucination():
    case = _case()
    metrics = _metrics(required_facts=False)
    no_sources = {"error_kind": None, "payload": {"sources": []}}
    with_sources = {"error_kind": None, "payload": {"sources": [{"id": "1"}]}}
    assert F.classify_case(case, metrics, no_sources) == "retrieval_failure"
    assert F.classify_case(case, metrics, with_sources) == "hallucination"


def test_infra_errors_are_not_failures():
    case = _case()
    assert F.classify_case(case, _metrics(required_facts=False),
                           {"error_kind": "infra", "payload": {}}) is None
    assert F.classify_case(case, _metrics(required_facts=True),
                           {"error_kind": None, "payload": {}}) is None


def test_degraded_run_produces_clusters_and_export(tmp_path):
    store = Store(tmp_path / "t.db")
    registry = AdapterRegistry()
    registry.register(MockAdapter(degraded=True))
    summary = Orchestrator(store, registry).run_evaluation(
        "mock-agent", ROOT / "datasets" / "mock.jsonl", profile="standard")
    clusters = store.get_failure_clusters(summary["run_id"])
    assert clusters, "degraded mock should produce failure clusters"
    assert all(c["hypothesis"] for c in clusters)
    cases_by_id = {c.id: c for c in load_dataset(ROOT / "datasets" / "mock.jsonl")}
    drafts = tmp_path / "drafts.jsonl"
    record = F.export_failure_as_case(store, summary["run_id"],
                                      clusters[0]["case_keys"][0], cases_by_id, drafts)
    assert record["id"].endswith("-reg")
    written = [json.loads(l) for l in drafts.read_text().splitlines()]
    assert written[0]["id"] == record["id"]
    assert "from_failure" in written[0]["tags"]
