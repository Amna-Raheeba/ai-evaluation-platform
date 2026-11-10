"""Orchestrator end-to-end (mock agent) + store round-trips + the
infra-vs-quality separation for the unavailable RAG adapter."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval_platform.adapter import AdapterRegistry
from eval_platform.adapters.mock import MockAdapter
from eval_platform.adapters.rag import RagAdapter
from eval_platform.orchestrator import Orchestrator
from eval_platform.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "test.db")


def _orchestrator(store, *adapters):
    registry = AdapterRegistry()
    for adapter in adapters:
        registry.register(adapter)
    return Orchestrator(store, registry)


def test_mock_run_end_to_end(store):
    orch = _orchestrator(store, MockAdapter())
    summary = orch.run_evaluation("mock-agent", ROOT / "datasets" / "mock.jsonl",
                                  profile="standard")
    assert summary["scored"] == 3 and summary["infra_errors"] == 0
    assert summary["pass_rate"] == 1.0
    assert summary["metric_means"]["required_facts"] == 1.0
    assert summary["latency_p95_s"] >= summary["latency_p50_s"]
    run = store.get_run(summary["run_id"])
    assert run["status"] == "complete"
    assert run["dataset_version"].startswith("mock@")
    assert len(store.get_test_results(summary["run_id"])) == 3
    assert store.get_metric_results(summary["run_id"])


def test_quick_profile_runs_subset(store):
    orch = _orchestrator(store, MockAdapter())
    summary = orch.run_evaluation("mock-agent", ROOT / "datasets" / "mock.jsonl",
                                  profile="quick")
    assert summary["cases_run"] <= 6


def test_repeats_are_recorded(store):
    orch = _orchestrator(store, MockAdapter())
    summary = orch.run_evaluation("mock-agent", ROOT / "datasets" / "mock.jsonl",
                                  profile="standard", repeats=2)
    assert summary["scored"] == 6
    assert summary["repeats"] == 2


def test_unavailable_agent_is_infra_never_scored(store, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    orch = _orchestrator(store, RagAdapter(repo_path=ROOT.parent / "production-rag-citations"))
    summary = orch.run_evaluation("rag-agent", ROOT / "datasets" / "rag.jsonl",
                                  profile="quick")
    assert summary["scored"] == 0
    assert summary["infra_errors"] == summary["cases_run"] > 0
    assert summary["pass_rate"] is None  # no quality verdict from an outage
    results = store.get_test_results(summary["run_id"])
    assert all(r["error_kind"] == "infra" for r in results)


def test_baseline_flag_sets_baseline(store):
    orch = _orchestrator(store, MockAdapter())
    summary = orch.run_evaluation("mock-agent", ROOT / "datasets" / "mock.jsonl",
                                  set_baseline_flag=True)
    baseline = store.get_baseline("mock-agent")
    assert baseline["id"] == summary["run_id"]


def test_store_agent_listing_and_trends(store):
    orch = _orchestrator(store, MockAdapter())
    orch.run_evaluation("mock-agent", ROOT / "datasets" / "mock.jsonl")
    agents = store.list_agents()
    assert any(a["name"] == "mock-agent" for a in agents)
    trends = store.trends("mock-agent")
    assert len(trends) == 1 and trends[0]["pass_rate"] == 1.0
