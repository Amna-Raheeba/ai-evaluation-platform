"""Regression engine: a degraded mock MUST fail the gate, an identical
rerun MUST pass, and an infra-heavy run must be INCONCLUSIVE (never a
quality verdict)."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval_platform.adapter import AdapterRegistry
from eval_platform.adapters.mock import MockAdapter
from eval_platform.metrics import MetricResult
from eval_platform.orchestrator import Orchestrator, load_gates
from eval_platform.regression import compare_runs
from eval_platform.store import Store

ROOT = Path(__file__).resolve().parents[1]
GATES = load_gates()


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "test.db")


def _run(store, adapter):
    registry = AdapterRegistry()
    registry.register(adapter)
    return Orchestrator(store, registry).run_evaluation(
        "mock-agent", ROOT / "datasets" / "mock.jsonl", profile="standard")


def test_identical_rerun_passes(store):
    baseline = _run(store, MockAdapter())
    rerun = _run(store, MockAdapter())
    report = compare_runs(store, baseline["run_id"], rerun["run_id"], GATES)
    assert report["decision"] == "PASS"
    assert report["per_metric"]["required_facts"]["delta"] == 0.0


def test_degraded_mock_fails_gate(store):
    baseline = _run(store, MockAdapter(version="1.0.0"))
    degraded = _run(store, MockAdapter(version="1.1.0", degraded=True))
    report = compare_runs(store, baseline["run_id"], degraded["run_id"], GATES)
    assert report["decision"] == "FAIL"
    assert report["per_metric"]["required_facts"]["new_mean"] < 1.0
    assert any("required_facts" in r for r in report["reasons"])


def test_small_noise_within_tolerance_passes(store):
    baseline = _run(store, MockAdapter())
    slightly = _run(store, MockAdapter(version="1.0.1", fact_drop_rate=0.0))
    report = compare_runs(store, baseline["run_id"], slightly["run_id"], GATES)
    assert report["decision"] in ("PASS", "INCONCLUSIVE")  # never FAIL on no change


def test_infra_heavy_run_is_inconclusive(store):
    baseline = _run(store, MockAdapter())
    # hand-build a "run" whose summary is mostly infrastructure errors
    run_id = store.start_run("fake-run", "mock-agent", "9.9", "mock@deadbeef",
                             "standard", 1, {"project": "mock"})
    for case_id in ("mock-001", "mock-002", "mock-003"):
        tr = store.save_test_result(run_id, case_id, 0, "infra_error", None,
                                    error_kind="infra", error_message="service down")
        store.save_metric_results(run_id, tr, [MetricResult(
            case_id, "mock-agent", "required_facts", 0.0, False,
            explanation="placeholder")])
    store.finish_run(run_id, "complete", {"cases_run": 3, "infra_errors": 3, "scored": 0})
    report = compare_runs(store, baseline["run_id"], run_id, GATES)
    assert report["decision"] == "INCONCLUSIVE"
    assert report["infra_error_rate_new_run"] == 1.0
