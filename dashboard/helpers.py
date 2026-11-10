"""Data-shaping helpers for the Streamlit dashboard.

Everything the dashboard displays is shaped here first: overview
counts, per-agent comparison rows, per-case detail, regression
comparisons against the stored baseline, failure-cluster aggregates,
and cost rows. This module is pure Python on purpose — no streamlit
import — so the shaping logic is unit-tested in
``tests/test_dashboard_helpers.py`` and the app file stays a thin
presentation layer.

The dashboard reads the same SQLite store the CLI writes
(``eval_platform.store.Store``). Reading helpers never create the
database file as a side effect: when the file does not exist they
return empty/zero shapes, and only ``run_mock_evaluation`` opens the
store for writing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_platform.adapter import AdapterRegistry  # noqa: E402
from eval_platform.adapters.mock import MockAdapter  # noqa: E402
from eval_platform.orchestrator import Orchestrator, load_gates  # noqa: E402
from eval_platform.regression import compare_runs  # noqa: E402
from eval_platform.store import Store  # noqa: E402


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    return default_db_path()


def _resolve_baseline_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return PROJECT_ROOT / "results" / "baseline_summary.json"


def default_db_path() -> Path:
    """Database path: ``$EVAL_PLATFORM_DB`` or project-root/eval_platform.db."""
    env = os.environ.get("EVAL_PLATFORM_DB")
    if env:
        return Path(env)
    return PROJECT_ROOT / "eval_platform.db"


def load_baseline_summary(path: str | Path | None = None) -> dict:
    """Load ``results/baseline_summary.json``; ``{}`` if missing/unparseable."""
    p = _resolve_baseline_path(path)
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _regression_decisions(db_path: Path) -> dict[str, int]:
    decisions = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT decision, COUNT(*) FROM regressions GROUP BY decision"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return decisions
    for decision, count in rows:
        if decision in decisions:
            decisions[decision] = int(count)
    return decisions


def overview(db_path: str | Path | None = None) -> dict[str, Any]:
    """Store-level counts for the overview strip.

    Returns ``db_exists=False`` and zeros when the file is absent,
    without creating it — checking existence happens before any
    ``Store``/sqlite connection is opened.
    """
    p = _resolve_db_path(db_path)
    if not p.exists():
        return {
            "db_exists": False,
            "agents": [],
            "total_runs": 0,
            "runs_by_status": {},
            "decisions": {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0},
            "total_cost_usd": 0.0,
            "latest_run_id": None,
        }
    store = Store(p)
    try:
        agents = [a["name"] for a in store.list_agents()]
        runs = store.list_runs()
        runs_by_status: dict[str, int] = {}
        total_cost = 0.0
        for run in runs:
            status = run.get("status") or "unknown"
            runs_by_status[status] = runs_by_status.get(status, 0) + 1
            cost = (run.get("summary") or {}).get("total_cost_usd")
            if isinstance(cost, (int, float)):
                total_cost += float(cost)
        latest_run_id = max((r["id"] for r in runs), default=None)
        decisions = _regression_decisions(p)
        return {
            "db_exists": True,
            "agents": agents,
            "total_runs": len(runs),
            "runs_by_status": runs_by_status,
            "decisions": decisions,
            "total_cost_usd": round(total_cost, 6),
            "latest_run_id": latest_run_id,
        }
    finally:
        store.close()


def agent_comparison(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """One row per agent, taken from that agent's latest run."""
    p = _resolve_db_path(db_path)
    if not p.exists():
        return []
    store = Store(p)
    try:
        runs = store.list_runs()
        agent_names = sorted({r["agent_name"] for r in runs})
        rows: list[dict[str, Any]] = []
        for name in agent_names:
            latest = store.latest_run(name)
            if latest is None:
                continue
            summary = latest.get("summary") or {}
            rows.append({
                "agent": name,
                "run_id": latest["id"],
                "agent_version": latest.get("agent_version"),
                "dataset_version": latest.get("dataset_version"),
                "profile": latest.get("profile"),
                "status": latest.get("status"),
                "pass_rate": summary.get("pass_rate"),
                "scored": summary.get("scored"),
                "infra_errors": summary.get("infra_errors"),
                "total_cost_usd": summary.get("total_cost_usd"),
                "latency_p95_s": summary.get("latency_p95_s"),
                "metric_means": summary.get("metric_means", {}),
            })
        return rows
    finally:
        store.close()


def run_options(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Run list for the detail selector, newest first."""
    p = _resolve_db_path(db_path)
    if not p.exists():
        return []
    store = Store(p)
    try:
        runs = store.list_runs()
        rows = []
        for run in runs:
            summary = run.get("summary") or {}
            rows.append({
                "id": run["id"],
                "agent_name": run.get("agent_name"),
                "profile": run.get("profile"),
                "status": run.get("status"),
                "started_at": run.get("started_at"),
                "pass_rate": summary.get("pass_rate"),
            })
        rows.sort(key=lambda r: r["id"], reverse=True)
        return rows
    finally:
        store.close()


def run_case_rows(db_path: str | Path | None, run_id: int) -> list[dict[str, Any]]:
    """Per-case rows for one run, joining test results and metric results."""
    p = _resolve_db_path(db_path)
    if not p.exists():
        return []
    store = Store(p)
    try:
        test_results = store.get_test_results(run_id)
        metric_results = store.get_metric_results(run_id)
        by_test_id: dict[Any, list[dict[str, Any]]] = {}
        by_case: dict[str, list[dict[str, Any]]] = {}
        for m in metric_results:
            shaped = {
                "metric_name": m.get("metric_name"),
                "score": m.get("score"),
                "passed": m.get("passed"),
                "explanation": m.get("explanation"),
                "evaluator_type": m.get("evaluator_type"),
            }
            tid = m.get("test_result_id")
            if tid is not None:
                by_test_id.setdefault(tid, []).append(shaped)
            by_case.setdefault(m.get("case_key"), []).append(shaped)
        rows: list[dict[str, Any]] = []
        for tr in test_results:
            metrics = by_test_id.get(tr.get("id"))
            if metrics is None:
                # Fallback for rows written without a test_result link.
                metrics = by_case.get(tr.get("case_key"), [])
            rows.append({
                "case_key": tr.get("case_key"),
                "status": tr.get("status"),
                "repeat_idx": tr.get("repeat_idx"),
                "execution_time_s": tr.get("execution_time_s"),
                "error_kind": tr.get("error_kind"),
                "metrics": list(metrics),
            })
        return rows
    finally:
        store.close()


def regression_rows(db_path: str | Path | None, agent_name: str) -> dict[str, Any] | None:
    """Compare an agent's latest run against its stored baseline.

    Returns ``None`` when the agent has no baseline, or when the
    latest run *is* the baseline run (nothing to compare yet).
    """
    p = _resolve_db_path(db_path)
    if not p.exists():
        return None
    store = Store(p)
    try:
        baseline = store.get_baseline(agent_name)
        if baseline is None:
            return None
        latest = store.latest_run(agent_name)
        if latest is None or latest["id"] == baseline["id"]:
            return None
        report = compare_runs(store, baseline["id"], latest["id"], load_gates())
        rows = []
        for metric in sorted(report.get("per_metric", {})):
            entry = report["per_metric"][metric]
            rows.append({
                "metric": metric,
                "baseline_mean": entry.get("baseline_mean"),
                "new_mean": entry.get("new_mean"),
                "delta": entry.get("delta"),
                "ci95_half_width": entry.get("ci95_half_width"),
                "paired_cases": entry.get("paired_cases"),
                "gate": entry.get("gate"),
            })
        return {
            "decision": report.get("decision"),
            "baseline_run_id": baseline["id"],
            "new_run_id": latest["id"],
            "reasons": report.get("reasons", []),
            "rows": rows,
        }
    finally:
        store.close()


def failure_clusters(db_path: str | Path | None, run_id: int) -> list[dict[str, Any]]:
    """Shaped failure-cluster rows for one run."""
    p = _resolve_db_path(db_path)
    if not p.exists():
        return []
    store = Store(p)
    try:
        clusters = store.get_failure_clusters(run_id)
        rows = []
        for c in clusters:
            case_keys = c.get("case_keys") or []
            rows.append({
                "category": c.get("category"),
                "signature": c.get("signature"),
                "case_count": c.get("case_count", len(case_keys)),
                "hypothesis": c.get("hypothesis"),
                "case_keys": list(case_keys),
            })
        return rows
    finally:
        store.close()


def failure_category_totals(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Failure clusters aggregated over every stored run, by category."""
    p = _resolve_db_path(db_path)
    if not p.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT category, COUNT(*) AS clusters, "
                "COALESCE(SUM(case_count), 0) AS cases "
                "FROM failure_clusters GROUP BY category ORDER BY category"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [
        {"category": category, "clusters": int(clusters), "cases": int(cases)}
        for category, clusters, cases in rows
    ]


def cost_rows(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Per-agent cost/performance aggregates over all stored runs."""
    p = _resolve_db_path(db_path)
    if not p.exists():
        return []
    store = Store(p)
    try:
        runs = store.list_runs()
        by_agent: dict[str, list[dict]] = {}
        for run in runs:
            by_agent.setdefault(run["agent_name"], []).append(run)
        rows: list[dict[str, Any]] = []
        for agent in sorted(by_agent):
            agent_runs = sorted(by_agent[agent], key=lambda r: r["id"])
            scored = 0
            total_cost = 0.0
            pass_rates: list[float] = []
            for run in agent_runs:
                summary = run.get("summary") or {}
                if isinstance(summary.get("scored"), (int, float)):
                    scored += int(summary["scored"])
                if isinstance(summary.get("total_cost_usd"), (int, float)):
                    total_cost += float(summary["total_cost_usd"])
                if isinstance(summary.get("pass_rate"), (int, float)):
                    pass_rates.append(float(summary["pass_rate"]))
            latest_summary = agent_runs[-1].get("summary") or {}
            rows.append({
                "agent": agent,
                "runs": len(agent_runs),
                "scored_cases": scored,
                "total_cost_usd": round(total_cost, 6),
                "mean_pass_rate": round(statistics.mean(pass_rates), 4) if pass_rates else None,
                "latest_latency_p95_s": latest_summary.get("latency_p95_s"),
            })
        return rows
    finally:
        store.close()


def run_mock_evaluation(db_path: str | Path | None = None) -> dict[str, Any]:
    """Run the healthy MockAdapter (quick profile) into the store at db_path.

    This genuinely executes the orchestrator — it is the same code
    path as ``python -m eval_platform run --agent mock-agent`` — and
    returns the run summary (including ``run_id``).
    """
    p = _resolve_db_path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    store = Store(p)
    try:
        registry = AdapterRegistry()
        registry.register(MockAdapter())
        orchestrator = Orchestrator(store, registry)
        return orchestrator.run_evaluation(
            "mock-agent", PROJECT_ROOT / "datasets" / "mock.jsonl", profile="quick"
        )
    finally:
        store.close()
