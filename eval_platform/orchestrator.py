"""Evaluation Orchestrator.

Pipeline per run: load dataset version → select profile cases → for
each case (× repeats) call the agent through its adapter
(timeout-guarded) → run metrics + judges → persist everything →
aggregate summary → failure analysis.

Partial failures are normal: an adapter timing out or an agent being
unavailable marks that case as an infrastructure error and the run
still completes. Infra errors are counted separately in the summary —
they are never averaged into quality scores.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from statistics import mean
from typing import Any

from . import failures as failures_mod
from .adapter import AdapterError, AdapterRegistry, check_contract
from .datasets import dataset_version, load_dataset, select_cases
from .judges import LexicalJudge
from .metrics import evaluate_case
from .store import Store

ROOT = Path(__file__).resolve().parent.parent


def load_pricing() -> dict:
    path = ROOT / "evaluation_configs" / "pricing.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("models", {})
    return {}


def load_gates() -> dict:
    path = ROOT / "evaluation_configs" / "quality_gates.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"metrics": {}, "ceilings": {}}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
    return values[idx]


class Orchestrator:
    def __init__(self, store: Store, registry: AdapterRegistry,
                 pricing: dict | None = None, judges: list | None = None):
        self.store = store
        self.registry = registry
        self.pricing = pricing if pricing is not None else load_pricing()
        self.judges = judges if judges is not None else [LexicalJudge()]

    def _call_adapter(self, adapter, input_data: dict, timeout_s: float) -> dict:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(adapter.run, dict(input_data))
            try:
                return future.result(timeout=timeout_s)
            except FutureTimeout as exc:
                raise AdapterError("timeout", f"adapter exceeded {timeout_s}s") from exc

    def run_evaluation(self, agent_name: str, dataset_path: str | Path,
                       profile: str = "standard", repeats: int = 1,
                       timeout_s: float = 60.0, set_baseline_flag: bool = False) -> dict:
        dataset_path = Path(dataset_path)
        cases_all = load_dataset(dataset_path)
        version = dataset_version(dataset_path)
        content_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        cases = select_cases(cases_all, profile)
        if profile == "full":
            repeats = max(repeats, 2)
        adapter = self.registry.get(agent_name)
        info = adapter.get_agent_info()
        project = cases_all[0].project if cases_all else ""
        self.store.register_dataset(dataset_path.stem, project, version,
                                    content_hash, [c.raw for c in cases_all])
        self.store.save_quality_gate_policy("default", load_gates())
        config = {"project": project, "dataset_path": str(dataset_path),
                  "dataset_version": version, "profile": profile,
                  "repeats": repeats, "timeout_s": timeout_s,
                  "evaluator_versions": {"metrics": "1.0", "judges": ["lexical@1.0"]},
                  "model": info.get("model", "")}
        run_key = f"{agent_name}-{version}-{profile}-{uuid.uuid4().hex[:6]}"
        run_id = self.store.start_run(run_key, agent_name, info["version"],
                                      version, profile, repeats, config)
        scored, passed_cases, infra_errors = 0, 0, 0
        metric_scores: dict[str, list[float]] = {}
        latencies: list[float] = []
        total_cost = 0.0
        try:
            for case in cases:
                for rep in range(repeats):
                    try:
                        result = self._call_adapter(adapter, case.input, timeout_s)
                        if result.get("status") == "unavailable":
                            reason = result.get("metadata", {}).get(
                                "unavailable_reason", "agent unavailable")
                            tr_id = self.store.save_test_result(
                                run_id, case.id, rep, "infra_error", result,
                                error_kind="infra", error_message=reason)
                            infra_errors += 1
                            continue
                        check_contract(result)
                    except AdapterError as exc:
                        self.store.save_test_result(
                            run_id, case.id, rep, "infra_error", None,
                            error_kind=exc.kind, error_message=exc.message)
                        infra_errors += 1
                        continue
                    tr_id = self.store.save_test_result(run_id, case.id, rep, "scored", result)
                    metrics = evaluate_case(case, result, agent_id=agent_name,
                                            pricing=self.pricing, judges=self.judges)
                    self.store.save_metric_results(run_id, tr_id, metrics)
                    scored += 1
                    if not any(m.passed is False for m in metrics):
                        passed_cases += 1
                    for m in metrics:
                        if m.score is not None:
                            metric_scores.setdefault(m.metric_name, []).append(m.score)
                        if m.metric_name == "latency":
                            latencies.append(m.score or 0.0)
                        if m.metric_name == "cost_estimate":
                            total_cost += m.score or 0.0
            summary = {
                "agent": agent_name, "agent_version": info["version"],
                "dataset_version": version, "profile": profile, "repeats": repeats,
                "cases_run": len(cases) * repeats, "scored": scored,
                "infra_errors": infra_errors,
                "pass_rate": round(passed_cases / scored, 4) if scored else None,
                "metric_means": {k: round(mean(v), 4) for k, v in sorted(metric_scores.items())},
                "total_cost_usd": round(total_cost, 6),
                "cost_per_successful_task": round(total_cost / passed_cases, 6) if passed_cases else None,
                "latency_p50_s": round(_percentile(latencies, 0.5), 4),
                "latency_p95_s": round(_percentile(latencies, 0.95), 4),
            }
            self.store.finish_run(run_id, "complete", summary)
            cases_by_id = {c.id: c for c in cases_all}
            clusters = failures_mod.analyse_run(self.store, run_id, cases_by_id)
            summary["failure_clusters"] = len(clusters)
            self.store.audit("evaluation_run", {"run_id": run_id, "agent": agent_name,
                                                "summary": summary})
            if set_baseline_flag:
                self.store.set_baseline(agent_name, run_id)
            summary["run_id"] = run_id
            return summary
        except Exception:
            self.store.finish_run(run_id, "failed", {"error": "orchestrator exception"})
            raise


def report_markdown(store: Store, run_id: int) -> str:
    run = store.get_run(run_id)
    if not run:
        return f"# Evaluation run {run_id}\n\nRun not found."
    s = run["summary"]
    lines = [f"# Evaluation Report — run {run_id}",
             "",
             f"- Agent: **{run['agent_name']}** v{run['agent_version']}",
             f"- Dataset: {run['dataset_version']} · profile: {run['profile']} · repeats: {run['repeats']}",
             f"- Status: {run['status']} · cases: {s.get('cases_run')} · scored: {s.get('scored')} · infra errors: {s.get('infra_errors')}",
             f"- Pass rate: {s.get('pass_rate')} · total estimated cost: ${s.get('total_cost_usd')} · latency p50/p95: {s.get('latency_p50_s')}s / {s.get('latency_p95_s')}s",
             "",
             "## Metric means",
             "",
             "| Metric | Mean |", "|---|---|"]
    for metric, value in s.get("metric_means", {}).items():
        lines.append(f"| {metric} | {value} |")
    clusters = store.get_failure_clusters(run_id)
    if clusters:
        lines += ["", "## Failure clusters", ""]
        for c in clusters:
            lines.append(f"- **{c['category']}** ({c['signature']}): {len(c['case_keys'])} case(s) "
                         f"{', '.join(c['case_keys'])} — hypothesis: {c['hypothesis']}")
    lines += ["", "_Costs are estimates from evaluation_configs/pricing.json, not invoices. "
              "Offline judge scores are lexical proxies._", ""]
    return "\n".join(lines)
