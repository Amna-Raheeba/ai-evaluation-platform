"""Regression Detection Engine.

Compares a new evaluation run against a saved baseline:

  - per-metric means and deltas (direction-aware: for latency, cost and
    token usage a *negative* delta is an improvement),
  - paired per-case deltas with a 95% confidence interval
    (normal approximation: mean ± 1.96 * sd / sqrt(n); the cases are
    the same in both runs, so pairing removes case-difficulty noise),
  - quality gates from evaluation_configs/quality_gates.json:
    minimum_score and maximum_regression per quality metric, ceilings
    for latency/cost,
  - a decision: PASS, FAIL, or INCONCLUSIVE.

Noise discipline: a metric only counts as regressed when the gate is
violated AND the paired CI excludes "no change" in the bad direction.
If the new run had too many infrastructure errors, or there are too
few paired cases to say anything, the decision is INCONCLUSIVE — an
outage must never be reported as a quality regression.
"""
from __future__ import annotations

import math
from statistics import mean, stdev

from .metrics import HIGHER_BETTER, LOWER_BETTER

MIN_PAIRED = 3
MAX_INFRA_RATE = 0.2


def _metric_scores(store, run_id: int) -> dict[str, dict[str, list[float]]]:
    """metric -> case_key -> [scores across repeats]"""
    out: dict[str, dict[str, list[float]]] = {}
    for row in store.get_metric_results(run_id):
        if row["score"] is None:
            continue
        out.setdefault(row["metric_name"], {}).setdefault(row["case_key"], []).append(row["score"])
    return out


def _ci95(deltas: list[float]) -> tuple[float, float]:
    if len(deltas) < 2:
        return (deltas[0], 0.0) if deltas else (0.0, 0.0)
    m = mean(deltas)
    half = 1.96 * stdev(deltas) / math.sqrt(len(deltas))
    return m, half


def compare_runs(store, baseline_run_id: int, new_run_id: int,
                 gates: dict) -> dict:
    base = _metric_scores(store, baseline_run_id)
    new = _metric_scores(store, new_run_id)
    base_run = store.get_run(baseline_run_id)
    new_run = store.get_run(new_run_id)
    reasons: list[str] = []
    per_metric: dict[str, dict] = {}
    gate_failures: list[str] = []

    infra_rate = (new_run["summary"].get("infra_errors", 0)
                  / max(1, new_run["summary"].get("cases_run", 1)))
    infra_override = infra_rate > MAX_INFRA_RATE
    if infra_override:
        reasons.append(f"infrastructure errors in {infra_rate:.0%} of new-run cases "
                       f"(> {MAX_INFRA_RATE:.0%}); cannot judge quality")

    gate_metrics = gates.get("metrics", {})
    ceilings = gates.get("ceilings", {})
    for metric in sorted(set(base) | set(new)):
        b_cases, n_cases = base.get(metric, {}), new.get(metric, {})
        b_mean = mean([mean(v) for v in b_cases.values()]) if b_cases else None
        n_mean = mean([mean(v) for v in n_cases.values()]) if n_cases else None
        paired_keys = sorted(set(b_cases) & set(n_cases))
        deltas = [mean(n_cases[k]) - mean(b_cases[k]) for k in paired_keys]
        delta, ci_half = _ci95(deltas)
        entry = {"baseline_mean": None if b_mean is None else round(b_mean, 4),
                 "new_mean": None if n_mean is None else round(n_mean, 4),
                 "delta": round(delta, 4) if paired_keys else None,
                 "ci95_half_width": round(ci_half, 4),
                 "paired_cases": len(paired_keys),
                 "direction": "higher" if metric in HIGHER_BETTER else "lower",
                 "gate": None}
        gate = gate_metrics.get(metric)
        if gate and n_mean is not None:
            problems = []
            if "minimum_score" in gate and metric in HIGHER_BETTER and n_mean < gate["minimum_score"]:
                problems.append(f"new mean {n_mean:.3f} < minimum {gate['minimum_score']}")
            if "maximum_regression" in gate and paired_keys:
                tol = gate["maximum_regression"]
                bad = delta < -tol if metric in HIGHER_BETTER else delta > tol
                ci_excludes_zero = (delta + ci_half < 0) if metric in HIGHER_BETTER else (delta - ci_half > 0)
                if bad and ci_excludes_zero:
                    problems.append(f"regression {delta:+.3f} beyond tolerance {tol} "
                                    f"(95% CI excludes no-change, normal approx, n={len(paired_keys)})")
                elif bad:
                    reasons.append(f"{metric}: delta {delta:+.3f} beyond tolerance but CI "
                                   f"includes no-change — treated as noise, not a regression")
            if problems:
                entry["gate"] = "FAIL"
                gate_failures.extend(f"{metric}: {p}" for p in problems)
            else:
                entry["gate"] = "PASS"
        if metric in ceilings and n_mean is not None and n_mean > ceilings[metric]:
            entry["gate"] = "FAIL"
            gate_failures.append(f"{metric}: new mean {n_mean:.4f} exceeds ceiling {ceilings[metric]}")
        per_metric[metric] = entry

    paired_counts = [v["paired_cases"] for v in per_metric.values() if v["paired_cases"]]
    if paired_counts and max(paired_counts) < MIN_PAIRED:
        reasons.append(f"only {max(paired_counts)} paired cases (< {MIN_PAIRED}); "
                       f"statistical comparison not meaningful")

    if infra_override:
        # An outage is not a quality regression: gate numbers computed
        # from the surviving cases are not trustworthy enough to block.
        decision = "INCONCLUSIVE"
        if gate_failures:
            reasons = gate_failures + reasons
    elif gate_failures:
        decision = "FAIL"
        reasons = gate_failures + reasons
    elif reasons:
        decision = "INCONCLUSIVE"
    else:
        decision = "PASS"
    return {"decision": decision, "baseline_run_id": baseline_run_id,
            "new_run_id": new_run_id, "per_metric": per_metric,
            "reasons": reasons, "ci_method": "normal approximation on paired per-case deltas",
            "infra_error_rate_new_run": round(infra_rate, 4)}
