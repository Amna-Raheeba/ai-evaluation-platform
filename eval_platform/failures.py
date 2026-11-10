"""Failure Analysis Engine.

A failed score says *that* an agent failed; this module says *what
kind* of failure it was, groups similar failures into clusters, and
attaches a root-cause hypothesis (a hypothesis — a human confirms or
rejects it via feedback; it is never presented as a diagnosis).

Taxonomy: retrieval_failure, hallucination, citation_failure,
reasoning_failure, tool_failure, routing_failure, safety_failure,
performance_failure, instruction_failure.

Confirmed failures can be exported as draft golden cases to a drafts
JSONL file — datasets themselves are immutable once versioned, so new
cases land in drafts for review, never silently inside a dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import case_to_jsonl

HYPOTHESES = {
    "retrieval_failure": "Retriever returned no/irrelevant evidence for this input; check retrieval coverage and ranking for the case's terms.",
    "hallucination": "Output asserts facts absent from the retrieved sources; generation is not constrained to evidence.",
    "citation_failure": "Citations are missing, malformed, or do not support the claims they are attached to.",
    "reasoning_failure": "Evidence was available but the conclusion does not follow; check the reasoning/planning step.",
    "tool_failure": "A tool was called outside the allowed set or with wrong usage; check tool schemas and the agent's tool policy.",
    "routing_failure": "The router sent this prompt to the wrong tier; check complexity signals and tier thresholds.",
    "safety_failure": "The agent answered (or refused) against the safety expectation for this category; check guardrails.",
    "performance_failure": "Latency or cost exceeded the configured limit; check model choice and payload size.",
    "instruction_failure": "Output violated the required structure/instructions; check output formatting and schema enforcement.",
}


def classify_case(case, metric_results: list[dict], test_result: dict) -> str | None:
    """Return a failure category for a failed case, else None."""
    if test_result.get("error_kind"):
        return None  # infrastructure, not a quality failure
    failed = {m["metric_name"] for m in metric_results if m.get("passed") is False}
    failed.discard("source_diversity")  # informational metric, never a failure
    if not failed:
        return None
    scores = {m["metric_name"]: m.get("score") for m in metric_results}
    if "routing_accuracy" in failed:
        return "routing_failure"
    if case.category in ("safety", "adversarial") and "abstention_correctness" in failed:
        return "safety_failure"
    if "latency" in failed:
        return "performance_failure"
    if "tool_correctness" in failed:
        return "tool_failure"
    if "schema_validity" in failed:
        return "instruction_failure"
    if {"citation_presence", "citation_format", "citation_support",
            "citation_precision", "citation_coverage"} & failed:
        return "citation_failure"
    if "required_facts" in failed or "claim_support_rate" in failed:
        sources = test_result.get("payload", {}).get("sources", [])
        if not sources:
            return "retrieval_failure"
        if (scores.get("required_facts") or 0) < 0.5:
            return "hallucination"
        return "reasoning_failure"
    if "abstention_correctness" in failed:
        return "safety_failure"
    return "reasoning_failure"


def analyse_run(store, run_id: int, cases_by_id: dict) -> list[dict]:
    """Classify + cluster all failed cases of a run; persist clusters."""
    clusters: dict[tuple[str, str], dict] = {}
    results = {r["case_key"]: r for r in store.get_test_results(run_id)}
    for case_key, test_result in results.items():
        case = cases_by_id.get(case_key)
        if case is None:
            continue
        metrics = store.get_metric_results(run_id, case_key)
        category = classify_case(case, metrics, test_result)
        if category is None:
            continue
        failed_metrics = sorted(m["metric_name"] for m in metrics if m.get("passed") is False)
        signature = "+".join(failed_metrics)
        key = (category, signature)
        cluster = clusters.setdefault(key, {"category": category, "signature": signature,
                                            "hypothesis": HYPOTHESES[category],
                                            "case_keys": []})
        if case_key not in cluster["case_keys"]:
            cluster["case_keys"].append(case_key)
    out = sorted(clusters.values(), key=lambda c: -len(c["case_keys"]))
    if out:
        store.save_failure_clusters(run_id, out)
    return out


def export_failure_as_case(store, run_id: int, case_key: str,
                           cases_by_id: dict, drafts_path: str | Path) -> dict:
    """Append the failed case to a drafts JSONL (for human review before
    it can join a versioned dataset). Returns the exported record."""
    case = cases_by_id[case_key]
    record = json.loads(case_to_jsonl(case))
    record["tags"] = sorted(set(record.get("tags", [])) | {"from_failure", f"run:{run_id}"})
    record["id"] = f"{case.id}-reg"
    path = Path(drafts_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    store.audit("failure_exported_as_case", {"run_id": run_id, "case": case_key,
                                             "draft": str(path)})
    return record
