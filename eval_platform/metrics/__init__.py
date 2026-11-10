"""Evaluation Engine: deterministic metrics + project-specific metrics.

No single overall score: every metric is recorded independently with
its evaluator type/version and a human-readable explanation, so a
failure can be diagnosed instead of just observed.

Deterministic metrics (evaluator_type="deterministic", version 1.0):
  schema_validity, required_facts, citation_presence, citation_format,
  citation_support, abstention_correctness, tool_correctness,
  latency, token_usage, cost_estimate

Project-specific metrics:
  rag:      citation_precision, citation_coverage
  research: claim_support_rate, source_diversity, contradiction_surfaced
  router:   routing_accuracy, expensive_call_rate

Directions: HIGHER_BETTER vs LOWER_BETTER — the regression engine uses
this to interpret deltas; cost/latency improvements are negative deltas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..datasets import TestCase

EVALUATOR_VERSION = "1.0"

HIGHER_BETTER = {"schema_validity", "required_facts", "citation_presence",
                 "citation_format", "citation_support", "abstention_correctness",
                 "tool_correctness", "citation_precision", "citation_coverage",
                 "claim_support_rate", "source_diversity",
                 "contradiction_surfaced", "routing_accuracy",
                 "lexical_faithfulness", "lexical_relevance"}
LOWER_BETTER = {"latency", "token_usage", "cost_estimate", "expensive_call_rate"}

_CITATION_RE = re.compile(r"\[(\d+|S\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")
REFUSAL_PHRASES = ("insufficient evidence", "could not find enough evidence",
                   "i will not guess", "not enough evidence", "cannot answer",
                   "i don't know", "no relevant evidence")


@dataclass
class MetricResult:
    test_case_id: str
    agent_id: str
    metric_name: str
    score: float | None
    passed: bool | None
    evaluator_type: str = "deterministic"
    evaluator_version: str = EVALUATOR_VERSION
    explanation: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"test_case_id": self.test_case_id, "agent_id": self.agent_id,
                "metric_name": self.metric_name, "score": self.score,
                "passed": self.passed, "evaluator_type": self.evaluator_type,
                "evaluator_version": self.evaluator_version,
                "explanation": self.explanation, "raw": self.raw}


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def is_abstention(result: dict) -> bool:
    if result.get("status") == "abstained":
        return True
    out = result.get("output", "").lower()
    return any(p in out for p in REFUSAL_PHRASES)


def citation_markers(output: str) -> list[str]:
    return _CITATION_RE.findall(output or "")


def claims_with_citations(output: str) -> list[tuple[str, list[str]]]:
    """Split output into sentences/lines, each with its citation markers.

    Citations written after the punctuation ("Sentence. [1]") are first
    re-attached to their sentence ("Sentence [1].") so the marker is not
    orphaned into a fragment of its own."""
    text = re.sub(r"([.!?])\s+(\[\w+\])", r" \2\1", output or "")
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n", text) if p.strip()]
    return [(p, citation_markers(p)) for p in parts]


# ----------------------------------------------------------------------
# deterministic metrics
# ----------------------------------------------------------------------

def m_schema_validity(case: TestCase, result: dict, ctx: dict) -> MetricResult:
    from ..adapter import STANDARD_KEYS
    missing = [k for k in STANDARD_KEYS if k not in result]
    ok = not missing and isinstance(result.get("output"), str)
    return MetricResult(case.id, ctx["agent_id"], "schema_validity",
                        1.0 if ok else 0.0, ok,
                        explanation="standard keys present" if ok else f"missing {missing}")


def m_required_facts(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    facts = case.expected.get("required_facts")
    if not facts:
        return None
    out = _norm(result.get("output", ""))
    found = [f for f in facts if _norm(f) in out]
    score = len(found) / len(facts)
    return MetricResult(case.id, ctx["agent_id"], "required_facts", round(score, 4),
                        score == 1.0,
                        explanation=f"{len(found)}/{len(facts)} required facts present",
                        raw={"found": found, "missing": [f for f in facts if f not in found]})


def m_citation_presence(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if not case.expected.get("must_include_citations"):
        return None
    markers = citation_markers(result.get("output", "")) or result.get("citations", [])
    ok = len(markers) > 0
    return MetricResult(case.id, ctx["agent_id"], "citation_presence",
                        1.0 if ok else 0.0, ok,
                        explanation=f"{len(markers)} citation marker(s) found" if ok
                        else "citations required but none present")


def m_citation_format(case: TestCase, result: dict, ctx: dict) -> MetricResult:
    markers = citation_markers(result.get("output", ""))
    bracketed = re.findall(r"\[[^\]]*\]", result.get("output", ""))
    if not bracketed:
        return MetricResult(case.id, ctx["agent_id"], "citation_format", 1.0, True,
                            explanation="no bracketed references to misformat")
    score = len(markers) / len(bracketed)
    return MetricResult(case.id, ctx["agent_id"], "citation_format", round(score, 4),
                        score == 1.0,
                        explanation=f"{len(markers)}/{len(bracketed)} bracketed refs are valid markers")


def _source_texts(result: dict) -> dict[str, str]:
    return {str(s.get("id")): (s.get("text") or s.get("summary") or "")
            for s in result.get("sources", [])}


def _supported(claim: str, source_text: str, threshold: float = 0.25) -> bool:
    ct, st = _tokens(claim), _tokens(source_text)
    if not ct or not st:
        return False
    return len(ct & st) / len(ct) >= threshold


def m_citation_support(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    sources = _source_texts(result)
    cited = [(c, ms) for c, ms in claims_with_citations(result.get("output", "")) if ms]
    if not sources or not cited:
        return None
    supported = 0
    for claim, markers in cited:
        texts = [sources.get(m, "") for m in markers]
        if any(_supported(claim, t) for t in texts if t):
            supported += 1
    score = supported / len(cited)
    return MetricResult(case.id, ctx["agent_id"], "citation_support", round(score, 4),
                        score >= 0.7,  # lexical proxy: paraphrase reads as partial support
                        explanation=f"{supported}/{len(cited)} cited claims lexically supported by their source",
                        raw={"cited_claims": len(cited), "supported": supported})


def m_abstention_correctness(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if "should_abstain" not in case.expected:
        return None
    abstained = is_abstention(result)
    ok = abstained == bool(case.expected["should_abstain"])
    return MetricResult(case.id, ctx["agent_id"], "abstention_correctness",
                        1.0 if ok else 0.0, ok,
                        explanation=f"expected abstain={case.expected['should_abstain']}, agent abstained={abstained}")


def m_tool_correctness(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    allowed = case.expected.get("allowed_tools")
    if allowed is None:
        return None
    calls = result.get("tool_calls", [])
    if not calls:
        return MetricResult(case.id, ctx["agent_id"], "tool_correctness", 1.0, True,
                            explanation="no tool calls made")
    good = [c for c in calls if c.get("name") in allowed]
    score = len(good) / len(calls)
    return MetricResult(case.id, ctx["agent_id"], "tool_correctness", round(score, 4),
                        score == 1.0,
                        explanation=f"{len(good)}/{len(calls)} tool calls within allowed set {allowed}")


def m_latency(case: TestCase, result: dict, ctx: dict) -> MetricResult:
    t = float(result.get("execution_time_s", 0.0))
    limit = case.expected.get("max_latency_s")
    return MetricResult(case.id, ctx["agent_id"], "latency", round(t, 4),
                        (t <= limit) if limit else None,
                        explanation=f"{t:.3f}s" + (f" (limit {limit}s)" if limit else ""))


def m_token_usage(case: TestCase, result: dict, ctx: dict) -> MetricResult:
    tu = result.get("token_usage", {})
    total = int(tu.get("input", 0)) + int(tu.get("output", 0))
    return MetricResult(case.id, ctx["agent_id"], "token_usage", float(total), None,
                        explanation=f"input {tu.get('input', 0)} + output {tu.get('output', 0)} tokens (estimated by adapter)")


def m_cost_estimate(case: TestCase, result: dict, ctx: dict) -> MetricResult:
    pricing = ctx.get("pricing", {})
    model = result.get("metadata", {}).get("model", "unknown")
    rates = pricing.get(model) or pricing.get("default", {"input": 0.0, "output": 0.0})
    tu = result.get("token_usage", {})
    cost = (tu.get("input", 0) * rates.get("input", 0.0)
            + tu.get("output", 0) * rates.get("output", 0.0)) / 1_000_000
    return MetricResult(case.id, ctx["agent_id"], "cost_estimate", round(cost, 6), None,
                        explanation=f"estimated ${cost:.6f} at listed rates for {model} (estimate, not an invoice)",
                        raw={"model": model, "rates_per_1m": rates})


# ----------------------------------------------------------------------
# project-specific metrics
# ----------------------------------------------------------------------

def m_citation_precision(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if case.project != "rag":
        return None
    support = m_citation_support(case, result, ctx)
    if support is None:
        return None
    support.metric_name = "citation_precision"
    support.explanation = "share of citations whose claim is supported by the cited evidence: " + support.explanation
    return support


def m_citation_coverage(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if case.project != "rag":
        return None
    claims = claims_with_citations(result.get("output", ""))
    if not claims:
        return None
    cited = sum(1 for _, ms in claims if ms)
    score = cited / len(claims)
    return MetricResult(case.id, ctx["agent_id"], "citation_coverage", round(score, 4),
                        score >= 0.8,
                        explanation=f"{cited}/{len(claims)} claims carry a citation")


def m_claim_support_rate(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if case.project != "research":
        return None
    structured = result.get("structured", {})
    findings = structured.get("findings", [])
    sources = _source_texts(result)
    if not findings or not sources:
        return None
    supported = 0
    for f in findings:
        texts = [sources.get(sid, "") for sid in f.get("source_ids", [])]
        if any(_supported(f.get("claim", ""), t) for t in texts if t):
            supported += 1
    score = supported / len(findings)
    return MetricResult(case.id, ctx["agent_id"], "claim_support_rate", round(score, 4),
                        score >= 0.8,
                        explanation=f"{supported}/{len(findings)} findings supported by their linked sources")


def m_source_diversity(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if case.project != "research":
        return None
    sources = result.get("sources", [])
    if not sources:
        # Informational only: for a missing-evidence case, returning no
        # sources is correct behaviour, not a failure.
        return MetricResult(case.id, ctx["agent_id"], "source_diversity", 0.0, None,
                            explanation="no sources returned")
    venues = {s.get("venue") or s.get("source_type") or "?" for s in sources}
    score = len(venues) / max(1, len(sources))
    return MetricResult(case.id, ctx["agent_id"], "source_diversity", round(score, 4), None,
                        explanation=f"{len(sources)} sources across {len(venues)} venue/type group(s)",
                        raw={"sources": len(sources), "groups": sorted(venues)})


def m_contradiction_surfaced(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if case.project != "research" or "has_contradictions" not in case.expected:
        return None
    structured = result.get("structured", {})
    found = bool(structured.get("contradictions")) or "onflicting Evidence" in result.get("output", "")
    ok = found == bool(case.expected["has_contradictions"])
    return MetricResult(case.id, ctx["agent_id"], "contradiction_surfaced",
                        1.0 if ok else 0.0, ok,
                        explanation=f"expected contradictions={case.expected['has_contradictions']}, surfaced={found}")


def m_routing_accuracy(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if case.project != "router" or "expected_tier" not in case.expected:
        return None
    tier = result.get("structured", {}).get("tier")
    ok = tier == case.expected["expected_tier"]
    return MetricResult(case.id, ctx["agent_id"], "routing_accuracy",
                        1.0 if ok else 0.0, ok,
                        explanation=f"expected tier {case.expected['expected_tier']}, routed to {tier}",
                        raw={"tier": tier, "complexity_score": result.get("structured", {}).get("complexity_score")})


def m_expensive_call_rate(case: TestCase, result: dict, ctx: dict) -> MetricResult | None:
    if case.project != "router":
        return None
    tier = result.get("structured", {}).get("tier")
    expensive = 1.0 if tier == "premium" else 0.0
    return MetricResult(case.id, ctx["agent_id"], "expensive_call_rate", expensive, None,
                        explanation=f"routed to {tier}")


ALL_METRICS = [m_schema_validity, m_required_facts, m_citation_presence,
               m_citation_format, m_citation_support, m_abstention_correctness,
               m_tool_correctness, m_latency, m_token_usage, m_cost_estimate,
               m_citation_precision, m_citation_coverage, m_claim_support_rate,
               m_source_diversity, m_contradiction_surfaced, m_routing_accuracy,
               m_expensive_call_rate]


def evaluate_case(case: TestCase, result: dict, *, agent_id: str,
                  pricing: dict | None = None, judges: list | None = None) -> list[MetricResult]:
    ctx = {"agent_id": agent_id, "pricing": pricing or {}}
    results: list[MetricResult] = []
    for fn in ALL_METRICS:
        mr = fn(case, result, ctx)
        if mr is not None:
            results.append(mr)
    for judge in judges or []:
        results.extend(judge.judge(case, result))
    return results
