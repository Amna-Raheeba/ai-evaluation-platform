"""Deterministic metrics + judges, on hand-built cases/results."""
from __future__ import annotations

import pytest

from eval_platform.adapter import AdapterError
from eval_platform.datasets import TestCase
from eval_platform.judges import LLMJudge, LexicalJudge
from eval_platform.metrics import evaluate_case


def _case(**overrides):
    base = {"id": "t-1", "project": "rag", "input": {"question": "vacation days?"},
            "expected": {"required_facts": ["twenty-six days", "public holidays"],
                         "must_include_citations": True},
            "category": "factual_qa", "difficulty": "easy", "tags": []}
    base.update(overrides)
    return TestCase(**base)


def _result(output, **extra):
    base = {"output": output, "execution_time_s": 1.25,
            "token_usage": {"input": 100, "output": 50}, "tool_calls": [],
            "trace_id": "x", "metadata": {"agent_name": "rag-agent",
                                          "agent_version": "1", "model": "gemini-flash-latest"},
            "status": "ok", "sources": [{"id": "1", "title": "p3",
                                         "text": "twenty-six days of paid vacation and public holidays"}]}
    base.update(extra)
    return base


def _by_name(metrics):
    return {m.metric_name: m for m in metrics}


def test_required_facts_fraction_and_citations():
    metrics = _by_name(evaluate_case(_case(), _result(
        "Employees get twenty-six days of paid vacation [1]."), agent_id="rag-agent"))
    assert metrics["required_facts"].score == 0.5  # one of two facts
    assert metrics["required_facts"].passed is False
    assert metrics["citation_presence"].score == 1.0
    assert metrics["citation_format"].score == 1.0
    assert metrics["schema_validity"].score == 1.0


def test_cost_and_latency_metrics_use_pricing_estimates():
    pricing = {"gemini-flash-latest": {"input": 0.30, "output": 2.50}}
    metrics = _by_name(evaluate_case(_case(), _result("x"), agent_id="rag-agent", pricing=pricing))
    expected = (100 * 0.30 + 50 * 2.50) / 1_000_000
    assert metrics["cost_estimate"].score == pytest.approx(expected)
    assert metrics["latency"].score == 1.25
    assert metrics["token_usage"].score == 150


def test_abstention_correctness_both_directions():
    case = _case(expected={"should_abstain": True})
    good = _by_name(evaluate_case(case, _result(
        "INSUFFICIENT EVIDENCE", status="abstained"), agent_id="rag-agent"))
    assert good["abstention_correctness"].passed is True
    bad = _by_name(evaluate_case(case, _result("The answer is forty-two."), agent_id="rag-agent"))
    assert bad["abstention_correctness"].passed is False


def test_router_metrics():
    case = _case(project="router", expected={"expected_tier": "cheap"})
    result = _result("Routed to tier 'balanced'", structured={"tier": "balanced"})
    metrics = _by_name(evaluate_case(case, result, agent_id="llm-router"))
    assert metrics["routing_accuracy"].score == 0.0
    assert metrics["expensive_call_rate"].score == 0.0


def test_tool_correctness():
    case = _case(expected={"allowed_tools": ["search"]})
    result = _result("x", tool_calls=[{"name": "search"}, {"name": "delete_everything"}])
    metrics = _by_name(evaluate_case(case, result, agent_id="rag-agent"))
    assert metrics["tool_correctness"].score == 0.5


def test_lexical_judge_returns_structured_scores():
    results = LexicalJudge().judge(_case(), _result(
        "Employees get twenty-six days of paid vacation and public holidays [1]."))
    names = {r.metric_name for r in results}
    assert "lexical_faithfulness" in names
    for r in results:
        assert r.evaluator_type == "lexical_judge"
        assert 0.0 <= r.score <= 1.0
        assert r.explanation


def test_llm_judge_without_key_is_infra_not_a_score(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_API_KEY", raising=False)
    with pytest.raises(AdapterError) as exc:
        LLMJudge(api_key=None).judge(_case(), _result("x"))
    assert exc.value.kind == "infra"
