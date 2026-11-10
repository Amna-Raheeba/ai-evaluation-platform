"""Judges: qualitative evaluation behind a small interface.

A judge returns MetricResult dicts shaped exactly like the roadmap's
judge output: {metric, score, passed, reason, evidence}.

Two implementations:
  - LexicalJudge — deterministic, offline. Overlap-based proxies for
    faithfulness (output sentences grounded in the returned sources or
    the expected facts) and relevance (output addresses the input).
    It is a proxy, not an understanding — see docs/metrics.md.
  - LLMJudge — optional. Without an API key it raises an *infra*
    AdapterError: an unavailable judge is an infrastructure problem,
    never a quality failure and never a fabricated score.
"""
from __future__ import annotations

import os
import re

from .adapter import AdapterError
from .datasets import TestCase
from .metrics import MetricResult, _tokens

JUDGE_VERSION = "1.0"


class Judge:
    name = "judge"

    def judge(self, case: TestCase, result: dict) -> list[MetricResult]:
        raise NotImplementedError


class LexicalJudge(Judge):
    name = "lexical"

    def judge(self, case: TestCase, result: dict) -> list[MetricResult]:
        agent_id = result.get("metadata", {}).get("agent_name", "unknown")
        output = result.get("output", "")
        if result.get("status") == "abstained":
            return []  # relevance/faithfulness proxies are meaningless on a refusal
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", output) if len(s.strip()) > 15]
        source_tokens: set[str] = set()
        for s in result.get("sources", []):
            source_tokens |= _tokens(s.get("text") or s.get("summary") or "")
        for fact in case.expected.get("required_facts", []):
            source_tokens |= _tokens(fact)
        results: list[MetricResult] = []
        if sentences and source_tokens:
            grounded = 0
            for sent in sentences:
                st = _tokens(sent)
                if st and len(st & source_tokens) / len(st) >= 0.3:
                    grounded += 1
            score = grounded / len(sentences)
            results.append(MetricResult(
                case.id, agent_id, "lexical_faithfulness", round(score, 4),
                score >= 0.6, evaluator_type="lexical_judge",
                evaluator_version=JUDGE_VERSION,
                explanation=f"{grounded}/{len(sentences)} output sentences share vocabulary with sources/expected facts",
                raw={"evidence": [f"{grounded} grounded sentences"]}))
        if "tier" in result.get("structured", {}):
            return results  # a routing decision is not a prose answer; relevance proxy n/a
        question = " ".join(str(v) for v in case.input.values())
        qt, ot = _tokens(question), _tokens(output)
        if qt and ot:
            score = len(qt & ot) / len(qt)
            results.append(MetricResult(
                case.id, agent_id, "lexical_relevance", round(score, 4),
                score >= 0.15, evaluator_type="lexical_judge",
                evaluator_version=JUDGE_VERSION,
                explanation=f"output covers {score:.0%} of the input's content words",
                raw={"evidence": [f"input/output token overlap {len(qt & ot)}/{len(qt)}"]}))
        return results


class LLMJudge(Judge):
    """Optional LLM-as-a-judge. Deliberately unimplemented without a key:
    scoring quality with a model you cannot call would be fabrication."""

    name = "llm"
    ENV_KEY = "EVAL_JUDGE_API_KEY"

    def __init__(self, api_key: str | None = None, model: str = "judge-model"):
        self.api_key = api_key or os.environ.get(self.ENV_KEY)
        self.model = model

    def judge(self, case: TestCase, result: dict) -> list[MetricResult]:
        if not self.api_key:
            raise AdapterError("infra", f"LLMJudge unavailable: no API key "
                                         f"(set {self.ENV_KEY})")
        raise AdapterError("infra", "LLMJudge transport not configured in the "
                                    "offline core; plug a provider client in here")
