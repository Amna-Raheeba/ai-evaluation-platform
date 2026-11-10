"""MockAdapter: a deterministic stand-in agent for testing the platform
itself and for regression experiments — no API calls, no cost.

Behaviour is driven by the test-case input convention used by the
platform's own test datasets: if the input carries ``mock_facts`` (a
list of fact strings) and ``mock_sources``, a healthy mock answers with
those facts, cited. Degradation is injectable:

  - degraded=True                         drops citations and most facts
  - fact_drop_rate=0..1                   fraction of facts omitted
  - add_hallucination=True                appends an unsupported claim

That is what makes the Week-1 experiment possible: run the good mock,
save a baseline, then run a degraded mock of the "same" agent and watch
the regression gate fire — with zero model spend.
"""
from __future__ import annotations

import time

from ..adapter import AgentAdapter, standard_result


class MockAdapter(AgentAdapter):
    name = "mock-agent"
    project = "mock"

    def __init__(self, version: str = "1.0.0", degraded: bool = False,
                 fact_drop_rate: float = 0.0, add_hallucination: bool = False,
                 latency_s: float = 0.01):
        self.version = version
        self.degraded = degraded
        self.fact_drop_rate = fact_drop_rate
        self.add_hallucination = add_hallucination
        self.latency_s = latency_s

    def run(self, input_data: dict) -> dict:
        started = time.time()
        facts = list(input_data.get("mock_facts", []))
        sources = list(input_data.get("mock_sources", []))
        if self.degraded:
            facts = facts[:1]
        elif self.fact_drop_rate:
            keep = max(0, int(len(facts) * (1 - self.fact_drop_rate)))
            facts = facts[:keep]
        sentences = []
        for i, fact in enumerate(facts):
            marker = f" [{(i % max(1, len(sources))) + 1}]" if sources and not self.degraded else ""
            sentences.append(fact.rstrip(".") + "." + marker)
        if self.add_hallucination:
            sentences.append("The organisation also offers unlimited free flights to all staff.")
        if input_data.get("mock_abstain"):
            output = "INSUFFICIENT EVIDENCE. I could not find enough evidence to answer."
            status = "abstained"
        else:
            output = " ".join(sentences) if sentences else "No information available."
            status = "ok"
        src_payload = [{"id": str(i + 1), "title": s.get("title", f"Source {i+1}"),
                        "text": s.get("text", "")} for i, s in enumerate(sources)]
        return standard_result(
            output, agent_name=self.name, agent_version=self.version,
            model="mock-deterministic", started=started,
            input_tokens=max(1, len(str(input_data)) // 4),
            output_tokens=max(1, len(output) // 4),
            tool_calls=[{"name": "search"}, {"name": "answer"}],
            status=status, sources=src_payload,
            citations=[str(i + 1) for i in range(len(src_payload))] if sentences and not self.degraded else [])
