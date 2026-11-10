"""Common Agent Adapter SDK.

The evaluation platform must never depend on an agent's internal
framework. Every agent is wrapped in a thin adapter implementing this
contract, and every adapter returns the same standardized result:

    {
        "output": str,                      # the agent's answer / report
        "execution_time_s": float,
        "token_usage": {"input": int, "output": int},
        "tool_calls": [{"name": str, ...}],
        "trace_id": str,
        "metadata": {"agent_name": str, "agent_version": str, "model": str},
        # optional, adapter-specific:
        "status": "ok" | "abstained" | "unavailable",
        "sources": [{"id": str, "title": str, "text": str}],
        "citations": [str],
        "structured": {...},                # e.g. router tier, findings
    }

Errors are normalized into AdapterError with a `kind`:
  - "timeout"   the agent did not answer in time
  - "infra"     infrastructure problem (missing service, key, dependency)
  - "contract"  the adapter/agent returned something off-contract

Infrastructure problems are NOT quality failures. The orchestrator
counts them separately and the regression engine can return
INCONCLUSIVE because of them, but they never lower a quality score.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

ERROR_KINDS = ("timeout", "infra", "contract")

STANDARD_KEYS = ("output", "execution_time_s", "token_usage", "tool_calls",
                 "trace_id", "metadata")


class AdapterError(Exception):
    def __init__(self, kind: str, message: str):
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown error kind {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.message = message


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def standard_result(output: str, *, agent_name: str, agent_version: str,
                    model: str, started: float, input_tokens: int = 0,
                    output_tokens: int = 0, tool_calls: list | None = None,
                    **extra: Any) -> dict:
    """Build a contract-conforming result dict."""
    result = {
        "output": output,
        "execution_time_s": round(time.time() - started, 4),
        "token_usage": {"input": int(input_tokens), "output": int(output_tokens)},
        "tool_calls": tool_calls or [],
        "trace_id": new_trace_id(),
        "metadata": {"agent_name": agent_name, "agent_version": agent_version,
                     "model": model},
        "status": "ok",
    }
    result.update(extra)
    return result


def check_contract(result: Any) -> None:
    """Raise AdapterError(kind='contract') if a result is off-contract."""
    if not isinstance(result, dict):
        raise AdapterError("contract", f"adapter returned {type(result).__name__}, expected dict")
    missing = [k for k in STANDARD_KEYS if k not in result]
    if missing:
        raise AdapterError("contract", f"result missing keys: {missing}")
    if not isinstance(result["output"], str):
        raise AdapterError("contract", "output must be a string")
    tu = result["token_usage"]
    if not isinstance(tu, dict) or "input" not in tu or "output" not in tu:
        raise AdapterError("contract", "token_usage must have input/output")
    md = result["metadata"]
    for key in ("agent_name", "agent_version", "model"):
        if key not in md:
            raise AdapterError("contract", f"metadata missing {key}")


class AgentAdapter(ABC):
    """One adapter per agent project. Thin: wrap, don't rewrite."""

    name: str = "unnamed"
    version: str = "0.0.0"
    project: str = ""          # dataset project this adapter serves

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """Execute the agent and return a standardized result dict."""
        raise NotImplementedError

    def get_agent_info(self) -> dict:
        return {"name": self.name, "version": self.version, "project": self.project}


class AdapterRegistry:
    def __init__(self):
        self._adapters: dict[str, AgentAdapter] = {}

    def register(self, adapter: AgentAdapter) -> AgentAdapter:
        self._adapters[adapter.name] = adapter
        return adapter

    def get(self, name: str) -> AgentAdapter:
        if name not in self._adapters:
            raise AdapterError("contract", f"no adapter registered as {name!r}; "
                                           f"known: {sorted(self._adapters)}")
        return self._adapters[name]

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def infos(self) -> list[dict]:
        return [self._adapters[n].get_agent_info() for n in self.names()]


def default_registry() -> AdapterRegistry:
    """Registry with every adapter that can be constructed offline."""
    from .adapters.mock import MockAdapter
    from .adapters.rag import RagAdapter
    from .adapters.research import ResearchAdapter
    from .adapters.router import RouterAdapter

    registry = AdapterRegistry()
    registry.register(MockAdapter())
    registry.register(ResearchAdapter())
    registry.register(RouterAdapter())
    registry.register(RagAdapter())
    return registry
