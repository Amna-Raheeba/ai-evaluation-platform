"""Experiment 02 — adapter smoke runs against the real agents.

Calls each real adapter once, directly (no orchestrator), and prints
what came back in the standardized format. The RAG adapter is expected
to report UNAVAILABLE offline — that is the contract working.

Run: python notebooks/02_adapter_smoke_runs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_platform.adapter import AdapterError             # noqa: E402
from eval_platform.adapters.rag import RagAdapter          # noqa: E402
from eval_platform.adapters.research import ResearchAdapter  # noqa: E402
from eval_platform.adapters.router import RouterAdapter    # noqa: E402


def show(name: str, fn) -> None:
    try:
        result = fn()
        meta = result["metadata"]
        print(f"{name}: status={result['status']} model={meta['model']} "
              f"tokens={result['token_usage']} output[:90]={result['output'][:90]!r}")
        if result["status"] == "unavailable":
            print(f"   unavailable reason: {meta.get('unavailable_reason')}")
    except AdapterError as exc:
        print(f"{name}: AdapterError kind={exc.kind}: {exc.message}")


def main() -> None:
    show("research", lambda: ResearchAdapter().run(
        {"topic": "explainable AI in healthcare"}))
    show("router", lambda: RouterAdapter().run(
        {"prompt": "Analyze the trade-off and justify under uncertainty."}))
    show("rag", lambda: RagAdapter().run(
        {"question": "How many vacation days do employees get?"}))


if __name__ == "__main__":
    main()
