"""Experiment 01 — can the platform tell quality, noise, and outage apart?

Three runs of the mock agent over datasets/mock.jsonl:
  1. healthy mock          -> saved as baseline
  2. healthy mock again    -> comparison must PASS (no change)
  3. degraded mock         -> comparison must FAIL the quality gate
                             (citations dropped, facts truncated)

No model calls, no cost: degradation is injected by MockAdapter.
Run: python notebooks/01_mock_regression_experiment.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_platform.adapter import AdapterRegistry          # noqa: E402
from eval_platform.adapters.mock import MockAdapter        # noqa: E402
from eval_platform.orchestrator import Orchestrator, load_gates  # noqa: E402
from eval_platform.regression import compare_runs          # noqa: E402
from eval_platform.store import Store                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(store, adapter):
    registry = AdapterRegistry()
    registry.register(adapter)
    return Orchestrator(store, registry).run_evaluation(
        "mock-agent", ROOT / "datasets" / "mock.jsonl", profile="standard")


def main() -> None:
    store = Store(Path(tempfile.mkdtemp()) / "experiment.db")
    gates = load_gates()
    baseline = run(store, MockAdapter(version="1.0.0"))
    same = run(store, MockAdapter(version="1.0.0"))
    degraded = run(store, MockAdapter(version="1.1.0", degraded=True))
    for label, summary in (("baseline", baseline), ("identical rerun", same),
                           ("degraded", degraded)):
        print(f"{label}: pass_rate={summary['pass_rate']} "
              f"required_facts={summary['metric_means'].get('required_facts')} "
              f"citation_presence={summary['metric_means'].get('citation_presence')}")
    for label, summary in (("identical rerun", same), ("degraded", degraded)):
        report = compare_runs(store, baseline["run_id"], summary["run_id"], gates)
        print(f"compare baseline vs {label}: {report['decision']}")
        for reason in report["reasons"]:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
