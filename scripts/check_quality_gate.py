#!/usr/bin/env python3
"""Quality gate: compare the agent's latest run against its baseline.

Exit codes: 0 = PASS, 1 = FAIL (a confirmed regression blocks the
merge), 2 = INCONCLUSIVE or no baseline (reported, not a block — an
outage or a too-small sample must not masquerade as a quality verdict).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_platform.orchestrator import load_gates  # noqa: E402
from eval_platform.regression import compare_runs  # noqa: E402
from eval_platform.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="mock-agent")
    parser.add_argument("--db", default="eval_platform.db")
    parser.add_argument("--baseline-run", type=int, default=None)
    parser.add_argument("--run", type=int, default=None)
    args = parser.parse_args()

    store = Store(args.db)
    baseline_id = args.baseline_run
    if baseline_id is None:
        baseline = store.get_baseline(args.agent)
        if not baseline:
            print(f"INCONCLUSIVE: no baseline saved for {args.agent}")
            return 2
        baseline_id = baseline["id"]
    run_id = args.run
    if run_id is None:
        latest = store.latest_run(args.agent)
        if not latest or latest["id"] == baseline_id:
            print("INCONCLUSIVE: no newer run to compare against the baseline")
            return 2
        run_id = latest["id"]
    report = compare_runs(store, baseline_id, run_id, load_gates())
    store.save_regression(baseline_id, run_id, report["decision"], report)
    print(json.dumps({"decision": report["decision"], "reasons": report["reasons"]}, indent=2))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report["decision"]]


if __name__ == "__main__":
    raise SystemExit(main())
