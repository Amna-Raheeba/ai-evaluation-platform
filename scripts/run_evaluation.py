#!/usr/bin/env python3
"""Run an evaluation profile and print the summary as JSON.

Used by CI (see .github/workflows/ci.yml) and locally:
    python scripts/run_evaluation.py --agent mock-agent --profile quick
Exit code is 0 when the run completes (infra errors are reported in
the summary, not hidden); quality decisions belong to
scripts/check_quality_gate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_platform.__main__ import DATASETS  # noqa: E402
from eval_platform.adapter import default_registry  # noqa: E402
from eval_platform.orchestrator import Orchestrator  # noqa: E402
from eval_platform.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="mock-agent")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--profile", default="quick",
                        choices=["quick", "standard", "full", "regression"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--db", default="eval_platform.db")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset or str(DATASETS.get(args.agent, ""))
    if not dataset:
        print(f"no dataset known for {args.agent}", file=sys.stderr)
        return 2
    store = Store(args.db)
    orch = Orchestrator(store, default_registry())
    summary = orch.run_evaluation(args.agent, dataset, profile=args.profile,
                                  repeats=args.repeats,
                                  set_baseline_flag=args.baseline)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
