"""CLI: python -m eval_platform <command>

  list-agents
  validate-dataset datasets/research.jsonl
  run --agent research-agent --dataset datasets/research.jsonl --profile standard [--repeats 2] [--baseline]
  compare --baseline-run 1 --run 2
  failures --run 2
  report --run 2 [--out results/report.md]

The store defaults to eval_platform.db in the current directory
(override with --db or EVAL_PLATFORM_DB).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapter import default_registry
from .datasets import dataset_version, load_dataset
from .orchestrator import Orchestrator, load_gates, report_markdown
from .regression import compare_runs
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {"rag-agent": ROOT / "datasets" / "rag.jsonl",
            "research-agent": ROOT / "datasets" / "research.jsonl",
            "llm-router": ROOT / "datasets" / "router.jsonl",
            "mock-agent": ROOT / "datasets" / "mock.jsonl"}


def _store(args) -> Store:
    return Store(args.db or os.environ.get("EVAL_PLATFORM_DB", "eval_platform.db"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_platform")
    parser.add_argument("--db", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-agents")
    p_val = sub.add_parser("validate-dataset")
    p_val.add_argument("dataset")
    p_run = sub.add_parser("run")
    p_run.add_argument("--agent", required=True)
    p_run.add_argument("--dataset", default=None)
    p_run.add_argument("--profile", default="standard",
                       choices=["quick", "standard", "full", "regression"])
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--timeout", type=float, default=60.0)
    p_run.add_argument("--baseline", action="store_true",
                       help="save this run as the agent's baseline")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--baseline-run", type=int, required=True)
    p_cmp.add_argument("--run", type=int, required=True)
    p_fail = sub.add_parser("failures")
    p_fail.add_argument("--run", type=int, required=True)
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--run", type=int, required=True)
    p_rep.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    store = _store(args)
    registry = default_registry()

    if args.command == "list-agents":
        for info in registry.infos():
            print(f"{info['name']:<16} project={info['project']:<9} version={info['version']}")
        return 0

    if args.command == "validate-dataset":
        cases = load_dataset(args.dataset)
        print(f"OK {args.dataset}: {len(cases)} cases, version {dataset_version(args.dataset)}")
        return 0

    if args.command == "run":
        dataset = args.dataset or str(DATASETS.get(args.agent, ""))
        if not dataset:
            print(f"no default dataset for agent {args.agent}; pass --dataset", file=sys.stderr)
            return 2
        orch = Orchestrator(store, registry)
        summary = orch.run_evaluation(args.agent, dataset, profile=args.profile,
                                      repeats=args.repeats, timeout_s=args.timeout,
                                      set_baseline_flag=args.baseline)
        print(json.dumps(summary, indent=2))
        if args.profile == "regression":
            baseline = store.get_baseline(args.agent)
            if baseline and baseline["id"] != summary["run_id"]:
                report = compare_runs(store, baseline["id"], summary["run_id"], load_gates())
                store.save_regression(baseline["id"], summary["run_id"],
                                      report["decision"], report)
                print(f"REGRESSION DECISION vs baseline run {baseline['id']}: {report['decision']}")
                for reason in report["reasons"]:
                    print(f"  - {reason}")
        return 0

    if args.command == "compare":
        report = compare_runs(store, args.baseline_run, args.run, load_gates())
        store.save_regression(args.baseline_run, args.run, report["decision"], report)
        print(json.dumps(report, indent=2))
        return 0 if report["decision"] != "FAIL" else 1

    if args.command == "failures":
        clusters = store.get_failure_clusters(args.run)
        if not clusters:
            print(f"no failure clusters recorded for run {args.run}")
        for c in clusters:
            print(f"{c['category']} ({c['signature']}): {len(c['case_keys'])} case(s): "
                  f"{', '.join(c['case_keys'])}")
            print(f"  hypothesis: {c['hypothesis']}")
        return 0

    if args.command == "report":
        md = report_markdown(store, args.run)
        if args.out:
            Path(args.out).write_text(md, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(md)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
