# Dashboard

A Streamlit front-end for the evaluation platform. It reads the
actual SQLite store the CLI writes — nothing is pre-generated — and
the sidebar button runs a real evaluation through the orchestrator.

## What it shows

- **Overview** — agent count, total runs, saved regression decisions
  (PASS / FAIL / INCONCLUSIVE), total estimated cost, and the latest
  run id, all from the store.
- **Agent comparison** — one row per agent from its latest run
  (version, dataset version, profile, pass rate, scored / infra-error
  counts, estimated cost, latency p95), with the full metric-means
  table for each agent in an expander.
- **Run details** — pick any stored run and expand each case to see
  its status, timing, and every metric result with score, pass/fail,
  evaluator type, and explanation.
- **Regression center** — for agents with a saved baseline, the
  latest run compared against that baseline by the regression
  engine: decision banner, per-metric baseline/new means, paired
  delta and 95% CI half-width, paired-case count, and gate outcome,
  plus the engine's reasons. When the latest run is the baseline
  itself, the section says so instead of comparing a run to itself.
- **Failures** — failure-cluster totals by category across all runs,
  plus the clusters (signature, case count, hypothesis, case keys)
  for a selected run.
- **Cost & performance** — per-agent run counts, scored cases, total
  estimated cost, mean pass rate, and latest latency p95, with a bar
  chart of total cost per agent.
- **Run quick evaluation (mock)** — sidebar button that executes the
  MockAdapter quick evaluation live via the orchestrator, writes the
  run to the store, shows its pass rate, and refreshes the page.

Costs shown are pricing-config estimates, not invoices, and offline
judge scores are lexical proxies — the header caption says so.

If there is no database yet, the dashboard falls back to
`results/baseline_summary.json` when it exists (with an info banner),
and otherwise shows a warning pointing at the CLI or the mock button
instead of failing.

## Run it

From the project root:

```bash
pip install streamlit
streamlit run dashboard/app.py
```

The database path honours the `EVAL_PLATFORM_DB` environment
variable, matching the CLI; otherwise it defaults to
`eval_platform.db` in the project root.

## Layout

```
dashboard/
├── app.py       # the Streamlit app (presentation only)
├── helpers.py   # pure-Python data shaping over the store, no
│                # streamlit import — unit-tested in
│                # tests/test_dashboard_helpers.py
└── README.md    # this file
```

Streamlit is a dashboard-only dependency (the `dashboard` extra in
`pyproject.toml`). The core `eval_platform` package stays
standard-library only.
