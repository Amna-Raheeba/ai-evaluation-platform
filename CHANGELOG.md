# Changelog

All notable changes to this project, in my own words. Dates are when the
work was done, not when it was tidied up for the repo.

## v0.2.0 — 2026-11-10

### Streamlit dashboard + regression-methods experiment

- **Streamlit dashboard** (`dashboard/`): a local app over the
  store's real data — overview counts and saved regression decisions,
  agent comparison from each agent's latest run, per-run case detail
  with metric explanations, a regression center comparing the latest
  run against the stored baseline (paired deltas, 95% CI half-widths,
  gate outcomes), failure-cluster totals and per-run clusters, and
  cost & performance per agent. The sidebar **Run quick evaluation
  (mock)** button executes the MockAdapter through the orchestrator
  live and writes the run to the same store. `dashboard/helpers.py`
  holds the shaping logic with no streamlit import and is covered by
  `tests/test_dashboard_helpers.py`; streamlit stays a dashboard-only
  extra and the core remains standard-library only. When no database
  exists the dashboard falls back to `results/baseline_summary.json`.
- **Regression-methods experiment** (`notebooks/03`): an experiment
  comparing regression-comparison methods; details in
  `docs/experiment_log.md`.

## v0.1.0 — 2026-11-10

### Added
- **Adapter SDK** (`eval_platform/adapter.py`): one standardized contract
  for every agent (output, timing, token usage, tool calls, trace id,
  metadata), an adapter registry, and normalized errors
  (timeout / infra / contract). Infrastructure problems are recorded as
  infra errors and never averaged into quality scores.
- **Four adapters**: MockAdapter (deterministic, with injectable
  degradation for experiments), ResearchAdapter (the real multi-agent
  research Supervisor, offline), RouterAdapter (the router's real
  `complexity_report` + `decide_tier`; no LLM calls), RagAdapter (the
  real ragcites CLI via subprocess when configured; a structured
  UNAVAILABLE result otherwise).
- **Golden Dataset Engine**: validated JSONL datasets with content-hash
  versions (`name@sha8`); rag (10 cases), research (8), router (12,
  expected tiers verified against the real routing logic), mock (3).
- **Evaluation Engine**: deterministic metrics (schema, required facts,
  citation presence/format/support, abstention, tool correctness,
  latency, tokens, estimated cost) and project-specific metrics (RAG
  citation precision/coverage; research claim support, source
  diversity, contradiction surfacing; router routing accuracy,
  expensive-call rate), plus a judge interface with an offline
  LexicalJudge and an optional LLMJudge that reports infra-unavailable
  without a key instead of fabricating scores.
- **Orchestrator** with quick/standard/full/regression profiles,
  timeout-guarded adapter calls, partial-failure handling, and run
  summaries (pass rate, metric means, estimated cost, latency p50/p95,
  cost per successful task).
- **Regression Detector**: baselines, paired per-case deltas with 95%
  CIs (normal approximation), quality gates, and PASS / FAIL /
  INCONCLUSIVE decisions; infra-heavy runs are INCONCLUSIVE by rule.
- **Failure Analyzer**: 9-category taxonomy, clustering by failure
  signature, root-cause hypotheses, and export of a failed case as a
  draft golden case for review.
- **SQLite store** mirroring the PostgreSQL schema table-for-table,
  with full reproducibility metadata per run; optional FastAPI layer;
  CLI; CI scripts (`run_evaluation.py`, `check_quality_gate.py`) and a
  GitHub Actions workflow.
- **First real baselines** in `results/`: research and router agents
  evaluated for real and compared (PASS); the RAG run recorded honestly
  as infrastructure-unavailable. 41 tests passing.
