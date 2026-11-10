# Experiment Log

Dated working notes. Claims here are only as strong as the runs behind
them; anything not yet measured says so.

## 2026-11-10 — v0.1.0 core and first baselines

**Built.** Adapter SDK + 4 adapters, golden dataset engine, deterministic
metrics + lexical judge, SQLite store, orchestrator (quick/standard/
full/regression profiles), regression detector with paired CIs and
quality gates, failure taxonomy + clustering, CLI + CI gate scripts.

**Real runs (offline, this machine).** Full numbers in
`results/baseline_summary.json`; sample report in
`results/sample_report.md`.

- Research agent (fixture corpus, 8 cases): required_facts 1.0,
  claim_support_rate 1.0, contradiction_surfaced 1.0,
  citation_support 0.79. Rerun vs baseline → **PASS** (identical,
  deterministic agent).
- LLM router (12 golden prompts, real `decide_tier`): routing_accuracy
  1.0, expensive_call_rate 0.167 (2 of 12 prompts genuinely need the
  premium tier). Rerun vs baseline → **PASS**.
- RAG agent: **unavailable** — its pipeline needs google-genai + a
  Gemini key. Run recorded as 10 infra errors, 0 scored. That is the
  infra/quality separation working as designed, not a failed agent.

**Findings worth keeping.**

1. *Lexical support under-reads paraphrase.* The research writer's
   prose summaries score 0.79 on `citation_support` while the
   findings-level `claim_support_rate` is 1.0. The gap is the proxy's
   limit, documented in docs/metrics.md — and the reason
   `citation_support` passes at 0.7 while claim support is gated
   separately. An entailment-based judge is the honest upgrade.
2. *Two proxies had to be scoped.* Lexical relevance is meaningless
   for a routing decision (a tier label is not an answer) and for an
   abstention (a refusal should not be punished for not repeating the
   question). Both are now skipped in those cases — recorded in the
   judge code, not silently.
3. *Missing evidence is not a failure.* The research case with an
   out-of-corpus topic correctly returns no sources; `source_diversity`
   is informational there. Early version failed the case for it —
   exactly the kind of false failure this platform must not produce.

**Controlled regression check (notebooks/01).** Good mock vs degraded
mock (citations dropped, facts truncated): the gate fires **FAIL** on
`required_facts`/`citation_presence`, and the degraded run's failures
cluster as citation_failure. Identical reruns pass; an infra-heavy run
returns **INCONCLUSIVE**. The detector distinguishes all three states
it claims to distinguish.

**Not yet measured.** Judge-vs-human agreement, adaptive test
selection, statistical power at larger n. The roadmap's experiments
(§8 of the project plan) stay open until there is data for them.

## 2026-11-10 — Experiment 03: fixed threshold vs paired-CI regression decisions

**Question.** On repeated runs of a stochastic agent, which regression
decision method is more trustworthy: a fixed threshold on the run-level
mean drop, or the paired-CI noise discipline `compare_runs` applies?
Script: `notebooks/03_regression_methods_experiment.py`; full data in
`results/regression_methods.csv` (+ `.png`).

**Setup.** `StochasticMockAdapter` drops each fact independently with
probability `p_drop` per call (the plain `MockAdapter` is deterministic,
so repeated runs would have zero variance to discipline). Generated
12-case dataset (4 facts/case, temp file — golden datasets stay
hand-reviewed), primary metric `required_facts`, tolerance 0.05,
12 repetitions per condition, fresh baseline + candidate run each
repetition, master seed 20261110 (re-running reproduces the CSV
exactly — verified by diffing two runs). Methods: **fixed_threshold**
FAILs iff run-level mean delta < −0.05; **paired_ci** FAILs iff mean
paired per-case delta < −0.05 AND delta + CI half-width < 0.

**Results (12 repetitions per condition).**

- *Healthy* (baseline p_drop=0.05 vs candidate p_drop=0.05 — any FAIL
  is a false alarm): fixed_threshold **1/12 false alarms (8.3%)** —
  repetition 7, delta −0.0625 with CI half-width 0.1493, i.e. pure
  noise past the tolerance line. paired_ci **0/12 (0.0%)**.
- *Subtle* (candidate p_drop=0.20, expected drop ≈ 0.15; observed mean
  delta −0.167): fixed_threshold detected **11/12 (91.7%)**, paired_ci
  **9/12 (75.0%)**. Both missed repetition 5 (delta −0.0417, inside
  tolerance). paired_ci additionally passed repetitions 8 and 9
  (delta −0.0833 each) because their CIs (half-widths 0.110 and 0.092)
  still included no-change.
- *Clear* (candidate degraded=True, mean pinned at 0.25, deltas
  −0.625 to −0.750): both methods **12/12 (100.0%)**.

**CI half-widths (paired, n=12 cases).** Healthy: mean 0.088, range
0.041–0.149. Subtle: mean 0.118, range 0.092–0.160. Clear: mean 0.053,
range 0.000–0.095.

**Reading.** paired_ci false-alarms less (0 vs 1 on healthy) — the CI
requirement absorbed exactly the one noise excursion that fooled the
fixed threshold. The fixed threshold detects subtle drops more often
(11 vs 9): at n=12 cases the CI is wide relative to the 0.05
tolerance, so genuinely real but borderline drops (−0.083) are
indistinguishable from noise and paired_ci lets them through. Neither
method dominates; it is the expected specificity/sensitivity trade,
and the lever is paired-case count, not the tolerance.

**Limitations.** Synthetic per-fact drop model with independent drops
(real degradations are usually correlated across facts and cases);
one metric only (`required_facts`); n=12 cases × 12 repetitions;
normal-approximation CI, as in `regression.py`; mock agent only — no
claim yet about real-model run-to-run variance, which is typically
larger and heavier-tailed.
