# Production-Grade AI Evaluation & Regression Platform

> Manually checking whether an agent "still gives good answers" does not scale. This platform runs structured golden test cases against heterogeneous AI agents, measures quality with independent metrics, compares new versions against saved baselines with statistical discipline, classifies failures, and turns the result into a pass/fail quality gate for CI.

**Author:** Amna Raheeba — BS-IT, Government College Women University Faisalabad · Research interests: explainable AI, LLMs, AI for healthcare

![CI](https://github.com/Amna-Raheeba/ai-evaluation-platform/actions/workflows/ci.yml/badge.svg)

---

## 1. The problems it solves

| Problem | Platform's answer |
|---|---|
| AI answers can be inaccurate | Deterministic metrics + judges, per case, with explanations |
| Citations can be unreliable | Citation presence, format, and *support* checked separately — a citation existing is not a citation supporting |
| Agent updates can silently reduce quality | Baselines + paired statistical comparison + quality gates |
| AI tests can be inconsistent | Versioned golden datasets (content hash), repeated runs, confidence intervals |
| Evaluation can be expensive | Profiles (quick/standard/full), cost tracking per run and per successful task |
| Failures are hard to diagnose | A failure taxonomy with clustering and root-cause hypotheses |

The central design rule: **infrastructure problems are not quality failures.** An unavailable agent or a missing API key is recorded as an infra error, excluded from quality scores, and can force a regression decision to INCONCLUSIVE — never to FAIL.

## 2. Architecture

Three agent repositories are integrated through **one common adapter interface** — the evaluation engine never learns which framework an agent uses:

- **RAG with citations** (`production-rag-citations`) — via its real CLI, when its own environment + API key are present
- **Multi-agent research system** (`multi-agent-research-assistant`) — the real Supervisor, run offline on its fixture corpus
- **Cost-optimized LLM router** (`llm-cost-optimized-router`) — its real complexity + tier-decision logic, no LLM calls

Plus a deterministic **MockAdapter** with injectable degradation, so the platform itself (and the regression detector) can be tested with zero model spend.

Full diagram and component table: [docs/architecture.md](docs/architecture.md), including the honest scoping — the v1 core uses **SQLite with the PostgreSQL schema mirrored table-for-table**; Postgres/Celery/Next.js are deployment steps, not silently-claimed features.

## 3. Quick Start

```bash
# Python 3.10+, standard library only for the core
python -m eval_platform list-agents
python -m eval_platform validate-dataset datasets/router.jsonl

# evaluate the real agents, offline
python -m eval_platform run --agent llm-router --profile standard --baseline
python -m eval_platform run --agent research-agent --profile standard
python -m eval_platform run --agent rag-agent --profile quick   # records infra-unavailable offline, honestly

# regression comparison + report
python -m eval_platform compare --baseline-run 1 --run 2
python -m eval_platform report --run 2
python -m eval_platform failures --run 2

pip install pytest && pytest -q    # 47 tests
pip install streamlit && streamlit run dashboard/app.py   # dashboard over the store
```

## 4. Project Structure

```
eval_platform/
├── adapter.py            # Adapter SDK: contract, registry, normalized errors
├── adapters/             # mock · research · router · rag (thin wrappers, agents not rewritten)
├── datasets.py           # Golden Dataset Engine: validation, name@sha8 versions
├── metrics/              # deterministic + project-specific metrics
├── judges.py             # judge interface · offline LexicalJudge · optional LLMJudge
├── orchestrator.py       # profiles, timeout-guarded runs, aggregation
├── regression.py         # baselines, paired 95% CIs, quality gates, PASS/FAIL/INCONCLUSIVE
├── failures.py           # failure taxonomy, clustering, failure -> draft golden case
├── store.py              # SQLite store with reproducibility metadata
├── api.py                # optional FastAPI layer (lazily imported)
└── __main__.py           # CLI
datasets/                 # rag.jsonl (10) · research.jsonl (8) · router.jsonl (12) · mock.jsonl (3)
evaluation_configs/       # quality_gates.json · pricing.json (labelled estimates)
scripts/                  # run_evaluation.py · check_quality_gate.py (CI)
notebooks/                # 01 mock regression experiment · 02 adapter smoke runs
results/                  # baseline_summary.json · sample_report.md (real runs)
docs/                     # architecture · metrics · api · experiment_log
dashboard/                # Streamlit dashboard over the store (app · helpers · README)
tests/                    # contract, dataset, metric, regression, failure, orchestrator tests
```

## 5. How evaluation works

1. **Golden datasets** — JSONL test cases with expected facts, citation requirements, abstention expectations, routing tiers. A dataset version is `name@sha8` of the file bytes; a version used in a run is immutable, so old results stay reproducible.
2. **Metrics, not one score** — schema validity, required facts, citation presence/format/support, abstention correctness, tool correctness, latency, tokens, estimated cost — plus project-specific ones (RAG citation precision/coverage; research claim support, source diversity, contradiction surfacing; router routing accuracy, expensive-call rate). Every result carries its evaluator type/version and an explanation. Details: [docs/metrics.md](docs/metrics.md).
3. **Regression detection** — a new run is compared to the agent's baseline with *paired* per-case deltas and 95% confidence intervals (normal approximation). A quality gate fails only when the drop exceeds both the configured tolerance and what noise explains. Decisions: **PASS / FAIL / INCONCLUSIVE**.
4. **Failure analysis** — failed cases are classified (retrieval, hallucination, citation, reasoning, tool, routing, safety, performance, instruction), clustered by signature, and given a root-cause *hypothesis* for a human to confirm. Confirmed failures export as draft golden cases — never silently into a versioned dataset.
5. **CI quality gate** — `scripts/run_evaluation.py` + `scripts/check_quality_gate.py` (exit 0/1/2) run in GitHub Actions; a confirmed regression can block a merge, an outage cannot.

## 6. First results (real, offline, 2026-11-10)

| Agent | Result |
|---|---|
| Research (8 cases) | required facts 1.0 · claim support 1.0 · contradictions surfaced 1.0 · citation support 0.79 (lexical proxy — see experiment log) · rerun vs baseline: **PASS** |
| Router (12 prompts) | routing accuracy 1.0 against the real `decide_tier` · premium-tier rate 0.167 · rerun vs baseline: **PASS** |
| RAG | unavailable offline (needs its own env + Gemini key) — recorded as infra errors, **not** scored |
| Mock (controlled) | healthy → PASS; degraded (citations dropped, facts truncated) → **FAIL** with citation-failure clusters; infra-heavy run → **INCONCLUSIVE** |

Full numbers: [results/baseline_summary.json](results/baseline_summary.json) · [results/sample_report.md](results/sample_report.md) · [docs/experiment_log.md](docs/experiment_log.md).

## 7. Limitations

- Offline judges are **lexical proxies**: paraphrase reads as partial support. The findings-level metrics are stricter; an entailment/LLM judge is the planned upgrade and its slot already reports "infra unavailable" rather than inventing scores.
- The RAG integration needs the RAG repo's own environment (google-genai + Gemini key); offline it is exercised by contract tests with a stubbed subprocess.
- SQLite is the local store; the PostgreSQL deployment, Celery/Redis execution, and a Next.js front-end are later deployment steps — the current dashboard is a local Streamlit app over the store's real data (see docs/architecture.md §3).
- Golden datasets are small and hand-written; statistical claims are limited accordingly, and the experiment log says so.

## 8. Roadmap

Statistical regression detection at larger n · judge calibration against human ratings · adaptive test selection (changed components, historical failures) with periodic full runs · automatic failure clustering by semantic similarity · cost-aware evaluation profiles · dashboard over the store's real data · PostgreSQL + Celery deployment.

## License

MIT © 2026 Amna Raheeba — see [LICENSE](LICENSE).
