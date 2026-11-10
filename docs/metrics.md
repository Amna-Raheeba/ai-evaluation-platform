# Metrics

Every metric result records: test case, agent, score, pass/fail,
evaluator type + version, and a human-readable explanation. There is
no single blended score — a blended score hides exactly the failures
this platform exists to diagnose.

## 1. Deterministic metrics (evaluator `deterministic@1.0`)

| Metric | What it checks |
|---|---|
| `schema_validity` | Adapter result follows the standardized contract |
| `required_facts` | Fraction of the case's required facts present in the output (normalized substring match) |
| `citation_presence` | Citations exist when the case requires them |
| `citation_format` | Bracketed references are valid markers (`[1]`, `[S3]`) |
| `citation_support` | Cited claims share vocabulary with their cited source (lexical overlap ≥ 0.25 = supported) |
| `abstention_correctness` | Agent abstains exactly when the case says it should |
| `tool_correctness` | Tool calls stay inside the case's allowed set |
| `latency` | Wall-clock seconds per case (lower is better) |
| `token_usage` | Input + output tokens as reported/estimated by the adapter |
| `cost_estimate` | Token cost at the list prices in `evaluation_configs/pricing.json` — **estimates, not invoices** |

## 2. Project-specific metrics

| Agent | Metrics |
|---|---|
| RAG | `citation_precision` (supported citations / citations), `citation_coverage` (claims carrying a citation / claims) |
| Research | `claim_support_rate` (findings supported by their linked sources), `source_diversity` (sources across venue/type groups, informational), `contradiction_surfaced` (conflicting evidence flagged exactly when expected) |
| Router | `routing_accuracy` (chosen tier = expected tier), `expensive_call_rate` (share routed to the premium tier, lower is better) |

## 3. Judges

Judge results use the roadmap's shape — `{metric, score, passed,
reason, evidence}` — stored as metric results with
`evaluator_type` set to the judge kind.

- **LexicalJudge** (`lexical_judge@1.0`, offline, deterministic):
  `lexical_faithfulness` = share of output sentences whose vocabulary
  overlaps the returned sources / expected facts;
  `lexical_relevance` = share of the input's content words covered by
  the output. These are *proxies*: paraphrase reads as partial
  support, which is why `citation_support` passes at 0.7 while the
  findings-based `claim_support_rate` is stricter. Skipped for
  abstentions and for routing decisions, where the proxies are
  meaningless.
- **LLMJudge** (optional): with no API key it raises an infrastructure
  error. An unavailable judge must never become a made-up score.

## 4. Quality gates

`evaluation_configs/quality_gates.json` sets, per quality metric, a
`minimum_score` for the new run and a `maximum_regression` for the
paired delta, plus ceilings for `latency` and `cost_estimate`. A gate
fails a run only when the violation is real: for regressions, the
paired 95% confidence interval (normal approximation) must also
exclude no-change. See `eval_platform/regression.py`.
