# API

Optional FastAPI layer (`pip install -e ".[api]"`, then serve
`eval_platform.api:create_app`). The core works fully without it via
the CLI. For a public deployment, put authentication in front of these
endpoints — the local v1 ships without auth by design and says so.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/agents` | Register an agent |
| GET | `/api/agents` | List adapters + registered agents |
| POST | `/api/evaluations/run` | Start an evaluation run (`{agent, dataset, profile, repeats, baseline}`) |
| GET | `/api/evaluations/{id}` | Run status + summary |
| GET | `/api/evaluations/{id}/results` | Test results + metric results |
| POST | `/api/regressions/compare` | Compare two runs (`{baseline_run, run}`) → PASS/FAIL/INCONCLUSIVE |
| GET | `/api/analytics/trends` | Quality trends across runs (`?agent=`) |
| GET | `/api/failures` | Failure clusters for a run (`?run_id=`) |

Every run records reproducibility metadata with its results: agent
version, dataset version (content hash), profile/repeats, evaluator
versions, and the config snapshot.
