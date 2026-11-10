"""Optional FastAPI layer over the platform core.

FastAPI is an optional dependency (`pip install -e ".[api]"`). This
module imports cleanly without it; create_app() raises a clear error
only when the app is actually requested. The core (CLI, orchestrator,
store) never depends on this module.
"""
from __future__ import annotations

import os

from .adapter import default_registry
from .orchestrator import Orchestrator, load_gates
from .regression import compare_runs
from .store import Store


def create_app(db_path: str | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("FastAPI layer requested but fastapi is not installed; "
                           "install with: pip install -e \".[api]\"") from exc

    store = Store(db_path or os.environ.get("EVAL_PLATFORM_DB", "eval_platform.db"))
    registry = default_registry()
    orch = Orchestrator(store, registry)
    app = FastAPI(title="AI Evaluation & Regression Platform", version="0.1.0")

    class RunRequest(BaseModel):
        agent: str
        dataset: str
        profile: str = "standard"
        repeats: int = 1
        baseline: bool = False

    class CompareRequest(BaseModel):
        baseline_run: int
        run: int

    @app.post("/api/agents")
    def register_agent(payload: dict):
        store.register_agent(payload["name"], payload.get("project", ""),
                             payload.get("version", "0.0.0"), payload.get("model", ""))
        return {"ok": True}

    @app.get("/api/agents")
    def list_agents():
        return {"agents": registry.infos(), "registered": store.list_agents()}

    @app.post("/api/evaluations/run")
    def run_evaluation(req: RunRequest):
        try:
            return orch.run_evaluation(req.agent, req.dataset, profile=req.profile,
                                       repeats=req.repeats,
                                       set_baseline_flag=req.baseline)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/evaluations/{run_id}")
    def get_evaluation(run_id: int):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/api/evaluations/{run_id}/results")
    def get_results(run_id: int):
        return {"test_results": store.get_test_results(run_id),
                "metric_results": store.get_metric_results(run_id)}

    @app.post("/api/regressions/compare")
    def compare(req: CompareRequest):
        report = compare_runs(store, req.baseline_run, req.run, load_gates())
        store.save_regression(req.baseline_run, req.run, report["decision"], report)
        return report

    @app.get("/api/analytics/trends")
    def trends(agent: str | None = None):
        return {"trends": store.trends(agent)}

    @app.get("/api/failures")
    def failures(run_id: int):
        return {"clusters": store.get_failure_clusters(run_id)}

    return app
