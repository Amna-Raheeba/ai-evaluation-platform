"""ResearchAdapter: thin wrapper around the multi-agent research
assistant (sibling repo `multi-agent-research-assistant`).

It imports the real Supervisor and runs it offline against its fixture
corpus — nothing is reimplemented here. The repo is located via the
RESEARCH_REPO environment variable or as a sibling checkout.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..adapter import AdapterError, AgentAdapter, standard_result

PLATFORM_ROOT = Path(__file__).resolve().parents[2]


def research_repo_path() -> Path:
    env = os.environ.get("RESEARCH_REPO")
    if env:
        return Path(env)
    return PLATFORM_ROOT.parent / "multi-agent-research-assistant"


class ResearchAdapter(AgentAdapter):
    name = "research-agent"
    version = "0.3.0"
    project = "research"

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = Path(repo_path) if repo_path else research_repo_path()
        self._supervisor_cls = None
        self._settings_cls = None

    def _load(self):
        if self._supervisor_cls is not None:
            return
        if not (self.repo_path / "research_team").exists():
            raise AdapterError("infra", f"research repo not found at {self.repo_path} "
                                        f"(set RESEARCH_REPO)")
        root = str(self.repo_path)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from research_team.config import Settings
            from research_team.supervisor import Supervisor
        except Exception as exc:  # missing deps etc. = infrastructure
            raise AdapterError("infra", f"cannot import research_team: {exc}") from exc
        self._supervisor_cls, self._settings_cls = Supervisor, Settings

    def run(self, input_data: dict) -> dict:
        self._load()
        topic = input_data.get("topic") or input_data.get("question") or ""
        if not topic:
            raise AdapterError("contract", "research input needs a 'topic'")
        started = time.time()
        import tempfile
        settings = self._settings_cls(
            runs_dir=__import__("pathlib").Path(tempfile.mkdtemp(prefix="eval-runs-")))
        supervisor = self._supervisor_cls(settings)
        try:
            state = supervisor.run(topic, save=False)
        except Exception as exc:
            raise AdapterError("infra", f"research run failed: {exc}") from exc
        budget = supervisor.budget.summary()
        in_tok = sum(v["input_tokens"] for v in budget["by_agent"].values())
        out_tok = sum(v["output_tokens"] for v in budget["by_agent"].values())
        sources = [{"id": s.source_id, "title": s.title,
                    "text": (s.summary + " " + " ".join(s.key_points)).strip(),
                    "venue": s.venue, "source_type": s.source_type}
                   for s in state.sources]
        findings = [{"claim": f.claim, "source_ids": f.source_ids,
                     "confidence": f.confidence} for f in state.findings]
        tool_calls = [{"name": ev.get("agent", "agent"), "action": ev.get("action", "")}
                      for ev in state.trace]
        return standard_result(
            state.draft or "", agent_name=self.name, agent_version=self.version,
            model=settings.model_name, started=started,
            input_tokens=in_tok, output_tokens=out_tok, tool_calls=tool_calls,
            status="ok" if state.status == "complete" else "abstained",
            sources=sources,
            structured={"findings": findings,
                        "contradictions": state.contradictions,
                        "status": state.status,
                        "critique_approved": bool(state.critique and state.critique.approved),
                        "budget": budget})
