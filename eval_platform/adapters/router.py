"""RouterAdapter: wraps the cost-optimized LLM router's *decision*
logic (sibling repo `llm-cost-optimized-router`).

No LLM is called. The adapter imports the router's real
``complexity_report`` and ``decide_tier`` and returns the routing
decision as its output, so the platform can measure routing accuracy
and the expensive-call rate across a golden prompt set.

Import note: the router's config module loads python-dotenv if present.
Offline, .env files are never used here, so when dotenv is not
installed the adapter installs a no-op shim for ``load_dotenv`` before
importing — the router code itself is untouched.
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

from ..adapter import AdapterError, AgentAdapter, standard_result

PLATFORM_ROOT = Path(__file__).resolve().parents[2]

TIER_MODELS = {"cheap": "gemini-flash-lite-latest",
               "balanced": "gemini-flash-latest",
               "premium": "gemini-pro-latest"}


def _mini_yaml_module():
    """Fallback `yaml` shim with a safe_load for the simple nested
    mapping subset used by the router's two config files. Only used
    when PyYAML is not installed; the real library is always preferred.
    Not a general YAML parser."""
    import re as _re

    def _scalar(value: str):
        value = value.strip().strip('"').strip("'")
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def safe_load(text: str):
        root: dict = {}
        stack: list[tuple[int, dict]] = [(-1, root)]
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            m = _re.match(r"\s*([^:]+):\s*(.*)$", line)
            if not m:
                continue
            key, value = m.group(1).strip(), m.group(2).strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value:
                parent[key] = _scalar(value)
            else:
                parent[key] = {}
                stack.append((indent, parent[key]))
        return root

    module = types.ModuleType("yaml")
    module.safe_load = safe_load
    return module


def router_repo_path() -> Path:
    env = os.environ.get("ROUTER_REPO")
    if env:
        return Path(env)
    return PLATFORM_ROOT.parent / "llm-cost-optimized-router"


class RouterAdapter(AgentAdapter):
    name = "llm-router"
    version = "1.0.0"
    project = "router"

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = Path(repo_path) if repo_path else router_repo_path()
        self._mods = None

    def _load(self):
        if self._mods is not None:
            return self._mods
        if not (self.repo_path / "app" / "router").exists():
            raise AdapterError("infra", f"router repo not found at {self.repo_path} "
                                        f"(set ROUTER_REPO)")
        root = str(self.repo_path)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import dotenv  # noqa: F401
        except ImportError:
            shim = types.ModuleType("dotenv")
            shim.load_dotenv = lambda *a, **k: False
            sys.modules.setdefault("dotenv", shim)
        try:
            import yaml  # noqa: F401
        except ImportError:
            sys.modules.setdefault("yaml", _mini_yaml_module())
        try:
            from app.router.complexity import complexity_report
            from app.router.engine import decide_tier
            from app.router.policies import load_policy
            policy = load_policy()
        except Exception as exc:
            raise AdapterError("infra", f"cannot import router decision logic: {exc}") from exc
        self._mods = (complexity_report, decide_tier, policy)
        return self._mods

    def run(self, input_data: dict) -> dict:
        complexity_report, decide_tier, policy = self._load()
        prompt = input_data.get("prompt") or input_data.get("question") or ""
        if not prompt:
            raise AdapterError("contract", "router input needs a 'prompt'")
        started = time.time()
        report = complexity_report(prompt)
        decision = decide_tier(prompt, input_data.get("quality"), policy)
        model = TIER_MODELS.get(decision.tier, "unknown")
        output = (f"Routed to tier '{decision.tier}' (model {model}) with "
                  f"complexity score {decision.complexity_score}/10. "
                  f"Reasons: {'; '.join(decision.reasons)}.")
        return standard_result(
            output, agent_name=self.name, agent_version=self.version,
            model=model, started=started,
            input_tokens=max(1, len(prompt) // 4), output_tokens=0, tool_calls=[],
            structured={"tier": decision.tier,
                        "complexity_score": decision.complexity_score,
                        "complexity_reasons": list(report.reasons),
                        "decision_reasons": list(decision.reasons)})
