"""RagAdapter: wraps the cited RAG system (sibling repo
`production-rag-citations`), whose pipeline needs the google-genai SDK
and a Gemini API key — neither is available in an offline environment.

Contract:
  - Configured (env RAGCITES_REPO points at the repo AND a Gemini key
    is present in the environment): run the real CLI
    ``python -m ragcites.pipeline "<question>"`` as a subprocess and
    parse its answer + citations into the standard result.
  - Not configured: return a structured UNAVAILABLE result. The
    orchestrator records it as an INFRASTRUCTURE error — never as a
    quality failure, never scored.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ..adapter import AdapterError, AgentAdapter, standard_result

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_CITATION_LINE = re.compile(r"\[(\d+)\]\s+page\s+(\d+):\s*(.*)", re.I)


def rag_repo_path() -> Path:
    env = os.environ.get("RAGCITES_REPO")
    if env:
        return Path(env)
    return PLATFORM_ROOT.parent / "production-rag-citations"


class RagAdapter(AgentAdapter):
    name = "rag-agent"
    version = "1.0.0"
    project = "rag"

    def __init__(self, repo_path: Path | None = None, timeout_s: float = 120.0,
                 runner=None):
        self.repo_path = Path(repo_path) if repo_path else rag_repo_path()
        self.timeout_s = timeout_s
        self._runner = runner or subprocess.run  # injectable for contract tests

    def configured(self) -> tuple[bool, str]:
        if not (self.repo_path / "ragcites" / "pipeline.py").exists():
            return False, f"ragcites repo not found at {self.repo_path} (set RAGCITES_REPO)"
        if not any(os.environ.get(k) for k in KEY_VARS):
            return False, "no Gemini API key in environment (GEMINI_API_KEY/GOOGLE_API_KEY)"
        return True, "configured"

    def run(self, input_data: dict) -> dict:
        question = input_data.get("question") or input_data.get("prompt") or ""
        if not question:
            raise AdapterError("contract", "rag input needs a 'question'")
        started = time.time()
        ok, reason = self.configured()
        if not ok:
            result = standard_result(
                "", agent_name=self.name, agent_version=self.version,
                model="gemini (ragcites pipeline)", started=started,
                status="unavailable")
            result["metadata"]["error_kind"] = "infra"
            result["metadata"]["unavailable_reason"] = reason
            return result
        try:
            proc = self._runner(
                [sys.executable, "-m", "ragcites.pipeline", question],
                cwd=str(self.repo_path), capture_output=True, text=True,
                timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise AdapterError("timeout", f"ragcites CLI exceeded {self.timeout_s}s") from exc
        except Exception as exc:
            raise AdapterError("infra", f"ragcites subprocess failed: {exc}") from exc
        if proc.returncode != 0:
            raise AdapterError("infra", f"ragcites CLI exited {proc.returncode}: "
                                        f"{(proc.stderr or '')[:300]}")
        return self._parse(proc.stdout or "", started)

    def _parse(self, stdout: str, started: float) -> dict:
        lines = stdout.splitlines()
        answer_lines, citations, sources = [], [], []
        in_citations = False
        ungrounded = None
        for line in lines:
            if line.strip().lower() == "citations:":
                in_citations = True
                continue
            if line.strip().startswith("(ungrounded:"):
                ungrounded = line.strip()
                continue
            m = _CITATION_LINE.search(line) if in_citations else None
            if m:
                citations.append(m.group(1))
                sources.append({"id": m.group(1), "title": f"page {m.group(2)}",
                                "text": m.group(3).removesuffix("...")})
            elif not in_citations:
                answer_lines.append(line)
        answer = "\n".join(answer_lines).strip()
        status = "abstained" if ungrounded else "ok"
        return standard_result(
            answer, agent_name=self.name, agent_version=self.version,
            model="gemini (ragcites pipeline)", started=started,
            input_tokens=max(1, len(answer) // 4), output_tokens=max(1, len(answer) // 4),
            status=status, sources=sources, citations=citations,
            structured={"ungrounded_note": ungrounded})
