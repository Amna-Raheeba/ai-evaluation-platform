"""Adapter contract tests: every adapter must return the standardized
result (or a structured unavailable / normalized error), whatever the
agent does internally."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eval_platform.adapter import AdapterError, check_contract
from eval_platform.adapters.mock import MockAdapter
from eval_platform.adapters.rag import RagAdapter
from eval_platform.adapters.research import ResearchAdapter, research_repo_path
from eval_platform.adapters.router import RouterAdapter, router_repo_path

ROOT = Path(__file__).resolve().parents[1]


def test_mock_contract_and_content():
    result = MockAdapter().run({
        "question": "leave?",
        "mock_facts": ["Employees receive twenty-six days of paid vacation per year"],
        "mock_sources": [{"title": "HB", "text": "twenty-six days of paid vacation"}]})
    check_contract(result)  # raises if off-contract
    assert "twenty-six days" in result["output"]
    assert result["citations"] == ["1"]
    assert result["metadata"]["agent_name"] == "mock-agent"


def test_mock_degraded_drops_citations_and_facts():
    result = MockAdapter(degraded=True).run({
        "mock_facts": ["fact one", "fact two", "fact three"],
        "mock_sources": [{"title": "S", "text": "fact one fact two fact three"}]})
    assert result["citations"] == []
    assert "fact three" not in result["output"]


def test_mock_abstention():
    result = MockAdapter().run({"mock_abstain": True})
    assert result["status"] == "abstained"
    assert "INSUFFICIENT EVIDENCE" in result["output"]


def test_adapter_error_kinds_are_normalized():
    with pytest.raises(AdapterError) as exc:
        MockAdapter().run({}) if False else (_ for _ in ()).throw(AdapterError("infra", "down"))
    assert exc.value.kind == "infra"
    with pytest.raises(ValueError):
        AdapterError("bogus-kind", "x")


@pytest.mark.skipif(not (research_repo_path() / "research_team").exists(),
                    reason="research sibling repo not checked out")
def test_research_adapter_real_run():
    result = ResearchAdapter().run({"topic": "explainable AI in healthcare"})
    check_contract(result)
    assert result["sources"], "expected fixture sources"
    assert result["structured"]["findings"], "expected findings"
    assert "Saliency" in result["output"]
    assert result["token_usage"]["input"] > 0


@pytest.mark.skipif(not (research_repo_path() / "research_team").exists(),
                    reason="research sibling repo not checked out")
def test_research_adapter_missing_topic_is_contract_error():
    with pytest.raises(AdapterError) as exc:
        ResearchAdapter().run({})
    assert exc.value.kind == "contract"


@pytest.mark.skipif(not (router_repo_path() / "app" / "router").exists(),
                    reason="router sibling repo not checked out")
def test_router_adapter_uses_real_decision_logic():
    adapter = RouterAdapter()
    simple = adapter.run({"prompt": "What is the capital of France?"})
    check_contract(simple)
    assert simple["structured"]["tier"] == "cheap"
    hard = adapter.run({"prompt": "Analyze and compare the trade-off, justify under "
                                   "uncertainty, derive a proof of the failure mechanism "
                                   "and root cause, then evaluate and optimize the design."})
    assert hard["structured"]["tier"] == "premium"
    assert hard["structured"]["complexity_score"] >= 7


RAG_REPO = ROOT.parent / "production-rag-citations"


@pytest.mark.skipif(not (RAG_REPO / "ragcites").exists(),
                    reason="rag sibling repo not checked out")
def test_rag_adapter_unavailable_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    adapter = RagAdapter(repo_path=ROOT.parent / "production-rag-citations")
    ok, reason = adapter.configured()
    assert not ok and "key" in reason
    result = adapter.run({"question": "How many vacation days do employees get?"})
    check_contract(result)
    assert result["status"] == "unavailable"
    assert result["metadata"]["error_kind"] == "infra"


def test_rag_adapter_missing_repo(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    adapter = RagAdapter(repo_path=ROOT / "no-such-repo")
    ok, reason = adapter.configured()
    assert not ok and "not found" in reason


@pytest.mark.skipif(not (RAG_REPO / "ragcites").exists(),
                    reason="rag sibling repo not checked out")
def test_rag_adapter_configured_parses_cli_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    stdout = ("Employees receive twenty-six days of paid vacation per year [1].\n\n"
              "Citations:\n  [1] page 3: All employees receive twenty-six days of paid vacation...\n")

    def fake_runner(cmd, **kwargs):
        assert cmd[1:] == ["-m", "ragcites.pipeline", "How many vacation days do employees get?"]
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    adapter = RagAdapter(repo_path=ROOT.parent / "production-rag-citations", runner=fake_runner)
    result = adapter.run({"question": "How many vacation days do employees get?"})
    check_contract(result)
    assert result["status"] == "ok"
    assert result["citations"] == ["1"]
    assert result["sources"][0]["title"] == "page 3"
    assert "twenty-six days" in result["output"]


@pytest.mark.skipif(not (RAG_REPO / "ragcites").exists(),
                    reason="rag sibling repo not checked out")
def test_rag_adapter_cli_failure_is_infra(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    def failing_runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

    adapter = RagAdapter(repo_path=ROOT.parent / "production-rag-citations", runner=failing_runner)
    with pytest.raises(AdapterError) as exc:
        adapter.run({"question": "anything"})
    assert exc.value.kind == "infra"
