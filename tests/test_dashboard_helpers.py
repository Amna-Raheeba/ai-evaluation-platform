"""Tests for dashboard/helpers.py — the dashboard's shaping logic.

The helpers module is loaded by file path because ``dashboard/`` is
not a package (Streamlit apps are run as scripts). Loading it this
way also proves the module has no streamlit import: this test run
has no streamlit installed.
"""
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "dashboard_helpers", PROJECT_ROOT / "dashboard" / "helpers.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)

from eval_platform.store import Store  # noqa: E402


def test_overview_missing_db_does_not_create_file(tmp_path):
    db = tmp_path / "missing.db"
    result = helpers.overview(db)
    assert result["db_exists"] is False
    assert result["agents"] == []
    assert result["total_runs"] == 0
    assert result["runs_by_status"] == {}
    assert result["decisions"] == {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
    assert result["total_cost_usd"] == 0.0
    assert result["latest_run_id"] is None
    assert not db.exists()


def test_load_baseline_summary_reads_real_file_and_missing_returns_empty(tmp_path):
    summary = helpers.load_baseline_summary(
        PROJECT_ROOT / "results" / "baseline_summary.json")
    assert summary["research_agent"]["regression_decision_vs_baseline"] == "PASS"
    assert helpers.load_baseline_summary(tmp_path / "nope.json") == {}


def test_run_mock_evaluation_and_agent_comparison(tmp_path):
    db = tmp_path / "eval.db"
    summary = helpers.run_mock_evaluation(db)
    assert summary["run_id"] is not None
    ov = helpers.overview(db)
    assert ov["db_exists"] is True
    assert ov["total_runs"] >= 1
    comparison = helpers.agent_comparison(db)
    mock_rows = [r for r in comparison if r["agent"] == "mock-agent"]
    assert len(mock_rows) == 1
    assert mock_rows[0]["pass_rate"] == 1.0


def test_run_case_rows_for_mock_run(tmp_path):
    db = tmp_path / "eval.db"
    summary = helpers.run_mock_evaluation(db)
    rows = helpers.run_case_rows(db, summary["run_id"])
    assert len(rows) == 3
    for row in rows:
        assert row["case_key"]
        assert row["metrics"], f"no metrics for {row['case_key']}"
        for metric in row["metrics"]:
            assert "metric_name" in metric
            assert "score" in metric
            assert "explanation" in metric


def test_regression_rows_after_setting_baseline(tmp_path):
    db = tmp_path / "eval.db"
    first = helpers.run_mock_evaluation(db)
    helpers.run_mock_evaluation(db)
    store = Store(db)
    try:
        store.set_baseline("mock-agent", first["run_id"])
    finally:
        store.close()
    result = helpers.regression_rows(db, "mock-agent")
    assert result is not None
    assert result["decision"] == "PASS"
    assert result["baseline_run_id"] == first["run_id"]
    assert result["rows"]
    for row in result["rows"]:
        assert "metric" in row
        assert "delta" in row
        assert "ci95_half_width" in row


def test_failure_category_totals_on_empty_store(tmp_path):
    db = tmp_path / "empty.db"
    store = Store(db)
    store.close()
    assert helpers.failure_category_totals(db) == []
