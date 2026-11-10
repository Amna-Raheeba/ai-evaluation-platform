"""Streamlit dashboard for the AI evaluation platform.

This is a presentation layer over the real SQLite store — every
number here comes from ``eval_platform.store.Store`` via
``dashboard/helpers.py``: run summaries, per-case metric results,
baseline comparisons, failure clusters, and cost estimates. The
sidebar button runs a genuine quick evaluation of the MockAdapter
through the orchestrator and writes it to the same store.

Run from the project root:

    pip install streamlit
    streamlit run dashboard/app.py

Streamlit is a dashboard-only dependency — the core package
(``eval_platform``) stays standard-library only.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts this script's folder on sys.path, not the project
# root, so `import eval_platform` / `import helpers` need both added.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st  # noqa: E402

import helpers  # noqa: E402

st.set_page_config(page_title="AI Evaluation Platform", layout="wide")
st.title("AI Evaluation Platform")
st.caption(
    "Local SQLite store (eval_platform.db). Costs are pricing-config "
    "estimates, not invoices. Offline judge scores are lexical proxies."
)

DB_PATH = helpers.default_db_path()
BASELINE_PATH = PROJECT_ROOT / "results" / "baseline_summary.json"

with st.sidebar:
    st.header("Store")
    st.write(f"**Database:** `{DB_PATH}`")
    st.caption(
        "The path honours the EVAL_PLATFORM_DB environment variable, "
        "matching the CLI."
    )
    if st.button("Run quick evaluation (mock)", type="primary"):
        with st.spinner("Running the mock-agent quick evaluation…"):
            summary = helpers.run_mock_evaluation(DB_PATH)
        pass_rate = summary.get("pass_rate")
        if pass_rate is not None:
            st.success(
                f"Run {summary.get('run_id')} finished — "
                f"pass rate {pass_rate:.0%} "
                f"({summary.get('scored')} scored, "
                f"{summary.get('infra_errors')} infra errors)."
            )
        else:
            st.success(f"Run {summary.get('run_id')} finished.")
        st.rerun()

ov = helpers.overview(DB_PATH)
baseline_summary = helpers.load_baseline_summary(BASELINE_PATH)

if not ov["db_exists"] and not baseline_summary:
    st.warning(
        "No evaluation store found yet and no fallback summary exists. "
        "Run the CLI (for example `python -m eval_platform run --agent "
        "llm-router --profile standard`) or press **Run quick evaluation "
        "(mock)** in the sidebar to create the first run."
    )
    st.stop()

if not ov["db_exists"] and baseline_summary:
    st.info(
        "No local database found — showing the saved fallback summary "
        "from `results/baseline_summary.json`. Run an evaluation (or the "
        "mock button in the sidebar) to create the live store."
    )
    for key in ("research_agent", "llm_router", "rag_agent"):
        entry = baseline_summary.get(key)
        if not entry:
            continue
        st.subheader(key)
        flat = {
            k: v for k, v in entry.items() if not isinstance(v, (dict, list))
        }
        if flat:
            st.dataframe([flat], use_container_width=True, hide_index=True)
        metric_means = entry.get("metric_means")
        if metric_means:
            st.dataframe(
                [{"metric": m, "mean": v} for m, v in sorted(metric_means.items())],
                use_container_width=True,
                hide_index=True,
            )
    st.stop()

# ------------------------------------------------------------- overview
st.header("Overview")
decisions = ov.get("decisions", {})
cols = st.columns(7)
cols[0].metric("Agents", len(ov.get("agents", [])))
cols[1].metric("Total runs", ov.get("total_runs", 0))
cols[2].metric("PASS decisions", decisions.get("PASS", 0))
cols[3].metric("FAIL decisions", decisions.get("FAIL", 0))
cols[4].metric("INCONCLUSIVE", decisions.get("INCONCLUSIVE", 0))
cols[5].metric("Total est. cost", f"${ov.get('total_cost_usd', 0.0):.4f}")
cols[6].metric("Latest run", ov.get("latest_run_id") if ov.get("latest_run_id") is not None else "—")
if ov.get("agents"):
    st.caption("Agents in store: " + ", ".join(ov["agents"]))

# ------------------------------------------------------ agent comparison
st.header("Agent comparison")
st.caption("Each agent's latest run only.")
comparison = helpers.agent_comparison(DB_PATH)
if comparison:
    st.dataframe(
        [
            {
                "agent": r["agent"],
                "run_id": r["run_id"],
                "agent_version": r["agent_version"],
                "dataset_version": r["dataset_version"],
                "profile": r["profile"],
                "status": r["status"],
                "pass_rate": r["pass_rate"],
                "scored": r["scored"],
                "infra_errors": r["infra_errors"],
                "total_cost_usd": r["total_cost_usd"],
                "latency_p95_s": r["latency_p95_s"],
            }
            for r in comparison
        ],
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Metric means by agent (latest run)"):
        for r in comparison:
            st.write(f"**{r['agent']}** (run {r['run_id']})")
            means = r.get("metric_means") or {}
            if means:
                st.dataframe(
                    [{"metric": m, "mean": v} for m, v in sorted(means.items())],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No scored metrics for this run.")
else:
    st.info("No runs in the store yet.")

# ----------------------------------------------------------- run details
st.header("Run details")
options = helpers.run_options(DB_PATH)
if options:
    labels = {
        o["id"]: (
            f"Run {o['id']} — {o['agent_name']} · {o['profile']} · "
            f"{o['status']} · pass rate {o['pass_rate']}"
        )
        for o in options
    }
    selected_run = st.selectbox(
        "Run", [o["id"] for o in options], format_func=lambda rid: labels[rid]
    )
    case_rows = helpers.run_case_rows(DB_PATH, selected_run)
    if case_rows:
        for row in case_rows:
            title = (
                f"{row['case_key']} — {row['status']} "
                f"(repeat {row['repeat_idx']})"
            )
            with st.expander(title):
                st.write(
                    f"Execution time: {row['execution_time_s']}s · "
                    f"error kind: {row['error_kind'] or '—'}"
                )
                if row["metrics"]:
                    st.dataframe(
                        row["metrics"],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No metric results for this case.")
    else:
        st.info("No per-case results stored for this run.")
else:
    st.info("No runs to inspect yet.")

# ------------------------------------------------------ regression center
st.header("Regression center")
st.caption(
    "Latest run vs the agent's saved baseline, with paired deltas and "
    "95% confidence intervals from the regression engine."
)
# Agents that actually have a baseline, checked via the store helpers.
_baseline_agents: list[str] = []
for _agent in ov.get("agents", []):
    # regression_rows returns None both for "no baseline" and for
    # "baseline == latest"; to list candidates we probe with a direct
    # store check through agent_comparison-independent logic: an agent
    # is a candidate if it appears in the comparison and a regression
    # probe or the baseline==latest case applies. We surface every
    # agent and let the section below explain the specific state.
    _baseline_agents.append(_agent)

# Narrow to agents with a stored baseline by inspecting the store
# through helpers' Store access pattern: reuse regression probe plus
# a baseline existence check via a lightweight helper call chain.
# (Kept in the app so helpers stay exactly as specified.)
try:
    from eval_platform.store import Store as _Store

    _store = _Store(DB_PATH)
    try:
        _with_baseline = [
            a for a in ov.get("agents", []) if _store.get_baseline(a) is not None
        ]
    finally:
        _store.close()
except Exception:
    _with_baseline = _baseline_agents

if _with_baseline:
    reg_agent = st.selectbox("Agent (with baseline)", _with_baseline)
    reg = helpers.regression_rows(DB_PATH, reg_agent)
    if reg is None:
        st.info(
            f"{reg_agent}: the latest run is the baseline run itself — "
            "run another evaluation to compare against it."
        )
    else:
        st.write(
            f"Baseline run {reg['baseline_run_id']} → new run {reg['new_run_id']}"
        )
        if reg["decision"] == "PASS":
            st.success(f"Decision: {reg['decision']}")
        elif reg["decision"] == "FAIL":
            st.error(f"Decision: {reg['decision']}")
        else:
            st.warning(f"Decision: {reg['decision']}")
        if reg["rows"]:
            st.dataframe(reg["rows"], use_container_width=True, hide_index=True)
        if reg.get("reasons"):
            st.write("**Reasons**")
            for reason in reg["reasons"]:
                st.write(f"- {reason}")
else:
    st.info("No agent has a saved baseline yet, so there is nothing to compare.")

# --------------------------------------------------------------- failures
st.header("Failures")
totals = helpers.failure_category_totals(DB_PATH)
if totals:
    st.subheader("Category totals (all runs)")
    st.dataframe(totals, use_container_width=True, hide_index=True)
else:
    st.info("No failure clusters stored — every scored run so far passed, "
            "or failures have not been clustered yet.")
if options:
    fail_run = st.selectbox(
        "Run for clusters",
        [o["id"] for o in options],
        format_func=lambda rid: labels[rid],
        key="failure_run_select",
    )
    clusters = helpers.failure_clusters(DB_PATH, fail_run)
    if clusters:
        st.dataframe(
            [
                {
                    "category": c["category"],
                    "signature": c["signature"],
                    "case_count": c["case_count"],
                    "hypothesis": c["hypothesis"],
                    "case_keys": ", ".join(c["case_keys"]),
                }
                for c in clusters
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(f"No failure clusters for run {fail_run}.")

# ---------------------------------------------------- cost & performance
st.header("Cost & performance")
st.caption("Costs are estimates from evaluation_configs/pricing.json, not invoices.")
costs = helpers.cost_rows(DB_PATH)
if costs:
    st.dataframe(costs, use_container_width=True, hide_index=True)
    chart_data = {r["agent"]: r["total_cost_usd"] for r in costs}
    if chart_data and any(v for v in chart_data.values()):
        st.bar_chart(chart_data)
else:
    st.info("No cost data yet.")
