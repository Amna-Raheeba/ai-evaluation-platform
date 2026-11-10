"""Results store: SQLite (stdlib) for the local core.

The schema mirrors the PostgreSQL design in docs/architecture.md
table-for-table, so moving to Postgres for deployment is a connection
change, not a redesign. Every evaluation run records the metadata
needed to reproduce it: agent version, dataset version (content hash),
profile/repeats, evaluator versions and the full config snapshot.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, project TEXT,
    created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS agent_versions (
    id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL REFERENCES agents(id),
    version TEXT NOT NULL, model TEXT, UNIQUE(agent_id, version));
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, project TEXT);
CREATE TABLE IF NOT EXISTS dataset_versions (
    id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    version TEXT NOT NULL, content_hash TEXT NOT NULL, case_count INTEGER,
    UNIQUE(dataset_id, version));
CREATE TABLE IF NOT EXISTS test_cases (
    id INTEGER PRIMARY KEY, dataset_version_id INTEGER NOT NULL,
    case_key TEXT NOT NULL, category TEXT, difficulty TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY, run_key TEXT UNIQUE NOT NULL,
    agent_id INTEGER NOT NULL, agent_version TEXT, dataset_version TEXT,
    profile TEXT, repeats INTEGER, status TEXT, started_at REAL, finished_at REAL,
    config TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, case_key TEXT NOT NULL,
    repeat_idx INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
    output TEXT, execution_time_s REAL, input_tokens INTEGER, output_tokens INTEGER,
    error_kind TEXT, error_message TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS metric_results (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, test_result_id INTEGER,
    case_key TEXT NOT NULL, metric_name TEXT NOT NULL, score REAL,
    passed INTEGER, evaluator_type TEXT, evaluator_version TEXT,
    explanation TEXT, raw TEXT);
CREATE TABLE IF NOT EXISTS baselines (
    id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL, run_id INTEGER NOT NULL,
    created_at REAL NOT NULL, UNIQUE(agent_id));
CREATE TABLE IF NOT EXISTS regressions (
    id INTEGER PRIMARY KEY, baseline_run_id INTEGER NOT NULL, new_run_id INTEGER NOT NULL,
    decision TEXT NOT NULL, report TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS failure_clusters (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, category TEXT NOT NULL,
    signature TEXT NOT NULL, case_count INTEGER, hypothesis TEXT, case_keys TEXT);
CREATE TABLE IF NOT EXISTS quality_gates (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, policy TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS judge_versions (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
    UNIQUE(name, version));
CREATE TABLE IF NOT EXISTS human_feedback (
    id INTEGER PRIMARY KEY, run_id INTEGER, case_key TEXT, verdict TEXT,
    note TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, event TEXT NOT NULL, detail TEXT);
"""


class Store:
    def __init__(self, path: str | Path = "eval_platform.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -- registry ------------------------------------------------------
    def register_agent(self, name: str, project: str, version: str, model: str = "") -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO agents(name, project, created_at) VALUES (?,?,?)",
                    (name, project, time.time()))
        agent_id = cur.execute("SELECT id FROM agents WHERE name=?", (name,)).fetchone()[0]
        cur.execute("INSERT OR IGNORE INTO agent_versions(agent_id, version, model) VALUES (?,?,?)",
                    (agent_id, version, model))
        self.conn.commit()
        return agent_id

    def list_agents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT a.name, a.project, av.version, av.model FROM agents a "
            "LEFT JOIN agent_versions av ON av.agent_id=a.id ORDER BY a.name").fetchall()
        return [dict(r) for r in rows]

    def agent_id(self, name: str) -> int | None:
        row = self.conn.execute("SELECT id FROM agents WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    # -- datasets --------------------------------------------------------
    def register_dataset(self, name: str, project: str, version: str,
                         content_hash: str, cases: list[dict]) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO datasets(name, project) VALUES (?,?)", (name, project))
        ds_id = cur.execute("SELECT id FROM datasets WHERE name=?", (name,)).fetchone()[0]
        existing = cur.execute("SELECT id FROM dataset_versions WHERE dataset_id=? AND version=?",
                               (ds_id, version)).fetchone()
        if existing:
            return  # immutable once registered: same version, same cases
        cur.execute("INSERT INTO dataset_versions(dataset_id, version, content_hash, case_count) "
                    "VALUES (?,?,?,?)", (ds_id, version, content_hash, len(cases)))
        dv_id = cur.lastrowid
        for case in cases:
            cur.execute("INSERT INTO test_cases(dataset_version_id, case_key, category, difficulty, payload) "
                        "VALUES (?,?,?,?,?)",
                        (dv_id, case["id"], case.get("category"), case.get("difficulty"),
                         json.dumps(case)))
        self.conn.commit()

    # -- runs ------------------------------------------------------------
    def start_run(self, run_key: str, agent_name: str, agent_version: str,
                  dataset_version: str, profile: str, repeats: int,
                  config: dict) -> int:
        agent_id = self.register_agent(agent_name, config.get("project", ""),
                                       agent_version, config.get("model", ""))
        cur = self.conn.execute(
            "INSERT INTO evaluation_runs(run_key, agent_id, agent_version, dataset_version, "
            "profile, repeats, status, started_at, config) VALUES (?,?,?,?,?,?, 'running', ?, ?)",
            (run_key, agent_id, agent_version, dataset_version, profile, repeats,
             time.time(), json.dumps(config)))
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, summary: dict) -> None:
        self.conn.execute("UPDATE evaluation_runs SET status=?, finished_at=?, summary=? WHERE id=?",
                          (status, time.time(), json.dumps(summary), run_id))
        self.conn.commit()

    def save_test_result(self, run_id: int, case_key: str, repeat_idx: int,
                         status: str, result: dict | None,
                         error_kind: str | None = None, error_message: str = "") -> int:
        result = result or {}
        cur = self.conn.execute(
            "INSERT INTO test_results(run_id, case_key, repeat_idx, status, output, "
            "execution_time_s, input_tokens, output_tokens, error_kind, error_message, payload) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, case_key, repeat_idx, status, result.get("output", ""),
             result.get("execution_time_s"), result.get("token_usage", {}).get("input"),
             result.get("token_usage", {}).get("output"), error_kind, error_message,
             json.dumps(result)))
        self.conn.commit()
        return cur.lastrowid

    def save_metric_results(self, run_id: int, test_result_id: int, metrics: list) -> None:
        self.conn.executemany(
            "INSERT INTO metric_results(run_id, test_result_id, case_key, metric_name, score, "
            "passed, evaluator_type, evaluator_version, explanation, raw) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(run_id, test_result_id, m.test_case_id, m.metric_name, m.score,
              None if m.passed is None else int(m.passed), m.evaluator_type,
              m.evaluator_version, m.explanation, json.dumps(m.raw)) for m in metrics])
        self.conn.commit()

    def get_run(self, run_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT r.*, a.name AS agent_name FROM evaluation_runs r "
            "JOIN agents a ON a.id=r.agent_id WHERE r.id=?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = json.loads(d["config"] or "{}")
        d["summary"] = json.loads(d["summary"] or "{}")
        return d

    def latest_run(self, agent_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT r.id FROM evaluation_runs r JOIN agents a ON a.id=r.agent_id "
            "WHERE a.name=? ORDER BY r.id DESC LIMIT 1", (agent_name,)).fetchone()
        return self.get_run(row[0]) if row else None

    def list_runs(self, agent_name: str | None = None) -> list[dict]:
        q = ("SELECT r.id, a.name AS agent_name, r.agent_version, r.dataset_version, "
             "r.profile, r.status, r.started_at, r.summary FROM evaluation_runs r "
             "JOIN agents a ON a.id=r.agent_id ")
        args: tuple = ()
        if agent_name:
            q += "WHERE a.name=? "
            args = (agent_name,)
        rows = self.conn.execute(q + "ORDER BY r.id", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["summary"] = json.loads(d["summary"] or "{}")
            out.append(d)
        return out

    def get_test_results(self, run_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM test_results WHERE run_id=? ORDER BY id",
                                 (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"] or "{}")
            out.append(d)
        return out

    def get_metric_results(self, run_id: int, case_key: str | None = None) -> list[dict]:
        q = "SELECT * FROM metric_results WHERE run_id=?"
        args: list = [run_id]
        if case_key:
            q += " AND case_key=?"
            args.append(case_key)
        rows = self.conn.execute(q + " ORDER BY id", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["passed"] = None if d["passed"] is None else bool(d["passed"])
            d["raw"] = json.loads(d["raw"] or "{}")
            out.append(d)
        return out

    # -- baselines & regressions ------------------------------------------
    def set_baseline(self, agent_name: str, run_id: int) -> None:
        agent_id = self.agent_id(agent_name)
        self.conn.execute("INSERT OR REPLACE INTO baselines(agent_id, run_id, created_at) "
                          "VALUES (?,?,?)", (agent_id, run_id, time.time()))
        self.conn.commit()
        self.audit("baseline_set", {"agent": agent_name, "run_id": run_id})

    def get_baseline(self, agent_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT b.run_id FROM baselines b JOIN agents a ON a.id=b.agent_id WHERE a.name=?",
            (agent_name,)).fetchone()
        return self.get_run(row[0]) if row else None

    def save_regression(self, baseline_run_id: int, new_run_id: int,
                        decision: str, report: dict) -> None:
        self.conn.execute("INSERT INTO regressions(baseline_run_id, new_run_id, decision, report, created_at) "
                          "VALUES (?,?,?,?,?)",
                          (baseline_run_id, new_run_id, decision, json.dumps(report), time.time()))
        self.conn.commit()

    # -- failures ----------------------------------------------------------
    def save_failure_clusters(self, run_id: int, clusters: list[dict]) -> None:
        self.conn.executemany(
            "INSERT INTO failure_clusters(run_id, category, signature, case_count, hypothesis, case_keys) "
            "VALUES (?,?,?,?,?,?)",
            [(run_id, c["category"], c["signature"], len(c["case_keys"]), c["hypothesis"],
              json.dumps(c["case_keys"])) for c in clusters])
        self.conn.commit()

    def get_failure_clusters(self, run_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM failure_clusters WHERE run_id=? ORDER BY id",
                                 (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["case_keys"] = json.loads(d["case_keys"] or "[]")
            out.append(d)
        return out

    # -- gates, feedback, audit ---------------------------------------------
    def save_quality_gate_policy(self, name: str, policy: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO quality_gates(name, policy) VALUES (?,?)",
                          (name, json.dumps(policy)))
        self.conn.commit()

    def add_feedback(self, run_id: int, case_key: str, verdict: str, note: str = "") -> None:
        self.conn.execute("INSERT INTO human_feedback(run_id, case_key, verdict, note, created_at) "
                          "VALUES (?,?,?,?,?)", (run_id, case_key, verdict, note, time.time()))
        self.conn.commit()

    def audit(self, event: str, detail: Any) -> None:
        self.conn.execute("INSERT INTO audit_events(ts, event, detail) VALUES (?,?,?)",
                          (time.time(), event, json.dumps(detail)))
        self.conn.commit()

    # -- analytics -----------------------------------------------------------
    def trends(self, agent_name: str | None = None) -> list[dict]:
        runs = self.list_runs(agent_name)
        return [{"run_id": r["id"], "agent": r["agent_name"],
                 "agent_version": r["agent_version"], "started_at": r["started_at"],
                 "pass_rate": r["summary"].get("pass_rate"),
                 "mean_scores": r["summary"].get("metric_means", {}),
                 "total_cost_usd": r["summary"].get("total_cost_usd")}
                for r in runs]
