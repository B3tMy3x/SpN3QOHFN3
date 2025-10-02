from __future__ import annotations

import json
import datetime
from decimal import Decimal
import os
import sqlite3
from typing import Any, Dict, List, Tuple


DB_PATH = os.path.join(os.path.dirname(__file__), "jobs.sqlite")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _json_default(o):
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    # Fallback to str to avoid breaking the job on exotic types
    return str(o)


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              phase TEXT,
              total INTEGER NOT NULL,
              done INTEGER NOT NULL DEFAULT 0,
              optimize INTEGER NOT NULL DEFAULT 0,
              jdbc_url TEXT,
              started_at REAL,
              finished_at REAL,
              job_error TEXT,
              metrics_json TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS job_items (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id TEXT NOT NULL,
              idx INTEGER NOT NULL,
              item_json TEXT NOT NULL,
              FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
        # Best-effort migrate: ensure job_error column exists
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(jobs)").fetchall()]
            if "job_error" not in cols:
                con.execute("ALTER TABLE jobs ADD COLUMN job_error TEXT")
            if "phase" not in cols:
                con.execute("ALTER TABLE jobs ADD COLUMN phase TEXT")
            if "metrics_json" not in cols:
                con.execute("ALTER TABLE jobs ADD COLUMN metrics_json TEXT")
        except Exception:
            pass


def create_job(job_id: string, total: int, optimize: bool, jdbc_url: str, started_at: float) -> None:  # type: ignore[name-defined]
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO jobs (id, status, total, done, optimize, jdbc_url, started_at) VALUES (?,?,?,?,?,?,?)",
            (job_id, "running", total, 0, 1 if optimize else 0, jdbc_url, started_at),
        )


def update_job_progress(job_id: str, done: int) -> None:
    with _conn() as con:
        con.execute("UPDATE jobs SET done=? WHERE id=?", (done, job_id))


def finish_job(job_id: str, finished_at: float) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='finished', finished_at=? WHERE id=?",
            (finished_at, job_id),
        )


def set_job_error(job_id: str, error: str) -> None:
    with _conn() as con:
        con.execute("UPDATE jobs SET job_error=? WHERE id=?", (error, job_id))


def set_job_phase(job_id: str, phase: str | None) -> None:
    with _conn() as con:
        con.execute("UPDATE jobs SET phase=? WHERE id=?", (phase, job_id))


def append_item(job_id: str, idx: int, item: Dict[str, Any]) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO job_items (job_id, idx, item_json) VALUES (?,?,?)",
            (job_id, idx, json.dumps(item, ensure_ascii=False, default=_json_default)),
        )


def list_jobs() -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute(
            "SELECT id, status, phase, total, done, optimize, started_at, finished_at, job_error, metrics_json FROM jobs ORDER BY started_at DESC"
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            d = dict(r)
            if d.get("metrics_json"):
                try:
                    d["metrics"] = json.loads(d["metrics_json"])  # type: ignore[index]
                except Exception:
                    d["metrics"] = None
            out.append(d)
        return out


def get_job(job_id: str) -> Dict[str, Any]:
    with _conn() as con:
        cur = con.execute(
            "SELECT id, status, phase, total, done, optimize, started_at, finished_at, jdbc_url, job_error, metrics_json FROM jobs WHERE id=?",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        job = dict(row)
        if job.get("metrics_json"):
            try:
                job["metrics"] = json.loads(job["metrics_json"])  # type: ignore[index]
            except Exception:
                job["metrics"] = None
        cur2 = con.execute(
            "SELECT idx, item_json FROM job_items WHERE job_id=? ORDER BY idx ASC",
            (job_id,),
        )
        progress = []
        for r in cur2.fetchall():
            try:
                item = json.loads(r["item_json"])  # type: ignore[index]
            except Exception:
                item = {"index": r["idx"], "error": "Corrupt item_json"}
            progress.append(item)
        job["progress"] = progress
        return job


def set_job_metrics(job_id: str, metrics: Dict[str, Any]) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET metrics_json=? WHERE id=?",
            (json.dumps(metrics, ensure_ascii=False), job_id),
        )


def delete_job(job_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM job_items WHERE job_id=?", (job_id,))
        con.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def get_job_queries(job_id: str) -> List[str]:
    with _conn() as con:
        cur = con.execute(
            "SELECT idx, item_json FROM job_items WHERE job_id=? ORDER BY idx ASC",
            (job_id,),
        )
        out: List[str] = []
        for r in cur.fetchall():
            try:
                obj = json.loads(r["item_json"])  # type: ignore[index]
                sql = obj.get("sql")
                if isinstance(sql, str):
                    out.append(sql)
            except Exception:
                continue
        return out


def set_setting(key: str, value: str) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_all_settings() -> Dict[str, str]:
    with _conn() as con:
        cur = con.execute("SELECT key, value FROM settings")
        return {row[0]: row[1] for row in cur.fetchall()}
