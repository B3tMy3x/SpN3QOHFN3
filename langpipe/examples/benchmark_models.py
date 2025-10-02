from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from langpipe.langpipe.llm import LLM
from langpipe.langpipe.sql import SQLExecutorModel
from langpipe.langpipe.trino_exec import TrinoExecutorModel, parse_trino_jdbc_url
from langpipe.webapp.pipeline_trace import optimize_sql_with_trace
from langpipe.langpipe.critic import call_enhance_prompt
from langpipe.langpipe.generator import call_generate_sql
from langpipe.langpipe.utils import schema_subset_for_sql


def _set_env(vars: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Temporarily set env vars, return previous values."""
    prev: Dict[str, Optional[str]] = {}
    for k, v in vars.items():
        prev[k] = os.getenv(k)
        if v is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = v
    return prev


def _restore_env(prev: Dict[str, Optional[str]]):
    for k, v in prev.items():
        if v is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = v


def _load_queries(path: Optional[str], single_sql: Optional[str]) -> List[str]:
    if single_sql:
        return [single_sql]
    if not path:
        raise SystemExit("Provide --sql or --queries (JSON list or JSON {queries:[...]})")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        return [str(x.get("sql") or x) for x in data["queries"]]
    raise SystemExit("Unsupported queries file format")


def _summarize_result(trace: Dict[str, Any]) -> Dict[str, Any]:
    steps = trace.get("steps", [])
    ok = False
    explain_ok = False
    err = trace.get("error")
    best_sql = ""
    for st in steps:
        if st.get("name") == "explain_rank":
            out = st.get("output") or []
            explain_ok = bool(out) and (out[0].get("explain_error") is False)
            if out:
                best_sql = out[0].get("sql") or best_sql
        if st.get("name") == "sql_executor":
            ok = st.get("output", {}).get("ok", False)
    return {"ok": ok, "explain_ok": explain_ok, "best_sql_preview": (best_sql or "")[:120], "error": err}


def run_profile(name: str, env_overrides: Dict[str, Optional[str]], queries: List[str], jdbc: Optional[str]) -> Dict[str, Any]:
    prev = _set_env(env_overrides)
    try:
        # Init LLM and executor
        llm = LLM()
        executor: TrinoExecutorModel | SQLExecutorModel
        if jdbc:
            cfg = parse_trino_jdbc_url(jdbc)
            executor = TrinoExecutorModel(cfg, safe_mode=True)
        else:
            executor = SQLExecutorModel(llm)
        results: List[Dict[str, Any]] = []
        t0 = time.monotonic()
        for q in queries:
            t = time.monotonic()
            trace = optimize_sql_with_trace(llm, executor, q, max_attempts=int(os.getenv("PIPE_MAX_ATTEMPTS", "2")), dialect=os.getenv("SQL_DIALECT", "trino"), optimize_mode=True)
            summ = _summarize_result(trace)
            summ["ms"] = int((time.monotonic() - t) * 1000)
            results.append(summ)
        total_ms = int((time.monotonic() - t0) * 1000)
        ok_count = sum(1 for r in results if r.get("ok") or r.get("explain_ok"))
        return {"profile": name, "count": len(results), "ok_or_explain_ok": ok_count, "total_ms": total_ms, "items": results}
    finally:
        _restore_env(prev)


def run_paired_oss_sqlcoder(text_model: str, sql_model: str, queries: List[str], jdbc: Optional[str]) -> Dict[str, Any]:
    prev = {
        "LLM_BACKEND": os.getenv("LLM_BACKEND"),
        "IS_OLLAMA_MODEL": os.getenv("IS_OLLAMA_MODEL"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        "OLLAMA_MODE": os.getenv("OLLAMA_MODE"),
        "OLLAMA_FORCE_JSON": os.getenv("OLLAMA_FORCE_JSON"),
    }
    try:
        # Init text LLM (OSS)
        os.environ["LLM_BACKEND"] = "ollama"
        os.environ["IS_OLLAMA_MODEL"] = "true"
        if os.getenv("OLLAMA_BASE_URL") is None:
            os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
        os.environ["OLLAMA_MODEL"] = text_model
        os.environ["OLLAMA_MODE"] = "chat"
        os.environ["OLLAMA_FORCE_JSON"] = "true"
        llm_text = LLM()

        # Init SQL LLM (sqlcoder); prompts in English
        os.environ["OLLAMA_MODEL"] = sql_model
        os.environ["OLLAMA_MODE"] = "chat"
        os.environ["OLLAMA_FORCE_JSON"] = "true"
        llm_sql = LLM()

        # Executor
        if jdbc:
            cfg = parse_trino_jdbc_url(jdbc)
            executor: TrinoExecutorModel | SQLExecutorModel = TrinoExecutorModel(cfg, safe_mode=True)
        else:
            executor = SQLExecutorModel(llm_sql)

        items: List[Dict[str, Any]] = []
        t0 = time.monotonic()
        for q in queries:
            # Step 1: English enhancement (text model)
            enh = call_enhance_prompt(
                llm=llm_text,
                user_query=("Respond in English. Optimize and rewrite equivalently:\n" + q),
                critique=None,
                issues=[],
                db_schema=None,
                chat_history=[],
            )
            prompt_en = enh.get("enhanced_prompt") or "Rewrite the SQL query to be equivalent and efficient. Return only valid SQL."
            # Step 2: SQL generation (sqlcoder)
            gen = call_generate_sql(
                llm=llm_sql,
                user_query="Rewrite in efficient SQL. Return JSON {sql_query, reasoning} or a single SQL.",
                current_prompt=prompt_en,
                db_schema=None,
                chat_history=[],
                dialect=os.getenv("SQL_DIALECT", "trino"),
            )
            sql = (gen.get("sql_query") or "").strip()
            ok = False
            explain_ok = False
            err = None
            # Execute if possible
            if sql and isinstance(executor, TrinoExecutorModel):
                res, err = executor.run(sql, None)
                ok = bool(res)
            elif sql:
                # Validate via simulator
                sim: SQLExecutorModel = executor if isinstance(executor, SQLExecutorModel) else SQLExecutorModel(llm_sql)
                res, err = sim.run(sql, None)
                ok = bool(res)
            items.append({
                "sql_preview": sql[:120],
                "ok": ok,
                "error": err,
                "enhanced_prompt_preview": prompt_en[:120],
            })
        total_ms = int((time.monotonic() - t0) * 1000)
        ok_count = sum(1 for r in items if r.get("ok"))
        return {"profile": f"paired:{text_model}->{sql_model}", "count": len(items), "ok": ok_count, "total_ms": total_ms, "items": items}
    finally:
        _restore_env(prev)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Benchmark model profiles and paired OSS+SQLCoder flow")
    ap.add_argument("--sql", type=str, default=None, help="Single SQL to test")
    ap.add_argument("--queries", type=str, default=None, help="Path to JSON file with queries")
    ap.add_argument("--jdbc", type=str, default=os.getenv("TRINO_JDBC_URL"), help="Trino JDBC URL")
    ap.add_argument("--out", type=str, default="debug_out/bench/report.json", help="Output JSON report path")
    args = ap.parse_args()

    queries = _load_queries(args.queries, args.sql)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    profiles = [
        {
            "name": "stable-generate-nojson-sqlcoder",
            "env": {
                "LLM_BACKEND": "ollama",
                "IS_OLLAMA_MODEL": "true",
                "OLLAMA_MODE": "generate",
                "OLLAMA_FORCE_JSON": "false",
                "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "sqlcoder:15b"),
                "LLM_HTTP_TIMEOUT": os.getenv("LLM_HTTP_TIMEOUT", "300"),
                "OLLAMA_MAX_RETRIES": os.getenv("OLLAMA_MAX_RETRIES", "3"),
                "OLLAMA_BACKOFF_SECONDS": os.getenv("OLLAMA_BACKOFF_SECONDS", "2"),
                "OLLAMA_NUM_PREDICT": os.getenv("OLLAMA_NUM_PREDICT", "512"),
                "LLM_SCHEMA_MAX_CHARS": os.getenv("LLM_SCHEMA_MAX_CHARS", "2000"),
                "LLM_PROMPT_MAX_CHARS": os.getenv("LLM_PROMPT_MAX_CHARS", "800"),
            },
        },
        {
            "name": "chat-json-qwen",
            "env": {
                "LLM_BACKEND": "ollama",
                "IS_OLLAMA_MODEL": "true",
                "OLLAMA_MODE": "chat",
                "OLLAMA_FORCE_JSON": "true",
                "OLLAMA_MODEL": "qwen2.5-coder:14b",
                "LLM_HTTP_TIMEOUT": os.getenv("LLM_HTTP_TIMEOUT", "300"),
                "OLLAMA_MAX_RETRIES": os.getenv("OLLAMA_MAX_RETRIES", "3"),
                "OLLAMA_BACKOFF_SECONDS": os.getenv("OLLAMA_BACKOFF_SECONDS", "2"),
                "OLLAMA_NUM_PREDICT": os.getenv("OLLAMA_NUM_PREDICT", "512"),
                "LLM_SCHEMA_MAX_CHARS": os.getenv("LLM_SCHEMA_MAX_CHARS", "2000"),
                "LLM_PROMPT_MAX_CHARS": os.getenv("LLM_PROMPT_MAX_CHARS", "800"),
            },
        },
    ]

    report: Dict[str, Any] = {"profiles": [], "paired": None}
    for p in profiles:
        rep = run_profile(p["name"], p["env"], queries, args.jdbc)
        report["profiles"].append(rep)

    # Paired mode: OSS (gpt-oss:latest) for English prompt + sqlcoder:15b for SQL
    paired = run_paired_oss_sqlcoder(
        text_model=os.getenv("OSS_TEXT_MODEL", "gpt-oss:latest"),
        sql_model=os.getenv("SQLCODER_MODEL", "sqlcoder:15b"),
        queries=queries,
        jdbc=args.jdbc,
    )
    report["paired"] = paired

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved {args.out}")
    # Print summary
    for pr in report["profiles"]:
        print(f"{pr['profile']}: ok_or_explain_ok={pr['ok_or_explain_ok']}/{pr['count']} total_ms={pr['total_ms']}")
    print(f"paired: ok={report['paired']['ok']}/{report['paired']['count']} total_ms={report['paired']['total_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

