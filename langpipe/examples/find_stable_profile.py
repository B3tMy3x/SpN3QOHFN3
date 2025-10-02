from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from langpipe.langpipe.llm import LLM
from langpipe.langpipe.sql import SQLExecutorModel
from langpipe.langpipe.trino_exec import TrinoExecutorModel, parse_trino_jdbc_url
from langpipe.webapp.pipeline_trace import optimize_sql_with_trace


def _set_env(vars: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
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


def _count_empty_steps(trace: Dict[str, Any]) -> int:
    cnt = 0
    for st in trace.get("steps", []):
        if st.get("name") == "generate_sql_empty":
            cnt += 1
    return cnt


def _explain_ok(trace: Dict[str, Any]) -> bool:
    for st in trace.get("steps", []):
        if st.get("name") == "explain_rank":
            out = st.get("output") or []
            if out and out[0].get("explain_error") is False:
                return True
    return False


def _exec_ok(trace: Dict[str, Any]) -> bool:
    for st in trace.get("steps", []):
        if st.get("name") == "sql_executor":
            if st.get("output", {}).get("ok"):
                return True
    return False


def _best_sql_preview(trace: Dict[str, Any]) -> str:
    for st in trace.get("steps", []):
        if st.get("name") == "explain_rank":
            out = st.get("output") or []
            if out:
                return (out[0].get("sql") or "")[:160]
    return ""


def _run_once(sql: str, jdbc: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    llm = LLM()
    if jdbc:
        cfg = parse_trino_jdbc_url(jdbc)
        executor = TrinoExecutorModel(cfg, safe_mode=True)
    else:
        executor = SQLExecutorModel(llm)
    t = time.monotonic()
    trace = optimize_sql_with_trace(
        llm,
        executor,
        sql,
        max_attempts=int(os.getenv("PIPE_MAX_ATTEMPTS", "2")),
        dialect=os.getenv("SQL_DIALECT", "trino"),
        optimize_mode=True,
    )
    ms = int((time.monotonic() - t) * 1000)
    summary = {
        "empty_steps": _count_empty_steps(trace),
        "explain_ok": _explain_ok(trace),
        "exec_ok": _exec_ok(trace),
        "ms": ms,
        "best_sql_preview": _best_sql_preview(trace),
    }
    return trace, summary


def build_profiles(base_url: Optional[str], model: Optional[str]) -> List[Dict[str, Optional[str]]]:
    base = {
        "LLM_BACKEND": "ollama",
        "IS_OLLAMA_MODEL": "true",
        "OLLAMA_BASE_URL": base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": model or os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
        "LLM_HTTP_TIMEOUT": os.getenv("LLM_HTTP_TIMEOUT", "300"),
        "OLLAMA_MAX_RETRIES": os.getenv("OLLAMA_MAX_RETRIES", "3"),
        "OLLAMA_BACKOFF_SECONDS": os.getenv("OLLAMA_BACKOFF_SECONDS", "2"),
        "OLLAMA_NUM_PREDICT": os.getenv("OLLAMA_NUM_PREDICT", "512"),
        "LLM_SCHEMA_MAX_CHARS": os.getenv("LLM_SCHEMA_MAX_CHARS", "2000"),
        "LLM_PROMPT_MAX_CHARS": os.getenv("LLM_PROMPT_MAX_CHARS", "800"),
        "PIPE_MAX_ATTEMPTS": os.getenv("PIPE_MAX_ATTEMPTS", "2"),
    }
    profiles: List[Dict[str, Optional[str]]] = []
    # 1) generate + no json (самый устойчивый к пустым JSON)
    p1 = dict(base)
    p1.update({"OLLAMA_MODE": "generate", "OLLAMA_FORCE_JSON": "false"})
    profiles.append(p1)
    # 2) chat + json (если модель стабильно отдает JSON)
    p2 = dict(base)
    p2.update({"OLLAMA_MODE": "chat", "OLLAMA_FORCE_JSON": "true"})
    profiles.append(p2)
    # 3) chat + no json (иногда лучше при длинных промптах)
    p3 = dict(base)
    p3.update({"OLLAMA_MODE": "chat", "OLLAMA_FORCE_JSON": "false"})
    profiles.append(p3)
    # 4) generate + json (на случай, если сервер оборачивает ответы как JSON)
    p4 = dict(base)
    p4.update({"OLLAMA_MODE": "generate", "OLLAMA_FORCE_JSON": "true"})
    profiles.append(p4)
    return profiles


def choose_best(results: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    # Лучшая: exec_ok или explain_ok и zero empty_steps; при равенстве — минимальный ms
    candidates = sorted(
        results,
        key=lambda x: (
            0 if (x[2]["exec_ok"] or x[2]["explain_ok"]) else 1,
            x[2]["empty_steps"],
            x[2]["ms"],
        ),
    )
    return candidates[0][0], candidates[0][2]


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Find a stable Ollama profile to run the pipeline without empty responses")
    ap.add_argument("--sql", type=str, required=True, help="SQL to optimize")
    ap.add_argument("--jdbc", type=str, default=os.getenv("TRINO_JDBC_URL"), help="Trino JDBC URL (optional)")
    ap.add_argument("--ollama-url", type=str, default=os.getenv("OLLAMA_BASE_URL"), help="Ollama base URL")
    ap.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"), help="Ollama model name")
    ap.add_argument("--out", type=str, default="debug_out/bench/stable_profile.json", help="Where to save the report")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    prof_envs = build_profiles(args.ollama_url, args.model)
    results: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    report: Dict[str, Any] = {"runs": []}

    for i, envs in enumerate(prof_envs, start=1):
        name = f"profile_{i}:{envs['OLLAMA_MODE']}/json={envs['OLLAMA_FORCE_JSON']}"
        prev = _set_env(envs)
        try:
            trace, summ = _run_once(args.sql, args.jdbc)
            results.append((name, trace, summ))
            report["runs"].append({"name": name, "env": envs, "summary": summ, "best_sql_preview": summ.get("best_sql_preview")})
            print(f"{name}: empty={summ['empty_steps']} explain_ok={summ['explain_ok']} exec_ok={summ['exec_ok']} ms={summ['ms']}")
        finally:
            _restore_env(prev)

    best_name, best_summ = choose_best(results)
    report["best"] = {"name": best_name, "summary": best_summ}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved {args.out}")
    print(f"Recommended: {best_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

