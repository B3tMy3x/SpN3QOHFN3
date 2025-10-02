from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from ..langpipe.llm import LLM
from ..langpipe.sql import SQLExecutorModel
from ..langpipe.trino_exec import TrinoExecutorModel, TrinoConfig, parse_trino_jdbc_url
from .. import build_graph
from .pipeline_trace import optimize_sql_with_trace
from . import db
from dotenv import load_dotenv  # type: ignore
from ..langpipe.json_utils import extract_json_best_effort


# ---------------------------
# Global runtime state
# ---------------------------

DEFAULT_TRINO_JDBC = os.getenv("TRINO_JDBC_URL", "")

state: Dict[str, Any] = {
    "jdbc_url": DEFAULT_TRINO_JDBC,
    "model_mode": "ollama",  # default to Ollama for stability; can be changed via UI
    "openrouter_model": os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-coder:6.7b-instruct"),
    "ollama_base": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
    "trino_safe_mode": True,
}

llm: Optional[LLM] = None
sql_sim: Optional[SQLExecutorModel] = None
trino_exec: Optional[TrinoExecutorModel] = None

jobs: Dict[str, Dict[str, Any]] = {}


def _init_models():
    global llm, sql_sim, trino_exec
    # Configure LLM either local or remote
    state.pop("llm_error", None)
    os.environ["IS_OPENROUTER_MODEL"] = "true" if state.get("model_mode") == "openrouter" else "false"
    os.environ["IS_OLLAMA_MODEL"] = "true" if state.get("model_mode") == "ollama" else "false"
    try:
        if state.get("model_mode") == "openrouter":
            if not os.getenv("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY is not set for remote model")
            os.environ["OPENROUTER_MODEL"] = state["openrouter_model"]
        elif state.get("model_mode") == "ollama":
            os.environ["OLLAMA_BASE_URL"] = state.get("ollama_base", "http://localhost:11434")
            os.environ["OLLAMA_MODEL"] = state.get("ollama_model", "gpt-oss:latest")
        llm_obj = LLM()
        llm = llm_obj
        sql_sim = SQLExecutorModel(llm_obj)
    except Exception as e:
        # Defer LLM init; allow service to start
        llm = None
        sql_sim = None
        state["llm_error"] = str(e)

    # Trino executor can be initialized regardless of LLM
    try:
        cfg = parse_trino_jdbc_url(state["jdbc_url"])
        trino_exec = TrinoExecutorModel(cfg, safe_mode=bool(state.get("trino_safe_mode", True)))
    except Exception as e:
        trino_exec = None
        state["trino_error"] = str(e)


def _ensure_ready():
    if not trino_exec:
        _init_models()


def _normalize_result(res: Dict[str, Any]) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cols = list(res.get("columns") or [])
    rows = list(res.get("rows") or [])
    # Ensure tuples for set-like comparisons
    rows_t = [tuple(r) if not isinstance(r, tuple) else r for r in rows]
    return cols, rows_t


def _trino_stats_from_part(part: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return ((part or {}).get("result") or {}).get("trino") or {}
    except Exception:
        return {}


def _resource_metric_from_trino(tri: Dict[str, Any]) -> Optional[float]:
    import os
    source = os.getenv("RESOURCE_METRIC_SOURCE", "auto").lower()
    # Normalize keys that can appear as camelCase or snake_case
    def get_first(keys: list[str]) -> Optional[float]:
        for k in keys:
            if k in tri and tri[k] is not None:
                try:
                    return float(tri[k])
                except Exception:
                    continue
        return None
    if source in ("physicalinputbytes", "physical_input_bytes"):
        return get_first(["physicalInputBytes", "physical_input_bytes"])  # type: ignore[list-item]
    if source in ("processedbytes", "processed_bytes"):
        return get_first(["processedBytes", "processed_bytes"])  # type: ignore[list-item]
    if source in ("processedrows", "processed_rows"):
        return get_first(["processedRows", "processed_rows"])  # type: ignore[list-item]
    if source in ("peakmemorybytes", "peak_memory_bytes"):
        return get_first(["peakMemoryBytes", "peak_memory_bytes"])  # type: ignore[list-item]
    # Cascade (auto): physicalInputBytes -> processedBytes -> processedRows
    return get_first(["physicalInputBytes", "physical_input_bytes", "processedBytes", "processed_bytes", "processedRows", "processed_rows"])  # type: ignore[list-item]


def _resource_metric_from_part(part: Optional[Dict[str, Any]]) -> Optional[float]:
    return _resource_metric_from_trino(_trino_stats_from_part(part))


def _compare_results(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a_cols, a_rows = _normalize_result(a)
    b_cols, b_rows = _normalize_result(b)
    same_schema = a_cols == b_cols
    a_set = set(a_rows)
    b_set = set(b_rows)
    only_a = list(a_set - b_set)
    only_b = list(b_set - a_set)
    match = same_schema and not only_a and not only_b and len(a_rows) == len(b_rows)
    return {
        "match": match,
        "same_schema": same_schema,
        "only_left": only_a[:20],
        "only_right": only_b[:20],
        "left_count": len(a_rows),
        "right_count": len(b_rows),
        "columns": a_cols if same_schema else {"left": a_cols, "right": b_cols},
    }


def _metric_x(A_ms: Optional[int], T_ms: Optional[int], Q: float = 1.0) -> Optional[float]:
    if not A_ms or not T_ms or T_ms <= 0:
        return None
    # For a single operation, x_i = (Q*A/T)/Q = A/T
    return float(A_ms) / float(T_ms)


def _percent_reduction(A_ms: Optional[int], T_ms: Optional[int]) -> Optional[float]:
    if not A_ms or not T_ms or A_ms <= 0:
        return None
    return 100.0 * (float(A_ms) - float(T_ms)) / float(A_ms)


def _metric_y(S: Optional[float], C: Optional[float]) -> Optional[float]:
    if not S or not C or C == 0:
        return None
    return float(S) / float(C)


def _metric_z(x: Optional[float], y: Optional[float]) -> Optional[float]:
    if x is None:
        return None if y is None else (y)
    if y is None:
        return (x ** (1.0/3.0))
    return (x ** (1.0/3.0)) * y


def _extract_sql_meta(entry: Any) -> Tuple[str, Dict[str, Any]]:
    """Accepts string or object; returns (sql, meta) where meta may contain q,S,C."""
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        sql = entry.get("sql") or entry.get("query") or ""
        meta = {}
        if "q" in entry:
            meta["q"] = entry.get("q")
        if "runquantity" in entry:
            meta["q"] = entry.get("runquantity")
        if "S" in entry:
            meta["S"] = entry.get("S")
        if "C" in entry:
            meta["C"] = entry.get("C")
        return sql, meta
    return str(entry), {}


app = FastAPI(title="LangPipe Web Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
def root_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/single", response_class=HTMLResponse)
def page_single(request: Request):
    return templates.TemplateResponse("single.html", {"request": request})

@app.get("/pair", response_class=HTMLResponse)
def page_pair(request: Request):
    return templates.TemplateResponse("pair.html", {"request": request})

@app.get("/queue", response_class=HTMLResponse)
def page_queue(request: Request):
    return templates.TemplateResponse("queue.html", {"request": request})

@app.get("/jobs", response_class=HTMLResponse)
def page_jobs(request: Request):
    # Use persisted view (DB) to include metrics on initial render
    try:
        items = db.list_jobs()
    except Exception:
        # Fallback to in-memory with no metrics
        items = [
            {
                "id": jid,
                "status": j.get("status"),
                "total": j.get("total"),
                "done": j.get("done"),
                "started_at": j.get("started_at"),
                "finished_at": j.get("finished_at"),
                "optimize": j.get("optimize"),
            }
            for jid, j in sorted(jobs.items(), key=lambda kv: kv[1].get("started_at", 0), reverse=True)
        ]
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": items})

@app.get("/jobs/view/{job_id}", response_class=HTMLResponse)
def page_job_detail(job_id: str, request: Request):
    return templates.TemplateResponse("job.html", {"request": request, "job_id": job_id})


@app.get("/config")
def get_config():
    return {
        "jdbc_url": state["jdbc_url"],
        "model_mode": state["model_mode"],
        "openrouter_model": state["openrouter_model"],
        "ollama_base": state["ollama_base"],
        "ollama_model": state["ollama_model"],
        "trino_safe_mode": bool(state.get("trino_safe_mode", True)),
        "llm_ready": bool(llm is not None),
        "trino_ready": bool(trino_exec is not None),
        "llm_error": state.get("llm_error"),
        "trino_error": state.get("trino_error"),
    }


@app.post("/config/model")
def set_model(mode: str = Form(...), openrouter_model: Optional[str] = Form(None), ollama_base: Optional[str] = Form(None), ollama_model: Optional[str] = Form(None)):
    state["model_mode"] = mode
    if openrouter_model:
        state["openrouter_model"] = openrouter_model
    if ollama_base:
        state["ollama_base"] = ollama_base
    if ollama_model:
        state["ollama_model"] = ollama_model
    # re-init models
    _init_models()
    # persist
    db.set_setting("model_mode", state["model_mode"]) 
    db.set_setting("openrouter_model", state["openrouter_model"])
    db.set_setting("ollama_base", state["ollama_base"])
    db.set_setting("ollama_model", state["ollama_model"])
    return {"ok": True, "model_mode": state["model_mode"], "openrouter_model": state["openrouter_model"], "ollama_base": state["ollama_base"], "ollama_model": state["ollama_model"]}


@app.post("/config/trino")
def set_trino(jdbc_url: str = Form(...)):
    state["jdbc_url"] = jdbc_url
    # re-init trino executor only
    cfg = parse_trino_jdbc_url(jdbc_url)
    global trino_exec
    trino_exec = TrinoExecutorModel(cfg, safe_mode=bool(state.get("trino_safe_mode", True)))
    # persist
    db.set_setting("jdbc_url", jdbc_url)
    return {"ok": True, "jdbc_url": jdbc_url}


@app.post("/config/trino_safety")
def set_trino_safety(safe_mode: bool = Form(True)):
    state["trino_safe_mode"] = bool(safe_mode)
    # re-init executor with new mode
    cfg = parse_trino_jdbc_url(state["jdbc_url"])
    global trino_exec
    trino_exec = TrinoExecutorModel(cfg, safe_mode=bool(safe_mode))
    db.set_setting("trino_safe_mode", "1" if safe_mode else "0")
    return {"ok": True, "trino_safe_mode": bool(safe_mode)}


@app.post("/run/single")
async def run_single(sql: str = Form(...), optimize: bool = Form(False)):
    _ensure_ready()
    assert trino_exec

    # create a job in feed
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "progress": [], "total": 1, "done": 0, "started_at": time.time(), "optimize": bool(optimize)}
    db.create_job(job_id, 1, bool(optimize), state["jdbc_url"], jobs[job_id]["started_at"])  # type: ignore[index]

    t0 = time.monotonic()
    db.set_job_phase(job_id, "executing_original")
    base_res, base_err = await trino_exec.run_async(sql, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
    base_ms_client = int((time.monotonic() - t0) * 1000)
    tri0 = (base_res or {}).get("trino", {}) if base_res else {}
    # Prefer CPU time (more stable) → execution → elapsed → client
    if tri0 and tri0.get("cpu_ms") is not None:
        base_ms = int(tri0.get("cpu_ms"))
    elif tri0 and tri0.get("execution_ms") is not None:
        base_ms = int(tri0.get("execution_ms"))
    elif tri0 and tri0.get("elapsed_ms") is not None:
        base_ms = int(tri0.get("elapsed_ms"))
    else:
        base_ms = base_ms_client

    tri0 = (base_res or {}).get("trino", {}) if base_res else {}
    out: Dict[str, Any] = {
        "original": {
            "error": base_err,
            "result": base_res,
            "ms": base_ms,
            "times": {
                "execution_ms": tri0.get("execution_ms"),
                "elapsed_ms": tri0.get("elapsed_ms"),
                "client_ms": base_ms_client,
                "queued_ms": tri0.get("queued_ms"),
                "cpu_ms": tri0.get("cpu_ms"),
                "scheduled_ms": tri0.get("scheduled_ms"),
            },
        }
    }

    if optimize and not base_err:
        # Full pipeline with step trace
        if not llm:
            out["error"] = "LLM is not initialized: " + str(state.get("llm_error", ""))
            # fall through without optimization
            trace = {"steps": []}
        else:
            db.set_job_phase(job_id, "optimizing")
            trace = await asyncio.to_thread(optimize_sql_with_trace, llm, trino_exec, sql, 3, "trino", True)
        opt_sql = trace.get("optimized_sql") or ""
        t1 = time.monotonic()
        db.set_job_phase(job_id, "executing_optimized")
        opt_res, opt_err = await trino_exec.run_async(opt_sql, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
        opt_ms_client = int((time.monotonic() - t1) * 1000)
        tri1 = (opt_res or {}).get("trino", {}) if opt_res else {}
        if tri1 and tri1.get("cpu_ms") is not None:
            opt_ms = int(tri1.get("cpu_ms"))
        elif tri1 and tri1.get("execution_ms") is not None:
            opt_ms = int(tri1.get("execution_ms"))
        elif tri1 and tri1.get("elapsed_ms") is not None:
            opt_ms = int(tri1.get("elapsed_ms"))
        else:
            opt_ms = opt_ms_client
        out["optimized_sql"] = opt_sql
        tri1 = (opt_res or {}).get("trino", {}) if opt_res else {}
        out["optimized"] = {
            "error": opt_err,
            "result": opt_res,
            "ms": opt_ms,
            "times": {
                "execution_ms": tri1.get("execution_ms"),
                "elapsed_ms": tri1.get("elapsed_ms"),
                "client_ms": opt_ms_client,
                "queued_ms": tri1.get("queued_ms"),
                "cpu_ms": tri1.get("cpu_ms"),
                "scheduled_ms": tri1.get("scheduled_ms"),
            },
        }
        out["trace"] = trace.get("steps", [])
        if opt_res and base_res:
            out["compare"] = _compare_results(base_res, opt_res)

    elif optimize and base_err:
        # Оригинал упал — сообщаем модели об ошибке и пробуем перегенерировать и выполнить
        if not llm:
            out["error"] = "LLM is not initialized: " + str(state.get("llm_error", ""))
        else:
            db.set_job_phase(job_id, "retrying")
            trace = await asyncio.to_thread(optimize_sql_with_trace, llm, trino_exec, sql, 3, "trino", True)
            retry_sql = trace.get("optimized_sql") or ""
            t1 = time.monotonic()
            db.set_job_phase(job_id, "executing_retry")
            retry_res, retry_err = await trino_exec.run_async(retry_sql, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
            retry_ms_client = int((time.monotonic() - t1) * 1000)
            triR = (retry_res or {}).get("trino", {}) if retry_res else {}
            if triR and triR.get("cpu_ms") is not None:
                retry_ms = int(triR.get("cpu_ms"))
            elif triR and triR.get("execution_ms") is not None:
                retry_ms = int(triR.get("execution_ms"))
            elif triR and triR.get("elapsed_ms") is not None:
                retry_ms = int(triR.get("elapsed_ms"))
            else:
                retry_ms = retry_ms_client
            out["retry_sql"] = retry_sql
            triR = (retry_res or {}).get("trino", {}) if retry_res else {}
            out["retry"] = {
                "error": retry_err,
                "result": retry_res,
                "ms": retry_ms,
                "times": {
                    "execution_ms": triR.get("execution_ms"),
                    "elapsed_ms": triR.get("elapsed_ms"),
                    "client_ms": retry_ms_client,
                    "queued_ms": triR.get("queued_ms"),
                    "cpu_ms": triR.get("cpu_ms"),
                    "scheduled_ms": triR.get("scheduled_ms"),
                },
            }
            out["trace"] = trace.get("steps", [])

    # persist single item into job feed
    item: Dict[str, Any] = {"index": 1, "sql": sql, "original": out.get("original")}
    if optimize:
        item["optimized_sql"] = out.get("optimized_sql")
        item["optimized"] = out.get("optimized")
        if "compare" in out:
            item["compare"] = out.get("compare")
        if "trace" in out:
            item["trace"] = out.get("trace")
        if "retry" in out:
            item["retry_sql"] = out.get("retry_sql")
            item["retry"] = out.get("retry")
        # metrics for single
        A = item.get("original", {}).get("ms") if item.get("original") else None
        T = None
        right_obj = None
        if item.get("optimized"):
            T = item["optimized"].get("ms")
            right_obj = item.get("optimized")
        elif item.get("retry"):
            T = item["retry"].get("ms")
            right_obj = item.get("retry")
        x_i = _metric_x(A, T, 1.0)
        pr = _percent_reduction(A, T)
        # y = S/C from resource metric (configurable via RESOURCE_METRIC_SOURCE)
        S = _resource_metric_from_part(item.get("original"))
        C = _resource_metric_from_part(right_obj)
        y_i = None
        try:
            if S is not None and C is not None and float(C) > 0:
                y_i = float(S) / float(C)
        except Exception:
            y_i = None
        z_i = (x_i ** (1.0/3.0)) * y_i if (x_i is not None and y_i is not None) else None
        item["metrics"] = {"x": x_i, "y": y_i, "z": z_i, "percent_time_reduction": pr, "A_ms": A, "T_ms": T}
    jobs[job_id]["progress"].append(item)
    db.append_item(job_id, 1, item)
    jobs[job_id]["done"] = 1
    db.update_job_progress(job_id, 1)
    jobs[job_id]["status"] = "finished"
    jobs[job_id]["finished_at"] = time.time()
    db.finish_job(job_id, jobs[job_id]["finished_at"])  # type: ignore[index]
    db.set_job_phase(job_id, None)
    # set job metrics for single (if computed)
    if item.get("metrics"):
        payload = {"x": item["metrics"].get("x"), "percent_time_reduction": item["metrics"].get("percent_time_reduction")}
        if item["metrics"].get("y") is not None:
            payload["y"] = item["metrics"].get("y")
        if item["metrics"].get("z") is not None:
            payload["z"] = item["metrics"].get("z")
        db.set_job_metrics(job_id, payload)

    out["job_id"] = job_id
    return out


@app.post("/run/pair")
async def run_pair(sql1: str = Form(...), sql2: str = Form(...)):
    _ensure_ready()
    assert trino_exec
    # create a job in feed
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "progress": [], "total": 2, "done": 0, "started_at": time.time(), "optimize": False}
    db.create_job(job_id, 2, False, state["jdbc_url"], jobs[job_id]["started_at"])  # type: ignore[index]
    t0 = time.monotonic()
    res1, err1 = await trino_exec.run_async(sql1, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
    ms1_client = int((time.monotonic() - t0) * 1000)
    triL = (res1 or {}).get("trino", {}) if res1 else {}
    if triL and triL.get("cpu_ms") is not None:
        ms1 = int(triL.get("cpu_ms"))
    elif triL and triL.get("execution_ms") is not None:
        ms1 = int(triL.get("execution_ms"))
    elif triL and triL.get("elapsed_ms") is not None:
        ms1 = int(triL.get("elapsed_ms"))
    else:
        ms1 = ms1_client

    t1 = time.monotonic()
    res2, err2 = await trino_exec.run_async(sql2, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
    ms2_client = int((time.monotonic() - t1) * 1000)
    triR2 = (res2 or {}).get("trino", {}) if res2 else {}
    if triR2 and triR2.get("cpu_ms") is not None:
        ms2 = int(triR2.get("cpu_ms"))
    elif triR2 and triR2.get("execution_ms") is not None:
        ms2 = int(triR2.get("execution_ms"))
    elif triR2 and triR2.get("elapsed_ms") is not None:
        ms2 = int(triR2.get("elapsed_ms"))
    else:
        ms2 = ms2_client

    triL = (res1 or {}).get("trino", {}) if res1 else {}
    triR = (res2 or {}).get("trino", {}) if res2 else {}
    out: Dict[str, Any] = {
        "left": {
            "error": err1,
            "result": res1,
            "ms": ms1,
            "times": {
                "execution_ms": triL.get("execution_ms"),
                "elapsed_ms": triL.get("elapsed_ms"),
                "client_ms": ms1_client,
                "queued_ms": triL.get("queued_ms"),
                "cpu_ms": triL.get("cpu_ms"),
                "scheduled_ms": triL.get("scheduled_ms"),
            },
        },
        "right": {
            "error": err2,
            "result": res2,
            "ms": ms2,
            "times": {
                "execution_ms": triR.get("execution_ms"),
                "elapsed_ms": triR.get("elapsed_ms"),
                "client_ms": ms2_client,
                "queued_ms": triR.get("queued_ms"),
                "cpu_ms": triR.get("cpu_ms"),
                "scheduled_ms": triR.get("scheduled_ms"),
            },
        },
    }
    if res1 and res2:
        out["compare"] = _compare_results(res1, res2)

    # persist as single combined pair item for clearer UI
    pair_item: Dict[str, Any] = {
        "index": 1,
        "left_sql": sql1,
        "right_sql": sql2,
        "left": out["left"],
        "right": out["right"],
    }
    if "compare" in out:
        pair_item["compare"] = out["compare"]
    jobs[job_id]["progress"].append(pair_item)
    db.append_item(job_id, 1, pair_item)
    jobs[job_id]["done"] = 2
    db.update_job_progress(job_id, 2)
    jobs[job_id]["status"] = "finished"
    jobs[job_id]["finished_at"] = time.time()
    db.finish_job(job_id, jobs[job_id]["finished_at"])  # type: ignore[index]
    # Persist job-level metrics for pair: x = A/T, y = S/C from processed_bytes, z = x^(1/3)*y
    if isinstance(ms1, (int, float)) and isinstance(ms2, (int, float)) and ms2 and ms2 > 0:
        x_pair = float(ms1) / float(ms2)
    else:
        x_pair = None
    S_pair = _resource_metric_from_part(out.get("left"))
    C_pair = _resource_metric_from_part(out.get("right"))
    y_pair = None
    try:
        if S_pair is not None and C_pair is not None and float(C_pair) > 0:
            y_pair = float(S_pair) / float(C_pair)
    except Exception:
        y_pair = None
    z_pair = ((x_pair ** (1.0/3.0)) * y_pair) if (x_pair is not None and y_pair is not None) else None
    jobs[job_id]["metrics"] = {"x": x_pair, "y": y_pair, "z": z_pair}
    try:
        payload = {}
        if x_pair is not None: payload["x"] = x_pair
        if y_pair is not None: payload["y"] = y_pair
        if z_pair is not None: payload["z"] = z_pair
        if payload:
            db.set_job_metrics(job_id, payload)
    except Exception:
        pass

    out["job_id"] = job_id
    return out


def _load_queries_from_json_obj(obj: Dict[str, Any]) -> Tuple[str, List[str]]:
    jdbc = obj.get("url") or state["jdbc_url"]
    queries: List[Any] = []
    if isinstance(obj.get("queries"), list):
        for q in obj["queries"]:
            if isinstance(q, dict) and ("query" in q or "sql" in q):
                entry = {
                    "sql": q.get("sql") or q.get("query"),
                }
                if "runquantity" in q:
                    entry["q"] = q.get("runquantity")
                if "S" in q:
                    entry["S"] = q.get("S")
                if "C" in q:
                    entry["C"] = q.get("C")
                queries.append(entry)
            elif isinstance(q, str):
                queries.append({"sql": q})
    return jdbc, queries


@app.get("/queues/predefined")
def queues_predefined(name: str, limit: int = 0):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f"{name}.json")
    if not os.path.isfile(path):
        return JSONResponse({"error": f"No such predefined: {name}"}, status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    jdbc, queries = _load_queries_from_json_obj(obj)
    if limit > 0:
        queries = queries[:limit]
    return {"jdbc_url": jdbc, "count": len(queries), "queries": queries}


@app.post("/queues/upload")
async def queues_upload(file: UploadFile = File(...), limit: int = Form(0)):
    data = await file.read()
    obj = json.loads(data.decode("utf-8"))
    jdbc, queries = _load_queries_from_json_obj(obj)
    if limit > 0:
        queries = queries[:limit]
    return {"jdbc_url": jdbc, "count": len(queries), "queries": queries}


@app.post("/queues/run")
async def queues_run(queries: str = Form(...), optimize: bool = Form(True), jdbc_url: Optional[str] = Form(None)):
    # queries is JSON-encoded list of SQL strings
    try:
        raw_list: List[Any] = json.loads(queries)
    except Exception:
        return JSONResponse({"error": "Invalid queries payload"}, status_code=400)

    if jdbc_url:
        state["jdbc_url"] = jdbc_url
    _ensure_ready()
    assert trino_exec

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "progress": [],  # list of per-query dicts
        "total": len(raw_list),
        "done": 0,
        "started_at": time.time(),
        "optimize": bool(optimize),
    }
    # persist job
    db.create_job(job_id, len(raw_list), bool(optimize), state["jdbc_url"], jobs[job_id]["started_at"])  # type: ignore[index]

    # normalize entries -> (sql, meta)
    qlist: List[Tuple[str, Dict[str, Any]]] = []
    for entry in raw_list:
        sql, meta = _extract_sql_meta(entry)
        if sql:
            qlist.append((sql, meta))

    async def worker(job_id: str, qlist: List[Tuple[str, Dict[str, Any]]], optimize: bool):
        try:
            sum_Q = 0.0
            sum_QA_over_T = 0.0
            sum_S = 0.0
            sum_C = 0.0
            def _trino_bytes_from_item_part(part: Optional[Dict[str, Any]]) -> Optional[float]:
                """Extract processed_bytes (prefer) or processed_rows from an item sub-dict {result: {trino: ...}}"""
                try:
                    tri = ((part or {}).get("result") or {}).get("trino") or {}
                    val = tri.get("processed_bytes")
                    if val is None:
                        val = tri.get("processedBytes")
                    if val is None:
                        val = tri.get("processed_rows")
                    if val is None:
                        val = tri.get("processedRows")
                    return float(val) if val is not None else None
                except Exception:
                    return None
            for idx, (q, meta) in enumerate(qlist, start=1):
                if jobs.get(job_id, {}).get("status") != "running":
                    break
                item: Dict[str, Any] = {"index": idx, "sql": q, "meta": meta}
                try:
                    db.set_job_phase(job_id, f"executing_original_{idx}")
                    t0 = time.monotonic()
                    res, err = await trino_exec.run_async(q, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
                    ms_client = int((time.monotonic() - t0) * 1000)
                    triA = (res or {}).get("trino", {}) if res else {}
                    if triA and triA.get("cpu_ms") is not None:
                        ms = int(triA.get("cpu_ms"))
                    elif triA and triA.get("execution_ms") is not None:
                        ms = int(triA.get("execution_ms"))
                    elif triA and triA.get("elapsed_ms") is not None:
                        ms = int(triA.get("elapsed_ms"))
                    else:
                        ms = ms_client
                    item["original"] = {"error": err, "result": res, "ms": ms}

                    if optimize and not err:
                        if not llm:
                            trace = {"steps": []}
                        else:
                            db.set_job_phase(job_id, f"optimizing_{idx}")
                            trace = await asyncio.to_thread(optimize_sql_with_trace, llm, trino_exec, q, 3, "trino", True)
                        opt_sql = trace.get("optimized_sql") or ""
                        db.set_job_phase(job_id, f"executing_optimized_{idx}")
                        t1 = time.monotonic()
                        opt_res, opt_err = await trino_exec.run_async(opt_sql, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
                        ms2_client = int((time.monotonic() - t1) * 1000)
                        triB = (opt_res or {}).get("trino", {}) if opt_res else {}
                        if triB and triB.get("cpu_ms") is not None:
                            ms2 = int(triB.get("cpu_ms"))
                        elif triB and triB.get("execution_ms") is not None:
                            ms2 = int(triB.get("execution_ms"))
                        elif triB and triB.get("elapsed_ms") is not None:
                            ms2 = int(triB.get("elapsed_ms"))
                        else:
                            ms2 = ms2_client
                        item["optimized_sql"] = opt_sql
                        item["optimized"] = {"error": opt_err, "result": opt_res, "ms": ms2}
                        if opt_res and res:
                            item["compare"] = _compare_results(res, opt_res)
                        item["trace"] = trace.get("steps", [])
                    elif optimize and err:
                        # Оригинальный запрос упал — пытаемся восстановить
                        if not llm:
                            trace = {"steps": []}
                        else:
                            db.set_job_phase(job_id, f"retrying_{idx}")
                            trace = await asyncio.to_thread(optimize_sql_with_trace, llm, trino_exec, q, 3, "trino", True)
                        retry_sql = trace.get("optimized_sql") or ""
                        db.set_job_phase(job_id, f"executing_retry_{idx}")
                        t1 = time.monotonic()
                        retry_res, retry_err = await trino_exec.run_async(retry_sql, allow_unsafe=not bool(state.get("trino_safe_mode", True)))
                        ms2_client = int((time.monotonic() - t1) * 1000)
                        triC = (retry_res or {}).get("trino", {}) if retry_res else {}
                        if triC and triC.get("cpu_ms") is not None:
                            ms2 = int(triC.get("cpu_ms"))
                        elif triC and triC.get("execution_ms") is not None:
                            ms2 = int(triC.get("execution_ms"))
                        elif triC and triC.get("elapsed_ms") is not None:
                            ms2 = int(triC.get("elapsed_ms"))
                        else:
                            ms2 = ms2_client
                        item["retry_sql"] = retry_sql
                        item["retry"] = {"error": retry_err, "result": retry_res, "ms": ms2}
                        item["trace"] = trace.get("steps", [])
                except Exception as e:
                    item["error"] = str(e)
                # Metrics per item
                Q = float(meta.get("q") or 1.0)
                A = item.get("original", {}).get("ms") if item.get("original") else None
                T = None
                if item.get("optimized"):
                    T = item["optimized"].get("ms")
                elif item.get("retry"):
                    T = item["retry"].get("ms")
                x_i = _metric_x(A, T, Q)
                pr = _percent_reduction(A, T)
                # y per item: prefer provided S/C; fallback to resource metric from Trino stats
                y_i = None
                if (meta.get("S") is not None and meta.get("C") is not None):
                    y_i = _metric_y(meta.get("S"), meta.get("C"))
                else:
                    Sb = _resource_metric_from_part(item.get("original"))
                    Cb = _resource_metric_from_part(item.get("optimized") or item.get("retry"))
                    if Sb is not None and Cb is not None and Cb > 0:
                        y_i = Sb / Cb
                z_i = _metric_z(x_i, y_i)
                item["metrics"] = {"x": x_i, "percent_time_reduction": pr, "y": y_i, "z": z_i, "Q": Q, "A_ms": A, "T_ms": T}
                if x_i is not None:
                    sum_QA_over_T += Q * (A or 0) / (T or 1)
                sum_Q += Q
                # Aggregate resources if provided (weighted by Q)
                # Aggregate resources S/C
                if meta.get("S") is not None and meta.get("C") is not None:
                    try:
                        s_val = float(meta.get("S"))
                        c_val = float(meta.get("C"))
                        if c_val > 0:
                            sum_S += Q * s_val
                            sum_C += Q * c_val
                    except Exception:
                        pass
                else:
                    Sb = _resource_metric_from_part(item.get("original"))
                    Cb = _resource_metric_from_part(item.get("optimized") or item.get("retry"))
                    if Cb is not None and Cb > 0 and Sb is not None:
                        sum_S += Q * Sb
                        sum_C += Q * Cb
                jobs[job_id]["progress"].append(item)
                db.append_item(job_id, idx, item)
                jobs[job_id]["done"] = idx
                db.update_job_progress(job_id, idx)
                await asyncio.sleep(0)  # yield control
        except Exception as e:
            jobs[job_id]["error"] = str(e)
            db.set_job_error(job_id, str(e))
        finally:
            jobs[job_id]["status"] = "finished"
            jobs[job_id]["finished_at"] = time.time()
            # persist job-level metrics
            x_overall = (sum_QA_over_T / sum_Q) if sum_Q>0 else None
            y_overall = (sum_S / sum_C) if (sum_S>0 and sum_C>0) else None
            z_overall = ((x_overall ** (1.0/3.0)) * y_overall) if (x_overall is not None and y_overall is not None) else (None if x_overall is None and y_overall is None else (y_overall if x_overall is None else (x_overall ** (1.0/3.0))))
            jobs[job_id]["metrics"] = {"x": x_overall, "y": y_overall, "z": z_overall}
            db.finish_job(job_id, jobs[job_id]["finished_at"])  # type: ignore[index]
            db.set_job_phase(job_id, None)
            # Persist whatever is available
            payload = {}
            if x_overall is not None:
                payload["x"] = x_overall
            if y_overall is not None:
                payload["y"] = y_overall
            if z_overall is not None:
                payload["z"] = z_overall
            if payload:
                db.set_job_metrics(job_id, payload)

    asyncio.create_task(worker(job_id, qlist, optimize))
    return {"job_id": job_id}


@app.get("/jobs/feed")
def jobs_feed():
    return {"jobs": db.list_jobs()}

@app.get("/jobs/{job_id}")
def jobs_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job

@app.delete("/jobs/{job_id}")
def jobs_delete(job_id: str):
    db.delete_job(job_id)
    jobs.pop(job_id, None)
    return {"ok": True}

@app.post("/jobs/{job_id}/rerun")
async def jobs_rerun(job_id: str, optimize: bool = Form(True), jdbc_url: Optional[str] = Form(None)):
    queries = db.get_job_queries(job_id)
    if not queries:
        return JSONResponse({"error": "no queries to rerun"}, status_code=400)
    if jdbc_url:
        state["jdbc_url"] = jdbc_url
        cfg = parse_trino_jdbc_url(jdbc_url)
        global trino_exec
        trino_exec = TrinoExecutorModel(cfg)
        db.set_setting("jdbc_url", jdbc_url)
    _ensure_ready()
    new_id = str(uuid.uuid4())
    jobs[new_id] = {
        "status": "running", "progress": [], "total": len(queries), "done": 0,
        "started_at": time.time(), "optimize": bool(optimize)
    }
    db.create_job(new_id, len(queries), bool(optimize), state["jdbc_url"], jobs[new_id]["started_at"])  # type: ignore[index]

    async def worker(new_id: str, qlist: List[str], optimize: bool):
        try:
            sum_Q = 0.0
            sum_QA_over_T = 0.0
            sum_S = 0.0
            sum_C = 0.0
            def _bytes_from_part(part: Optional[Dict[str, Any]]) -> Optional[float]:
                try:
                    tri = ((part or {}).get("result") or {}).get("trino") or {}
                    val = tri.get("processed_bytes") or tri.get("processedBytes") or tri.get("processed_rows") or tri.get("processedRows")
                    return float(val) if val is not None else None
                except Exception:
                    return None
            for idx, q in enumerate(qlist, start=1):
                if jobs.get(new_id, {}).get("status") != "running":
                    break
                item: Dict[str, Any] = {"index": idx, "sql": q}
                try:
                    t0 = time.monotonic()
                    res, err = await trino_exec.run_async(q)
                    ms_client = int((time.monotonic() - t0) * 1000)
                    triD = (res or {}).get("trino", {}) if res else {}
                    if triD and triD.get("cpu_ms") is not None:
                        ms = int(triD.get("cpu_ms"))
                    elif triD and triD.get("execution_ms") is not None:
                        ms = int(triD.get("execution_ms"))
                    elif triD and triD.get("elapsed_ms") is not None:
                        ms = int(triD.get("elapsed_ms"))
                    else:
                        ms = ms_client
                    item["original"] = {
                        "error": err,
                        "result": res,
                        "ms": ms,
                        "times": {
                            "execution_ms": triD.get("execution_ms"),
                            "elapsed_ms": triD.get("elapsed_ms"),
                            "client_ms": ms_client,
                            "queued_ms": triD.get("queued_ms"),
                            "cpu_ms": triD.get("cpu_ms"),
                            "scheduled_ms": triD.get("scheduled_ms"),
                        },
                    }
                    right = None
                    if optimize and not err:
                        assert llm and trino_exec
                        trace = await asyncio.to_thread(optimize_sql_with_trace, llm, trino_exec, q, 3, "trino", True)
                        opt_sql = trace.get("optimized_sql") or ""
                        t1 = time.monotonic()
                        opt_res, opt_err = await trino_exec.run_async(opt_sql)
                        ms2_client = int((time.monotonic() - t1) * 1000)
                        triE = (opt_res or {}).get("trino", {}) if opt_res else {}
                        if triE and triE.get("cpu_ms") is not None:
                            ms2 = int(triE.get("cpu_ms"))
                        elif triE and triE.get("execution_ms") is not None:
                            ms2 = int(triE.get("execution_ms"))
                        elif triE and triE.get("elapsed_ms") is not None:
                            ms2 = int(triE.get("elapsed_ms"))
                        else:
                            ms2 = ms2_client
                        item["optimized_sql"] = opt_sql
                        item["optimized"] = {
                            "error": opt_err,
                            "result": opt_res,
                            "ms": ms2,
                            "times": {
                                "execution_ms": triE.get("execution_ms"),
                                "elapsed_ms": triE.get("elapsed_ms"),
                                "client_ms": ms2_client,
                                "queued_ms": triE.get("queued_ms"),
                                "cpu_ms": triE.get("cpu_ms"),
                                "scheduled_ms": triE.get("scheduled_ms"),
                            },
                        }
                        right = item["optimized"]
                        if opt_res and res:
                            item["compare"] = _compare_results(res, opt_res)
                        item["trace"] = trace.get("steps", [])
                    # Per-item metrics
                    A = item.get("original", {}).get("ms")
                    T = (right or {}).get("ms")
                    x_i = _metric_x(A, T, 1.0)
                    pr = _percent_reduction(A, T)
                    Sb = _bytes_from_part(item.get("original"))
                    Cb = _bytes_from_part(right)
                    y_i = (Sb / Cb) if (Sb is not None and Cb is not None and Cb > 0) else None
                    z_i = _metric_z(x_i, y_i)
                    item["metrics"] = {"x": x_i, "y": y_i, "z": z_i, "percent_time_reduction": pr, "A_ms": A, "T_ms": T}
                    if x_i is not None and T and T > 0 and A is not None:
                        sum_QA_over_T += 1.0 * (A / T)
                    sum_Q += 1.0
                    if Sb is not None and Cb is not None and Cb > 0:
                        sum_S += Sb
                        sum_C += Cb
                except Exception as e:
                    item["error"] = str(e)
                jobs[new_id]["progress"].append(item)
                db.append_item(new_id, idx, item)
                jobs[new_id]["done"] = idx
                db.update_job_progress(new_id, idx)
                await asyncio.sleep(0)
        except Exception as e:
            jobs[new_id]["error"] = str(e)
            db.set_job_error(new_id, str(e))
        finally:
            jobs[new_id]["status"] = "finished"
            jobs[new_id]["finished_at"] = time.time()
            db.finish_job(new_id, jobs[new_id]["finished_at"])  # type: ignore[index]
            # Persist job-level metrics
            x_overall = (sum_QA_over_T / sum_Q) if sum_Q>0 else None
            y_overall = (sum_S / sum_C) if (sum_S>0 and sum_C>0) else None
            z_overall = ((x_overall ** (1.0/3.0)) * y_overall) if (x_overall is not None and y_overall is not None) else None
            payload = {}
            if x_overall is not None: payload["x"] = x_overall
            if y_overall is not None: payload["y"] = y_overall
            if z_overall is not None: payload["z"] = z_overall
            if payload:
                db.set_job_metrics(new_id, payload)

    asyncio.create_task(worker(new_id, queries, bool(optimize)))
    return {"job_id": new_id}


# Entry for `uvicorn langpipe.webapp.main:app --reload`
@app.on_event("startup")
def _startup():
    # Load .env defaults for maximum-quality profile and other settings
    try:
        load_dotenv()
    except Exception:
        pass
    db.init_db()
    # load persisted settings
    settings = db.get_all_settings()
    if settings.get("jdbc_url"):
        state["jdbc_url"] = settings["jdbc_url"]
    if settings.get("model_mode"):
        state["model_mode"] = settings["model_mode"]
    if settings.get("openrouter_model"):
        state["openrouter_model"] = settings["openrouter_model"]
    if settings.get("ollama_base"):
        state["ollama_base"] = settings["ollama_base"]
    if settings.get("ollama_model"):
        state["ollama_model"] = settings["ollama_model"]
    if settings.get("trino_safe_mode") is not None:
        state["trino_safe_mode"] = settings["trino_safe_mode"] in ("1", "true", "True")
    _init_models()
# ---------------------------
# Asynchronous analysis API (non-visual service)
# ---------------------------

analysis_tasks: Dict[str, Dict[str, Any]] = {}


def _extract_create_table_names(ddl_statements: List[str]) -> List[str]:
    import re
    names: List[str] = []
    pat = re.compile(r"create\s+table\s+([\w\.]+)", re.IGNORECASE)
    for st in ddl_statements:
        m = pat.search(st or "")
        if m:
            full = m.group(1)
            # Keep only table name if qualified
            tbl = full.split(".")[-1]
            names.append(tbl)
    return list(sorted(set(names)))


def _rewrite_query_to_new_schema(sql: str, catalog: str, old_schema: Optional[str], new_schema: str, known_tables: List[str]) -> str:
    import re
    s = str(sql or "")
    # Replace FQN occurrences catalog.old_schema.tbl -> catalog.new_schema.tbl
    if old_schema:
        s = re.sub(rf"\b{re.escape(catalog)}\.{re.escape(old_schema)}\.", f"{catalog}.{new_schema}.", s)
    # Qualify known bare table names to catalog.new_schema.table (best-effort)
    for t in known_tables:
        # Skip if already qualified in this SQL
        # Replace only when word is not preceded by '.'
        s = re.sub(rf"(?<!\.)\b{re.escape(t)}\b", f"{catalog}.{new_schema}.{t}", s)
    return s


async def _run_analysis_task(task_id: str, payload: Dict[str, Any]):
    analysis_tasks[task_id]["status"] = "RUNNING"
    try:
        t_start = time.monotonic()
        max_seconds = 1200.0  # 20 minutes default
        try:
            max_seconds = float(os.getenv("ANALYSIS_MAX_SECONDS", "1200"))
        except Exception:
            pass
        # Parse JDBC to extract catalog/schema
        jdbc = payload.get("url") or payload.get("jdbc")
        catalog = None
        schema = None
        try:
            cfg = parse_trino_jdbc_url(jdbc)
            catalog = cfg.catalog
            schema = cfg.schema
        except Exception:
            pass
        if not catalog:
            catalog = os.getenv("TRINO_CATALOG", "data")
        old_schema = schema or os.getenv("TRINO_SCHEMA")
        new_schema = os.getenv("ANALYSIS_NEW_SCHEMA", "optimized")
        ddl_list = [str(item.get("statement") or "") for item in (payload.get("ddl") or []) if isinstance(item, dict)]
        known_tables = _extract_create_table_names(ddl_list)

        # Initialize LLM + executor for optimization pipeline
        llm_obj = LLM()
        exec_model: TrinoExecutorModel | SQLExecutorModel
        try:
            if jdbc:
                cfg_exec = parse_trino_jdbc_url(jdbc)
                # Respect safety
                exec_model = TrinoExecutorModel(cfg_exec, safe_mode=True)
            else:
                exec_model = SQLExecutorModel(llm_obj)
        except Exception:
            exec_model = SQLExecutorModel(llm_obj)

        out_ddl: List[Dict[str, str]] = []
        out_migrations: List[Dict[str, str]] = []
        out_queries: List[Dict[str, str]] = []

        # Optional schema advisor stage (unless only_query=true)
        only_query = bool(payload.get("only_query", False))
        need_schema_changes = False
        if not only_query:
            system = (
                "You are a database schema advisor. Given a JDBC URL (with catalog/schema), current DDLs and a list of frequent SQL queries, "
                "decide if schema changes are beneficial for performance and propose textual DDL + migrations. Do NOT connect to the DB. Return JSON only."
            )
            user = {
                "jdbc": jdbc,
                "catalog": catalog,
                "schema": old_schema,
                "ddl": ddl_list[:20],
                "queries": [
                    {"id": str(q.get("queryid") or q.get("id") or ""), "runquantity": int(q.get("runquantity") or 0), "query": str(q.get("query") or "")[:4000]}
                    for q in (payload.get("queries") or []) if isinstance(q, dict)
                ][:50],
                "instructions": (
                    "Return JSON: need_schema_changes (bool), new_schema (string), ddl (array of statements), migrations (array of statements). "
                    "If need_schema_changes=false, keep ddl/migrations empty. If true: first DDL MUST be CREATE SCHEMA <catalog>.<new_schema>. "
                    "All statements MUST use fully qualified names <catalog>.<new_schema>.<table> and migrations must read from <catalog>.<old_schema>.<table>."
                ),
            }
            # run blocking LLM call in a worker thread to avoid blocking the event loop
            advisor_resp = await asyncio.to_thread(
                llm_obj.chat,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                None,
                True,
            )
            adv = extract_json_best_effort(advisor_resp)
            need_schema_changes = bool(adv.get("need_schema_changes")) if isinstance(adv, dict) else False
            if need_schema_changes and isinstance(adv, dict):
                ns = adv.get("new_schema")
                if isinstance(ns, str) and ns.strip():
                    new_schema = ns.strip()
                for st in (adv.get("ddl") or []):
                    if isinstance(st, str) and st.strip():
                        out_ddl.append({"statement": st.strip()})
                if not out_ddl:
                    out_ddl.append({"statement": f"CREATE SCHEMA {catalog}.{new_schema}"})
                for st in (adv.get("migrations") or []):
                    if isinstance(st, str) and st.strip():
                        out_migrations.append({"statement": st.strip()})
                # Avoid executing against non-existent schema
                exec_model = SQLExecutorModel(llm_obj)
        else:
            # Backward-compatible stub for new schema
            out_ddl.append({"statement": f"CREATE SCHEMA {catalog}.{new_schema}"})

        # Optimize each query via the same pipeline as UI, then re-qualify to the (possibly) new schema
        # track progress for visibility
        queries_in = [q for q in (payload.get("queries") or []) if isinstance(q, dict)]
        analysis_tasks[task_id]["total"] = len(queries_in)
        analysis_tasks[task_id]["done"] = 0

        for q in queries_in:
            if time.monotonic() - t_start > max_seconds:
                raise TimeoutError("analysis timeout exceeded")
            if not isinstance(q, dict):
                continue
            qid = str(q.get("queryid") or q.get("id") or "")
            sql_in = str(q.get("query") or "")
            # Run optimize/trace
            trace = await asyncio.to_thread(
                optimize_sql_with_trace,
                llm_obj,
                exec_model,
                sql_in,
                int(os.getenv("PIPE_MAX_ATTEMPTS", "3")),
                os.getenv("SQL_DIALECT", "trino"),
                True,
            )
            sql_opt = trace.get("optimized_sql") or ""
            if not sql_opt:
                # Try best from explain_rank
                steps = trace.get("steps", [])
                for st in steps:
                    if st.get("name") == "explain_rank":
                        out = st.get("output") or []
                        if out:
                            sql_opt = out[0].get("sql") or sql_opt
                        break
            if not sql_opt:
                # Fallback: original SQL
                sql_opt = sql_in
            # Re-qualify schema if needed
            if (not only_query and need_schema_changes) or only_query:
                sql_new = _rewrite_query_to_new_schema(sql_opt, catalog=catalog or "data", old_schema=old_schema, new_schema=new_schema, known_tables=known_tables)
            else:
                sql_new = sql_opt
            out_queries.append({"queryid": qid, "query": sql_new})
            analysis_tasks[task_id]["done"] = analysis_tasks[task_id].get("done", 0) + 1
            # yield control to event loop to keep API responsive
            await asyncio.sleep(0)

        analysis_tasks[task_id]["result"] = {"ddl": out_ddl, "migrations": out_migrations, "queries": out_queries}
        analysis_tasks[task_id]["status"] = "DONE"
    except Exception as e:
        analysis_tasks[task_id]["status"] = "FAILED"
        analysis_tasks[task_id]["error"] = str(e)


@app.post("/new")
async def api_new(payload: Dict[str, Any]):
    # Minimal validation
    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid payload"}, status_code=400)
    task_id = str(uuid.uuid4())
    analysis_tasks[task_id] = {"status": "PENDING", "created_at": time.time()}
    asyncio.create_task(_run_analysis_task(task_id, payload))
    return {"taskid": task_id}


@app.get("/status")
async def api_status(task_id: str):
    t = analysis_tasks.get(task_id)
    if not t:
        return JSONResponse({"error": "unknown task"}, status_code=404)
    return {"status": t.get("status", "PENDING")}


@app.get("/getresult")
async def api_getresult(task_id: str):
    t = analysis_tasks.get(task_id)
    if not t:
        return JSONResponse({"error": "unknown task"}, status_code=404)
    st = t.get("status")
    if st == "RUNNING" or st == "PENDING":
        return JSONResponse({"error": "task not finished", "status": st}, status_code=409)
    if st == "FAILED":
        return JSONResponse({"error": t.get("error") or "failed"}, status_code=500)
    return t.get("result") or {"ddl": [], "migrations": [], "queries": []}
