from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv  # type: ignore


def load_queries(path: str, limit: int = 5) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: List[str] = []
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict) and isinstance(it.get("sql") or it.get("query"), str):
                out.append(it.get("sql") or it.get("query"))
    elif isinstance(obj, dict):
        q = obj.get("queries")
        if isinstance(q, list):
            for it in q:
                if isinstance(it, str):
                    out.append(it)
                elif isinstance(it, dict) and isinstance(it.get("sql") or it.get("query"), str):
                    out.append(it.get("sql") or it.get("query"))
    return out[:limit]


def extract_step(steps: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for s in steps:
        if s.get("name") == name:
            return s
    return None


def extract_cpu_ms(step: Optional[Dict[str, Any]]) -> Optional[int]:
    if not step:
        return None
    # We record preferred CPU time into step.ms when available
    ms = step.get("ms")
    try:
        return int(ms) if ms is not None else None
    except Exception:
        return None


def extract_trino_stats(step: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        out = ((step or {}).get("output") or {}).get("trino") or {}
        if not out and isinstance(step, dict):
            # sometimes we stored the whole result in step["output"]
            out = (step.get("output") or {}).get("result", {}).get("trino") or {}
        return out
    except Exception:
        return {}


def resource_metric(tri: Dict[str, Any], source: str = "processedBytes") -> Optional[float]:
    source = (source or "processedBytes").lower()
    def pick(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in tri and tri[k] is not None:
                try:
                    return float(tri[k])
                except Exception:
                    continue
        return None
    if source in ("physicalinputbytes", "physical_input_bytes"):
        return pick(["physicalInputBytes", "physical_input_bytes"])
    if source in ("processedbytes", "processed_bytes"):
        return pick(["processedBytes", "processed_bytes"])
    if source in ("processedrows", "processed_rows"):
        return pick(["processedRows", "processed_rows"])
    if source in ("peakmemorybytes", "peak_memory_bytes"):
        return pick(["peakMemoryBytes", "peak_memory_bytes"])
    return pick(["processedBytes", "processed_bytes", "processedRows", "processed_rows"])


def main() -> int:
    load_dotenv()
    # Make sure Python can import our package
    _LP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _REPO_ROOT = os.path.dirname(_LP_DIR)
    for p in (_REPO_ROOT, _LP_DIR):
        if p not in sys.path:
            sys.path.append(p)
    from langpipe.langpipe.llm import LLM
    from langpipe.langpipe.trino_exec import parse_trino_jdbc_url, TrinoExecutorModel
    from langpipe.webapp.pipeline_trace import optimize_sql_with_trace

    jdbc = os.getenv("TRINO_JDBC_URL") or os.getenv("HARDCODE_TRINO_URL") or (
        "jdbc:trino://trino.czxqx2r9.data.bizmrg.com:443?user=hackuser&password=dovq(ozaq8ngt)oS&catalog=quests&schema=public"
    )
    source_metric = os.getenv("RESOURCE_METRIC_SOURCE", "processedBytes")
    cfg = parse_trino_jdbc_url(jdbc)
    execm = TrinoExecutorModel(cfg, safe_mode=True)
    llm = LLM()

    # choose input file
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(_LP_DIR), "questsH.json")
    if not os.path.isfile(path):
        path = os.path.join(os.path.dirname(_LP_DIR), "questsH.json")
    if not os.path.isfile(path):
        print("Cannot find questsH.json; pass path as argument.")
        return 2
    queries = load_queries(path, limit=int(os.getenv("BULK_LIMIT", "5")))
    if not queries:
        print("No queries found in", path)
        return 2

    rows: List[Dict[str, Any]] = []
    for idx, q in enumerate(queries, start=1):
        print(f"Running {idx}/{len(queries)} ...")
        res = optimize_sql_with_trace(llm, execm, q, max_attempts=3, dialect="trino", optimize_mode=True)
        steps = res.get("steps") or []
        s0 = extract_step(steps, "exec_original")
        s1 = extract_step(steps, "sql_executor")
        A = extract_cpu_ms(s0)
        T = extract_cpu_ms(s1)
        tri0 = extract_trino_stats(s0)
        tri1 = extract_trino_stats(s1)
        S = resource_metric(tri0, source_metric)
        C = resource_metric(tri1, source_metric)
        x = (float(A)/float(T)) if (A and T and T>0) else None
        y = (float(S)/float(C)) if (S and C and C>0) else None
        z = ((x ** (1.0/3.0)) * y) if (x is not None and y is not None) else None
        ve = extract_step(steps, "verify_equivalence")
        mismatch = False
        if ve and isinstance(ve.get("output"), dict):
            out = ve["output"]
            mismatch = not (out.get("same_cols") and out.get("rowcount_equal"))
        rows.append({
            "index": idx,
            "A_cpu_ms": A,
            "T_cpu_ms": T,
            "x": round(x, 3) if x is not None else None,
            "y": round(y, 3) if y is not None else None,
            "z": round(z, 3) if z is not None else None,
            "mismatch": mismatch,
        })

    # Print summary
    print("\nSummary (A/T=x, z=x^(1/3)*y):")
    for r in rows:
        print(f"#{r['index']:>2} A={r['A_cpu_ms']} T={r['T_cpu_ms']} x={r['x']} y={r['y']} z={r['z']} mismatch={r['mismatch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
