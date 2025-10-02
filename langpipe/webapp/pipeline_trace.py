from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..langpipe.llm import LLM
from ..langpipe.sql import SQLExecutorModel
from ..langpipe.trino_exec import TrinoExecutorModel
from ..langpipe.trino_schema import introspect_schema
import logging
from ..langpipe.utils import schema_subset_for_sql, clip_text
import os
from ..langpipe.critic import call_critic, call_enhance_prompt, call_fix_error_prompt
from ..langpipe.generator import call_generate_sql
from ..langpipe.utils import extract_fqn_tables


def optimize_sql_with_trace(
    llm: LLM,
    executor: TrinoExecutorModel | SQLExecutorModel,
    user_sql: str,
    max_attempts: int = 3,
    dialect: str = "trino",
    optimize_mode: bool = True,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {
        "user_query": f"Оптимизируй и перепиши эквивалентно:\n{user_sql}",
        "chat_history": [],
        "db_schema": None,
        "dialect": dialect,
        "trino_catalog": "",
        "trino_schema": "",
    }

    # Allow env to override attempts used inside this function
    try:
        max_attempts = int(os.getenv("PIPE_MAX_ATTEMPTS", str(max_attempts)))
    except Exception:
        pass

    # 1) Introspect schema if Trino
    if isinstance(executor, TrinoExecutorModel):
        t0 = time.monotonic()
        schema_text = introspect_schema(executor.cfg)
        if not schema_text:
            logging.warning("schema_introspect_trino: empty schema (catalog=%s schema=%s)", executor.cfg.catalog, executor.cfg.schema)
        steps.append({
            "name": "schema_introspect_trino",
            "ms": int((time.monotonic() - t0) * 1000),
            "output": {
                "ok": bool(schema_text),
                "schema": schema_text or "",
                "chars": len(schema_text) if schema_text else 0,
            },
        })
        if schema_text:
            state["db_schema"] = schema_text
            state["trino_catalog"] = executor.cfg.catalog or ""
            state["trino_schema"] = executor.cfg.schema or ""

    executed_ok = False
    # 2) Если есть исходный SQL — попробуем выполнить его напрямую и при ошибке сразу включим цикл фиксации
    if user_sql.strip():
        t0 = time.monotonic()
        sub_schema = schema_subset_for_sql(user_sql, state.get("db_schema"))
        res0, err0 = executor.run(user_sql, sub_schema)
        client_ms0 = int((time.monotonic() - t0) * 1000)
        # choose ms by cpu -> execution -> elapsed -> client
        tri0 = (res0 or {}).get("trino") if isinstance(res0, dict) else None
        if tri0 and tri0.get("cpu_ms") is not None:
            ms0 = int(tri0.get("cpu_ms"))
        elif tri0 and tri0.get("execution_ms") is not None:
            ms0 = int(tri0.get("execution_ms"))
        elif tri0 and tri0.get("elapsed_ms") is not None:
            ms0 = int(tri0.get("elapsed_ms"))
        else:
            ms0 = client_ms0
        steps.append({
            "name": "exec_original",
            "ms": ms0,
            "output": {
                "error": err0,
                "ok": bool(res0),
                "rowcount": (res0 or {}).get("rowcount") if isinstance(res0, dict) else None,
                "columns": (res0 or {}).get("columns") if isinstance(res0, dict) else None,
                "trino": tri0,
                "client_ms": client_ms0,
            }
        })
        if not err0:
            # Исходный SQL валиден — считаем его оптимизированным (эквивалентным),
            # но НЕ выходим сразу — дадим пройти через критика, чтобы было видно решение модели
            state["sql_query"] = user_sql
            optimized_sql = user_sql
            error = None
            executed_ok = True
        else:
            # подготовим fixed_prompt от ошибки, чтобы первая генерация была нацелена на исправление
            tfix = time.monotonic()
            fix0 = call_fix_error_prompt(
                llm=llm,
                user_query=state.get("user_query", ""),
                sql_query=user_sql,
                error=err0,
                db_schema=sub_schema,
                chat_history=state.get("chat_history"),
            )
            steps.append({"name": "fix_error_prompt", "ms": int((time.monotonic() - tfix) * 1000), "output": fix0})
            state["current_prompt"] = fix0.get("fixed_prompt") or state.get("current_prompt")

    # 3) Critic
    t0 = time.monotonic()
    # Prepare clipped schema for LLM to avoid overlong context
    _schema_for_critic = schema_subset_for_sql(state.get("sql_query", ""), state.get("db_schema")) or state.get("db_schema")
    _schema_for_critic = clip_text(_schema_for_critic, int(os.getenv("LLM_SCHEMA_MAX_CHARS", "2000")))
    _user_query_clipped = clip_text(state.get("user_query", ""), int(os.getenv("LLM_USER_MAX_CHARS", "2000"))) or ""

    critic = call_critic(
        llm=llm,
        user_query=_user_query_clipped,
        sql_query=state.get("sql_query"),
        db_schema=_schema_for_critic,
        chat_history=state.get("chat_history"),
    )
    if not critic.get("route"):
        logging.error("llm_critic: missing route; raw_len=%s", len(critic.get("raw") or ""))
    steps.append({
        "name": "llm_critic",
        "ms": int((time.monotonic() - t0) * 1000),
        "input": {
            "user_query": state.get("user_query", ""),
            "has_sql": bool(state.get("sql_query")),
            "schema_chars": len(state.get("db_schema") or "")
        },
        "output": critic,
    })
    route = critic.get("route") or "enhance_prompt"
    if optimize_mode:
        route = "enhance_prompt"

    # Даже если исходный SQL выполнился, в режиме оптимизации продолжаем к enhance/generate

    # 4) Branches
    attempt = 0
    fixed_prompt = None
    optimized_sql = None
    error: Optional[str] = None

    def _gen_sql(prompt_override: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        t = time.monotonic()
        eff_prompt = prompt_override if prompt_override is not None else (fixed_prompt or state.get("current_prompt"))
        _prompt = clip_text(eff_prompt, int(os.getenv("LLM_PROMPT_MAX_CHARS", "800")))
        _schema_for_gen = schema_subset_for_sql(state.get("sql_query", ""), state.get("db_schema")) or state.get("db_schema")
        _schema_for_gen = clip_text(_schema_for_gen, int(os.getenv("LLM_SCHEMA_MAX_CHARS", "2000")))
        res = call_generate_sql(
            llm=llm,
            user_query=_user_query_clipped,
            current_prompt=_prompt,
            db_schema=_schema_for_gen,
            chat_history=state.get("chat_history"),
            dialect=state.get("dialect", "trino"),
            catalog=state.get("trino_catalog"),
            schema=state.get("trino_schema"),
        )
        return int((time.monotonic() - t) * 1000), res

    def _collect_metadata_for_tables(sql: str):
        if not isinstance(executor, TrinoExecutorModel):
            return []
        tables = extract_fqn_tables(sql)
        results = []
        for fqn in tables[: int(os.getenv("PIPE_MAX_META_TABLES", "5"))]:
            try:
                # DESCRIBE
                t0 = time.monotonic()
                desc_res, desc_err = executor.run(f"DESCRIBE {fqn}", None)
                ms_desc = int((time.monotonic() - t0) * 1000)
                # SHOW CREATE TABLE
                t1 = time.monotonic()
                sct_res, sct_err = executor.run(f"SHOW CREATE TABLE {fqn}", None)
                ms_sct = int((time.monotonic() - t1) * 1000)
                results.append({
                    "table": fqn,
                    "describe": None if desc_err else (desc_res or {}).get("rows"),
                    "show_create": None if sct_err else (sct_res or {}).get("rows"),
                    "ms": {"describe": ms_desc, "show_create": ms_sct},
                })
            except Exception as e:
                results.append({"table": fqn, "error": str(e)})
        return results

    def _explain_text_for(sql: str):
        if not isinstance(executor, TrinoExecutorModel):
            return None, "not_trino"
        try:
            res, err = executor.run(f"EXPLAIN (TYPE DISTRIBUTED) {sql}", None)
            if err:
                return None, err
            rows = (res or {}).get("rows") or []
            # EXPLAIN returns a single column with plan text; join lines
            plan = "\n".join(r[0] for r in rows if r)
            return plan, None
        except Exception as e:
            return None, str(e)

    def _score_explain(plan_text: str) -> float:
        # Lower is better; rough heuristics
        if not plan_text:
            return 1e9
        txt = plan_text
        tl = txt.lower()
        exchanges = tl.count("exchange")
        repartitions = tl.count("repartition") + tl.count("repartitioning")
        dynamic = tl.count("dynamic filter") + tl.count("dynamicfilter")
        # TableScan counting per table
        import re
        scans = {}
        for m in re.finditer(r"TableScan\[(.*?)\]", txt):
            entry = m.group(1)
            # try to extract table name like catalog.schema.table
            nm = None
            mm = re.search(r"(\w+\.\w+\.\w+)", entry)
            if mm:
                nm = mm.group(1).lower()
            else:
                nm = entry.strip().lower()
            scans[nm] = scans.get(nm, 0) + 1
        repeat_scans = sum(max(0, c - 1) for c in scans.values())
        # Sort/TopN at root (rough): penalize if present and исходник без ORDER BY — пока у нас нет исходного флага, просто penalize
        root_sort = 1 if ("Sort[" in txt or "TopN[" in txt) else 0
        # Combine: penalize exchanges/repartitions/repeat_scans/root_sort, reward dynamic filtering
        score = exchanges * 10 + repartitions * 5 + repeat_scans * 8 + root_sort * 6 - dynamic * 3
        return float(score)

    def _validate_sql_plausible(sql: str) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        s = sql.strip()
        # Reject backslashes that often appear as artifacts of escaping
        if "\\" in s:
            reasons.append("contains backslashes")
        # Parentheses balance
        if s.count("(") != s.count(")"):
            reasons.append("unbalanced parentheses")
        # Unclosed single quotes (account for doubled quotes '')
        tmp = s.replace("''", "")
        if tmp.count("'") % 2 == 1:
            reasons.append("unclosed quotes")
        import re
        # Trailing dangling keywords/operators
        if re.search(r"(?is)\b(AND|OR|ON|WHERE|JOIN|CASE|WHEN|THEN|ELSE|=|,)\s*$", s):
            reasons.append("dangling keyword at end")
        # LIMIT with no number: we'll fix later in ensure_safe_sql; not a hard reject
        return (len(reasons) == 0), reasons

    if route == "enhance_prompt":
        t0 = time.monotonic()
        # Clip schema and inputs for enhance
        _schema_for_enh = clip_text(state.get("db_schema"), int(os.getenv("LLM_SCHEMA_MAX_CHARS", "4096")))
        enh = call_enhance_prompt(
            llm=llm,
            user_query=_user_query_clipped,
            critique=critic.get("raw"),
            issues=critic.get("issues", []),
            db_schema=_schema_for_enh,
            chat_history=state.get("chat_history"),
        )
        if not (enh.get("enhanced_prompt") or "").strip():
            logging.error("enhance_prompt: empty enhanced_prompt; raw_len=%s", len(enh.get("raw") or ""))
        steps.append({
            "name": "enhance_prompt",
            "ms": int((time.monotonic() - t0) * 1000),
            "input": {
                "user_query": state.get("user_query", ""),
                "issues_count": len(critic.get("issues", [])),
                "schema_chars": len(state.get("db_schema") or ""),
            },
            "output": enh,
        })
        # Ensure we always have some guidance prompt (fallback to user query)
        state["current_prompt"] = (enh.get("enhanced_prompt", "") or "").strip() or _user_query_clipped
        # Collect table metadata (DESCRIBE/SHOW CREATE) for referenced tables
        meta = _collect_metadata_for_tables(user_sql)
        if meta:
            steps.append({"name": "collect_metadata", "ms": 0, "output": {"tables": [m.get("table") for m in meta], "details": meta}})

        while attempt < max_attempts:
            ms_gen, gen = _gen_sql()
            if not (gen.get("sql_query") or "").strip():
                # If generator produced no SQL, avoid spawning strategy variants immediately
                logging.error("generate_sql: empty sql_query; raw_len=%s", len(gen.get("raw") or ""))
                steps.append({
                    "name": "generate_sql_empty",
                    "ms": ms_gen,
                    "output": {"raw_len": len(gen.get("raw") or ""), "reason": "empty_generation"},
                })
                # Treat as a generation error and try to nudge via fix_error_prompt once
                t2 = time.monotonic()
                fix = call_fix_error_prompt(
                    llm=llm,
                    user_query=_user_query_clipped,
                    sql_query=state.get("sql_query", ""),
                    error="empty_generation",
                    db_schema=_schema_for_enh,
                    chat_history=state.get("chat_history"),
                )
                steps.append({
                    "name": "fix_error_prompt",
                    "ms": int((time.monotonic() - t2) * 1000),
                    "input": {"error": "empty_generation", "schema_chars": len(state.get("db_schema") or "")},
                    "output": fix,
                })
                fixed_prompt = fix.get("fixed_prompt")
                attempt += 1
                # Try a single regeneration after fix; then continue loop
                continue
            steps.append({
                "name": "generate_sql",
                "ms": ms_gen,
                "input": {
                    "prompt_chars": len((fixed_prompt or state.get("current_prompt") or "")),
                    "schema_chars": len(state.get("db_schema") or ""),
                },
                "output": gen,
            })
            # Build candidate set: deterministic + initial + strategy variants
            base_sql = (gen.get("sql_query", "") or "").strip()
            # Optional SQL normalization via sqlglot
            try:
                from ..langpipe.utils import maybe_sqlglot_format
                _norm = maybe_sqlglot_format(base_sql, dialect=state.get("dialect", "trino"))
                if _norm:
                    base_sql = _norm
            except Exception:
                pass
            candidates = []
            if not base_sql:
                # Do not attempt strategy variants when base is empty
                steps.append({"name": "generate_candidates", "ms": 0, "output": {"count": 0}})
                # Go fix prompt and retry loop
                t2 = time.monotonic()
                fix = call_fix_error_prompt(
                    llm=llm,
                    user_query=_user_query_clipped,
                    sql_query=state.get("sql_query", ""),
                    error="empty_generation",
                    db_schema=_schema_for_enh,
                    chat_history=state.get("chat_history"),
                )
                steps.append({
                    "name": "fix_error_prompt",
                    "ms": int((time.monotonic() - t2) * 1000),
                    "input": {"error": "empty_generation", "schema_chars": len(state.get("db_schema") or "")},
                    "output": fix,
                })
                fixed_prompt = fix.get("fixed_prompt")
                attempt += 1
                continue
            # Plausibility gate for base SQL
            ok_plausible, reasons = _validate_sql_plausible(base_sql)
            if not ok_plausible:
                steps.append({"name": "plausibility_check", "ms": 0, "output": {"ok": False, "reasons": reasons}})
                # First try a quick single-SQL regeneration without JSON and prose
                mini = ("Верни только валидный SQL под Trino. "
                        "Без пояснений, без кодовых блоков, без кавычек вокруг всего запроса. ")
                ms_g3, gen3 = _gen_sql(prompt_override=(state.get("current_prompt") or "") + "\n" + mini)
                sql3 = (gen3.get("sql_query") or "").strip()
                if sql3:
                    ok3, reasons3 = _validate_sql_plausible(sql3)
                    steps.append({"name": "regen_single_sql", "ms": ms_g3, "output": {"ok": ok3, "reasons": reasons3, "preview": sql3[:100]}})
                    if ok3:
                        base_sql = sql3
                    else:
                        # If still invalid, fall back to fix_error_prompt
                        t2 = time.monotonic()
                        fix = call_fix_error_prompt(
                            llm=llm,
                            user_query=_user_query_clipped,
                            sql_query=state.get("sql_query", ""),
                            error="invalid_sql",
                            db_schema=_schema_for_enh,
                            chat_history=state.get("chat_history"),
                        )
                        steps.append({"name": "fix_error_prompt", "ms": int((time.monotonic() - t2) * 1000), "input": {"error": "invalid_sql"}, "output": fix})
                        fixed_prompt = fix.get("fixed_prompt")
                        attempt += 1
                        continue
            # Optional EXPLAIN gate for base candidate (Trino only)
            base_gated = False
            base_plan_text = None
            if isinstance(executor, TrinoExecutorModel):
                try:
                    gate = float(os.getenv("PIPE_STRATEGY_GATE_SCORE", "10"))
                except Exception:
                    gate = 10.0
                plan, errp = _explain_text_for(base_sql)
                base_plan_text = plan
                score = _score_explain(plan or "")
                steps.append({"name": "explain_base", "ms": 0, "output": {"score": score, "explain_error": errp is not None}})
                if score is not None and score <= gate:
                    candidates.append({"sql": base_sql, "why": "base_gated"})
                    base_gated = True
            # No deterministic SQL candidates; rely on model + fallbacks only
            if base_sql and not base_gated:
                candidates.append({"sql": base_sql, "why": "base"})
            # Strategy variants to diversify plans
            strategies = [
                "Вынеси предагрегацию до JOIN, если это возможно",
                "Переставь порядок JOIN так, чтобы сначала шли более селективные таблицы",
                "Максимально протолкни предикаты по partition колонкам перед JOIN",
            ]
            # Limit number of strategies via env and disable if base_gated
            try:
                smax = int(os.getenv("PIPE_STRATEGY_MAX", "2"))
            except Exception:
                smax = 2
            if base_gated:
                strategies = []
            else:
                strategies = strategies[: max(0, min(len(strategies), smax))]
            # If base EXPLAIN indicates issues, add plan-derived hint as one strategy
            if not base_gated and base_plan_text and len(strategies) < max(1, smax):
                try:
                    from ..langpipe.utils import analyze_explain_hints
                    hints = analyze_explain_hints(base_plan_text)
                    if hints:
                        strategies.append("Подсказки из плана: " + "; ".join(hints))
                except Exception:
                    pass
            for s in strategies:
                sp = (state.get("current_prompt") or "") + "\nСтратегия: " + s
                ms_g2, gen2 = _gen_sql(prompt_override=sp)
                sql2 = (gen2.get("sql_query") or "").strip()
                # Optional normalization for strategy outputs
                try:
                    from ..langpipe.utils import maybe_sqlglot_format
                    _norm2 = maybe_sqlglot_format(sql2, dialect=state.get("dialect", "trino"))
                    if _norm2:
                        sql2 = _norm2
                except Exception:
                    pass
                if sql2 and all(sql2 != c["sql"] for c in candidates):
                    candidates.append({"sql": sql2, "why": s})
            # Always include original user SQL as a baseline candidate to ensure metrics
            if user_sql and all(user_sql.strip() != c["sql"].strip() for c in candidates):
                candidates.append({"sql": user_sql, "why": "original"})
            steps.append({"name": "generate_candidates", "ms": 0, "output": {"count": len(candidates)}})
            if not candidates:
                # Mini-prompt fallback: попросим вернуть только валидный SQL без пояснений
                mini = ("Верни только валидный SQL под Trino. "
                        "Без пояснений, без кодовых блоков, без кавычек вокруг всего запроса. ")
                ms_g3, gen3 = _gen_sql(prompt_override=(state.get("current_prompt") or "") + "\n" + mini)
                sql3 = (gen3.get("sql_query") or "").strip()
                if sql3:
                    candidates = [{"sql": sql3, "why": "mini_prompt"}]
                if not candidates:
                    # Guard against empty SQL from model: fallback to original user SQL to avoid empty execution
                    state["sql_query"] = user_sql
                    candidates = [{"sql": state["sql_query"], "why": "fallback_original"}]
            # Filter out obviously invalid candidates to reduce wasted EXPLAIN calls
            filtered: List[Dict[str, Any]] = []
            for c in candidates:
                ok_c, _reasons_c = _validate_sql_plausible(c["sql"])  # quick plausibility
                if ok_c:
                    filtered.append(c)
            if not filtered:
                # Ensure at least original user SQL present
                filtered = [{"sql": user_sql, "why": "original"}]
            candidates = filtered
            # Rank candidates by EXPLAIN (memoized per-run)
            _explain_cache: Dict[str, Dict[str, Any]] = {}
            scored = []
            for c in candidates:
                sql_key = c["sql"].strip()
                plan = None
                errp = None
                if isinstance(executor, TrinoExecutorModel):
                    cached = _explain_cache.get(sql_key)
                    if cached is not None:
                        plan = cached.get("plan")
                        errp = cached.get("err")
                        score = cached.get("score")
                    else:
                        plan, errp = _explain_text_for(sql_key)
                        score = _score_explain(plan or "")
                        _explain_cache[sql_key] = {"plan": plan, "err": errp, "score": score}
                else:
                    score = _score_explain("")
                scored.append({"sql": c["sql"], "why": c["why"], "score": score, "explain_error": errp is not None})
            scored.sort(key=lambda x: x["score"])
            steps.append({"name": "explain_rank", "ms": 0, "output": scored[:3]})
            # Micro-probing: execute top-K candidates and pick the fastest successful
            probe_top = 1
            try:
                probe_top = max(1, int(os.getenv("PIPE_PROBE_TOP", "2")))
            except Exception:
                probe_top = 2
            to_probe = scored[: min(len(scored), probe_top)] if scored else []
            probe_results: List[Dict[str, Any]] = []
            best_idx = -1
            best_ms: Optional[int] = None
            best_res: Optional[Dict[str, Any]] = None
            best_err: Optional[str] = None
            for idx, cand in enumerate(to_probe):
                sql_c = cand["sql"]
                t1 = time.monotonic()
                res_c, err_c = executor.run(sql_c, schema_subset_for_sql(sql_c, state.get("db_schema")) or state.get("db_schema"))
                ms_client = int((time.monotonic() - t1) * 1000)
                tri_ms = None
                if res_c and isinstance(res_c, dict):
                    tri = (res_c.get("trino") or {})
                    if tri.get("cpu_ms") is not None:
                        tri_ms = int(tri.get("cpu_ms"))
                    elif tri.get("execution_ms") is not None:
                        tri_ms = int(tri.get("execution_ms"))
                    elif tri.get("elapsed_ms") is not None:
                        tri_ms = int(tri.get("elapsed_ms"))
                eff_ms = tri_ms if isinstance(tri_ms, int) else ms_client
                probe_results.append({
                    "sql_preview": sql_c[:120],
                    "ok": bool(res_c),
                    "error": err_c,
                    "ms": eff_ms,
                })
                if not err_c:
                    if best_ms is None or eff_ms < best_ms:
                        best_ms = eff_ms
                        best_idx = idx
                        best_res = res_c if isinstance(res_c, dict) else None
                        best_err = None
            steps.append({"name": "probe_execute", "ms": 0, "output": probe_results})
            # Choose winner or fallback to first candidate result
            if best_idx >= 0:
                chosen_sql = to_probe[best_idx]["sql"]
                state["sql_query"] = chosen_sql
                res = best_res
                err = best_err
                ms_ex = best_ms or 0
                trino_ms = best_ms
            else:
                # Fallback: execute best by explain score if not already executed
                chosen_sql = (scored[0]["sql"] if scored else candidates[0]["sql"]) if candidates else user_sql
                state["sql_query"] = chosen_sql
                t1 = time.monotonic()
                res, err = executor.run(state["sql_query"], schema_subset_for_sql(state.get("sql_query", ""), state.get("db_schema")) or state.get("db_schema"))
                ms_ex = int((time.monotonic() - t1) * 1000)
                trino_ms = None
                if res and isinstance(res, dict):
                    tri = (res.get("trino") or {})
                    if tri.get("cpu_ms") is not None:
                        trino_ms = tri.get("cpu_ms")
                    elif tri.get("execution_ms") is not None:
                        trino_ms = tri.get("execution_ms")
                    else:
                        trino_ms = tri.get("elapsed_ms")
            steps.append({
                "name": "sql_executor",
                "ms": int(trino_ms) if isinstance(trino_ms, (int, float)) else ms_ex,
                "output": {
                    "error": err,
                    "ok": bool(res),
                    "rowcount": (res or {}).get("rowcount") if isinstance(res, dict) else None,
                    "columns": (res or {}).get("columns") if isinstance(res, dict) else None,
                    "trino": (res or {}).get("trino") if isinstance(res, dict) else None,
                    "client_ms": ms_ex,
                }
            })
            if not err:
                optimized_sql = state["sql_query"]
                # Quick equivalence check: compare columns + rowcount with original if it executed ok
                if executed_ok and user_sql:
                    try:
                        res_o, _ = executor.run(user_sql, schema_subset_for_sql(user_sql, state.get("db_schema")) or state.get("db_schema"))
                        eq = {
                            "same_cols": (res_o or {}).get("columns") == (res or {}).get("columns"),
                            "rowcount_equal": (res_o or {}).get("rowcount") == (res or {}).get("rowcount"),
                        }
                        steps.append({"name": "verify_equivalence", "ms": 0, "output": eq})
                    except Exception:
                        pass
                error = None
                break
            # Fix prompt
            t2 = time.monotonic()
            # Clip schema for fix prompt
            _schema_for_fix = clip_text(schema_subset_for_sql(state.get("sql_query", ""), state.get("db_schema")) or state.get("db_schema"), int(os.getenv("LLM_SCHEMA_MAX_CHARS", "4096")))
            fix = call_fix_error_prompt(
                llm=llm,
                user_query=_user_query_clipped,
                sql_query=state.get("sql_query", ""),
                error=err,
                db_schema=_schema_for_fix,
                chat_history=state.get("chat_history"),
            )
            if not (fix.get("fixed_prompt") or "").strip():
                logging.error("fix_error_prompt: empty fixed_prompt; raw_len=%s", len(fix.get("raw") or ""))
            steps.append({
                "name": "fix_error_prompt",
                "ms": int((time.monotonic() - t2) * 1000),
                "input": {
                    "error": err,
                    "schema_chars": len(state.get("db_schema") or ""),
                },
                "output": fix,
            })
            fixed_prompt = fix.get("fixed_prompt")
            attempt += 1
            error = err
    else:  # execute_sql branch
        state["sql_query"] = state.get("sql_query", user_sql)
        t1 = time.monotonic()
        res, err = executor.run(state["sql_query"], schema_subset_for_sql(state.get("sql_query", ""), state.get("db_schema")) or state.get("db_schema"))
        if err:
            logging.warning("sql_executor error: %s", err)
        ms_ex = int((time.monotonic() - t1) * 1000)
        trino_ms = None
        if res and isinstance(res, dict):
            tri = (res.get("trino") or {})
            if tri.get("cpu_ms") is not None:
                trino_ms = tri.get("cpu_ms")
            elif tri.get("execution_ms") is not None:
                trino_ms = tri.get("execution_ms")
            else:
                trino_ms = tri.get("elapsed_ms")
        steps.append({
            "name": "sql_executor",
            "ms": int(trino_ms) if isinstance(trino_ms, (int, float)) else ms_ex,
            "output": {
                "error": err,
                "ok": bool(res),
                "rowcount": (res or {}).get("rowcount") if isinstance(res, dict) else None,
                "columns": (res or {}).get("columns") if isinstance(res, dict) else None,
                "trino": (res or {}).get("trino") if isinstance(res, dict) else None,
                "client_ms": ms_ex,
            }
        })
        if not err:
            optimized_sql = state["sql_query"]
        else:
            # minimal fix loop
            t2 = time.monotonic()
            _schema_for_fix = clip_text(state.get("db_schema"), int(os.getenv("LLM_SCHEMA_MAX_CHARS", "4096")))
            fix = call_fix_error_prompt(
                llm=llm,
                user_query=_user_query_clipped,
                sql_query=state.get("sql_query", ""),
                error=err,
                db_schema=_schema_for_fix,
                chat_history=state.get("chat_history"),
            )
            if not (fix.get("fixed_prompt") or "").strip():
                logging.error("fix_error_prompt: empty fixed_prompt; raw_len=%s", len(fix.get("raw") or ""))
            steps.append({
                "name": "fix_error_prompt",
                "ms": int((time.monotonic() - t2) * 1000),
                "input": {
                    "error": err,
                    "schema_chars": len(state.get("db_schema") or ""),
                },
                "output": fix,
            })
            fixed_prompt = fix.get("fixed_prompt")
            ms_gen, gen = _gen_sql()
            steps.append({"name": "generate_sql", "ms": ms_gen, "output": gen})
            optimized_sql = gen.get("sql_query")

    return {"optimized_sql": optimized_sql or "", "steps": steps, "attempts": attempt, "error": error}
