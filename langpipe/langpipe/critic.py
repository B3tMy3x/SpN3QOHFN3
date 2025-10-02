from __future__ import annotations

import json
from typing import Any, Dict, Optional, List
import logging

from .llm import LLM
import os
import time
from .json_utils import extract_json_best_effort
from .utils import classify_trino_error


def _extract_json(s: str) -> Dict[str, Any]:
    return extract_json_best_effort(s)


def call_critic(
    llm: LLM,
    user_query: str,
    sql_query: Optional[str] = None,
    db_schema: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Ask LLM to critique the current state and choose a route.
    Returns dict with keys: route (enhance_prompt|execute_sql), rationale, issues.
    """
    system = (
        "You are a cautious SQL pipeline critic. Decide whether to improve the prompt "
        "or proceed to execute the SQL. Prefer enhance_prompt when the SQL looks incomplete or risky. "
        "Respond in compact JSON."
    )
    user = {
        "user_query": user_query,
        "sql_query": sql_query,
        "db_schema": db_schema,
        "chat_history": chat_history or [],
        "instructions": "Return JSON: route ('enhance_prompt'|'execute_sql'), rationale, issues[]",
        "signals": {
            "has_select": bool(sql_query and 'select' in (sql_query or '').lower()),
            "has_with": bool(sql_query and 'with' in (sql_query or '').lower()),
        },
        "guidance": (
            "If no SQL exists or it's likely incomplete/unsafe, choose 'enhance_prompt'. "
            "If SQL looks valid and safe for the described task, choose 'execute_sql'."
        ),
    }

    content = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        force_json=True,
    )
    # Optional dump for debugging model I/O
    try:
        if str(os.getenv("LLM_DUMP", "false")).lower() in ("1","true","yes","on"):
            os.makedirs("debug_out/llm", exist_ok=True)
            ts = int(time.time()*1000)
            with open(f"debug_out/llm/critic_{ts}.txt", "w", encoding="utf-8") as f:
                f.write("-- critic request --\n")
                f.write(json.dumps(user, ensure_ascii=False)[:2000])
                f.write("\n-- critic response --\n")
                f.write(content)
    except Exception:
        pass
    parsed = _extract_json(content)
    if not parsed:
        logging.error("llm_critic: empty/invalid JSON; content_len=%s", len(content or ""))
        # Heuristic fallback decision
        import re
        sql_text = (sql_query or user_query or "").lower()
        has_sql = bool(re.search(r"\b(select|with)\b", sql_text))
        parsed = {"route": ("execute_sql" if has_sql else "enhance_prompt"), "rationale": "fallback", "issues": []}
    route = parsed.get("route")
    if route not in ("enhance_prompt", "execute_sql"):
        # Heuristic fallback
        route = "enhance_prompt" if not (sql_query and sql_query.strip()) else "execute_sql"
    return {
        "route": route,
        "rationale": parsed.get("rationale", ""),
        "issues": parsed.get("issues", []),
        "raw": content,
    }


def call_enhance_prompt(
    llm: LLM,
    user_query: str,
    critique: Optional[str] = None,
    issues: Optional[list] = None,
    db_schema: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generate an improved instruction prompt for SQL generation.
    Returns dict with keys: enhanced_prompt, notes.
    """
    system = (
        "You help craft concise, precise prompts that guide an SQL generator. "
        "Produce a short, actionable prompt. Return JSON only."
    )
    user_payload = {
        "user_query": user_query,
        "db_schema": db_schema,
        "chat_history": chat_history or [],
        "critique": critique,
        "issues": issues or [],
        "instructions": (
            "Return JSON with: enhanced_prompt (string), notes (array of strings). "
            "The enhanced_prompt should include constraints and schema hints when useful."
        ),
    }

    content = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        force_json=True,
    )
    try:
        if str(os.getenv("LLM_DUMP", "false")).lower() in ("1","true","yes","on"):
            os.makedirs("debug_out/llm", exist_ok=True)
            ts = int(time.time()*1000)
            with open(f"debug_out/llm/enhance_{ts}.txt", "w", encoding="utf-8") as f:
                f.write("-- enhance request --\n")
                f.write(json.dumps(user_payload, ensure_ascii=False)[:2000])
                f.write("\n-- enhance response --\n")
                f.write(content)
    except Exception:
        pass
    parsed = _extract_json(content)
    if not parsed.get("enhanced_prompt"):
        logging.error("enhance_prompt: missing enhanced_prompt; content_len=%s", len(content or ""))
        # Minimal fallback: pass-through user_query as guidance
        fallback_prompt = (user_query or "").strip() or "Оптимизируй эквивалентно исходный SQL."
        parsed = {"enhanced_prompt": fallback_prompt, "notes": ["fallback"]}
    return {
        "enhanced_prompt": parsed.get("enhanced_prompt", ""),
        "notes": parsed.get("notes", []),
        "raw": content,
    }


def call_fix_error_prompt(
    llm: LLM,
    user_query: str,
    sql_query: Optional[str],
    error: str,
    db_schema: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Given an execution error, produce a revised prompt that helps the SQL generator
    fix the query. Returns dict with keys: fixed_prompt, reasoning.
    """
    system = (
        "You repair SQL-generation prompts using execution errors. "
        "Return JSON only."
    )
    user_payload = {
        "user_query": user_query,
        "db_schema": db_schema,
        "chat_history": chat_history or [],
        "sql_query": sql_query,
        "error": error,
        "instructions": (
            "Return JSON with: fixed_prompt (string), reasoning (short string). "
            "Keep fixed_prompt focused on addressing the error and schema constraints."
        ),
    }
    content = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        force_json=True,
    )
    try:
        if str(os.getenv("LLM_DUMP", "false")).lower() in ("1","true","yes","on"):
            os.makedirs("debug_out/llm", exist_ok=True)
            ts = int(time.time()*1000)
            with open(f"debug_out/llm/fix_{ts}.txt", "w", encoding="utf-8") as f:
                f.write("-- fix_error request --\n")
                f.write(json.dumps(user_payload, ensure_ascii=False)[:2000])
                f.write("\n-- fix_error response --\n")
                f.write(content)
    except Exception:
        pass
    parsed = _extract_json(content)
    if not parsed.get("fixed_prompt"):
        logging.error("fix_error_prompt: missing fixed_prompt; content_len=%s", len(content or ""))
        # Fallback: construct a stricter repair prompt for the SQL generator
        etype, hints = classify_trino_error(error)
        tips = "; ".join(hints) if hints else ""
        minimal = (
            "Исправь и оптимизируй запрос под Trino, сохранив семантику. "
            f"Ошибка выполнения: {error}. Тип: {etype}. "
            + (f"Подсказки: {tips}. " if tips else "")
            + "Ограничения: используй только таблицы/колонки, которые присутствуют в db_schema, строго теми же именами; "
            + "не оставляй висячих запятых; не добавляй обратных слэшей; верни один корректный SELECT/WITH; "
            + "не меняй типы JOIN; избегай DISTINCT без необходимости; добавь LIMIT, если он уместен. "
            + "Цель оптимизации: минимизируй Exchanges/Repartition, избегай повторных TableScan, не добавляй Sort/TopN у корня без запроса, стремись уменьшить processedBytes/processedRows."
        )
        parsed = {"fixed_prompt": minimal, "reasoning": "fallback"}
    return {
        "fixed_prompt": parsed.get("fixed_prompt", ""),
        "reasoning": parsed.get("reasoning", ""),
        "raw": content,
    }
