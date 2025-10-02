from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import logging

from .llm import LLM
from .json_utils import extract_json_best_effort, extract_sql_from_text, sanitize_sql_string
from .utils import maybe_sqlglot_format, clip_text
import os
import time


def _extract_json(s: str) -> Dict[str, Any]:
    return extract_json_best_effort(s)


def call_generate_sql(
    llm: LLM,
    user_query: str,
    current_prompt: Optional[str],
    db_schema: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    dialect: str = "trino",
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Генерация SQL на основе текущего промпта/запроса пользователя и схемы БД.
    Возвращает: { sql_query, reasoning }
    """
    target = "Trino" if dialect.lower() == "trino" else "SQL"
    qual = (
        f"Старайся использовать полные имена таблиц {catalog}.{schema}.table, если это уместно."
        if catalog and schema else "Старайся корректно квалифицировать имена таблиц."
    )
    # Мягкие предпочтения и формат ответа
    system = (
        f"Ты генератор SQL для {target}. Твоя задача — вернуть эквивалентный, по возможности более эффективный запрос. "
        f"{qual} Предпочитай date_trunc и FILTER (WHERE ...) вместо CASE в агрегатах, если это эквивалентно. "
        f"Избегай DISTINCT без необходимости. По возможности добавь LIMIT, если он отсутствует. "
        f"Синтаксис должен быть корректным: не оставляй висячих запятых, не вставляй обратные слэши, завершай предложения полностью. "
        f"Используй только существующие таблицы/колонки строго по именам из db_schema. "
        f"Цель оптимизации: минимизируй количество Exchanges/Repartition, избегай повторных TableScan для одних и тех же таблиц, не добавляй Sort/TopN у корня без явного запроса, стремись уменьшить processedBytes/processedRows. "
        f"Верни только JSON с ключами: sql_query, reasoning."
    )
    # Clip inputs to reduce empty/trimmed responses on remote backends
    import os as _os
    max_schema = int(_os.getenv("LLM_SCHEMA_MAX_CHARS", "2000"))
    max_prompt = int(_os.getenv("LLM_PROMPT_MAX_CHARS", "800"))
    clipped_schema = clip_text(db_schema, max_schema)
    clipped_prompt = clip_text(current_prompt, max_prompt)

    user_payload = {
        "user_query": user_query,
        "db_schema": clipped_schema,
        "chat_history": chat_history or [],
        "prompt": clipped_prompt,
        "constraints": [
            "Сохраняй семантику и порядок столбцов результата",
            "Если возможно, не меняй имена алиасов",
            "Не добавляй ORDER BY, если явно не просили",
            "Для Trino: не группируй по алиасу столбца из SELECT; повторяй выражение или вынеси вычисление в подзапрос и группируй по нему",
            "Старайся объединять однотипные источники через CTE с UNION ALL и затем делать один проход JOIN (например, платежи из нескольких таблиц → один JOIN к клиенту/гео)",
            "По возможности выполняй предагрегацию до JOIN, чтобы уменьшить объём данных",
            "Проталкивай селективные предикаты по partition/дате как можно раньше",
            "Минимизируй количество обменов/переразбиений (exchanges/repartitions)",
            "Предпочитай UNION ALL вместо UNION, если дубликаты допустимы",
        ],
        "prohibitions": [
            "Не меняй типы JOIN (особенно LEFT/RIGHT/OUTER) и не переносите предикаты через внешние соединения",
            "Сохраняй поведение по NULL (не добавляй COALESCE/IS NOT NULL без необходимости)",
            "Не удаляй/не упрощай оконные функции, если они определяют выборку/сэмплирование (например, ROW_NUMBER() OVER (...) с фильтрацией по rn)",
            "Избегай DISTINCT без необходимости",
        ],
        "instructions": "Верни только JSON с sql_query и коротким reasoning",
    }
    # Strict JSON contract
    content = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        force_json=True,
    )
    # Optional dump for debugging
    try:
        if str(os.getenv("LLM_DUMP", "false")).lower() in ("1","true","yes","on"):
            os.makedirs("debug_out/llm", exist_ok=True)
            ts = int(time.time()*1000)
            with open(f"debug_out/llm/generate_{ts}.txt", "w", encoding="utf-8") as f:
                f.write("-- generate request --\n")
                f.write(json.dumps(user_payload, ensure_ascii=False)[:3000])
                f.write("\n-- generate response --\n")
                f.write(content)
    except Exception:
        pass
    data = _extract_json(content)
    sql = data.get("sql_query", "") if isinstance(data, dict) else ""
    reasoning = data.get("reasoning", "") if isinstance(data, dict) else ""

    # Fallback: try to salvage SQL from free-form text
    if not sql:
        extracted = extract_sql_from_text(content)
        if extracted and extracted.lower().startswith(("select", "with")):
            sql = extracted
            if not reasoning:
                reasoning = "extracted SQL from free-form response"
        else:
            # One internal retry with stricter, shorter prompt and without forced JSON
            retry_payload = {
                "user_query": user_query,
                "db_schema": (db_schema or "")[:2000],
                "prompt": (current_prompt or "")[:800],
                "instructions": "Верни только JSON {\"sql_query\": str, \"reasoning\": str} или, если не можешь, один корректный SQL-блок без текста.",
            }
            retry_system = (
                f"Ты генератор SQL для {target}. Отвечай строго JSON или одним SQL. Никакого другого текста."
            )
            try:
                content2 = llm.chat(
                    [
                        {"role": "system", "content": retry_system},
                        {"role": "user", "content": json.dumps(retry_payload, ensure_ascii=False)},
                    ],
                    force_json=False,
                )
                data2 = _extract_json(content2)
                if isinstance(data2, dict):
                    sql = data2.get("sql_query") or ""
                    reasoning = data2.get("reasoning") or reasoning
                if not sql:
                    extracted2 = extract_sql_from_text(content2)
                    if extracted2 and extracted2.lower().startswith(("select", "with")):
                        sql = extracted2
                        if not reasoning:
                            reasoning = "extracted SQL from retry response"
                # Final fallback: ask for a single valid SQL only, no JSON, no prose
                if not sql:
                    fb_system = (
                        f"Ты генератор SQL для {target}. Верни ОДИН корректный запрос SELECT/WITH без пояснений, без кодовых блоков, без кавычек вокруг всего запроса."
                    )
                    fb_user = (
                        (current_prompt or user_query or "").strip()
                        + "\n\nЕсли исходный запрос валиден — верни эквивалентный. Иначе перепиши корректно."
                    )
                    content3 = llm.chat(
                        [
                            {"role": "system", "content": fb_system},
                            {"role": "user", "content": fb_user},
                        ],
                        force_json=False,
                    )
                    extracted3 = extract_sql_from_text(content3)
                    if extracted3 and extracted3.lower().startswith(("select", "with")):
                        sql = extracted3
                        if not reasoning:
                            reasoning = "fallback single-SQL response"
                if not sql:
                    logging.error(
                        "generate_sql: empty sql_query after retry; content_len=%s prompt_chars=%s schema_chars=%s",
                        len((content2 or "")), len(current_prompt or ""), len(db_schema or ""),
                    )
            except Exception as e:
                logging.error("generate_sql retry failed: %s", e)

    # (no second attempt by design — we want to surface issues)

    # Санитайз артефактов (экранированные переводы строк, хвосты кодовых блоков и т.д.)
    sql = sanitize_sql_string(sql or "")
    # Попробовать форматировать/нормализовать SQL через sqlglot (опционально)
    sql_fmt = maybe_sqlglot_format(sql, dialect=(dialect or "trino"))
    if sql_fmt:
        sql = sql_fmt
    return {"sql_query": sql or "", "reasoning": reasoning, "raw": content}
