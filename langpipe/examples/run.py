from __future__ import annotations

import os

from dotenv import load_dotenv
try:
    from langpipe import build_graph, LLM, SQLExecutorModel, TrinoExecutorModel, TrinoConfig
    from langpipe.langpipe.trino_exec import parse_trino_jdbc_url
except Exception:
    # Fallback when running from different CWDs
    import sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.append(_root)
    from langpipe import build_graph, LLM, SQLExecutorModel, TrinoExecutorModel, TrinoConfig
    from langpipe.langpipe.trino_exec import parse_trino_jdbc_url


def main():
    # Load .env if present
    load_dotenv()
    # Configure local LLM. Ensure the model is available under kaggle/input/deepseek-coder
    # or pass model_dir to LLM(...).
    llm = LLM(model_dir=os.getenv("LOCAL_MODEL_DIR", "kaggle/input/deepseek-coder"))
    # Выбор исполнителя: Trino (если передан JDBC URL) или симулятор
    jdbc_url = os.getenv("TRINO_JDBC_URL")
    if not jdbc_url:
        # Пример: можно передать URL напрямую, если хотите жестко прописать:
        jdbc_url = os.getenv("HARDCODE_TRINO_URL")

    if jdbc_url:
        cfg = parse_trino_jdbc_url(jdbc_url)
        # Allow override of catalog/schema via env
        catalog = os.getenv("TRINO_CATALOG") or cfg.catalog
        schema = os.getenv("TRINO_SCHEMA") or cfg.schema
        cfg.catalog = catalog
        cfg.schema = schema
        exec_model = TrinoExecutorModel(cfg)
        print("Using Trino executor:", cfg)
    else:
        exec_model = SQLExecutorModel(llm)
        print("Using LLM SQL simulator executor")

    # Build graph
    app = build_graph(llm, exec_model)

    # Пример 1: нет SQL -> критик пошлёт на enhance → generate → execute (Trino)
    state = {
        "user_query": "Покажи 3 последних рейса из BWI в ORD по дате",
        # db_schema будет заполнена автоматически из Trino, если доступно
        "chat_history": [
            {"role": "user", "content": "Нужно посмотреть недавние перелёты BWI → ORD"},
            {"role": "assistant", "content": "Окей, сортируем по дате? Сколько строк?"},
            {"role": "user", "content": "Да, последние 3 по дате"},
        ],
        # Диалект можно задать через .env (SQL_DIALECT)
        "dialect": os.getenv("SQL_DIALECT", "trino" if isinstance(exec_model, TrinoExecutorModel) else "postgres"),
    }
    print("\n=== Run 1: critic → enhance_prompt → generate_sql → sql_executor ===")
    # Добавим лимит попыток
    state["max_attempts"] = 3
    out = app.invoke(state)
    print(out)

    # Пример 2: намеренно ошибочный SQL → executor error → fix_error_prompt → регенерация
    state2 = {
        "user_query": "Покажи 3 последних рейса из BWI в ORD по дате",
        # db_schema автозаполнится при наличии Trino
        "chat_history": [
            {"role": "user", "content": "Нужно посмотреть недавние перелёты BWI → ORD"},
            {"role": "assistant", "content": "Окей, сортируем по дате? Сколько строк?"},
            {"role": "user", "content": "Да, последние 3 по дате"},
        ],
        # Ошибка: неверное имя колонки 'depdelayminute' (правильно depdelayminutes)
        "sql_query": (
            "SELECT flightdate, airline, origin, dest, depdelayminute "
            "FROM flights.public.flights "
            "WHERE origin='BWI' AND dest='ORD' "
            "ORDER BY flightdate DESC LIMIT 3;"
        ),
        "dialect": os.getenv("SQL_DIALECT", "trino" if isinstance(exec_model, TrinoExecutorModel) else "postgres"),
    }
    print("\n=== Run 2: critic → sql_executor → fix_error_prompt → generate_sql (цикл) ===")
    state2["max_attempts"] = 3
    out2 = app.invoke(state2)
    print(out2)


if __name__ == "__main__":
    main()
