from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

# Ensure repo root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)


def ensure_jdbc_has_catalog_schema(url: str) -> str:
    if "catalog=" in url and "schema=" in url:
        return url
    sep = '&' if '?' in url else '?'
    # Default to placeholders; require user to set env correctly
    return f"{url}{sep}catalog=default&schema=public"


def main() -> int:
    load_dotenv()
    # Respect existing env for model routing; do not force remote defaults here
    os.environ.setdefault("LLM_HTTP_TIMEOUT", os.getenv("LLM_HTTP_TIMEOUT", "120"))

    jdbc = os.getenv("TRINO_JDBC_URL") or os.getenv("HARDCODE_TRINO_URL")
    if not jdbc:
        print("TRINO_JDBC_URL is not set. Please provide a JDBC URL via env.")
        return 1
    jdbc = ensure_jdbc_has_catalog_schema(jdbc)

    from langpipe import build_graph, LLM, TrinoExecutorModel
    from langpipe.langpipe.trino_exec import parse_trino_jdbc_url
    cfg = parse_trino_jdbc_url(jdbc)

    llm = LLM()
    exec_model = TrinoExecutorModel(cfg, safe_mode=True)
    app = build_graph(llm, exec_model)

    # Run one of the problem queries
    user_sql = (
        "SELECT CASE WHEN sci.age < 25 THEN '18-24' WHEN sci.age < 35 THEN '25-34' WHEN sci.age < 45 THEN '35-44' ELSE '45+' END AS age_group, "
        "COUNT(ep.excursion_id) AS excursion_purchases, COUNT(qp.quest_id) AS quest_purchases "
        "FROM quests.public.s_client_personal_info sci "
        "JOIN quests.public.l_payment_client pc ON sci.client_id = pc.client_id "
        "LEFT JOIN quests.public.l_excursion_payment ep ON pc.payment_id = ep.payment_id "
        "LEFT JOIN quests.public.l_quest_payment qp ON pc.payment_id = qp.payment_id "
        "GROUP BY CASE WHEN sci.age < 25 THEN '18-24' WHEN sci.age < 35 THEN '25-34' WHEN sci.age < 45 THEN '35-44' ELSE '45+' END "
        "ORDER BY COUNT(ep.excursion_id) + COUNT(qp.quest_id) DESC;"
    )

    state: Dict[str, Any] = {
        "user_query": f"Оптимизируй и перепиши эквивалентно:\n{user_sql}",
        "chat_history": [],
        "dialect": "trino",
        "max_attempts": 3,
    }
    out = app.invoke(state)
    # Save full state for inspection
    os.makedirs("debug_out", exist_ok=True)
    with open("debug_out/graph_out.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved debug_out/graph_out.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
