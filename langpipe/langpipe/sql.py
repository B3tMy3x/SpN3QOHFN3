from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from .llm import LLM
from .json_utils import extract_json_best_effort


def _extract_json(s: str) -> Dict[str, Any]:
    return extract_json_best_effort(s)


class SQLExecutorModel:
    """
    LLM-based "SQL executor" that does not connect to a database.

    It validates SQL against a provided DB schema and returns either:
      - success: true, analysis + optional pseudo_result
      - success: false, error string suitable for prompt-fixing

    This is useful when actual DB access is not allowed; the model performs
    static checks (table/column presence, syntax plausibility) and simulates
    execution feedback for the fix-error loop.
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, sql: str, db_schema: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        system = (
            "You are a strict SQL execution simulator. Given a DB schema and an SQL query, "
            "determine if the query is valid and safe. Do NOT invent new schema. "
            "Return compact JSON only."
        )
        user = {
            "db_schema": db_schema or "",
            "sql": sql,
            "instructions": (
                "Return JSON with keys: success (bool), "
                "analysis (short string), error (string or null), pseudo_result (optional array). "
                "If a table/column does not exist, set success=false and specify error clearly. "
                "If valid but no DB to run, set success=true and provide analysis; pseudo_result may be omitted."
            ),
        }
        content = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ])
        # Best-effort JSON parse
        data = _extract_json(content)
        success = bool(data.get("success"))
        if success:
            return data, None
        # Error message for fix-error flow
        err = data.get("error") or data.get("analysis") or "Unknown SQL error"
        return None, str(err)
