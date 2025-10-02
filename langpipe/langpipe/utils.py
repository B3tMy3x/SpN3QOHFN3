from __future__ import annotations

import re
from typing import Optional


_tbl_pat = re.compile(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b")


def schema_subset_for_sql(sql: str, full_schema_text: Optional[str]) -> Optional[str]:
    """
    Return a minimal text subset of the provided schema that includes only
    the catalog.schema.table entries referenced in the SQL.
    Falls back to the original text if extraction fails.
    """
    if not full_schema_text:
        return full_schema_text
    wanted = set(m.group(0) for m in _tbl_pat.finditer(sql or ""))
    if not wanted:
        return full_schema_text
    lines = full_schema_text.splitlines()
    keep: list[str] = []
    for ln in lines:
        for w in wanted:
            if w in ln:
                keep.append(ln)
                break
    # If we trimmed too much, keep original as fallback
    return "\n".join(keep) if keep else full_schema_text


def clip_text(text: Optional[str], max_chars: int = 4096) -> Optional[str]:
    """Return text limited to max_chars (prefix). None passes through.
    Keeps behavior simple and explicit to avoid surprising truncation in callers.
    """
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars]


def extract_fqn_tables(sql: str) -> list[str]:
    """Return list of catalog.schema.table mentioned in SQL (best-effort)."""
    if not sql:
        return []
    return sorted({m.group(0) for m in _tbl_pat.finditer(sql)})


def classify_trino_error(err: Optional[str]) -> tuple[str, list[str]]:
    """
    Classify Trino error string into a coarse type and return targeted hints.
    Returns: (type, hints)
    """
    e = (err or "").lower()
    if not e:
        return "unknown", []
    hints: list[str] = []
    if "column_not_found" in e or "column" in e and "cannot be resolved" in e:
        hints.append("Проверь имена колонок по db_schema; используй точные имена без новых полей")
        return "column_not_found", hints
    if "table" in e and ("not found" in e or "does not exist" in e):
        hints.append("Проверь имена таблиц и FQN (catalog.schema.table)")
        return "table_not_found", hints
    if "type_mismatch" in e or "must evaluate to a boolean" in e or "cannot apply operator" in e:
        hints.append("Приведи типы явно (CAST), убедись что предикаты имеют boolean тип")
        return "type_mismatch", hints
    if "syntax_error" in e or "mismatched input" in e:
        hints.append("Верни полный корректный SQL: без висячих запятых, без обратных слэшей, закрывай скобки")
        return "syntax_error", hints
    if "time" in e and ("out" in e or "limit" in e):
        hints.append("Добавь LIMIT и предагрегацию до JOIN для снижения объёма")
        return "timeout", hints
    if "memory" in e:
        hints.append("Снизь скан: предагрегируй, протолкни предикаты, избегай сортировок у корня")
        return "oom", hints
    return "unknown", []


def maybe_sqlglot_format(sql: Optional[str], dialect: str = "trino") -> Optional[str]:
    """
    Optionally format/normalize SQL via sqlglot if available and enabled.
    Controlled by env SQLGLOT_ENABLE (default: true). Returns input on failure.
    """
    if not sql:
        return sql
    import os
    if str(os.getenv("SQLGLOT_ENABLE", "true")).lower() not in ("1", "true", "yes", "on"):
        return sql
    try:
        import sqlglot  # type: ignore
        from sqlglot import exp
        # Parse with best-effort; transpile to target dialect
        tree = sqlglot.parse_one(sql, read=dialect)
        # Remove unnecessary parentheses/format
        formatted = tree.sql(dialect=dialect, pretty=True)
        # Ensure trailing semicolons removed
        return formatted.rstrip().rstrip(";")
    except Exception:
        return sql


def analyze_explain_hints(plan_text: Optional[str]) -> list[str]:
    """Generate human-readable hints from EXPLAIN plan text."""
    if not plan_text:
        return []
    t = plan_text.lower()
    hints: list[str] = []
    if t.count("exchange") >= 2 or "repartition" in t:
        hints.append("Снизь количество Exchanges/Repartition: предагрегируй до JOIN, упрощай пайплайн JOIN")
    if t.count("tablescan[") >= 2:
        hints.append("Избегай повторных TableScan: вынеси общие источники в CTE и переиспользуй результат")
    if "dynamic filter" not in t and "dynamicfilter" not in t:
        hints.append("Обеспечь селективные фильтры/условия JOIN для динамической фильтрации")
    if "topn[" in plan_text or "sort[" in plan_text:
        hints.append("Не добавляй глобальную сортировку без явной необходимости")
    return hints
