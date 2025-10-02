from __future__ import annotations

from typing import Optional, Dict, Any

from .trino_exec import TrinoConfig


def introspect_schema(cfg: TrinoConfig) -> str:
    """
    Запрашивает список таблиц и колонок через information_schema и возвращает
    компактное текстовое описание схемы для LLM, формат:
      catalog.schema.table(col type, col type, ...)
    """
    import trino
    from trino.auth import BasicAuthentication

    if not cfg.catalog or not cfg.schema:
        # Без catalog/schema генерация FQN затруднительна — вернем пустую строку
        return ""

    conn = trino.dbapi.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        http_scheme=cfg.http_scheme,
        auth=BasicAuthentication(cfg.user, cfg.password) if cfg.password else None,
        catalog=cfg.catalog,
        schema=cfg.schema,
        verify=cfg.verify,
    )
    cur = conn.cursor()

    cur.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = ?
        ORDER BY table_name, ordinal_position
        """,
        params=[cfg.schema],
    )
    rows = cur.fetchall()
    if not rows:
        return ""

    tables: Dict[str, list[str]] = {}
    for table, col, dtype in rows:
        tables.setdefault(str(table), []).append(f"{col} {dtype}")

    lines = []
    for table in sorted(tables):
        cols = ", ".join(tables[table])
        lines.append(f"{cfg.catalog}.{cfg.schema}.{table}({cols})")
    return "\n".join(lines)
