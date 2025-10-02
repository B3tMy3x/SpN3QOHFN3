from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _has_limit(sql: str) -> bool:
    # very rough check for LIMIT in the last clause
    return bool(re.search(r"\blimit\b\s+\d+\s*$", sql.strip(), flags=re.IGNORECASE))


def _is_read_only(sql: str) -> bool:
    s = sql.strip().lstrip("(")
    # allow WITH ... SELECT ...
    if re.match(r"^(with|select|explain|show|describe)\b", s, flags=re.IGNORECASE):
        # Reject common DDL/DML keywords anywhere to be safe
        banned = r"\b(insert|update|delete|merge|truncate|create|alter|drop|grant|revoke|vacuum|analyze)\b"
        return re.search(banned, s, flags=re.IGNORECASE) is None
    return False


def _strip_trailing_semicolons(s: str) -> str:
    i = len(s) - 1
    while i >= 0 and s[i].isspace():
        i -= 1
    while i >= 0 and s[i] == ';':
        i -= 1
        while i >= 0 and s[i].isspace():
            i -= 1
    return s[: i + 1]


def _first_unquoted_semicolon_index(s: str) -> int:
    """Return True if there's a semicolon outside quotes/comments.
    Handles: single quotes ('..', with '' escapes), double quotes (".."),
    line comments (-- ..\n), block comments (/* .. */).
    """
    in_squote = False
    in_dquote = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        nxt = s[i + 1] if i + 1 < n else ''

        if in_line_comment:
            if ch == '\n':
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == '*' and nxt == '/':
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_squote:
            if ch == "'":
                # SQL escape: '' inside string
                if nxt == "'":
                    i += 2
                else:
                    in_squote = False
                    i += 1
            else:
                i += 1
            continue
        if in_dquote:
            if ch == '"':
                in_dquote = False
            i += 1
            continue

        # Not in any quoted/comment state
        if ch == '-' and nxt == '-':
            in_line_comment = True
            i += 2
            continue
        if ch == '/' and nxt == '*':
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_squote = True
            i += 1
            continue
        if ch == '"':
            in_dquote = True
            i += 1
            continue
        if ch == ';':
            return i
        i += 1
    return -1


def _ensure_safe_sql(sql: str, default_limit: int = 1000, safe_mode: bool = True) -> Tuple[str, Optional[str]]:
    if safe_mode and not _is_read_only(sql):
        return sql, "Only read-only SELECT/WITH queries are allowed"
    s = sql.strip()
    # Remove trailing semicolons/spaces
    s = _strip_trailing_semicolons(s)
    # If multiple statements present, trim to the first statement only (safer than erroring)
    if safe_mode:
        idx = _first_unquoted_semicolon_index(s)
        if idx != -1:
            s = s[:idx]
    # Append LIMIT if safe_mode enabled and absent, only for SELECT/WITH (not EXPLAIN/SHOW/DESCRIBE)
    if safe_mode:
        starts = s.strip().lower()
        if re.match(r"^(select|with)\b", starts) and not _has_limit(s):
            s = f"{s} LIMIT {default_limit}"
    return s, None


@dataclass
class TrinoConfig:
    host: str
    port: int = 443
    user: str = "anonymous"
    password: Optional[str] = None
    http_scheme: str = "https"
    catalog: Optional[str] = None
    schema: Optional[str] = None
    verify: bool = True
    # request_timeout не используется напрямую для совместимости с разными версиями клиента
    request_timeout: Optional[float] = None


def parse_trino_jdbc_url(jdbc_url: str) -> TrinoConfig:
    # Example: jdbc:trino://host:443?user=u&password=p
    assert jdbc_url.startswith("jdbc:trino://"), "Unsupported JDBC URL"
    raw = jdbc_url[len("jdbc:"):]
    parsed = urllib.parse.urlparse(raw)
    host = parsed.hostname or "localhost"
    port = parsed.port or 443
    qs = urllib.parse.parse_qs(parsed.query)
    user = (qs.get("user") or ["anonymous"])[0]
    password = (qs.get("password") or [None])[0]
    catalog = (qs.get("catalog") or [None])[0]
    schema = (qs.get("schema") or [None])[0]
    # http scheme overrides via query: http_scheme=http|https|httpScheme
    http_scheme = (
        (qs.get("http_scheme") or qs.get("httpScheme") or qs.get("scheme") or [None])[0]
        or "https"
    )
    http_scheme = http_scheme.lower()
    if http_scheme not in ("http", "https"):
        http_scheme = "https"
    # SSL verify flag (optional): verify=false or insecure=true
    verify_param = (qs.get("verify") or [None])[0]
    insecure = (qs.get("insecure") or ["false"])[0]
    verify = True
    if isinstance(verify_param, str):
        verify = verify_param.lower() not in ("0", "false", "no")
    if isinstance(insecure, str) and insecure.lower() in ("1", "true", "yes"):
        verify = False
    # Allow env fallback for catalog/schema if not provided in JDBC
    import os as _os
    if not catalog:
        env_cat = _os.getenv("TRINO_CATALOG")
        if env_cat:
            catalog = env_cat
    if not schema:
        env_sch = _os.getenv("TRINO_SCHEMA")
        if env_sch:
            schema = env_sch

    return TrinoConfig(
        host=host,
        port=port,
        user=user,
        password=password,
        catalog=catalog,
        schema=schema,
        http_scheme=http_scheme,
        verify=verify,
    )


class TrinoExecutorModel:
    """Исполнитель запросов для Trino с базовой безопасностью (только SELECT)."""

    def __init__(self, cfg: TrinoConfig, safe_mode: bool = True) -> None:
        self.cfg = cfg
        self.safe_mode = safe_mode

    def run(self, sql: str, _db_schema: Optional[str] = None, allow_unsafe: Optional[bool] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        safe = self.safe_mode if allow_unsafe is None else not allow_unsafe
        safe_sql, err = _ensure_safe_sql(sql, safe_mode=safe)
        if err:
            return None, err
        try:
            import trino
            from trino.auth import BasicAuthentication

            session_properties = {
                # Safe defaults that can help planner; strings per Trino expectations
                "enable_dynamic_filtering": "true",
                "join_distribution_type": "AUTOMATIC",
            }
            # Allow env overrides
            import os
            jdt = os.getenv("TRINO_JOIN_DISTRIBUTION")
            if jdt:
                session_properties["join_distribution_type"] = jdt
            edf = os.getenv("TRINO_ENABLE_DF")
            if edf is not None:
                session_properties["enable_dynamic_filtering"] = "true" if str(edf).lower() in ("1","true","yes","on") else "false"

            conn = trino.dbapi.connect(
                host=self.cfg.host,
                port=self.cfg.port,
                user=self.cfg.user,
                http_scheme=self.cfg.http_scheme,
                auth=BasicAuthentication(self.cfg.user, self.cfg.password) if self.cfg.password else None,
                catalog=self.cfg.catalog,
                schema=self.cfg.schema,
                verify=self.cfg.verify,
                session_properties=session_properties,
            )
            cur = conn.cursor()
            cur.execute(safe_sql)
            rows = cur.fetchall()
            # columns metadata
            cols = [d[0] for d in cur.description] if cur.description else []
            # Try to extract Trino stats for precise timings
            stats: Dict[str, Any] = {}
            try:
                raw_stats = getattr(cur, "stats", {}) or {}
                qid = getattr(cur, "query_id", None) or raw_stats.get("queryId")
                # Normalize millis if available; fall back to string durations
                def _to_ms(v: Any) -> Optional[int]:
                    if v is None:
                        return None
                    if isinstance(v, (int, float)):
                        # assume already in ms if fairly large, or seconds if small
                        # but Trino exposes *Millis, so treat as ms
                        return int(v)
                    if isinstance(v, str):
                        # Parse formats like '1.23s', '123ms', '00:01:02.345'
                        import re
                        m = re.match(r"^(\d+)(?:ms)$", v)
                        if m:
                            return int(m.group(1))
                        m = re.match(r"^(\d+(?:\.\d+)?)s$", v)
                        if m:
                            return int(float(m.group(1)) * 1000)
                        # HH:MM:SS.mmm
                        m = re.match(r"^(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$", v)
                        if m:
                            hh, mm, ss, ms = m.groups()
                            base = (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000
                            extra = int(ms.ljust(3, '0')) if ms else 0
                            return base + extra
                    return None
                elapsed_ms = _to_ms(raw_stats.get("elapsedTimeMillis") or raw_stats.get("elapsedTime"))
                execution_ms = _to_ms(raw_stats.get("executionTimeMillis") or raw_stats.get("executionTime"))
                cpu_ms = _to_ms(raw_stats.get("cpuTimeMillis") or raw_stats.get("cpuTime"))
                queued_ms = _to_ms(raw_stats.get("queuedTimeMillis") or raw_stats.get("queuedTime"))
                scheduled_ms = _to_ms(raw_stats.get("scheduledTimeMillis") or raw_stats.get("scheduledTime"))
                processed_bytes = raw_stats.get("processedBytes") or raw_stats.get("processed_bytes")
                processed_rows = raw_stats.get("processedRows") or raw_stats.get("processed_rows")
                stats = {
                    "query_id": qid,
                    "elapsed_ms": elapsed_ms,
                    "execution_ms": execution_ms,
                    "cpu_ms": cpu_ms,
                    "queued_ms": queued_ms,
                    "scheduled_ms": scheduled_ms,
                    "processed_bytes": processed_bytes,
                    "processed_rows": processed_rows,
                    "raw": raw_stats,
                }
            except Exception:
                # Be resilient if client lib changes; stats remain optional
                stats = {}
            return {
                "success": True,
                "columns": cols,
                "rows": rows,
                "rowcount": len(rows),
                "query": safe_sql,
                "trino": stats,
            }, None
        except Exception as e:
            return None, str(e)

    async def run_async(self, sql: str, _db_schema: Optional[str] = None, allow_unsafe: Optional[bool] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        import asyncio
        return await asyncio.to_thread(self.run, sql, _db_schema, allow_unsafe)
