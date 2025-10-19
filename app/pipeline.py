from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections import Counter
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .models import NewTaskRequest, ResultDDL, ResultMigration, ResultQuery, TaskResult, QueryItem


# --------- Catalog/Schema detection ---------
_FQ_TBL = r"[A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*"


def _iter_fq_pairs_from_sql(sql: str) -> Iterable[str]:
    if not sql:
        return ()
    for pat in (rf"(?i)\bFROM\s+({_FQ_TBL})", rf"(?i)\bJOIN\s+({_FQ_TBL})"):
        for m in re.finditer(pat, sql):
            yield ".".join(m.group(1).split(".")[:2])
    for m in re.finditer(rf"(?i)\b(CREATE|ALTER|INSERT\s+INTO)\s+(?:TABLE\s+)?({_FQ_TBL})", sql):
        yield ".".join(m.group(2).split(".")[:2])


def _collect_catalog_schema_pairs(data: Dict) -> Counter:
    pairs: Counter = Counter()
    for d in (data.get("ddl") or []):
        stmt = (d or {}).get("statement") or ""
        for pair in _iter_fq_pairs_from_sql(stmt):
            pairs[pair] += 1
    for q in (data.get("queries") or []):
        sql = (q or {}).get("query") or ""
        for pair in _iter_fq_pairs_from_sql(sql):
            pairs[pair] += 1
    return pairs


def _decide_by_vote(pairs: Counter, majority: float = 0.8) -> Tuple[str, str]:
    if not pairs:
        raise ValueError("No fully-qualified table names found to determine <catalog>.<schema>")
    most, cnt = pairs.most_common(1)[0]
    total = sum(pairs.values())
    if cnt == total or (total and cnt / total >= majority):
        cat, sch = most.split(".")
        return cat, sch
    raise ValueError(f"Ambiguous schemas: {dict(pairs)} (no majority >= {majority})")


def detect_catalog_schema(data: Dict, majority: float = 0.8) -> Tuple[str, str]:
    pairs = _collect_catalog_schema_pairs(data)
    return _decide_by_vote(pairs, majority)


# --------- SQL helpers ---------
def split_sql(sql: str) -> List[str]:
    out: List[str] = []
    current: List[str] = []
    in_quote = False
    for ch in sql or "":
        if ch == "'" and (not current or current[-1] != "\\"):
            in_quote = not in_quote
        if ch == ";" and not in_quote:
            stmt = "".join(current).strip()
            if stmt:
                out.append(stmt)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        out.append(tail)
    return out


def _strip_name(name: str) -> str:
    return name.strip().strip("`").strip('"')


def ensure_target(catalog: str, target_schema: str, name: str) -> str:
    parts = [p for p in _strip_name(name).split(".") if p]
    tbl = parts[-1] if parts else _strip_name(name)
    return f"{catalog}.{target_schema}.{tbl}"


def ensure_source(catalog: str, source_schema: str, name: str) -> str:
    parts = [p for p in _strip_name(name).split(".") if p]
    tbl = parts[-1] if parts else _strip_name(name)
    return f"{catalog}.{source_schema}.{tbl}"


def normalize_ctas_order(stmt: str) -> str:
    pattern = re.compile(
        r"(?i)(CREATE\s+TABLE\s+[^\s]+)\s+AS\s+(SELECT\b.*)\s+WITH\s*\(([^)]*)\)"
    )

    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        select_sql = match.group(2)
        props = match.group(3)
        return f"{prefix} WITH ({props}) AS {select_sql}"

    return re.sub(pattern, repl, stmt)


def rewrite_targets_sources(catalog: str, source_schema: str, target_schema: str, stmt: str) -> str:
    s = stmt
    # Ensure CREATE/INSERT target tables point to target schema
    s = re.sub(
        r"(?i)\bCREATE\s+TABLE\s+([A-Za-z0-9_\.]+)",
        lambda m: f"CREATE TABLE {ensure_target(catalog, target_schema, m.group(1))}",
        s,
    )
    s = re.sub(
        r"(?i)\bINSERT\s+INTO\s+([A-Za-z0-9_\.]+)",
        lambda m: f"INSERT INTO {ensure_target(catalog, target_schema, m.group(1))}",
        s,
    )
    # Rewrite FROM/JOIN sources to use source schema explicitly
    s = re.sub(
        rf"(?i)\bFROM\s+({_FQ_TBL}|[A-Za-z_]\w*)",
        lambda m: f"FROM {ensure_source(catalog, source_schema, m.group(1))}",
        s,
    )
    s = re.sub(
        rf"(?i)\bJOIN\s+({_FQ_TBL}|[A-Za-z_]\w*)",
        lambda m: f"JOIN {ensure_source(catalog, source_schema, m.group(1))}",
        s,
    )
    return normalize_ctas_order(s)


def qualify_query_to_target(catalog: str, source_schema: str, target_schema: str, sql: str) -> str:
    return (sql or "").replace(f"{catalog}.{source_schema}.", f"{catalog}.{target_schema}.")


def qualify_query_full_to_target(catalog: str, target_schema: str, sql: str) -> str:
    s = sql or ""

    # Collect CTE names to avoid qualifying them as tables
    cte_names = set()
    for m in re.finditer(r"(?i)\bWITH\s+([A-Za-z_]\w*)\s+AS\b", s):
        cte_names.add(m.group(1).lower())
    for m in re.finditer(r"(?i),\s*([A-Za-z_]\w*)\s+AS\b", s):
        cte_names.add(m.group(1).lower())

    def _replace_from(m: re.Match) -> str:
        name = m.group(1)
        if name and name.lower() in cte_names:
            return f"FROM {name}"
        return f"FROM {ensure_target(catalog, target_schema, name)}"

    def _replace_join(m: re.Match) -> str:
        name = m.group(1)
        if name and name.lower() in cte_names:
            return f"JOIN {name}"
        return f"JOIN {ensure_target(catalog, target_schema, name)}"

    s = re.sub(rf"(?i)\bFROM\s+({_FQ_TBL}|[A-Za-z_]\w*)", _replace_from, s)
    s = re.sub(rf"(?i)\bJOIN\s+({_FQ_TBL}|[A-Za-z_]\w*)", _replace_join, s)
    return s


def clean_sql(s: str) -> str:
    if s is None:
        return ""
    s2 = s.strip()
    s2 = s2.replace("```sql", "```")
    if s2.startswith("```") and s2.endswith("```"):
        s2 = s2[3:-3].strip()
    return s2


# --------- LLM Client (OpenAI-compatible) ---------
class LLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        # Priority: use custom OpenAI-compatible base URL (e.g., https://cloud.m1r0.ru/v1) without requiring API key
        self.model = model or os.getenv("OPENAI_MODEL", "qwen3-coder:30b")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip()
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()

    async def acomplete(self, system: str, user: str) -> str:
        # If custom base_url provided, call it directly via httpx without mandatory API key
        if self.base_url:
            import httpx

            url = self.base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            # Send Authorization only if a key is actually provided
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                # OpenAI-compatible schema
                return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")

        # Fallback: use official OpenAI SDK only if no base_url and an API key is given
        from openai import OpenAI
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OPENAI_BASE_URL provided and OPENAI_API_KEY is missing")
        client = OpenAI(api_key=key)

        loop = asyncio.get_event_loop()

        def _call():
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content or ""

        return await loop.run_in_executor(None, _call)


# --------- Prompt builders (concise, aligned with task) ---------
def prompt_query_analyze(catalog: str, source_schema: str) -> str:
    return (
        "Ты — аналитик SQL для Trino + Iceberg. "
        "Проанализируй батч из до 5 запросов. Определи чаще используемые таблицы, пары JOIN и фильтры. "
        "Сообщи ключевые наблюдения кратко (без кода)."
        f"\nCatalog: {catalog}\nSchema: {source_schema}\n"
    )


def prompt_query_summarize(catalog: str, source_schema: str) -> str:
    return (
        "Объедини несколько блоков анализа SQL в единый сводный отчет для DDL-оптимизации. "
        "Выдели hot таблицы, частые JOIN, фильтры и тяжёлые операции. Без кода."
        f"\nCatalog: {catalog}\nSchema: {source_schema}\n"
    )


def prompt_ddl_optimize(catalog: str, source_schema: str, target_schema: str) -> str:
    return (
        "Ты — DDL-оптимизатор под Trino + Iceberg. Верни ТОЛЬКО DDL (без текста), "
        "начиная с CREATE SCHEMA {catalog}.{target_schema}. Все таблицы полностью квалифицированы. "
        "CTAS в форме: CREATE TABLE <...> WITH (format='PARQUET') AS SELECT ..."
    ).format(catalog=catalog, target_schema=target_schema)


def prompt_migrations(catalog: str, source_schema: str, target_schema: str) -> str:
    return (
        "Сгенерируй ТОЛЬКО DML миграции (INSERT/UPDATE/MERGE) для переноса данных из старой схемы в новую. "
        "Полная квалификация таблиц обязательна. Без комментариев."
        f"\nSource: {catalog}.{source_schema}\nTarget: {catalog}.{target_schema}\n"
    )


def prompt_query_optimize(catalog: str, source_schema: str, target_schema: str) -> str:
    return (
        "Оптимизируй запрос для Trino, заменив схему на целевую и сохранив семантику. Верни один SQL без комментариев."
        f"\nTarget schema: {catalog}.{target_schema}\n"
    )


def prompt_query_critic(catalog: str, source_schema: str, target_schema: str) -> str:
    return (
        "Проверь SQL на соответствие Trino и корректность: схемы, алиасы, GROUP BY, синтаксис. "
        "Верни 'OK' или полный исправленный SQL."
        f"\nTarget schema: {catalog}.{target_schema}\n"
    )


# --------- Pipeline ---------
@dataclass
class PipelineConfig:
    majority_threshold: float = 0.8
    per_batch: int = 5
    timeout_sec: int = 60 * 20  # 20 minutes
    target_suffix: str = "_v2"
    concurrency: int = 2


class Pipeline:
    def __init__(self, cfg: PipelineConfig | None = None, llm: LLM | None = None):
        self.cfg = cfg or PipelineConfig()
        self.llm = llm or LLM()
        self.log = logging.getLogger(self.__class__.__name__)

    async def run(self, req: NewTaskRequest, task_id: str | None = None) -> TaskResult:
        t0 = time.monotonic()
        # Detect catalog/schema
        data_dict = {
            "ddl": [s.model_dump() for s in req.ddl],
            "queries": [q.model_dump() for q in req.queries],
        }
        self.log.info("task=%s pipeline start | ddl=%d | queries=%d", task_id, len(req.ddl), len(req.queries))
        catalog, source_schema = detect_catalog_schema(data_dict, self.cfg.majority_threshold)
        target_schema = f"{source_schema}{self.cfg.target_suffix}"
        self.log.info("task=%s detected catalog.schema=%s.%s -> target=%s", task_id, catalog, source_schema, target_schema)

        # Prepare inputs
        ddl_orig = "\n".join([i.statement for i in req.ddl])
        querys = req.queries

        # 1) Analyze queries in small batches
        analyze_blocks: List[str] = []
        sys_analyze = prompt_query_analyze(catalog, source_schema)
        self.log.info("task=%s stage=analyze batches=%d per_batch=%d", task_id, (len(querys)+self.cfg.per_batch-1)//self.cfg.per_batch, self.cfg.per_batch)
        for i in range(0, len(querys), self.cfg.per_batch):
            chunk = querys[i : i + self.cfg.per_batch]
            user = "\n\n".join(
                [
                    f"sql: {q.query}\nrunquantity: {q.runquantity or 0}\nexecutiontime: {q.executiontime or 'UNKNOWN'}"
                    for q in chunk
                ]
            )
            ts = time.monotonic()
            block = await self.llm.acomplete(sys_analyze, user)
            self.log.debug("task=%s stage=analyze batch_done in %.2fs", task_id, time.monotonic()-ts)
            analyze_blocks.append(block)

        # 2) Summarize analysis
        sys_sum = prompt_query_summarize(catalog, source_schema)
        ts = time.monotonic()
        summary = await self.llm.acomplete(sys_sum, "\n\n".join(analyze_blocks))
        self.log.info("task=%s stage=summarize done in %.2fs", task_id, time.monotonic()-ts)

        # 3) New DDL
        sys_ddl = prompt_ddl_optimize(catalog, source_schema, target_schema)
        ddl_user = (
            f"[АНАЛИЗ]\n{summary}\n\n[ИСХОДНЫЕ DDL]\n{ddl_orig}"
        )
        ts = time.monotonic()
        new_ddl_sql = clean_sql(await self.llm.acomplete(sys_ddl, ddl_user))
        self.log.info("task=%s stage=ddl done in %.2fs", task_id, time.monotonic()-ts)

        # 4) Migrations
        sys_mig = prompt_migrations(catalog, source_schema, target_schema)
        mig_user = (
            f"[АНАЛИЗ]\n{summary}\n\n[НОВЫЕ DDL]\n{new_ddl_sql}"
        )
        ts = time.monotonic()
        migrations_sql = clean_sql(await self.llm.acomplete(sys_mig, mig_user))
        self.log.info("task=%s stage=migrations done in %.2fs", task_id, time.monotonic()-ts)

        # 5) Optimize queries + critic (concurrently)
        sys_qopt = prompt_query_optimize(catalog, source_schema, target_schema)
        sys_qcrit = prompt_query_critic(catalog, source_schema, target_schema)

        sem = asyncio.Semaphore(self.cfg.concurrency)

        async def _process(q: QueryItem) -> ResultQuery:
            async with sem:
                ts1 = time.monotonic()
                opt_sql = clean_sql(await self.llm.acomplete(sys_qopt, q.query)) or q.query
                t_opt = time.monotonic() - ts1
            async with sem:
                ts2 = time.monotonic()
                critic_out = clean_sql(await self.llm.acomplete(sys_qcrit, opt_sql))
                t_crit = time.monotonic() - ts2
            if critic_out and critic_out.strip().upper() != "OK":
                opt_sql = critic_out
            final_sql = qualify_query_full_to_target(
                catalog,
                target_schema,
                qualify_query_to_target(catalog, source_schema, target_schema, opt_sql),
            )
            self.log.debug("task=%s stage=query id=%s opt=%.2fs crit=%.2fs", task_id, q.queryid, t_opt, t_crit)
            return ResultQuery(queryid=q.queryid, query=final_sql)

        tasks = [asyncio.create_task(_process(q)) for q in querys]
        ts = time.monotonic()
        result_queries = await asyncio.gather(*tasks)
        self.log.info("task=%s stage=queries processed=%d in %.2fs", task_id, len(result_queries), time.monotonic()-ts)

        # Assemble outputs
        ddl_entries: List[ResultDDL] = []
        for stmt in split_sql(new_ddl_sql):
            norm = rewrite_targets_sources(catalog, source_schema, target_schema, stmt)
            if norm and not norm.strip().upper().startswith("SELECT"):
                ddl_entries.append(ResultDDL(statement=norm))

        # Ensure first DDL creates the target schema explicitly
        if not ddl_entries or not ddl_entries[0].statement.strip().upper().startswith("CREATE SCHEMA"):
            ddl_entries.insert(0, ResultDDL(statement=f"CREATE SCHEMA {catalog}.{target_schema}"))

        migration_entries: List[ResultMigration] = []
        for stmt in split_sql(migrations_sql):
            norm = rewrite_targets_sources(catalog, source_schema, target_schema, stmt)
            up = norm.strip().upper()
            if up.startswith(("INSERT", "UPDATE", "MERGE")):
                migration_entries.append(ResultMigration(statement=norm))

        total = time.monotonic() - t0
        self.log.info("task=%s pipeline finished in %.2fs | ddl=%d | mig=%d | queries=%d", task_id, total, len(ddl_entries), len(migration_entries), len(result_queries))
        return TaskResult(ddl=ddl_entries, migrations=migration_entries, queries=result_queries)


# Simple in-memory task registry
@dataclass
class TaskState:
    status: str
    result: TaskResult | None = None
    error: str | None = None


class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, TaskState] = {}
        self.log = logging.getLogger(self.__class__.__name__)

    def create(self) -> str:
        tid = str(uuid.uuid4())
        self._tasks[tid] = TaskState(status="RUNNING")
        self.log.info("task=%s created", tid)
        return tid

    def set_done(self, tid: str, result: TaskResult):
        self._tasks[tid] = TaskState(status="DONE", result=result)
        self.log.info("task=%s marked DONE", tid)

    def set_failed(self, tid: str, error: str):
        self._tasks[tid] = TaskState(status="FAILED", error=error)
        self.log.warning("task=%s marked FAILED: %s", tid, error)

    def get(self, tid: str) -> TaskState | None:
        return self._tasks.get(tid)
