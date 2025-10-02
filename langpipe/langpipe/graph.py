from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END  # type: ignore

from .critic import call_critic, call_enhance_prompt, call_fix_error_prompt
from .llm import LLM
from .sql import SQLExecutorModel
from .generator import call_generate_sql
from .trino_exec import TrinoExecutorModel
from .trino_schema import introspect_schema


class PipelineState(TypedDict, total=False):
    user_query: str
    db_schema: Optional[str]
    chat_history: List[Dict[str, str]]
    current_prompt: str  # latest guidance prompt for SQL generator
    sql_query: str       # actual SQL to "execute" (model-based checker)
    execution_result: Any
    error: Optional[str]
    critique: str
    issues: List[str]
    rationale: str
    route: Literal["enhance_prompt", "execute_sql"]
    fixed_prompt: str
    notes: List[str]
    # Итерации генерации/исправления
    attempt: int
    max_attempts: int
    reasoning: str
    # Диалект и квалификация
    dialect: str
    trino_catalog: str
    trino_schema: str

def make_schema_introspect_trino_node(exec_model: Optional[SQLExecutorModel | TrinoExecutorModel]):
    def node(state: PipelineState) -> Dict[str, Any]:
        if state.get("db_schema"):
            return {}
        if not isinstance(exec_model, TrinoExecutorModel):
            return {}
        cfg = exec_model.cfg
        schema_text = introspect_schema(cfg)
        out: Dict[str, Any] = {}
        if schema_text:
            out["db_schema"] = schema_text
            out["trino_catalog"] = cfg.catalog or ""
            out["trino_schema"] = cfg.schema or ""
            if not state.get("dialect"):
                out["dialect"] = "trino"
        return out
    return node


def make_llm_critic_node(llm: LLM):
    def node(state: PipelineState) -> Dict[str, Any]:
        res = call_critic(
            llm=llm,
            user_query=state.get("user_query", ""),
            sql_query=state.get("sql_query"),
            db_schema=state.get("db_schema"),
            chat_history=state.get("chat_history"),
        )
        return {
            "route": res.get("route"),
            "rationale": res.get("rationale", ""),
            "issues": res.get("issues", []),
            "critique": res.get("raw", ""),
        }
    return node


def make_enhance_prompt_node(llm: LLM):
    def node(state: PipelineState) -> Dict[str, Any]:
        res = call_enhance_prompt(
            llm=llm,
            user_query=state.get("user_query", ""),
            critique=state.get("critique"),
            issues=state.get("issues", []),
            db_schema=state.get("db_schema"),
            chat_history=state.get("chat_history"),
        )
        return {
            "current_prompt": res.get("enhanced_prompt", ""),
            "notes": res.get("notes", []),
        }
    return node


def make_generate_sql_node(llm: LLM):
    def node(state: PipelineState) -> Dict[str, Any]:
        # Инициализация счётчиков
        attempt = int(state.get("attempt") or 0)
        max_attempts = int(state.get("max_attempts") or 3)
        res = call_generate_sql(
            llm=llm,
            user_query=state.get("user_query", ""),
            current_prompt=state.get("fixed_prompt") or state.get("current_prompt"),
            db_schema=state.get("db_schema"),
            chat_history=state.get("chat_history"),
            dialect=state.get("dialect", "trino"),
            catalog=state.get("trino_catalog"),
            schema=state.get("trino_schema"),
        )
        out = {
            "sql_query": res.get("sql_query", ""),
            "reasoning": res.get("reasoning", ""),
            "attempt": attempt,
            "max_attempts": max_attempts,
        }
        return out
    return node


def make_sql_executor_node(exec_model: SQLExecutorModel):
    def node(state: PipelineState) -> Dict[str, Any]:
        sql = state.get("sql_query")
        if not sql:
            return {"execution_result": None, "error": "No SQL to execute"}
        res, err = exec_model.run(sql, state.get("db_schema"))
        if err:
            return {"execution_result": None, "error": err}
        return {"execution_result": res, "error": None}
    return node


def make_fix_error_prompt_node(llm: LLM):
    def node(state: PipelineState) -> Dict[str, Any]:
        res = call_fix_error_prompt(
            llm=llm,
            user_query=state.get("user_query", ""),
            sql_query=state.get("sql_query"),
            error=state.get("error", "Unknown error"),
            db_schema=state.get("db_schema"),
            chat_history=state.get("chat_history"),
        )
        # Увеличим счётчик попыток здесь (внутри узла, а не в условии)
        attempt = int(state.get("attempt") or 0)
        max_attempts = int(state.get("max_attempts") or 3)
        return {
            "fixed_prompt": res.get("fixed_prompt", ""),
            "attempt": attempt + 1,
            "max_attempts": max_attempts,
        }
    return node


def _route_from_critic(state: PipelineState) -> Literal["enhance_prompt", "execute_sql"]:
    route = state.get("route") or "enhance_prompt"
    return route  # type: ignore


def _error_to_fix_or_end(state: PipelineState) -> Literal["fix_error_prompt", "__end__"]:
    return "fix_error_prompt" if state.get("error") else END  # type: ignore


def _retry_or_end(state: PipelineState) -> Literal["generate_sql", "__end__"]:
    attempt = int(state.get("attempt") or 0)
    max_attempts = int(state.get("max_attempts") or 3)
    return "generate_sql" if attempt < max_attempts else END  # type: ignore


def build_graph(llm: LLM, exec_model: SQLExecutorModel | TrinoExecutorModel):
    """
    Build the LangGraph pipeline with the required flows:
      - llm_critic -> enhance_prompt
      - llm_critic -> sql_executor -> fix_error_prompt
    """
    g = StateGraph(PipelineState)

    g.add_node("schema_introspect_trino", make_schema_introspect_trino_node(exec_model))
    g.add_node("llm_critic", make_llm_critic_node(llm))
    g.add_node("enhance_prompt", make_enhance_prompt_node(llm))
    g.add_node("sql_executor", make_sql_executor_node(exec_model))
    g.add_node("fix_error_prompt", make_fix_error_prompt_node(llm))
    g.add_node("generate_sql", make_generate_sql_node(llm))

    g.set_entry_point("schema_introspect_trino")
    g.add_edge("schema_introspect_trino", "llm_critic")

    # Branch based on critic decision
    g.add_conditional_edges(
        "llm_critic",
        _route_from_critic,
        {
            "enhance_prompt": "enhance_prompt",
            "execute_sql": "sql_executor",
        },
    )

    # If execution errors, go fix_error_prompt; otherwise end
    g.add_conditional_edges(
        "sql_executor",
        _error_to_fix_or_end,
        {
            "fix_error_prompt": "fix_error_prompt",
            END: END,
        },
    )

    # После enhance_prompt генерируем SQL
    g.add_edge("enhance_prompt", "generate_sql")

    # После генерации пробуем исполнить
    g.add_edge("generate_sql", "sql_executor")

    # После фикса ошибки — либо новый цикл генерации, либо конец по лимиту
    g.add_conditional_edges(
        "fix_error_prompt",
        _retry_or_end,
        {
            "generate_sql": "generate_sql",
            END: END,
        },
    )

    return g.compile()
