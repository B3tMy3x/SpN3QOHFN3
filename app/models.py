from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class DDLStatement(BaseModel):
    statement: str = Field(..., description="SQL DDL statement")


class QueryItem(BaseModel):
    queryid: str
    query: str
    runquantity: Optional[int] = None
    executiontime: Optional[float] = Field(
        default=None,
        description="Average execution time per run (optional)",
    )


class NewTaskRequest(BaseModel):
    url: str = Field(..., description="JDBC URL with credentials")
    ddl: List[DDLStatement] = Field(default_factory=list)
    queries: List[QueryItem] = Field(default_factory=list)


class NewTaskResponse(BaseModel):
    taskid: str


class StatusResponse(BaseModel):
    status: str = Field(description="RUNNING | DONE | FAILED")


class ResultDDL(BaseModel):
    statement: str


class ResultMigration(BaseModel):
    statement: str


class ResultQuery(BaseModel):
    queryid: str
    query: str


class TaskResult(BaseModel):
    ddl: List[ResultDDL]
    migrations: List[ResultMigration]
    queries: List[ResultQuery]
