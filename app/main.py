from __future__ import annotations

import asyncio
import os
from fastapi import FastAPI, HTTPException, Query
import logging
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .models import (
    NewTaskRequest,
    NewTaskResponse,
    StatusResponse,
)
from .pipeline import Pipeline, PipelineConfig, TaskRegistry
from .logging_config import setup_logging


app = FastAPI(title="DB Structure Analysis Service", version="1.0.0")

# Global registry and pipeline
registry = TaskRegistry()
logger = logging.getLogger(__name__)

def _mk_pipeline() -> Pipeline:
    # Read overrides from environment
    def _int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except Exception:
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except Exception:
            return default

    cfg = PipelineConfig(
        majority_threshold=_float("PIPELINE_MAJORITY", 0.8),
        per_batch=_int("PIPELINE_PER_BATCH", 5),
        timeout_sec=_int("PIPELINE_TIMEOUT_SEC", 60 * 20),
        target_suffix=os.getenv("TARGET_SCHEMA_SUFFIX", "_v2"),
        concurrency=_int("PIPELINE_CONCURRENCY", 2),
    )
    return Pipeline(cfg)


@app.post("/new", response_model=NewTaskResponse)
async def start_new_task(req: NewTaskRequest):
    tid = registry.create()
    logger.info("/new accepted | task=%s | ddl=%d | queries=%d", tid, len(req.ddl), len(req.queries))

    async def _runner():
        try:
            pipe = _mk_pipeline()
            logger.info("task=%s started", tid)
            result = await asyncio.wait_for(pipe.run(req, task_id=tid), timeout=PipelineConfig().timeout_sec)
            registry.set_done(tid, result)
            logger.info("task=%s done", tid)
        except asyncio.TimeoutError:
            registry.set_failed(tid, "Timed out after 20 minutes")
            logger.warning("task=%s failed by timeout", tid)
        except Exception as e:
            logger.exception("task=%s failed: %s", tid, e)
            registry.set_failed(tid, f"{type(e).__name__}: {e}")

    asyncio.create_task(_runner())
    return NewTaskResponse(taskid=tid)


@app.get("/status", response_model=StatusResponse)
async def get_status(task_id: str = Query(..., alias="task_id")):
    st = registry.get(task_id)
    if not st:
        raise HTTPException(status_code=404, detail="task not found")
    logger.info("/status | task=%s -> %s", task_id, st.status)
    return StatusResponse(status=st.status)


@app.get("/getresult")
async def get_result(task_id: str = Query(..., alias="task_id")):
    st = registry.get(task_id)
    if not st:
        raise HTTPException(status_code=404, detail="task not found")
    if st.status == "RUNNING":
        return JSONResponse(status_code=425, content={"detail": "task is still running"})
    if st.status == "FAILED":
        raise HTTPException(status_code=500, detail=st.error or "task failed")
    # DONE
    logger.info("/getresult | task=%s -> DONE result returned", task_id)
    return JSONResponse(content=st.result.model_dump())


# Convenience root
@app.get("/")
def root():
    return {"service": app.title, "version": app.version}


@app.on_event("startup")
async def _startup():
    setup_logging()
    logger.info("Service started: %s %s", app.title, app.version)
