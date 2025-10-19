DB Structure Analysis Service

Description
- Async FastAPI service that accepts a DB structure analysis request and returns proposed DDL changes, data migrations, and updated queries. The pipeline is ported from the notebook to plain Python and uses an OpenAI‑compatible LLM API.

Key Features
- Async job handling with task statuses and 20‑minute timeout.
- Automatic detection of `<catalog>.<schema>` from fully qualified names; target schema is `<schema>_v2` (configurable).
- Fully qualified SQL in all outputs; first DDL is `CREATE SCHEMA <catalog>.<target_schema>`.
- Config via `.env`, safe for Docker (secrets are not baked into images).
- Dockerfile and docker‑compose for simple deployment; buildx support for linux/amd64.

Project Layout
- `app/main.py` — FastAPI app and endpoints.
- `app/models.py` — Pydantic models for requests/responses.
- `app/pipeline.py` — LLM pipeline, SQL helpers, task registry and orchestration.
- `app/logging_config.py` — logging setup from env.
- `requirements.txt` — Python dependencies.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore` — containerization.
- `.env.example` — sample environment variables.

API Endpoints
- POST `/new` — start a task. Request body:
  {
    "url": "jdbc://...",
    "ddl": [{"statement": "CREATE TABLE catalog.schema.t (...)"}],
    "queries": [
      {"queryid": "...", "query": "SELECT ...", "runquantity": 1, "executiontime": 1.23}
    ]
  }
  Response: {"taskid": "..."}

- GET `/status?task_id=<id>` — returns {"status": "RUNNING|DONE|FAILED"}

- GET `/getresult?task_id=<id>` — returns:
  {"ddl": [{"statement": "..."}], "migrations": [{"statement": "..."}], "queries": [{"queryid": "...", "query": "..."}]}

Quick Start
- Prepare env file: copy `.env.example` to `.env` and fill values.
- Run with Docker Compose:
  - `docker compose up -d --build`
  - Open `http://localhost:8000/docs` for Swagger UI.
- Run with docker run:
  - `docker run --rm --env-file .env -p 8000:8000 docker.io/<namespace>/db-structure-analysis:latest`
- Local dev (Python):
  - `pip install -r requirements.txt`
  - `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

Environment (.env)
- `OPENAI_BASE_URL` — OpenAI‑compatible endpoint (priority, e.g. https://cloud.m1r0.ru/v1)
- `OPENAI_MODEL` — model name (default qwen3-coder:30b)
- `PIPELINE_MAJORITY` — optional (default 0.8)
- `PIPELINE_PER_BATCH` — optional (default 5)
- `PIPELINE_TIMEOUT_SEC` — optional (default 1200 = 20 minutes)
- `PIPELINE_CONCURRENCY` — optional (default 2)
- `TARGET_SCHEMA_SUFFIX` — optional (default `_v2`)
- `LOG_LEVEL` — optional (default INFO)

Usage Examples
- Start a task with `test.json`:
  - `curl -s -X POST http://localhost:8000/new -H 'Content-Type: application/json' --data @test.json`
  - Response: `{"taskid":"<uuid>"}`
- Poll status:
  - `curl -s "http://localhost:8000/status?task_id=<uuid>"`
- Get result when DONE:
  - `curl -s "http://localhost:8000/getresult?task_id=<uuid>"`

Build and Push (linux/amd64)
- Using Makefile:
  - `docker login`
  - `make push-amd64 IMAGE=docker.io/<namespace>/db-structure-analysis TAG=v0.1.0`
- Using buildx directly:
  - `docker buildx create --use --name builder || true`
  - `docker buildx build --platform linux/amd64 -t docker.io/<namespace>/db-structure-analysis:v0.1.0 -f Dockerfile --push .`

Troubleshooting
- 500/FAILED early: check LLM endpoint; verify `OPENAI_BASE_URL`/`OPENAI_MODEL` are passed to container (`--env-file .env`).
- Ambiguous or missing catalog.schema: ensure input SQL uses fully qualified names like `<catalog>.<schema>.<table>`.
- Port not reachable on localhost: ensure `-p 8000:8000` is used or compose service exposes `8000`.
