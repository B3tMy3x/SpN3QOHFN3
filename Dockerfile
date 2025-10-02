# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (optional but useful for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY langpipe/requirements.txt /app/langpipe/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/langpipe/requirements.txt

# Copy project
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Default environment (can be overridden via --env/--env-file)
ENV UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    LLM_HTTP_TIMEOUT=300

# Run the web service
CMD ["sh", "-c", "uvicorn langpipe.webapp.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8000}"]

