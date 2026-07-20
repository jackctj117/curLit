# curLit engine image — CANONICAL Dockerfile (CL-oluv).
#
# This replaces deploy/Dockerfile (removed). One image serves every app
# process (engine, watchdog, nlp, research dashboard, web API); the
# command is set per-service in docker-compose.app.yml.
#
#   docker build . -t curlit-app:latest
#
# What's baked in: core deps + [polymarket] extra (the engine's broker
# modes need it). Deliberately NOT baked in: [audio]/[train] (whisper /
# tensorboard-scale deps) and CUDA torch — torch is pinned to the CPU
# wheel index below to keep the image small. Train/audio workloads run
# natively, not in this image.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for wheels that fall back to source on edge platforms
# (psycopg2, hmmlearn, some EVM-stack packages).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — copied wholesale into the runtime stage. Using a venv
# (not --prefix) lets the second pip run see torch already installed so
# it doesn't re-resolve the multi-GB CUDA build from PyPI.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch FIRST (transformers needs torch; the default PyPI linux
# wheel drags in the full CUDA stack — several GB we never use).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /build
COPY pyproject.toml ./
# Minimal package stub so `pip install .` resolves without copying all of
# src/ (keeps this layer cacheable across source-only changes).
COPY src/__init__.py ./src/__init__.py
RUN pip install ".[polymarket]"


FROM python:3.12-slim AS runtime

# libpq (psycopg2 runtime) + curl for healthchecks + tini for signal
# handling / zombie reaping; nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/curlit
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY migrations/ ./migrations/
COPY pyproject.toml ./

# Run as non-root.
RUN useradd --uid 10001 --create-home --shell /bin/bash curlit \
    && mkdir -p /opt/curlit/logs /opt/curlit/reports \
    && chown -R curlit:curlit /opt/curlit
USER curlit

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/opt/curlit \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Engine liveness: unauthenticated /health on the embedded FastAPI app.
# CURLIT_API_PORT is the same env var live_engine.py reads, so overriding
# the port keeps the healthcheck in sync. Services that run a different
# process override/disable this in docker-compose.app.yml.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${CURLIT_API_PORT:-8200}/health" || exit 1

# tini reaps zombies + forwards SIGTERM so graceful_shutdown can run.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "src.runtime.run_engine", "--broker", "paper"]
