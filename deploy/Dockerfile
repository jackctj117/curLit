# Single image for all curLit application processes (CL-mz7).
#
# All four app services (engine, watchdog, nlp, web) share this image —
# the command to run is set per-service in docker-compose.app.yml.
# Multi-stage to keep the runtime image lean by dropping the build
# toolchain after wheels are compiled.

FROM python:3.11-slim AS builder

WORKDIR /build

# Build deps for any wheels that need compilation (numpy/pandas/scipy
# usually ship wheels, but psycopg2 and statsmodels can fall back to
# source on edge platforms).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install -e ".[dev]" \
    || pip install --no-cache-dir --prefix=/install -e .


FROM python:3.11-slim AS runtime

# libpq (psycopg2 runtime) + curl for healthchecks; nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/curlit

COPY --from=builder /install /usr/local
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

ENV PYTHONPATH=/opt/curlit \
    PYTHONUNBUFFERED=1

# tini reaps zombies + handles SIGTERM so graceful_shutdown can run.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "src.runtime.run_engine", "--broker", "paper"]
