#!/usr/bin/env bash
# curLit bootstrap (CL-oluv) — minimal first-run setup.
#
# Checks docker + compose, seeds .env from .env.example, prints next steps.
# Safe to re-run; never overwrites an existing .env.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

fail() { echo "ERROR: $*" >&2; exit 1; }

# ── Prerequisites ────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 \
    || fail "docker not found — install Docker Desktop (mac) or docker-ce (linux)"

docker info >/dev/null 2>&1 \
    || fail "docker daemon not running — start Docker and re-run"

docker compose version >/dev/null 2>&1 \
    || fail "'docker compose' plugin not found — install docker-compose-plugin v2"

echo "ok: docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?')"
echo "ok: $(docker compose version --short 2>/dev/null | sed 's/^/compose /')"

# ── .env ─────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    echo "ok: .env already exists — leaving it untouched"
else
    cp .env.example .env
    echo "created .env from .env.example"
fi

# ── Next steps ───────────────────────────────────────────────────────────
cat <<'EOF'

Next steps:
  1. Edit .env — at minimum POSTGRES_PASSWORD and WEB_API_SECRET
     (broker/API keys can wait; the engine degrades gracefully without them).

  2. Start a stack:
       light (postgres only):
         docker compose up -d
       light + containerized engine:
         docker compose -f docker-compose.yml -f docker-compose.app.yml up -d
       full (observability + airflow, the previous default):
         docker compose --profile full up -d

  3. Native engine (recommended for development) — see docs/BOOT.md:
       python -m venv .venv && .venv/bin/pip install -e '.[dev]'
       .venv/bin/python -m src.runtime.run_engine --broker paper
EOF
