"""Shared Postgres connection-URL builder (CL-8lv6).

One place for the env-var convention every daemon uses — and one place
that WARNS when the well-known default password is in effect, instead of
each call site silently connecting with ``changeme``. Rotation itself is
an operator action (CL-1esk runbook); this makes the debt loud at every
process start until it happens.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_PASSWORD = "changeme"  # noqa: S105 — the known-bad default we warn about
_warned = False


def build_db_url() -> str:
    """Postgres URL from env (DATABASE_URL wins), warning once per process
    when the password is the well-known default."""
    global _warned  # noqa: PLW0603 — once-per-process warning latch
    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit
    user = os.environ.get("POSTGRES_USER", "fx")
    password = os.environ.get("POSTGRES_PASSWORD", _DEFAULT_PASSWORD)
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "fx")
    if password == _DEFAULT_PASSWORD and not _warned:
        _warned = True
        logger.warning(
            "Postgres is using the WELL-KNOWN default password — rotate it "
            "(bead CL-1esk runbook: ALTER USER + .env + Airflow env + "
            "coordinated restart)",
        )
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
