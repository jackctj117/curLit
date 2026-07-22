"""scripts/rotate_secrets.py verifier log scrubbing (CL-8lv6 P1).

Failure paths must log the exception TYPE (and a redacted DSN) only —
never str(exc), which can embed the candidate secret (FRED key in the
URL query string, Postgres password in the DSN).
"""

from __future__ import annotations

import logging

import httpx
from scripts import rotate_secrets


def test_verify_postgres_failure_never_logs_password(monkeypatch, caplog):
    secret = "sup3r-s3cret-pw"  # noqa: S105 — test fixture value

    def boom(*args, **kwargs):
        raise RuntimeError(
            f"connection refused for postgresql+psycopg2://fx:{secret}@localhost:5432/fx"
        )

    monkeypatch.setattr("sqlalchemy.create_engine", boom)
    with caplog.at_level(logging.ERROR, logger="scripts.rotate_secrets"):
        assert rotate_secrets._verify_postgres(secret) is False

    assert secret not in caplog.text  # the actual finding
    assert "RuntimeError" in caplog.text  # type survives for debuggability
    assert ":***@" in caplog.text  # redacted DSN


def test_verify_postgres_dsn_survives_awkward_password(monkeypatch, caplog):
    """URL.create must escape passwords the old f-string DSN corrupted."""
    secret = "p@ss:word/with#specials"

    def boom(*args, **kwargs):
        raise RuntimeError(f"dsn was ...:{secret}@...")

    monkeypatch.setattr("sqlalchemy.create_engine", boom)
    with caplog.at_level(logging.ERROR, logger="scripts.rotate_secrets"):
        assert rotate_secrets._verify_postgres(secret) is False
    assert secret not in caplog.text


def test_verify_fred_failure_never_logs_api_key(monkeypatch, caplog):
    secret = "FREDSECRETKEY123"

    def boom(*args, **kwargs):
        raise httpx.ConnectError(
            f"GET https://api.stlouisfed.org/fred/series?api_key={secret} failed"
        )

    monkeypatch.setattr(rotate_secrets.httpx, "get", boom)
    with caplog.at_level(logging.ERROR, logger="scripts.rotate_secrets"):
        assert rotate_secrets._verify_fred(secret) is False

    assert secret not in caplog.text
    assert "ConnectError" in caplog.text


def test_verify_oanda_failure_never_logs_key(monkeypatch, caplog):
    secret = "oanda-bearer-token"
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "001-001-1234567-001")

    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout(f"timeout; headers included Bearer {secret}")

    monkeypatch.setattr(rotate_secrets.httpx, "get", boom)
    with caplog.at_level(logging.ERROR, logger="scripts.rotate_secrets"):
        assert rotate_secrets._verify_oanda(secret) is False

    assert secret not in caplog.text
    assert "ConnectTimeout" in caplog.text


def test_verify_oanda_success_path_still_works(monkeypatch):
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "001-001-1234567-001")

    class _Resp:
        status_code = 200

    monkeypatch.setattr(rotate_secrets.httpx, "get", lambda *a, **k: _Resp())
    assert rotate_secrets._verify_oanda("whatever") is True
