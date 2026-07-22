"""Web API auth hardening regressions (CL-k55b, CL-pu7i header-only)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.web.api import app

client = TestClient(app)


def test_default_secret_refuses_everyone(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "curlit-dev")
    # Even someone WHO KNOWS the default gets 503 — fail closed.
    r = client.get("/api/system", headers={"X-API-Key": "curlit-dev"})
    assert r.status_code == 503


def test_unset_secret_refuses_everyone(monkeypatch):
    monkeypatch.delenv("WEB_API_SECRET", raising=False)
    assert client.get("/api/system").status_code == 503


def test_wrong_secret_403(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    r = client.get("/api/system", headers={"X-API-Key": "nope"})
    assert r.status_code == 403


def test_missing_header_403(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    assert client.get("/api/system").status_code == 403


def test_query_param_no_longer_authenticates(monkeypatch):
    """CL-pu7i: the legacy ?secret= query param is dead — header only."""
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    r = client.get("/api/system", params={"secret": "a-real-secret-value"})
    assert r.status_code == 403


def test_header_works(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    r = client.get("/api/system", headers={"X-API-Key": "a-real-secret-value"})
    assert r.status_code == 200


def test_length_mismatch_is_clean_403(monkeypatch):
    """Hash-then-compare: unequal-length keys must 403, never 500, and the
    comparison runs over equal-length SHA-256 digests (no length leak)."""
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    for supplied in ("x", "a-real-secret-value-plus-extra-length", ""):
        r = client.get("/api/system", headers={"X-API-Key": supplied})
        assert r.status_code == 403, supplied


def test_health_stays_open(monkeypatch):
    monkeypatch.delenv("WEB_API_SECRET", raising=False)
    assert client.get("/health").status_code == 200
