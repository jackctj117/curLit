"""Web API auth hardening regressions (CL-k55b)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.web.api import app

client = TestClient(app)


def test_default_secret_refuses_everyone(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "curlit-dev")
    # Even someone WHO KNOWS the default gets 503 — fail closed.
    r = client.get("/api/system", params={"secret": "curlit-dev"})
    assert r.status_code == 503


def test_unset_secret_refuses_everyone(monkeypatch):
    monkeypatch.delenv("WEB_API_SECRET", raising=False)
    assert client.get("/api/system", params={"secret": ""}).status_code == 503


def test_wrong_secret_403(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    assert client.get("/api/system", params={"secret": "nope"}).status_code == 403


def test_query_param_still_works(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    r = client.get("/api/system", params={"secret": "a-real-secret-value"})
    assert r.status_code == 200


def test_header_preferred(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", "a-real-secret-value")
    r = client.get("/api/system", headers={"X-API-Key": "a-real-secret-value"})
    assert r.status_code == 200
    # header wins over a wrong query param
    r2 = client.get("/api/system", params={"secret": "wrong"},
                    headers={"X-API-Key": "a-real-secret-value"})
    assert r2.status_code == 200


def test_health_stays_open(monkeypatch):
    monkeypatch.delenv("WEB_API_SECRET", raising=False)
    assert client.get("/health").status_code == 200
