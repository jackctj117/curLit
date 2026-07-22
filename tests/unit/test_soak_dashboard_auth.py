"""Soak-dashboard auth hardening (CL-94n2).

Covers:

  * every endpoint (reads included) requires the X-API-Key header —
    no header and wrong header are rejected, the right one is accepted;
  * the legacy ?secret= query param no longer authenticates;
  * fail-closed 503 when WEB_API_SECRET is unset or a known default;
  * _require_boot_secret refuses process start on unset/default secrets;
  * _engine_api_get sends the secret as an X-API-Key header (never a
    query param) and skips the call entirely without a real secret;
  * the embedded browser JS carries no query-param secret / default.

Hermetic: /api/soak's data helpers are stubbed; no DB, engine API,
process table, or real log files are touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from scripts import soak_dashboard

SECRET = "unit-test-secret-7f3a"
AUTH = {"X-API-Key": SECRET}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with a real secret configured and every data source
    stubbed out so /api/soak is hermetic."""
    monkeypatch.setenv("WEB_API_SECRET", SECRET)
    monkeypatch.setattr(soak_dashboard, "SOAK_LOG", tmp_path / "absent.jsonl")
    monkeypatch.setattr(soak_dashboard, "_find_engine_pid", lambda: None)
    monkeypatch.setattr(soak_dashboard, "_read_samples", lambda: [])
    monkeypatch.setattr(soak_dashboard, "_db_counts", lambda: {})
    monkeypatch.setattr(soak_dashboard, "_recent_errors", lambda n=20: [])
    monkeypatch.setattr(soak_dashboard, "_engine_api_get", lambda path: None)
    monkeypatch.setattr(soak_dashboard, "_recent_trades", lambda n=15: [])
    monkeypatch.setattr(
        soak_dashboard, "DEFAULT_STATE_PATH", tmp_path / "state.json",
    )
    return TestClient(soak_dashboard.app)


# --- header auth on every endpoint -----------------------------------------


@pytest.mark.parametrize("path", ["/api/soak", "/api/approvals"])
def test_rejects_without_header(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 403


@pytest.mark.parametrize("path", ["/api/soak", "/api/approvals"])
def test_rejects_wrong_secret(client: TestClient, path: str) -> None:
    assert client.get(path, headers={"X-API-Key": "nope"}).status_code == 403


def test_rejects_query_param_secret(client: TestClient) -> None:
    """The RIGHT secret in the URL must not authenticate (CL-94n2)."""
    assert client.get(f"/api/soak?secret={SECRET}").status_code == 403


def test_accepts_right_secret_on_read(client: TestClient) -> None:
    resp = client.get("/api/soak", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"]["alive"] is False
    assert body["verdict"]["status"] == "RED"


def test_accepts_right_secret_on_approvals(client: TestClient) -> None:
    # State file doesn't exist -> handler reports the error field, but
    # auth passed and the route served.
    resp = client.get("/api/approvals", headers=AUTH)
    assert resp.status_code == 200
    assert "pending" in resp.json()


def test_post_approvals_rejects_without_header(client: TestClient) -> None:
    resp = client.post(
        "/api/approvals/some-slug", json={"gate": 1, "action": "APPROVE"},
    )
    assert resp.status_code == 403


def test_html_shell_stays_open(client: TestClient) -> None:
    """The static page (no data) must load so the browser can prompt
    for the key; all data routes stay gated."""
    assert client.get("/").status_code == 200


# --- fail-closed on unset/default secret -----------------------------------


@pytest.mark.parametrize(
    "bad", ["", "curlit-dev", "change-me-to-a-random-string", "CHANGE_ME_ABC"],
)
def test_serves_503_when_secret_unset_or_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    if bad:
        monkeypatch.setenv("WEB_API_SECRET", bad)
    else:
        monkeypatch.delenv("WEB_API_SECRET", raising=False)
    # Even a client SENDING the default value gets 503, not a pass.
    resp = client.get("/api/soak", headers={"X-API-Key": bad})
    assert resp.status_code == 503


@pytest.mark.parametrize(
    "bad", ["", "curlit-dev", "change-me-to-a-random-string", "CHANGE_ME_ABC"],
)
def test_refuses_boot_on_default_secret(
    monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    if bad:
        monkeypatch.setenv("WEB_API_SECRET", bad)
    else:
        monkeypatch.delenv("WEB_API_SECRET", raising=False)
    with pytest.raises(SystemExit, match="WEB_API_SECRET"):
        soak_dashboard._require_boot_secret()


def test_boot_check_passes_with_real_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_API_SECRET", SECRET)
    soak_dashboard._require_boot_secret()  # must not raise


# --- engine-API client side ------------------------------------------------


def test_engine_api_get_sends_header_not_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_API_SECRET", SECRET)
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"equity": 1.0}

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(soak_dashboard.httpx, "get", fake_get)
    assert soak_dashboard._engine_api_get("/api/account") == {"equity": 1.0}
    (call,) = calls
    assert call["headers"] == {"X-API-Key": SECRET}
    assert "params" not in call
    assert "secret" not in call["url"]


@pytest.mark.parametrize("bad", ["", "curlit-dev"])
def test_engine_api_get_skips_call_without_real_secret(
    monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    if bad:
        monkeypatch.setenv("WEB_API_SECRET", bad)
    else:
        monkeypatch.delenv("WEB_API_SECRET", raising=False)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("httpx.get must not be called without a secret")

    monkeypatch.setattr(soak_dashboard.httpx, "get", boom)
    assert soak_dashboard._engine_api_get("/api/account") is None


# --- browser JS regression guards ------------------------------------------


def test_html_has_no_query_param_secret_or_default() -> None:
    assert "curlit-dev" not in soak_dashboard._HTML
    assert "?secret" not in soak_dashboard._HTML
    assert "secret=" not in soak_dashboard._HTML
    assert "X-API-Key" in soak_dashboard._HTML
    assert "sessionStorage" in soak_dashboard._HTML
