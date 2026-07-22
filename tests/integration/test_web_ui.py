"""Web UI integration tests — API endpoints (X-API-Key header auth)."""

import pytest


class TestWebAPI:
    def test_account_with_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from src.web.api import app
        monkeypatch.setenv("WEB_API_SECRET", "integration-test-secret")
        client = TestClient(app)
        resp = client.get(
            "/api/account", headers={"X-API-Key": "integration-test-secret"},
        )
        assert resp.status_code == 200

    def test_auth_rejected_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from src.web.api import app
        monkeypatch.setenv("WEB_API_SECRET", "integration-test-secret")
        client = TestClient(app)
        resp = client.get("/api/account")
        assert resp.status_code == 403

    def test_query_param_secret_rejected(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CL-pu7i: the legacy ?secret= query param must not authenticate."""
        from fastapi.testclient import TestClient

        from src.web.api import app
        monkeypatch.setenv("WEB_API_SECRET", "integration-test-secret")
        client = TestClient(app)
        resp = client.get("/api/account?secret=integration-test-secret")
        assert resp.status_code == 403
