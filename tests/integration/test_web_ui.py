"""Web UI integration tests — API endpoints."""

import pytest


class TestWebAPI:
    def test_health_endpoint(self) -> None:
        from src.web.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/api/account?secret=curlit-dev")
        assert resp.status_code == 200

    def test_auth_rejected_no_secret(self) -> None:
        from src.web.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/api/account")
        assert resp.status_code == 403
