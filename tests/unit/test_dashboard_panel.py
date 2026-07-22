"""Unit tests for approvals-panel helpers + dashboard endpoints (CL-7t8d).

Covers:

  * list_pending_approvals flattens GATE 1 + GATE 2 pending entries
    (and ignores non-pending ones)
  * apply_decision validates inputs, mutates state, appends to log
  * apply_decision raises DecisionError on bad gate/action/already-
    decided cases
  * /api/approvals lists pending entries via FastAPI TestClient
  * /api/approvals/{slug} POST flows through apply_decision and
    enforces the WEB_API_SECRET
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.research.dashboard_panel import (
    DecisionError,
    apply_decision,
    list_pending_approvals,
)
from src.research.loop import LoopState, save_state


def _state_with(
    *,
    ideas: dict[str, dict] | None = None,
    debates: dict[str, dict] | None = None,
) -> LoopState:
    return LoopState(
        ideas_processed=dict(ideas or {}),
        debates_completed=dict(debates or {}),
    )


# --------------------------------------------------------------------- #
# list_pending_approvals
# --------------------------------------------------------------------- #


class TestListPending:
    def test_flattens_both_gates(self) -> None:
        state = _state_with(
            ideas={
                "h1": {
                    "status": "PENDING_OPERATOR_APPROVAL",
                    "slug": "alpha",
                    "pending_since": "2026-04-28T00:00:00",
                    "hypothesis_path": "docs/research/hypotheses/alpha.md",
                },
            },
            debates={
                "beta": {
                    "verdict": "PROMOTE",
                    "deploy_status": "PENDING_DEPLOY_CONFIRMATION",
                    "pending_since": "2026-04-28T01:00:00",
                    "transcript_path": "docs/research/debates/beta/transcript.md",
                    "candidate_report_path": "reports/candidates/beta.json",
                    "bull": "PROMOTE",
                    "bear": "PROMOTE",
                    "reason": "all gates pass",
                },
            },
        )
        out = list_pending_approvals(state)
        assert len(out) == 2
        gate1 = next(e for e in out if e.gate == 1)
        assert gate1.slug == "alpha"
        assert "hypothesis_path" in gate1.extra
        gate2 = next(e for e in out if e.gate == 2)
        assert gate2.slug == "beta"
        assert gate2.extra["bull"] == "PROMOTE"

    def test_skips_non_pending(self) -> None:
        state = _state_with(
            ideas={
                "h1": {"status": "APPROVED", "slug": "alpha"},
                "h2": {"status": "DECLINED", "slug": "beta"},
            },
            debates={
                "g": {"verdict": "PROMOTE", "deploy_status": "DEPLOYED"},
                "h": {"verdict": "REJECT"},
            },
        )
        assert list_pending_approvals(state) == []

    def test_to_json_includes_extras(self) -> None:
        state = _state_with(
            ideas={
                "h1": {
                    "status": "PENDING_OPERATOR_APPROVAL",
                    "slug": "alpha",
                    "pending_since": "2026-04-28T00:00:00",
                },
            },
        )
        json_form = list_pending_approvals(state)[0].to_json()
        assert json_form["gate"] == 1
        assert json_form["slug"] == "alpha"


# --------------------------------------------------------------------- #
# apply_decision
# --------------------------------------------------------------------- #


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "decisions.log"


class TestApplyDecisionGate1:
    def test_approve_gate1(self, state_path: Path, log_path: Path) -> None:
        save_state(_state_with(
            ideas={"h1": {"status": "PENDING_OPERATOR_APPROVAL", "slug": "alpha"}},
        ), state_path)
        result = apply_decision(
            state_path=state_path, gate=1, slug="alpha",
            action="APPROVE", reason="looks reasonable",
            decisions_log=log_path,
        )
        assert result["new_status"] == "APPROVED"
        # State persisted
        from src.research.loop import load_state
        state = load_state(state_path)
        assert state.ideas_processed["h1"]["status"] == "APPROVED"
        # Log written
        log_text = log_path.read_text()
        assert "GATE1" in log_text
        assert "APPROVE" in log_text
        assert "alpha" in log_text
        assert "looks reasonable" in log_text

    def test_reject_gate1(self, state_path: Path, log_path: Path) -> None:
        save_state(_state_with(
            ideas={"h1": {"status": "PENDING_OPERATOR_APPROVAL", "slug": "alpha"}},
        ), state_path)
        apply_decision(
            state_path=state_path, gate=1, slug="alpha",
            action="REJECT", reason="duplicates X",
            decisions_log=log_path,
        )
        from src.research.loop import load_state
        state = load_state(state_path)
        assert state.ideas_processed["h1"]["status"] == "SKIPPED"
        assert "duplicates X" in state.ideas_processed["h1"]["reason"]


class TestApplyDecisionGate2:
    def test_approve_gate2(self, state_path: Path, log_path: Path) -> None:
        save_state(_state_with(
            debates={"alpha": {
                "verdict": "PROMOTE",
                "deploy_status": "PENDING_DEPLOY_CONFIRMATION",
            }},
        ), state_path)
        apply_decision(
            state_path=state_path, gate=2, slug="alpha", action="APPROVE",
            decisions_log=log_path,
        )
        from src.research.loop import load_state
        state = load_state(state_path)
        assert state.debates_completed["alpha"]["deploy_status"] == (
            "DEPLOY_APPROVED"
        )

    def test_reject_gate2(self, state_path: Path, log_path: Path) -> None:
        save_state(_state_with(
            debates={"alpha": {
                "verdict": "PROMOTE",
                "deploy_status": "PENDING_DEPLOY_CONFIRMATION",
            }},
        ), state_path)
        apply_decision(
            state_path=state_path, gate=2, slug="alpha", action="REJECT",
            reason="not yet", decisions_log=log_path,
        )
        from src.research.loop import load_state
        state = load_state(state_path)
        assert state.debates_completed["alpha"]["deploy_status"] == (
            "DEPLOY_REJECTED"
        )


class TestApplyDecisionErrors:
    def test_unknown_gate(self, state_path: Path, log_path: Path) -> None:
        save_state(LoopState(), state_path)
        with pytest.raises(DecisionError, match="unknown gate"):
            apply_decision(
                state_path=state_path, gate=3, slug="x", action="APPROVE",
                decisions_log=log_path,
            )

    def test_unknown_action(self, state_path: Path, log_path: Path) -> None:
        save_state(LoopState(), state_path)
        with pytest.raises(DecisionError, match="unknown action"):
            apply_decision(
                state_path=state_path, gate=1, slug="x", action="PONDER",
                decisions_log=log_path,
            )

    def test_no_matching_slug_gate1(
        self, state_path: Path, log_path: Path,
    ) -> None:
        save_state(_state_with(
            ideas={"h1": {"status": "PENDING_OPERATOR_APPROVAL",
                          "slug": "other"}},
        ), state_path)
        with pytest.raises(DecisionError, match="no GATE 1 entry"):
            apply_decision(
                state_path=state_path, gate=1, slug="ghost",
                action="APPROVE", decisions_log=log_path,
            )

    def test_already_decided_gate1(
        self, state_path: Path, log_path: Path,
    ) -> None:
        save_state(_state_with(
            ideas={"h1": {"status": "APPROVED", "slug": "alpha"}},
        ), state_path)
        with pytest.raises(DecisionError, match="not 'PENDING_OPERATOR"):
            apply_decision(
                state_path=state_path, gate=1, slug="alpha",
                action="APPROVE", decisions_log=log_path,
            )

    def test_no_matching_slug_gate2(
        self, state_path: Path, log_path: Path,
    ) -> None:
        save_state(LoopState(), state_path)
        with pytest.raises(DecisionError, match="no GATE 2"):
            apply_decision(
                state_path=state_path, gate=2, slug="ghost",
                action="APPROVE", decisions_log=log_path,
            )

    def test_already_deployed_gate2(
        self, state_path: Path, log_path: Path,
    ) -> None:
        save_state(_state_with(
            debates={"alpha": {"verdict": "PROMOTE", "deploy_status": "DEPLOYED"}},
        ), state_path)
        with pytest.raises(DecisionError, match="not 'PENDING_DEPLOY"):
            apply_decision(
                state_path=state_path, gate=2, slug="alpha",
                action="APPROVE", decisions_log=log_path,
            )


# --------------------------------------------------------------------- #
# FastAPI endpoints
# --------------------------------------------------------------------- #


@pytest.fixture
def dashboard_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Boot a TestClient against the soak_dashboard FastAPI app with the
    state file pointed at tmp_path. The dashboard module reads
    DEFAULT_STATE_PATH at request time, so we monkeypatch the constant
    in the dashboard module's namespace."""
    from scripts import soak_dashboard

    state_path = tmp_path / "state.json"
    log_path = tmp_path / "decisions.log"
    monkeypatch.setattr(soak_dashboard, "DEFAULT_STATE_PATH", state_path)
    monkeypatch.setattr(soak_dashboard, "_DECISIONS_LOG", log_path)
    monkeypatch.setenv("WEB_API_SECRET", "test-secret")
    save_state(LoopState(), state_path)
    return TestClient(soak_dashboard.app)


#: Auth is via the X-API-Key header (CL-94n2) — never a query param.
_AUTH = {"X-API-Key": "test-secret"}


class TestApprovalsEndpoints:
    def test_list_empty(self, dashboard_client: TestClient) -> None:
        resp = dashboard_client.get("/api/approvals", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"pending": []}

    def test_list_pending(
        self, dashboard_client: TestClient, tmp_path: Path,
    ) -> None:
        # Seed state via the dashboard's monkey-patched path
        from scripts import soak_dashboard
        save_state(_state_with(
            ideas={"h1": {
                "status": "PENDING_OPERATOR_APPROVAL", "slug": "alpha",
                "pending_since": "2026-04-28T00:00:00",
            }},
        ), soak_dashboard.DEFAULT_STATE_PATH)
        resp = dashboard_client.get("/api/approvals", headers=_AUTH)
        body = resp.json()
        assert len(body["pending"]) == 1
        assert body["pending"][0]["slug"] == "alpha"

    def test_post_approve(
        self, dashboard_client: TestClient,
    ) -> None:
        from scripts import soak_dashboard
        save_state(_state_with(
            ideas={"h1": {"status": "PENDING_OPERATOR_APPROVAL", "slug": "alpha"}},
        ), soak_dashboard.DEFAULT_STATE_PATH)
        resp = dashboard_client.post(
            "/api/approvals/alpha", headers=_AUTH,
            json={"gate": 1, "action": "APPROVE", "reason": "looks good"},
        )
        assert resp.status_code == 200
        assert resp.json()["new_status"] == "APPROVED"
        # decisions.log written
        log_text = soak_dashboard._DECISIONS_LOG.read_text()
        assert "GATE1" in log_text and "APPROVE" in log_text

    def test_post_rejects_bad_secret(
        self, dashboard_client: TestClient,
    ) -> None:
        from scripts import soak_dashboard
        save_state(_state_with(
            ideas={"h1": {"status": "PENDING_OPERATOR_APPROVAL", "slug": "alpha"}},
        ), soak_dashboard.DEFAULT_STATE_PATH)
        resp = dashboard_client.post(
            "/api/approvals/alpha", headers={"X-API-Key": "wrong"},
            json={"gate": 1, "action": "APPROVE"},
        )
        assert resp.status_code == 403

    def test_post_rejects_legacy_query_param_secret(
        self, dashboard_client: TestClient,
    ) -> None:
        """?secret= no longer authenticates anything (CL-94n2)."""
        resp = dashboard_client.post(
            "/api/approvals/alpha?secret=test-secret",
            json={"gate": 1, "action": "APPROVE"},
        )
        assert resp.status_code == 403

    def test_post_rejects_already_decided(
        self, dashboard_client: TestClient,
    ) -> None:
        from scripts import soak_dashboard
        save_state(_state_with(
            ideas={"h1": {"status": "APPROVED", "slug": "alpha"}},
        ), soak_dashboard.DEFAULT_STATE_PATH)
        resp = dashboard_client.post(
            "/api/approvals/alpha", headers=_AUTH,
            json={"gate": 1, "action": "APPROVE"},
        )
        assert resp.status_code == 400
        assert "PENDING_OPERATOR" in resp.json()["detail"]

    def test_post_unknown_gate_returns_400(
        self, dashboard_client: TestClient,
    ) -> None:
        resp = dashboard_client.post(
            "/api/approvals/alpha", headers=_AUTH,
            json={"gate": 9, "action": "APPROVE"},
        )
        assert resp.status_code == 400
