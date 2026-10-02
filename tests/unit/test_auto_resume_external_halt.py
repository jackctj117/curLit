"""Auto-resume must only lift halts it owns (CL-d7ex).

Incident (2026-09-10 / 2026-09-23): the FX engine started entry-paused
(``CURLIT_START_ENTRY_PAUSED=1`` -> ``oms.halt_new_trades()``), then a
``stale_prices`` trip followed by stream recovery made
``KillSwitchManager.attempt_auto_resume`` see a cause set of exactly
``{"stale_prices"}``, drain it, and call ``oms.resume_trades()`` — lifting a
halt it never created. Requirement: a halt whose cause set includes anything
other than a data gate the manager recorded itself stays in force until an
explicit ``/api/system/resume``; a pure data-gate halt still auto-resumes.

Oracle: the REAL ``OrderManager``'s engine-local halt flag, read through
``src.web.api._oms_halted`` — exactly what ``/api/system`` reports as
``oms_halted`` and what the fleet watchdog's ``engine_halted`` reads — plus
whether ``oms.resume_trades`` is ever reached. Neither depends on the cause
bookkeeping under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import src.web.api as api
from src.execution.oms import OrderManager
from src.risk.kill_switches import KillSwitchManager
from src.runtime.live_engine import LiveEngine

pytestmark = pytest.mark.unit

SECRET = "a-real-secret-value"
AUTH = {"X-API-Key": SECRET}

# 700 s > the 600 s stale_prices threshold (_PRICE_STREAM_STALE_SEC); 5 s is
# a healthy stream. Everything else sits well inside every other switch.
_STALE_AGE_SEC = 700
_FRESH_AGE_SEC = 5


def _ctx(price_age: float = _FRESH_AGE_SEC) -> dict[str, object]:
    return {
        "portfolio_dd": -0.05,
        "daily_pnl_pct": 0.01,
        "vix_level": 22,
        "vix_change_1d": 0.1,
        "price_stream_age_sec": price_age,
        "cvix_zscore": 0.5,
        "position_mismatch": False,
    }


class _CountingOMS(OrderManager):
    """Real OrderManager halt semantics; counts resume calls."""

    def __init__(self) -> None:
        broker = Mock()
        broker.get_positions.return_value = []
        super().__init__(broker)
        self.resume_calls = 0

    def resume_trades(self) -> None:
        self.resume_calls += 1
        super().resume_trades()


def _mgr() -> tuple[KillSwitchManager, _CountingOMS]:
    oms = _CountingOMS()
    broker = Mock()
    broker.get_positions.return_value = []
    # Flat equity: keeps equity_trailing_stop quiet (no peak drawdown).
    broker.get_account.return_value = SimpleNamespace(equity=100_000.0)
    return KillSwitchManager(broker, oms, {}, trailing_state_path=None), oms


def _stale_trip_then_recover(mgr: KillSwitchManager) -> bool:
    """Fire stale_prices, then present a recovered stream to auto-resume."""
    fired = {t["switch"] for t in mgr.check(_ctx(_STALE_AGE_SEC))}
    assert "stale_prices" in fired  # precondition: the data gate really tripped
    return mgr.attempt_auto_resume(_ctx(_FRESH_AGE_SEC))


def _engine(
    oms: OrderManager,
    mgr: KillSwitchManager,
    reconciler: Any,
) -> LiveEngine:
    return LiveEngine([], oms, Mock(), cold_start_reconciler=reconciler, kill_switch_manager=mgr)


def _clean_reconciler() -> Mock:
    rec = Mock()
    rec.reconcile.return_value = SimpleNamespace(entries=[], has_mismatches=False)
    return rec


# ------------------------------------------------- pure data gate (unchanged)


def test_pure_data_gate_halt_still_auto_resumes() -> None:
    mgr, oms = _mgr()
    assert api._oms_halted(oms) is False
    assert _stale_trip_then_recover(mgr) is True
    assert oms.resume_calls == 1
    assert api._oms_halted(oms) is False


def test_pure_data_gate_after_clean_startup_auto_resumes(monkeypatch) -> None:
    monkeypatch.delenv("CURLIT_START_ENTRY_PAUSED", raising=False)
    mgr, oms = _mgr()
    _engine(oms, mgr, _clean_reconciler())._reconcile_startup()
    assert api._oms_halted(oms) is False  # clean start: nothing paused
    assert _stale_trip_then_recover(mgr) is True
    assert api._oms_halted(oms) is False


# ------------------------------------------------------ startup halts (bead)


def test_entry_paused_startup_survives_stale_trip_and_recovery(monkeypatch) -> None:
    monkeypatch.setenv("CURLIT_START_ENTRY_PAUSED", "1")
    mgr, oms = _mgr()
    _engine(oms, mgr, _clean_reconciler())._reconcile_startup()
    assert api._oms_halted(oms) is True
    assert _stale_trip_then_recover(mgr) is False
    assert oms.resume_calls == 0
    assert api._oms_halted(oms) is True
    # The data gate itself was still cleared and re-armed.
    assert "stale_prices" not in mgr.active_halt_causes()
    # And it stays halted across repeated trip/recover cycles (9 recurrences).
    for _ in range(3):
        assert _stale_trip_then_recover(mgr) is False
    assert oms.resume_calls == 0
    assert api._oms_halted(oms) is True


@pytest.mark.parametrize("mode", ["mismatch", "raises", "unavailable"])
def test_cold_start_reconciliation_halt_survives_stale_recovery(mode, monkeypatch) -> None:
    monkeypatch.delenv("CURLIT_START_ENTRY_PAUSED", raising=False)
    mgr, oms = _mgr()
    reconciler: Mock | None = None if mode == "unavailable" else Mock()
    if reconciler is not None:
        if mode == "raises":
            reconciler.reconcile.side_effect = OSError("account unavailable")
        else:
            reconciler.reconcile.return_value = SimpleNamespace(
                entries=["mismatch"], has_mismatches=True
            )
    _engine(oms, mgr, reconciler)._reconcile_startup()
    assert api._oms_halted(oms) is True
    assert _stale_trip_then_recover(mgr) is False
    assert oms.resume_calls == 0
    assert api._oms_halted(oms) is True


def test_rollover_does_not_drop_external_cause(monkeypatch) -> None:
    monkeypatch.setenv("CURLIT_START_ENTRY_PAUSED", "1")
    mgr, oms = _mgr()
    _engine(oms, mgr, _clean_reconciler())._reconcile_startup()
    mgr.check(_ctx(_STALE_AGE_SEC))
    mgr.reset_daily(clear_causes=False)  # automatic UTC rollover path
    assert mgr.attempt_auto_resume(_ctx(_FRESH_AGE_SEC)) is False
    assert api._oms_halted(oms) is True


def test_unregistered_prior_halt_is_not_auto_resumed() -> None:
    # A halt applied by some path that never recorded a cause must still not
    # be lifted once a data gate fires on top of it and then clears.
    mgr, oms = _mgr()
    oms.halt_new_trades()
    assert _stale_trip_then_recover(mgr) is False
    assert oms.resume_calls == 0
    assert api._oms_halted(oms) is True


def test_external_halt_recorded_after_auto_resume_reapplies_halt() -> None:
    # Ordering race: auto-resume drained and resumed first, then the operator
    # halt is recorded — the record call itself re-halts the OMS.
    mgr, oms = _mgr()
    assert _stale_trip_then_recover(mgr) is True
    assert api._oms_halted(oms) is False
    mgr.record_external_halt("operator_api_halt")
    assert api._oms_halted(oms) is True
    assert mgr.attempt_auto_resume(_ctx(_FRESH_AGE_SEC)) is False
    assert api._oms_halted(oms) is True


# --------------------------------------------------------- API halt / resume


@pytest.fixture
def wired(monkeypatch) -> Iterator[tuple[KillSwitchManager, _CountingOMS]]:
    monkeypatch.setenv("WEB_API_SECRET", SECRET)
    saved = dict(api._runtime)
    mgr, oms = _mgr()
    api._runtime.update(
        {
            "broker": None,
            "oms": oms,
            "strategies": [],
            "kill_switch_manager": mgr,
            "halt_store": None,
        }
    )
    yield mgr, oms
    api._runtime.clear()
    api._runtime.update(saved)


def test_manual_api_halt_survives_stale_trip_and_recovery(wired) -> None:
    mgr, oms = wired
    client = TestClient(api.app)
    r = client.post("/api/system/halt", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["oms_halted"] is True
    assert _stale_trip_then_recover(mgr) is False
    assert oms.resume_calls == 0
    assert api._oms_halted(oms) is True


def test_manual_api_halt_during_data_gate_halt_survives_recovery(wired) -> None:
    # stale_prices halts first; the operator halts on top of it; the stream
    # recovers — the operator's halt must still hold.
    mgr, oms = wired
    client = TestClient(api.app)
    mgr.check(_ctx(_STALE_AGE_SEC))
    assert client.post("/api/system/halt", headers=AUTH).status_code == 200
    assert mgr.attempt_auto_resume(_ctx(_FRESH_AGE_SEC)) is False
    assert oms.resume_calls == 0
    assert api._oms_halted(oms) is True


def test_explicit_api_resume_clears_everything(wired, monkeypatch) -> None:
    monkeypatch.setenv("CURLIT_START_ENTRY_PAUSED", "1")
    mgr, oms = wired
    client = TestClient(api.app)
    _engine(oms, mgr, Mock(reconcile=Mock(side_effect=OSError("down"))))._reconcile_startup()
    assert client.post("/api/system/halt", headers=AUTH).status_code == 200
    assert _stale_trip_then_recover(mgr) is False
    assert api._oms_halted(oms) is True

    r = client.post("/api/system/resume", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["oms_halted"] is False
    assert api._oms_halted(oms) is False
    assert mgr.active_halt_causes() == frozenset()
    # After the explicit resume, a pure data-gate halt auto-resumes again.
    assert _stale_trip_then_recover(mgr) is True
    assert api._oms_halted(oms) is False


class _ResumeOnRelease:
    """Wraps the manager's cause lock: on the first release it runs a
    manual resume (``reset_daily``) — the deterministic worst case of a
    concurrent /api/system/resume grabbing the lock the instant
    ``record_external_halt`` lets go."""

    def __init__(self, mgr: KillSwitchManager) -> None:
        self._inner = mgr._cause_lock
        self._mgr = mgr
        self.armed = True

    def __enter__(self) -> None:
        self._inner.__enter__()

    def __exit__(self, *exc: object) -> None:
        self._inner.__exit__(*exc)
        if self.armed:
            self.armed = False
            self._mgr.reset_daily()


def test_record_external_halt_tolerates_resume_racing_lock_release() -> None:
    # Codex round 1: a resume clearing the cause right after the lock is
    # released must not turn the operator's halt call into an exception
    # (which would 500 /api/system/halt before the durable halt write).
    mgr, oms = _mgr()
    racer = _ResumeOnRelease(mgr)
    mgr._cause_lock = racer  # type: ignore[assignment]
    mgr.record_external_halt("operator_api_halt")  # must not raise
    assert racer.armed is False  # the racing resume really ran
    assert api._oms_halted(oms) is True  # local brake applied
