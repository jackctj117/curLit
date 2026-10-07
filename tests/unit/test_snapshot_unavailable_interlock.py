"""CL-oqos: an unknown broker/account snapshot is never "flat".

Covers every caller of the broker position snapshot that can confirm/reject
pending entries or classify orphans (cold-start ``reconcile``, periodic
``check_alignment``, the event book's per-tick ``reconcile`` →
``confirm_entries``), the visible-reason surface (``/api/system`` and the
fleet watchdog's halt page), and verified recovery (the account-read halt
stays until an explicit operator resume).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import src.web.api as api
from src.execution.broker import Position
from src.execution.oms import OrderManager
from src.monitoring.fleet_watch import decide_halt_alert, describe_halt_causes
from src.portfolio.reconciler import PositionReconciler, SnapshotUnavailableError
from src.risk.kill_switches import KillSwitchManager
from src.runtime.live_engine import LiveEngine
from src.strategies.event_book import EventBook, EventPosition
from src.strategies.event_driven import EventDrivenStrategy

pytestmark = pytest.mark.unit

SECRET = "a-real-secret-value"
AUTH = {"X-API-Key": SECRET}


class _Strategy:
    id = "event_driven"
    state = None

    def __init__(self, book: Any) -> None:
        self.book = book

    @property
    def open_positions(self) -> Any:
        return self.book.held_positions()


def _book(tmp_path: Path, grace: int = 120) -> EventBook:
    return EventBook(
        state_path=str(tmp_path / "book.json"),
        max_loss_pct=0.5,
        per_instrument_max_pct=1.0,
        haven_max_pct=1.0,
        max_holding_hours=48,
        reconcile_grace_sec=grace,
    )


def _pending(book: EventBook, submitted: datetime) -> None:
    book.record_entry(
        EventPosition(
            symbol="USD_CAD",
            event_id=1,
            entry_ts=submitted,
            entry_price=1.36,
            quantity=-500.0,
            direction=-1,
            stop_price=1.40,
            headline="x",
        ),
        [],  # broker readable and flat at submit → baseline 0
        submitted,
    )


_MALFORMED: list[Any] = [
    None,
    {"USD_CAD": -500.0},
    [Position("USD_CAD", float("nan"), 1.36)],
    [Position("USD_CAD", float("inf"), 1.36)],
    [Position("USD_CAD", True, 1.36)],  # type: ignore[arg-type]
    [Position("USD_CAD", -500.0, 1.36), Position("USDCAD", -500.0, 1.36)],
    [Position("", -500.0, 1.36)],
    [object()],
]


# ------------------------------------------------- cold-start reconcile


@pytest.mark.parametrize("snapshot", _MALFORMED)
def test_cold_start_malformed_snapshot_is_typed_unavailable(tmp_path: Path, snapshot: Any) -> None:
    book = _book(tmp_path)
    _pending(book, datetime.now(UTC) - timedelta(hours=1))  # past grace
    broker = Mock()
    broker.get_positions.return_value = snapshot
    oms, journal = Mock(), Mock()
    recon = PositionReconciler(broker, oms, Mock(), [_Strategy(book)], journal=journal)
    with pytest.raises(SnapshotUnavailableError) as info:
        recon.reconcile()
    assert info.value.reason and info.value.at.tzinfo is not None
    assert "USD_CAD" in book.pending_entries  # never rejected against fake flat
    oms.submit_intent.assert_not_called()  # no orphan flatten
    journal.record.assert_not_called()


def test_cold_start_broker_exception_is_typed_and_runtime_error() -> None:
    broker = Mock()
    broker.get_positions.side_effect = OSError("broker offline")
    recon = PositionReconciler(broker, Mock(), Mock(), [SimpleNamespace(id="s", book=Mock())])
    with pytest.raises(RuntimeError, match="snapshot unavailable") as info:
        recon.reconcile()
    assert isinstance(info.value, SnapshotUnavailableError)
    assert "OSError" in info.value.reason


def test_confirm_entries_failure_aborts_before_orphan_flatten() -> None:
    # A book that cannot confirm its pending legs has UNKNOWN ownership: the
    # real broker fill must not be classified orphaned and flattened.
    book = Mock()
    book.confirm_entries.side_effect = ValueError("book broke")
    book.held_positions.return_value = {}
    broker = Mock()
    broker.get_positions.return_value = [Position("USD_CAD", -500.0, 1.36)]
    oms = Mock()
    strat = SimpleNamespace(id="event_driven", book=book, state=None, open_positions={})
    recon = PositionReconciler(broker, oms, None, [strat])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="pending-entry confirmation failed"):
        recon.reconcile()
    oms.submit_intent.assert_not_called()


# ------------------------------------------------- periodic alignment


@pytest.mark.parametrize("snapshot", _MALFORMED)
def test_alignment_malformed_snapshot_is_unknown_not_mismatch(
    tmp_path: Path, snapshot: Any
) -> None:
    book = _book(tmp_path)
    _pending(book, datetime.now(UTC) - timedelta(hours=1))
    broker = Mock()
    broker.get_positions.return_value = snapshot
    oms = Mock()
    recon = PositionReconciler(broker, oms, Mock(), [_Strategy(book)])
    assert recon.check_alignment() is None
    assert "USD_CAD" in book.pending_entries
    oms.submit_intent.assert_not_called()


def test_engine_alignment_unavailable_keeps_mismatch_unknown_and_visible() -> None:
    engine = LiveEngine([], Mock(), Mock(), cold_start_reconciler=Mock())
    engine._record_alignment_report(None)
    assert engine._current_position_mismatch() is None
    recon = engine.safety_status()["last_reconciliation"]
    assert recon["source"] == "alignment" and recon["status"] == "unavailable"


# ------------------------------------------------- event book per-tick path


@pytest.mark.parametrize("snapshot", _MALFORMED)
def test_event_book_tick_malformed_snapshot_confirms_and_prunes_nothing(
    tmp_path: Path, snapshot: Any
) -> None:
    # Codex r1: {} read as flat (reject), empty symbol (reject) and duplicate
    # dialect rows (promote at double size) all reached confirm_entries.
    book = _book(tmp_path, grace=60)
    _pending(book, datetime.now(UTC) - timedelta(hours=1))
    broker = Mock()
    broker.get_positions.return_value = snapshot
    now = datetime.now(UTC)
    # EventDrivenStrategy.generate_intents: confirm only on a non-None snapshot.
    assert book.reconcile(broker, now) is None
    assert "USD_CAD" in book.pending_entries
    assert "USD_CAD" not in book.open_positions


@pytest.mark.parametrize("qty", [float("nan"), float("inf"), True])
def test_confirm_entries_never_rejects_on_unreadable_qty(tmp_path: Path, qty: Any) -> None:
    # Defense in depth for a caller handing confirm_entries a raw snapshot.
    book = _book(tmp_path, grace=60)
    _pending(book, datetime.now(UTC) - timedelta(hours=1))
    book.confirm_entries([Position("USD_CAD", qty, 1.36)], datetime.now(UTC))
    assert "USD_CAD" in book.pending_entries
    assert "USD_CAD" not in book.open_positions


@pytest.mark.parametrize("snapshot", _MALFORMED)
def test_entry_baseline_from_malformed_snapshot_is_none(snapshot: Any) -> None:
    broker = Mock()
    broker.get_positions.return_value = snapshot
    assert EventDrivenStrategy._fetch_entry_baseline(broker) is None


def test_event_book_valid_snapshot_still_confirms(tmp_path: Path) -> None:
    book = _book(tmp_path, grace=60)
    _pending(book, datetime.now(UTC) - timedelta(seconds=5))
    broker = Mock()
    broker.get_positions.return_value = [Position("USD_CAD", -500.0, 1.36)]
    now = datetime.now(UTC)
    snapshot = book.reconcile(broker, now)
    assert snapshot is not None
    book.confirm_entries(snapshot, now)
    assert book.open_positions["USD_CAD"].quantity == -500.0


def test_event_book_tick_unreadable_broker_confirms_nothing(tmp_path: Path) -> None:
    book = _book(tmp_path, grace=60)
    _pending(book, datetime.now(UTC) - timedelta(hours=1))
    broker = Mock()
    broker.get_positions.side_effect = OSError("down")
    assert book.reconcile(broker, datetime.now(UTC)) is None
    assert "USD_CAD" in book.pending_entries


# ------------------------------------------------- engine + visible reason


def _mgr() -> tuple[KillSwitchManager, OrderManager, Mock]:
    broker = Mock()
    broker.get_positions.return_value = []
    oms = OrderManager(broker)
    return KillSwitchManager(broker, oms, {}, trailing_state_path=None), oms, broker


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("WEB_API_SECRET", SECRET)
    monkeypatch.delenv("CURLIT_START_ENTRY_PAUSED", raising=False)
    saved = dict(api._runtime)
    yield
    api._runtime.clear()
    api._runtime.update(saved)


def test_cold_start_unavailable_halts_with_visible_reason(tmp_path: Path, wired: None) -> None:
    mgr, oms, broker = _mgr()
    book = _book(tmp_path)
    _pending(book, datetime.now(UTC) - timedelta(hours=1))
    broker.get_positions.side_effect = OSError("broker offline")
    recon = PositionReconciler(broker, oms, Mock(), [_Strategy(book)])
    engine = LiveEngine([], oms, broker, cold_start_reconciler=recon, kill_switch_manager=mgr)
    engine._reconcile_startup()
    assert oms.is_halted
    assert "external:cold_start_snapshot_unavailable" in mgr.active_halt_causes()
    assert "USD_CAD" in book.pending_entries

    api.set_runtime(broker, oms, [], kill_switch_manager=mgr, safety_status=engine.safety_status)
    body = TestClient(api.app).get("/api/system", headers=AUTH).json()
    # Pre-existing fields unchanged.
    assert body["engine"] == "running" and body["oms_wired"] is True
    assert body["oms_halted"] is True and body["kill_switch_manager_wired"] is True
    assert "account_halt" in body
    # Additive CL-oqos fields.
    assert body["halt_causes"] == ["external:cold_start_snapshot_unavailable"]
    assert body["account_read_failures"] == 0
    recon_status = body["last_reconciliation"]
    assert recon_status["status"] == "unavailable"
    assert recon_status["source"] == "cold_start"
    assert "OSError" in recon_status["reason"]
    assert datetime.fromisoformat(recon_status["at"]).tzinfo is not None

    page = describe_halt_causes(body)
    assert page is not None and "external:cold_start_snapshot_unavailable" in page
    msg, halted = decide_halt_alert(True, False, page)
    assert halted and msg is not None and "cold_start_snapshot_unavailable" in msg


def test_api_system_unwired_reason_fields_are_none(wired: None) -> None:
    api._runtime.clear()
    api._runtime.update(
        {"broker": None, "oms": None, "strategies": [], "kill_switch_manager": None}
    )
    body = TestClient(api.app).get("/api/system", headers=AUTH).json()
    assert body["halt_causes"] is None
    assert body["account_read_failures"] is None
    assert body["last_reconciliation"] is None


def test_account_read_halt_needs_operator_resume_after_recovery(wired: None) -> None:
    mgr, oms, broker = _mgr()
    engine = LiveEngine([], oms, broker, kill_switch_manager=mgr)
    api.set_runtime(broker, oms, [], kill_switch_manager=mgr, safety_status=engine.safety_status)
    client = TestClient(api.app)

    broker.get_account.side_effect = RuntimeError("account unavailable")
    engine._health_tick()
    engine._health_tick()
    body = client.get("/api/system", headers=AUTH).json()
    assert body["account_read_failures"] == 2 and body["oms_halted"] is False
    engine._health_tick()
    body = client.get("/api/system", headers=AUTH).json()
    assert body["oms_halted"] is True
    assert body["account_read_failures"] == 3
    assert body["halt_causes"] == ["external:account_snapshot_unavailable"]
    page = describe_halt_causes(body)
    assert page is not None
    assert "account_snapshot_unavailable" in page and "x3" in page

    # Account reads recover: counter resets, but the halt holds — neither the
    # health tick's auto-resume nor several clean ticks lift it.
    broker.get_account.side_effect = None
    broker.get_account.return_value = SimpleNamespace(equity=100_000.0)
    for _ in range(3):
        engine._health_tick()
    body = client.get("/api/system", headers=AUTH).json()
    assert body["account_read_failures"] == 0
    assert body["oms_halted"] is True
    assert body["halt_causes"] == ["external:account_snapshot_unavailable"]

    # Explicit operator resume (CL-d7ex) is the only way out.
    assert client.post("/api/system/resume", headers=AUTH).status_code == 200
    body = client.get("/api/system", headers=AUTH).json()
    assert body["oms_halted"] is False and body["halt_causes"] == []
    engine._health_tick()  # healthy read after resume: no account cause re-recorded
    assert "external:account_snapshot_unavailable" not in mgr.active_halt_causes()


def test_resume_while_account_still_unreadable_rehalts() -> None:
    mgr, oms, broker = _mgr()
    engine = LiveEngine([], oms, broker, kill_switch_manager=mgr)
    broker.get_account.side_effect = RuntimeError("account unavailable")
    for _ in range(3):
        engine._health_tick()
    mgr.reset_daily()
    oms.resume_trades()
    engine._health_tick()  # 4th consecutive failure: still >= 3
    assert oms.is_halted
    assert "external:account_snapshot_unavailable" in mgr.active_halt_causes()


# ------------------------------------------------- watchdog decision text


def test_describe_halt_causes_handles_old_and_partial_payloads() -> None:
    assert describe_halt_causes(None) is None
    assert describe_halt_causes({"oms_halted": True}) is None  # pre-CL-oqos engine
    assert (
        describe_halt_causes(
            {"halt_causes": [], "account_read_failures": 0, "last_reconciliation": None}
        )
        is None
    )
    text = describe_halt_causes(
        {
            "halt_causes": ["stale_prices"],
            "account_read_failures": True,  # bool is not a count
            "last_reconciliation": {
                "source": "alignment",
                "status": "mismatch",
                "reason": None,
                "at": "2026-10-06T00:00:00+00:00",
            },
        }
    )
    assert text == (
        "causes: stale_prices; last reconciliation mismatch [alignment @ 2026-10-06T00:00:00+00:00]"
    )
