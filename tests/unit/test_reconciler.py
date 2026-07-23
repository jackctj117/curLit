"""Unit tests — portfolio.reconciler: cold-start position reconciliation (CL-sp3r)."""

from __future__ import annotations

from typing import Any

import pytest

from src.execution.broker import Position
from src.execution.oms import OrderIntent
from src.execution.paper_broker import PaperBroker
from src.portfolio.reconciler import (
    PositionReconciler,
    ReconciliationPolicy,
    ReconciliationStatus,
)

# =============================================================================
# Test fakes
# =============================================================================


class _RecordingOMS:
    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    def submit_intent(self, intent: OrderIntent) -> str:
        self.submitted.append(intent)
        return intent.intent_id

    def halt_new_trades(self) -> None:
        pass


class _FakeStateStore:
    """State store that returns canned per-strategy current positions."""

    def __init__(self, positions: dict[str, dict[str, Any] | None]) -> None:
        # positions: strategy_id → position dict (or None for flat)
        self._positions = positions

    def get_current_position(self, strategy_id: str) -> dict[str, Any] | None:
        return self._positions.get(strategy_id)


class _StrategyDouble:
    def __init__(self, sid: str) -> None:
        self.id = sid


# =============================================================================
# Builders
# =============================================================================


def _make_broker_with(
    positions: list[Position] | None = None,
    capital: float = 100_000.0,
) -> PaperBroker:
    broker = PaperBroker(initial_capital=capital)
    broker.set_price("EURUSD", 1.0999, 1.1001)
    broker.set_price("USDJPY", 149.99, 150.01)
    broker.set_price("GBPUSD", 1.2499, 1.2501)
    if positions:
        for p in positions:
            broker._positions[p.symbol] = p  # type: ignore[attr-defined]
    return broker


# =============================================================================
# Classification tests (one per outcome)
# =============================================================================


class TestReconciliationOutcomes:
    def test_matched_when_sizes_align(self) -> None:
        broker = _make_broker_with([
            Position(symbol="EURUSD", quantity=1000.0, avg_price=1.10),
        ])
        state = _FakeStateStore({
            "s1": {"symbol": "EURUSD", "size": 1000.0},
        })
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.reconcile()
        statuses = [e.status for e in report.entries]
        assert ReconciliationStatus.MATCHED in statuses
        assert not report.has_mismatches

    def test_underscore_broker_symbol_matches_canonical_internal(self) -> None:
        # CL-n5xk (P0): a paper broker holds an event leg in OANDA-underscore
        # form (USD_CAD); the strategy claims the same leg. Both sides must
        # canonicalize to USDCAD and MATCH — NOT be double-orphaned and
        # flattened (the false-orphan-flatten money-path bug).
        broker = _make_broker_with([
            Position(symbol="USD_CAD", quantity=-500.0, avg_price=1.36),
        ])
        state = _FakeStateStore({"s1": {"symbol": "USD_CAD", "size": -500.0}})
        oms = _RecordingOMS()
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.reconcile()
        statuses = [e.status for e in report.entries]
        assert ReconciliationStatus.MATCHED in statuses
        assert ReconciliationStatus.ORPHANED_BROKER not in statuses
        assert ReconciliationStatus.ORPHANED_INTERNAL not in statuses
        assert oms.submitted == []  # nothing falsely flattened

    def test_size_mismatch_detected(self) -> None:
        broker = _make_broker_with([
            Position(symbol="EURUSD", quantity=1500.0, avg_price=1.10),
        ])
        state = _FakeStateStore({
            "s1": {"symbol": "EURUSD", "size": 1000.0},
        })
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.reconcile()
        mismatch = [e for e in report.entries if e.status == ReconciliationStatus.SIZE_MISMATCH]
        assert len(mismatch) == 1
        assert mismatch[0].broker_quantity == pytest.approx(1500.0)
        assert mismatch[0].internal_quantity == pytest.approx(1000.0)

    def test_orphaned_broker_detected(self) -> None:
        # Broker has a position no strategy claims.
        broker = _make_broker_with([
            Position(symbol="USDJPY", quantity=500.0, avg_price=150.0),
        ])
        state = _FakeStateStore({"s1": None})  # s1 is flat
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.reconcile()
        orphans = report.by_status(ReconciliationStatus.ORPHANED_BROKER)
        assert len(orphans) == 1
        assert orphans[0].symbol == "USDJPY"
        assert orphans[0].broker_quantity == pytest.approx(500.0)
        assert orphans[0].contributing_strategies == []

    def test_orphaned_internal_detected(self) -> None:
        # Internal claims a position broker doesn't have.
        broker = _make_broker_with([])
        state = _FakeStateStore({
            "s1": {"symbol": "EURUSD", "size": 1000.0},
        })
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.reconcile()
        orphans = report.by_status(ReconciliationStatus.ORPHANED_INTERNAL)
        assert len(orphans) == 1
        assert orphans[0].symbol == "EURUSD"
        assert "s1" in orphans[0].contributing_strategies


# =============================================================================
# Multi-strategy aggregation
# =============================================================================


class TestMultiStrategyAggregation:
    def test_two_strategies_share_symbol_aggregate_to_match_broker(self) -> None:
        broker = _make_broker_with([
            Position(symbol="EURUSD", quantity=1500.0, avg_price=1.10),
        ])
        state = _FakeStateStore({
            "s1": {"symbol": "EURUSD", "size": 1000.0},
            "s2": {"symbol": "EURUSD", "size": 500.0},
        })
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1"), _StrategyDouble("s2")],
        )
        report = recon.reconcile()
        eur = [e for e in report.entries if e.symbol == "EURUSD"]
        assert len(eur) == 1
        assert eur[0].status == ReconciliationStatus.MATCHED
        assert set(eur[0].contributing_strategies) == {"s1", "s2"}

    def test_two_strategies_offset_each_other(self) -> None:
        # s1 long 1000, s2 short 1000 → net 0; broker must be flat to match.
        broker = _make_broker_with([])
        state = _FakeStateStore({
            "s1": {"symbol": "EURUSD", "size": 1000.0},
            "s2": {"symbol": "EURUSD", "size": -1000.0},
        })
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1"), _StrategyDouble("s2")],
        )
        report = recon.reconcile()
        # Net internal qty = 0 → no entry should be created (no symbol claimed).
        eur_entries = [e for e in report.entries if e.symbol == "EURUSD"]
        # Symbol may show up because internal_positions has it; but classified as
        # matched-flat (both sides effectively flat via netting).
        if eur_entries:
            assert eur_entries[0].status == ReconciliationStatus.MATCHED


# =============================================================================
# Policy actions
# =============================================================================


class TestPolicyActions:
    def test_orphaned_broker_flatten_submits_zero_intent(self) -> None:
        broker = _make_broker_with([
            Position(symbol="USDJPY", quantity=500.0, avg_price=150.0),
        ])
        oms = _RecordingOMS()
        state = _FakeStateStore({"s1": None})
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
            policy=ReconciliationPolicy(on_orphaned_broker="flatten"),
        )
        report = recon.reconcile()
        # Flatten intent submitted to OMS.
        assert len(oms.submitted) == 1
        assert oms.submitted[0].symbol == "USDJPY"
        assert oms.submitted[0].target_position == 0.0
        assert any("flattened" in a for a in report.actions_taken)

    def test_orphaned_broker_hold_does_not_submit(self) -> None:
        broker = _make_broker_with([
            Position(symbol="USDJPY", quantity=500.0, avg_price=150.0),
        ])
        oms = _RecordingOMS()
        state = _FakeStateStore({"s1": None})
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
            policy=ReconciliationPolicy(on_orphaned_broker="hold"),
        )
        report = recon.reconcile()
        assert oms.submitted == []
        assert any("held" in a for a in report.actions_taken)

    def test_orphaned_broker_alert_only_logs_only(self) -> None:
        broker = _make_broker_with([
            Position(symbol="USDJPY", quantity=500.0, avg_price=150.0),
        ])
        oms = _RecordingOMS()
        state = _FakeStateStore({"s1": None})
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
            policy=ReconciliationPolicy(on_orphaned_broker="alert_only"),
        )
        report = recon.reconcile()
        assert oms.submitted == []
        assert any("alert" in a for a in report.actions_taken)

    def test_invalid_policy_raises(self) -> None:
        with pytest.raises(AssertionError):
            ReconciliationPolicy(on_orphaned_broker="bogus")

    def test_orphaned_internal_does_not_emit_orders(self) -> None:
        # Internal-state cleanup is alert-only — never emits orders to broker.
        broker = _make_broker_with([])
        oms = _RecordingOMS()
        state = _FakeStateStore({
            "s1": {"symbol": "EURUSD", "size": 1000.0},
        })
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.reconcile()
        assert oms.submitted == []
        assert any("internal claim" in a for a in report.actions_taken)


# =============================================================================
# Robustness
# =============================================================================


class TestRobustness:
    def test_state_exception_on_one_strategy_does_not_stop_others(self) -> None:
        class _RaisingState:
            def get_current_position(self, sid: str) -> dict[str, Any] | None:
                if sid == "broken":
                    raise RuntimeError("DB down for this strategy")
                return None

        broker = _make_broker_with([
            Position(symbol="EURUSD", quantity=1000.0, avg_price=1.10),
        ])
        recon = PositionReconciler(
            broker, _RecordingOMS(), _RaisingState(),  # type: ignore[arg-type]
            strategies=[_StrategyDouble("broken"), _StrategyDouble("good")],
        )
        # Should still produce a report (broker positions visible).
        report = recon.reconcile()
        # EURUSD shows up as orphaned_broker because no good state record.
        assert any(e.symbol == "EURUSD" for e in report.entries)

    def test_broker_exception_returns_empty(self) -> None:
        class _BrokenBroker:
            def get_positions(self) -> list[Position]:
                raise RuntimeError("broker offline")

            def get_account(self) -> Any:
                raise RuntimeError

            def get_price(self, sym: str) -> tuple[float, float]:
                raise RuntimeError

            def place_order(self, order: Any) -> Any:
                raise RuntimeError

            def cancel_order(self, oid: str) -> bool:
                return False

            def get_order(self, oid: str) -> Any:
                raise RuntimeError

            async def stream_prices(self, symbols: list[str]) -> Any:
                raise RuntimeError

        recon = PositionReconciler(
            _BrokenBroker(), _RecordingOMS(),  # type: ignore[arg-type]
            _FakeStateStore({"s1": None}),  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.reconcile()
        # No entries because broker had no positions and internal is flat.
        assert report.entries == []

    def test_no_strategies_rejected(self) -> None:
        broker = _make_broker_with([])
        with pytest.raises(AssertionError):
            PositionReconciler(
                broker, _RecordingOMS(), _FakeStateStore({}),  # type: ignore[arg-type]
                strategies=[],
            )


# =============================================================================
# Multi-position strategy books (CL-8s1e)
# =============================================================================


class _BookStrategyDouble:
    """Strategy double with an event_driven-style multi-position book."""

    def __init__(self, sid: str, book: dict[str, Any]) -> None:
        self.id = sid
        self.open_positions = book


class _BookPos:
    def __init__(self, quantity: float) -> None:
        self.quantity = quantity


class TestMultiPositionBook:
    def test_book_legs_match_broker_and_are_not_flattened(self) -> None:
        """The live incident: event legs held at the broker but absent from
        the single-position store were flattened as orphaned_broker. With the
        book consulted, they MATCH (note the underscore normalization:
        USD_JPY -> USDJPY) and the flatten never fires."""
        broker = _make_broker_with([
            Position(symbol="USDJPY", quantity=-500.0, avg_price=150.0),
            Position(symbol="EURUSD", quantity=1000.0, avg_price=1.10),
        ])
        state = _FakeStateStore({"ev": None})  # store knows nothing
        oms = _RecordingOMS()
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_BookStrategyDouble("ev", {
                "USD_JPY": _BookPos(-500.0),
                "EUR_USD": _BookPos(1000.0),
            })],
        )
        report = recon.reconcile()
        assert not report.has_mismatches
        assert all(
            e.status == ReconciliationStatus.MATCHED for e in report.entries
        )
        assert oms.submitted == []  # nothing flattened

    def test_store_position_suppresses_book_double_count(self) -> None:
        """A strategy present in the store must not ALSO contribute its book
        (double counting would misreport a size mismatch)."""
        broker = _make_broker_with([
            Position(symbol="EURUSD", quantity=1000.0, avg_price=1.10),
        ])
        state = _FakeStateStore({
            "s1": {"symbol": "EURUSD", "size": 1000.0},
        })
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_BookStrategyDouble("s1", {
                "EUR_USD": _BookPos(1000.0),  # same position, book form
            })],
        )
        report = recon.reconcile()
        assert not report.has_mismatches  # 1000 vs 1000, not 2000 vs 1000

    def test_true_orphan_still_flattened(self) -> None:
        """The safety net stays intact: a broker position in NO store and NO
        book is still classified orphaned_broker and flattened."""
        broker = _make_broker_with([
            Position(symbol="GBPUSD", quantity=700.0, avg_price=1.25),
        ])
        state = _FakeStateStore({"ev": None})
        oms = _RecordingOMS()
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_BookStrategyDouble("ev", {"USD_JPY": _BookPos(-1.0)})],
            policy=ReconciliationPolicy(on_orphaned_broker="flatten"),
        )
        report = recon.reconcile()
        statuses = {e.symbol: e.status for e in report.entries}
        assert statuses["GBPUSD"] == ReconciliationStatus.ORPHANED_BROKER
        assert any(i.symbol == "GBPUSD" for i in oms.submitted)  # flattened

    def test_empty_or_missing_book_is_harmless(self) -> None:
        broker = _make_broker_with([])
        state = _FakeStateStore({"a": None, "b": None})
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[
                _BookStrategyDouble("a", {}),
                _StrategyDouble("b"),  # no open_positions attr at all
            ],
        )
        report = recon.reconcile()
        assert report.entries == []


# =============================================================================
# check_alignment — classification-only periodic pass (CL-i4tx)
# =============================================================================


class TestCheckAlignment:
    """check_alignment feeds the reconciliation_failure kill switch: same
    classification as reconcile(), but NO policy actions (nothing flattened
    mid-session) and None — not a mismatch — when the broker is unreachable."""

    def test_aligned_book_reports_clean(self) -> None:
        broker = _make_broker_with([
            Position(symbol="EURUSD", quantity=1000.0, avg_price=1.10),
        ])
        state = _FakeStateStore({"s1": {"symbol": "EURUSD", "size": 1000.0}})
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        report = recon.check_alignment()
        assert report is not None
        assert not report.has_mismatches

    def test_mismatch_detected_without_actions(self) -> None:
        # Orphaned broker position: reconcile() would flatten it; the
        # periodic alignment check must ONLY report it.
        broker = _make_broker_with([
            Position(symbol="GBPUSD", quantity=700.0, avg_price=1.25),
        ])
        state = _FakeStateStore({"s1": None})
        oms = _RecordingOMS()
        recon = PositionReconciler(
            broker, oms, state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
            policy=ReconciliationPolicy(on_orphaned_broker="flatten"),
        )
        report = recon.check_alignment()
        assert report is not None
        assert report.has_mismatches
        assert report.actions_taken == []
        assert oms.submitted == []  # nothing flattened

    def test_broker_failure_returns_none_not_mismatch(self) -> None:
        class _DeadBroker:
            def get_positions(self):  # noqa: ANN202
                raise ConnectionError("stream down")

        state = _FakeStateStore({"s1": {"symbol": "EURUSD", "size": 1000.0}})
        recon = PositionReconciler(
            _DeadBroker(), _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_StrategyDouble("s1")],
        )
        assert recon.check_alignment() is None

    def test_multi_leg_book_matches_across_dialects(self) -> None:
        # Event book keys OANDA-underscore; broker keys compact — the
        # alignment pass must use the same canonicalization as cold start.
        broker = _make_broker_with([
            Position(symbol="USDJPY", quantity=-500.0, avg_price=150.0),
        ])
        state = _FakeStateStore({"ev": None})
        recon = PositionReconciler(
            broker, _RecordingOMS(), state,  # type: ignore[arg-type]
            strategies=[_BookStrategyDouble("ev", {"USD_JPY": _BookPos(-500.0)})],
        )
        report = recon.check_alignment()
        assert report is not None
        assert not report.has_mismatches
