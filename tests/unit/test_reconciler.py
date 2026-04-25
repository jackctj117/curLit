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
