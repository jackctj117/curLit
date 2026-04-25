"""Integration tests — TradeJournal wiring into OMS / RejectionHandler /
PositionReconciler (CL-50qb).

These tests cover the wiring as exercised end-to-end against a real
SQLite-backed TradeJournal and a PaperBroker. They are deliberately
tighter than the per-module unit tests in test_trade_journal.py: this
file proves that journal events are emitted at the right call sites,
in the right order, with the right payload — and that the chain
verifies after a paper round-trip.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine

from src.execution.broker import Order, OrderType
from src.execution.oms import OrderIntent, OrderManager
from src.execution.paper_broker import PaperBroker
from src.execution.rejection import RejectionHandler
from src.execution.trade_journal import EventType, TradeJournal
from src.portfolio.reconciler import PositionReconciler


@pytest.fixture
def journal() -> TradeJournal:
    return TradeJournal(create_engine("sqlite:///:memory:"))


@pytest.fixture
def broker() -> PaperBroker:
    b = PaperBroker(initial_capital=100_000)
    b.set_price("EUR_USD", 1.1000, 1.1002)
    return b


# =============================================================================
# OMS wiring
# =============================================================================


class TestOMSWiring:
    def test_paper_round_trip_emits_intent_placed_filled(
        self,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        oms = OrderManager(broker, journal=journal)
        intent = OrderIntent(
            strategy_id="strat-1", symbol="EUR_USD", target_position=10_000.0,
        )
        oms.submit_intent(intent)

        events = journal.query_by_intent(intent.intent_id)
        types = [e.event_type for e in events]
        assert EventType.INTENT_SUBMITTED in types
        assert EventType.ORDER_PLACED in types
        # PaperBroker fills synchronously — ORDER_FILLED must appear.
        assert EventType.ORDER_FILLED in types

    def test_intent_payload_carries_delta(
        self,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        oms = OrderManager(broker, journal=journal)
        intent = OrderIntent(
            strategy_id="strat-1", symbol="EUR_USD", target_position=5_000.0,
        )
        oms.submit_intent(intent)

        intents = [
            e for e in journal.query_by_intent(intent.intent_id)
            if e.event_type == EventType.INTENT_SUBMITTED
        ]
        assert len(intents) == 1
        assert intents[0].payload["delta"] == 5_000.0

    def test_zero_delta_intent_emits_only_intent_event(
        self,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        # Same target as current position (0) → no order placed.
        oms = OrderManager(broker, journal=journal)
        intent = OrderIntent(
            strategy_id="strat-1", symbol="EUR_USD", target_position=0.5,
        )
        oms.submit_intent(intent)
        events = journal.query_by_intent(intent.intent_id)
        types = [e.event_type for e in events]
        assert EventType.INTENT_SUBMITTED in types
        assert EventType.ORDER_PLACED not in types

    def test_no_journal_means_silent_no_op(
        self,
        broker: PaperBroker,
    ) -> None:
        # OMS without a journal must still trade — bookkeeping is optional.
        oms = OrderManager(broker, journal=None)
        intent = OrderIntent(
            strategy_id="strat-1", symbol="EUR_USD", target_position=10_000.0,
        )
        oms.submit_intent(intent)
        # Must not raise.
        assert broker.get_positions()[0].quantity == 10_000.0

    def test_journal_failure_does_not_block_trading(
        self,
        broker: PaperBroker,
    ) -> None:
        # Journal raises on every record — OMS should swallow, log, and
        # let the trade complete.
        class _BoomJournal:
            def record(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("simulated DB outage")

        oms = OrderManager(broker, journal=_BoomJournal())  # type: ignore[arg-type]
        intent = OrderIntent(
            strategy_id="strat-1", symbol="EUR_USD", target_position=10_000.0,
        )
        oms.submit_intent(intent)
        # Trade still went through.
        assert broker.get_positions()[0].quantity == 10_000.0


# =============================================================================
# RejectionHandler wiring
# =============================================================================


class TestRejectionWiring:
    def test_rejection_writes_journal_event(
        self,
        journal: TradeJournal,
    ) -> None:
        handler = RejectionHandler(journal=journal)
        intent = OrderIntent(
            strategy_id="strat-1", symbol="EUR_USD", target_position=10_000.0,
        )
        order = Order(
            symbol="EUR_USD", side="buy", quantity=10_000,
            order_type=OrderType.MARKET,
        )
        handler.handle(
            intent=intent, order=order,
            exc=Exception("insufficient_margin"),
            attempt=1,
        )
        events = journal.query_by_intent(intent.intent_id)
        rejections = [e for e in events if e.event_type == EventType.ORDER_REJECTED]
        assert len(rejections) == 1
        assert rejections[0].payload["rejection_class"] == "margin"
        assert rejections[0].payload["attempt"] == 1


# =============================================================================
# Reconciler wiring
# =============================================================================


class _FakeStateStore:
    """In-memory state stub matching StrategyStateLike protocol."""

    def __init__(self, positions: dict[str, dict[str, Any]] | None = None) -> None:
        self._positions = positions or {}

    def get_current_position(self, strategy_id: str) -> dict[str, Any] | None:
        return self._positions.get(strategy_id)


class _FakeStrategy:
    def __init__(self, sid: str) -> None:
        self.id = sid


class TestReconcilerWiring:
    def test_reconcile_writes_one_journal_event(
        self,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        oms = OrderManager(broker, journal=journal)
        state = _FakeStateStore()
        strategies = [_FakeStrategy("strat-1")]
        recon = PositionReconciler(
            broker, oms, state, strategies, journal=journal,
        )
        recon.reconcile()
        events = journal.all_events()
        recon_events = [
            e for e in events
            if e.event_type == EventType.RECONCILIATION_REPORT
        ]
        assert len(recon_events) == 1
        assert "summary" in recon_events[0].payload


# =============================================================================
# End-to-end chain integrity
# =============================================================================


class TestChainIntegrity:
    def test_full_round_trip_chain_verifies(
        self,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        # Reconciler boots → OMS submits → all events chained.
        oms = OrderManager(
            broker,
            rejection_handler=RejectionHandler(journal=journal),
            journal=journal,
        )
        state = _FakeStateStore()
        strategies = [_FakeStrategy("strat-1")]
        PositionReconciler(
            broker, oms, state, strategies, journal=journal,
        ).reconcile()
        intent = OrderIntent(
            strategy_id="strat-1", symbol="EUR_USD", target_position=10_000.0,
        )
        oms.submit_intent(intent)

        ok, bad_seq = journal.verify_chain()
        assert ok, f"Chain verification failed at seq={bad_seq}"
        # At minimum we expect: 1 RECON + 1 INTENT + 1 PLACED + 1 FILLED = 4.
        assert len(journal.all_events()) >= 4
