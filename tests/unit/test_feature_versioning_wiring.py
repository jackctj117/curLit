"""Integration tests — feature_versioning wiring through OMS journal (CL-xpw9).

Proves the end-to-end loop:
    strategy → builds FeatureSnapshot, attaches snapshot_id to OrderIntent
    OMS    → records INTENT_SUBMITTED with snapshot_id in payload
    journal → can be queried by intent_id
    reconstruct_features → returns the original FeatureSnapshot bit-exactly
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine

from src.execution.oms import OrderIntent, OrderManager
from src.execution.paper_broker import PaperBroker
from src.execution.trade_journal import EventType, TradeJournal
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
    reconstruct_features,
)


@pytest.fixture
def engine() -> Any:
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def journal(engine: Any) -> TradeJournal:
    return TradeJournal(engine)


@pytest.fixture
def store(engine: Any) -> FeatureSnapshotStore:
    return FeatureSnapshotStore(engine)


@pytest.fixture
def broker() -> PaperBroker:
    b = PaperBroker(initial_capital=100_000)
    b.set_price("EUR_USD", 1.1000, 1.1002)
    return b


# =============================================================================
# OrderIntent metadata schema
# =============================================================================


class TestIntentMetadata:
    def test_default_is_empty_dict(self) -> None:
        intent = OrderIntent(
            strategy_id="s1",
            symbol="EUR_USD",
            target_position=10_000,
        )
        assert intent.metadata == {}

    def test_metadata_round_trip(self) -> None:
        intent = OrderIntent(
            strategy_id="s1",
            symbol="EUR_USD",
            target_position=10_000,
            metadata={"snapshot_id": "abc", "trigger": "entry"},
        )
        assert intent.metadata["snapshot_id"] == "abc"


# =============================================================================
# OMS merges metadata into journal payload
# =============================================================================


class TestOMSMetadataMerge:
    def test_intent_metadata_appears_in_journal_payload(
        self,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        oms = OrderManager(broker, journal=journal)
        intent = OrderIntent(
            strategy_id="s1",
            symbol="EUR_USD",
            target_position=10_000,
            metadata={"snapshot_id": "deadbeef", "trigger": "entry"},
        )
        oms.submit_intent(intent)
        rows = journal.query_by_intent(intent.intent_id)
        intent_events = [r for r in rows if r.event_type == EventType.INTENT_SUBMITTED]
        assert len(intent_events) == 1
        # Standard payload keys still present.
        assert "delta" in intent_events[0].payload
        # Metadata merged in.
        assert intent_events[0].payload["snapshot_id"] == "deadbeef"
        assert intent_events[0].payload["trigger"] == "entry"

    def test_empty_metadata_does_not_pollute_payload(
        self,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        oms = OrderManager(broker, journal=journal)
        intent = OrderIntent(
            strategy_id="s1",
            symbol="EUR_USD",
            target_position=10_000,
        )
        oms.submit_intent(intent)
        rows = journal.query_by_intent(intent.intent_id)
        intent_events = [r for r in rows if r.event_type == EventType.INTENT_SUBMITTED]
        # No surprise keys leaking in.
        for event in intent_events:
            assert set(event.payload.keys()) <= {
                "target_position",
                "current_position",
                "delta",
                "urgency",
                "max_slippage_bps",
            }


# =============================================================================
# End-to-end: strategy → store → journal → reconstruct
# =============================================================================


class TestReconstructRoundTrip:
    def test_strategy_emits_snapshot_then_reconstruct_bit_exact(
        self,
        store: FeatureSnapshotStore,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        # Mimic what a strategy does at signal-emit time.
        from datetime import UTC, datetime

        feature_values = {
            "z": 2.5,
            "price": 1.1001,
            "spread": 0.045,
            "direction": -1,
            "vol": 0.08,
        }
        snapshot = FeatureSnapshot.create(
            feature_set_name="rate_diff_mr",
            feature_set_version="v1",
            data_snapshot_id="live",
            model_version="alpha=0.95,beta=-0.12",
            ts=datetime.now(UTC),
            values=feature_values,
        )
        store.store(snapshot)
        meta = attach_snapshot_payload(snapshot)

        oms = OrderManager(broker, journal=journal)
        intent = OrderIntent(
            strategy_id="rate_diff_mr",
            symbol="EUR_USD",
            target_position=10_000,
            metadata=meta,
        )
        oms.submit_intent(intent)

        # Reconstruct from intent_id.
        reconstructed = reconstruct_features(journal, store, intent.intent_id)
        assert reconstructed is not None
        assert reconstructed.snapshot_id == snapshot.snapshot_id
        assert reconstructed.values == feature_values
        assert reconstructed.feature_set_name == "rate_diff_mr"
        assert reconstructed.model_version == "alpha=0.95,beta=-0.12"

    def test_reconstruct_returns_none_for_intent_with_no_snapshot(
        self,
        store: FeatureSnapshotStore,
        journal: TradeJournal,
        broker: PaperBroker,
    ) -> None:
        oms = OrderManager(broker, journal=journal)
        intent = OrderIntent(
            strategy_id="s1",
            symbol="EUR_USD",
            target_position=10_000,
        )
        oms.submit_intent(intent)
        # No snapshot stored, no metadata attached → reconstruct returns None.
        assert reconstruct_features(journal, store, intent.intent_id) is None


# =============================================================================
# Strategy-level wiring: emit_snapshot helpers degrade gracefully
# =============================================================================


class TestStrategySnapshotEmit:
    def test_rate_diff_no_store_returns_empty_meta(self) -> None:
        from src.strategies.rate_diff_mean_reversion import RateDiffMRStrategy

        s = RateDiffMRStrategy(snapshot_store=None)
        assert s._emit_snapshot({"z": 1.0}) == {}

    def test_rate_diff_with_store_returns_snapshot_payload(
        self,
        store: FeatureSnapshotStore,
    ) -> None:
        from src.strategies.rate_diff_mean_reversion import RateDiffMRStrategy

        s = RateDiffMRStrategy(snapshot_store=store)
        s._model = {"alpha": 0.95, "beta": -0.12, "residual_std": 0.03, "r_squared": 0.7}
        meta = s._emit_snapshot({"z": 1.5, "price": 1.1, "spread": 0.04})
        assert "snapshot_id" in meta
        assert meta["feature_set_name"] == "rate_diff_mr"
        # The snapshot was actually persisted.
        snap = store.fetch(meta["snapshot_id"])
        assert snap is not None
        assert snap.values["z"] == 1.5

    def test_carry_no_store_returns_empty_meta(self) -> None:
        from src.strategies.carry_vol_filter import CarryVolFilterStrategy

        s = CarryVolFilterStrategy(snapshot_store=None)
        assert s._emit_snapshot({"vol_z": 1.0}) == {}

    def test_carry_with_store_persists_snapshot(
        self,
        store: FeatureSnapshotStore,
    ) -> None:
        from src.strategies.carry_vol_filter import CarryVolFilterStrategy

        s = CarryVolFilterStrategy(snapshot_store=store)
        meta = s._emit_snapshot({"vol_z": -0.5, "exposure": 1.0})
        assert "snapshot_id" in meta
        snap = store.fetch(meta["snapshot_id"])
        assert snap is not None
        assert snap.values["vol_z"] == -0.5

    def test_cb_sentiment_no_store_returns_empty_meta(self) -> None:
        from src.strategies.cb_sentiment_shift import CBSentimentShiftStrategy

        s = CBSentimentShiftStrategy(snapshot_store=None)
        assert s._emit_snapshot({"shift": 0.5}) == {}

    def test_cb_sentiment_with_store_persists_snapshot(
        self,
        store: FeatureSnapshotStore,
    ) -> None:
        from src.strategies.cb_sentiment_shift import CBSentimentShiftStrategy

        s = CBSentimentShiftStrategy(snapshot_store=store)
        meta = s._emit_snapshot({"cb": "fed", "shift": 0.45, "direction": -1})
        assert "snapshot_id" in meta
        snap = store.fetch(meta["snapshot_id"])
        assert snap is not None
        assert snap.values["shift"] == 0.45
