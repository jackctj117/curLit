"""Unit tests — models.feature_versioning (CL-vswx)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine

from src.execution.trade_journal import EventType, TradeJournal
from src.models.feature_versioning import (
    FeatureSetDefinition,
    FeatureSnapshot,
    FeatureSnapshotStore,
    FeatureVersionRegistry,
    attach_snapshot_payload,
    canonical_snapshot_id,
    reconstruct_features,
)


@pytest.fixture
def store() -> FeatureSnapshotStore:
    engine = create_engine("sqlite:///:memory:")
    return FeatureSnapshotStore(engine)


@pytest.fixture
def journal() -> TradeJournal:
    engine = create_engine("sqlite:///:memory:")
    return TradeJournal(engine)


def _make_snapshot(seed: int = 0) -> FeatureSnapshot:
    return FeatureSnapshot.create(
        feature_set_name="rate_diff_features",
        feature_set_version="1.0.0",
        data_snapshot_id=f"data_{seed}",
        model_version="ols_v3",
        ts=datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC),
        values={
            "rate_spread_z": 1.5 + seed * 0.1,
            "us_2y": 0.045,
            "de_2y": 0.025,
        },
    )


# =============================================================================
# Snapshot ID determinism
# =============================================================================


class TestSnapshotId:
    def test_same_inputs_yield_same_id(self) -> None:
        ts = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
        a = canonical_snapshot_id("set", "v1", "data", "model", ts, {"x": 1.0})
        b = canonical_snapshot_id("set", "v1", "data", "model", ts, {"x": 1.0})
        assert a == b

    def test_different_values_yield_different_ids(self) -> None:
        ts = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
        a = canonical_snapshot_id("set", "v1", "data", "model", ts, {"x": 1.0})
        b = canonical_snapshot_id("set", "v1", "data", "model", ts, {"x": 1.5})
        assert a != b

    def test_dict_order_does_not_matter(self) -> None:
        ts = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
        a = canonical_snapshot_id(
            "set",
            "v1",
            "data",
            "model",
            ts,
            {"a": 1.0, "b": 2.0},
        )
        b = canonical_snapshot_id(
            "set",
            "v1",
            "data",
            "model",
            ts,
            {"b": 2.0, "a": 1.0},
        )
        assert a == b

    def test_create_assigns_id(self) -> None:
        snap = _make_snapshot()
        assert snap.snapshot_id  # non-empty
        # Snapshot id matches the canonical helper.
        expected = canonical_snapshot_id(
            snap.feature_set_name,
            snap.feature_set_version,
            snap.data_snapshot_id,
            snap.model_version,
            snap.ts,
            snap.values,
        )
        assert snap.snapshot_id == expected


# =============================================================================
# Store
# =============================================================================


class TestFeatureSnapshotStore:
    def test_store_then_fetch_round_trip(self, store: FeatureSnapshotStore) -> None:
        snap = _make_snapshot()
        sid = store.store(snap)
        assert sid == snap.snapshot_id
        fetched = store.fetch(sid)
        assert fetched is not None
        assert fetched.snapshot_id == snap.snapshot_id
        assert fetched.feature_set_name == snap.feature_set_name
        assert fetched.values == snap.values

    def test_idempotent_store(self, store: FeatureSnapshotStore) -> None:
        snap = _make_snapshot()
        store.store(snap)
        store.store(snap)  # second store is no-op
        assert store.count() == 1

    def test_unknown_snapshot_returns_none(self, store: FeatureSnapshotStore) -> None:
        assert store.fetch("nonexistent") is None

    def test_round_trip_preserves_values_bit_exact(
        self,
        store: FeatureSnapshotStore,
    ) -> None:
        snap = FeatureSnapshot.create(
            feature_set_name="set",
            feature_set_version="v1",
            data_snapshot_id="d",
            model_version="m",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
            values={
                "float_value": 3.141592653589793,
                "int_value": 42,
                "string_value": "hello",
                "list_value": [1, 2, 3],
                "nested": {"a": 1, "b": [1.0, 2.0]},
            },
        )
        store.store(snap)
        fetched = store.fetch(snap.snapshot_id)
        assert fetched is not None
        assert fetched.values == snap.values

    def test_count(self, store: FeatureSnapshotStore) -> None:
        for i in range(5):
            store.store(_make_snapshot(seed=i))
        assert store.count() == 5


# =============================================================================
# Registry
# =============================================================================


class TestRegistry:
    def test_register_and_lookup(self) -> None:
        reg = FeatureVersionRegistry()
        defn = FeatureSetDefinition(
            name="rate_diff_features",
            version="1.0.0",
            schema={"rate_spread_z": float},
        )
        reg.register(defn)
        assert reg.current_version("rate_diff_features") == "1.0.0"
        assert reg.get_definition("rate_diff_features") is defn

    def test_multiple_versions_coexist(self) -> None:
        reg = FeatureVersionRegistry()
        v1 = FeatureSetDefinition(
            name="rd",
            version="1.0.0",
            schema={"a": float},
        )
        v2 = FeatureSetDefinition(
            name="rd",
            version="2.0.0",
            schema={"a": float, "b": int},
        )
        reg.register(v1)
        reg.register(v2)
        assert reg.current_version("rd") == "2.0.0"
        assert reg.get_definition("rd", version="1.0.0") is v1

    def test_missing_set_raises(self) -> None:
        reg = FeatureVersionRegistry()
        with pytest.raises(KeyError):
            reg.current_version("missing")
        with pytest.raises(KeyError):
            reg.get_definition("missing")

    def test_validate_passes_on_match(self) -> None:
        defn = FeatureSetDefinition(
            name="rd",
            version="1.0.0",
            schema={"a": float, "b": int},
        )
        defn.validate({"a": 1.5, "b": 3})

    def test_validate_rejects_missing_field(self) -> None:
        defn = FeatureSetDefinition(
            name="rd",
            version="1.0.0",
            schema={"a": float, "b": int},
        )
        with pytest.raises(ValueError, match="missing"):
            defn.validate({"a": 1.5})

    def test_validate_rejects_wrong_type(self) -> None:
        defn = FeatureSetDefinition(
            name="rd",
            version="1.0.0",
            schema={"a": float},
        )
        with pytest.raises(ValueError, match="expected"):
            defn.validate({"a": "not a float"})


# =============================================================================
# Reconstruction
# =============================================================================


class TestReconstruction:
    def test_reconstruct_finds_snapshot_via_payload(
        self,
        store: FeatureSnapshotStore,
        journal: TradeJournal,
    ) -> None:
        snap = _make_snapshot()
        store.store(snap)

        # Strategy records intent with snapshot reference in payload.
        journal.record(
            EventType.INTENT_SUBMITTED,
            payload=attach_snapshot_payload(
                snap,
                extra={"target_position": 1000.0},
            ),
            intent_id="intent-1",
            strategy_id="rate_diff_mr",
            symbol="EURUSD",
        )
        recovered = reconstruct_features(journal, store, "intent-1")
        assert recovered is not None
        assert recovered.snapshot_id == snap.snapshot_id
        # Bit-exact reconstruction.
        assert recovered.values == snap.values

    def test_reconstruct_returns_none_when_no_payload_ref(
        self,
        store: FeatureSnapshotStore,
        journal: TradeJournal,
    ) -> None:
        # Record an intent without a snapshot reference.
        journal.record(
            EventType.INTENT_SUBMITTED,
            payload={"target_position": 1000.0},
            intent_id="bare",
            strategy_id="x",
            symbol="EURUSD",
        )
        assert reconstruct_features(journal, store, "bare") is None

    def test_reconstruct_returns_none_for_unknown_intent(
        self,
        store: FeatureSnapshotStore,
        journal: TradeJournal,
    ) -> None:
        assert reconstruct_features(journal, store, "never-seen") is None

    def test_attach_snapshot_payload_includes_required_keys(self) -> None:
        snap = _make_snapshot()
        payload = attach_snapshot_payload(snap, extra={"target_position": 100})
        for key in (
            "snapshot_id",
            "feature_set_name",
            "feature_set_version",
            "data_snapshot_id",
            "model_version",
            "feature_ts",
        ):
            assert key in payload
        assert payload["target_position"] == 100


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_snapshot_to_dict(self) -> None:
        snap = _make_snapshot()
        d = snap.to_dict()
        assert d["snapshot_id"] == snap.snapshot_id
        assert d["values"] == snap.values
