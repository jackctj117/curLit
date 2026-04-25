"""Feature + data version pinning per trade (CL-vswx) — full reproducibility.

Every entry signal must record (feature_set_version, data_snapshot_id,
model_version) tuple AND a snapshot of the feature values that drove it.
Without this, six months from now we cannot reproduce why a March 2026
trade was placed — making debugging mystery losses, regulatory defense,
and edge-testing replay impossible.

Three pieces:

    FeatureSnapshot          — immutable record of (model, ts, feature values)
                               keyed by content-addressable snapshot_id.

    FeatureSnapshotStore     — append-only storage. PostgreSQL/SQLite via
                               SQLAlchemy (matches TradeJournal pattern).
                               Snapshot_id is sha256 of canonical content;
                               re-storing the same content is a no-op.

    FeatureVersionRegistry   — tracks the *definition* of each feature set
                               (name → current version, schema). The actual
                               values for one specific call live in a
                               FeatureSnapshot.

    reconstruct_features     — given a trade_id (intent_id from
                               TradeJournal), returns the FeatureSnapshot
                               that produced it. Used by debugging tools
                               and the edge-testing framework.

Reference: CL-vswx (gap-issue draft).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.execution.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


# =============================================================================
# Snapshot
# =============================================================================


def _canonical_snapshot_id(
    feature_set_name: str,
    feature_set_version: str,
    data_snapshot_id: str,
    model_version: str,
    ts: datetime,
    values: dict[str, Any],
) -> str:
    """Content-addressable snapshot id — sha256 over canonical JSON."""
    payload = json.dumps(
        {
            "feature_set_name": feature_set_name,
            "feature_set_version": feature_set_version,
            "data_snapshot_id": data_snapshot_id,
            "model_version": model_version,
            "ts": ts.isoformat(),
            "values": values,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureSnapshot:
    """Immutable record of one feature evaluation.

    snapshot_id is derived from the content via sha256 — re-creating with
    identical inputs produces the same id, enabling deduplication and
    bit-exact reconstruction. Frozen because mutating a stored snapshot
    would invalidate the id.
    """

    snapshot_id: str
    feature_set_name: str
    feature_set_version: str
    data_snapshot_id: str
    model_version: str
    ts: datetime
    values: dict[str, Any]

    @classmethod
    def create(
        cls,
        feature_set_name: str,
        feature_set_version: str,
        data_snapshot_id: str,
        model_version: str,
        ts: datetime,
        values: dict[str, Any],
    ) -> FeatureSnapshot:
        """Build a snapshot with content-addressable id."""
        snap_id = _canonical_snapshot_id(
            feature_set_name=feature_set_name,
            feature_set_version=feature_set_version,
            data_snapshot_id=data_snapshot_id,
            model_version=model_version,
            ts=ts,
            values=values,
        )
        return cls(
            snapshot_id=snap_id,
            feature_set_name=feature_set_name,
            feature_set_version=feature_set_version,
            data_snapshot_id=data_snapshot_id,
            model_version=model_version,
            ts=ts,
            values=dict(values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "feature_set_name": self.feature_set_name,
            "feature_set_version": self.feature_set_version,
            "data_snapshot_id": self.data_snapshot_id,
            "model_version": self.model_version,
            "ts": self.ts.isoformat(),
            "values": dict(self.values),
        }


# =============================================================================
# Store
# =============================================================================


class FeatureSnapshotStore:
    """Append-only SQLAlchemy-backed store for FeatureSnapshots.

    Re-storing the same content (same snapshot_id) is a no-op (ON CONFLICT DO
    NOTHING). This means callers can safely re-record snapshots when re-running
    deterministic feature pipelines without polluting the store.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._create_tables()

    def _create_tables(self) -> None:
        dialect = self.engine.dialect.name
        json_type = "JSONB" if dialect == "postgresql" else "TEXT"
        ts_type = "TIMESTAMPTZ" if dialect == "postgresql" else "TIMESTAMP"
        on_conflict = (
            "ON CONFLICT (snapshot_id) DO NOTHING"
            if dialect == "postgresql"
            else "ON CONFLICT (snapshot_id) DO NOTHING"  # SQLite supports same syntax
        )
        # Cache for the on-conflict suffix used in store().
        self._on_conflict_clause = on_conflict
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS feature_snapshots (
                        snapshot_id           TEXT PRIMARY KEY,
                        feature_set_name      TEXT NOT NULL,
                        feature_set_version   TEXT NOT NULL,
                        data_snapshot_id      TEXT NOT NULL,
                        model_version         TEXT NOT NULL,
                        ts                    {ts_type} NOT NULL,
                        values_json           {json_type} NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS feature_snapshots_set_ts_idx
                    ON feature_snapshots (feature_set_name, ts)
                    """
                )
            )

    def store(self, snapshot: FeatureSnapshot) -> str:
        """Store a snapshot. Returns the snapshot_id (idempotent on duplicates)."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO feature_snapshots (
                        snapshot_id, feature_set_name, feature_set_version,
                        data_snapshot_id, model_version, ts, values_json
                    ) VALUES (:sid, :name, :ver, :dsid, :mver, :ts, :vals)
                    {self._on_conflict_clause}
                    """
                ),
                {
                    "sid": snapshot.snapshot_id,
                    "name": snapshot.feature_set_name,
                    "ver": snapshot.feature_set_version,
                    "dsid": snapshot.data_snapshot_id,
                    "mver": snapshot.model_version,
                    "ts": snapshot.ts,
                    "vals": json.dumps(snapshot.values, default=str),
                },
            )
        logger.debug(
            "Stored feature snapshot %s (set=%s ver=%s)",
            snapshot.snapshot_id, snapshot.feature_set_name,
            snapshot.feature_set_version,
        )
        return snapshot.snapshot_id

    def fetch(self, snapshot_id: str) -> FeatureSnapshot | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT snapshot_id, feature_set_name, feature_set_version,
                           data_snapshot_id, model_version, ts, values_json
                    FROM feature_snapshots WHERE snapshot_id = :sid
                    """
                ),
                {"sid": snapshot_id},
            ).fetchone()
        if row is None:
            return None
        sid, name, ver, dsid, mver, ts, vals = row
        # Dialect normalization (matching TradeJournal pattern).
        ts_obj = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        values = json.loads(vals) if isinstance(vals, str) else vals
        return FeatureSnapshot(
            snapshot_id=sid,
            feature_set_name=name,
            feature_set_version=ver,
            data_snapshot_id=dsid,
            model_version=mver,
            ts=ts_obj,
            values=values,
        )

    def count(self) -> int:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) FROM feature_snapshots"),
            ).fetchone()
        return int(row[0]) if row else 0


# =============================================================================
# Registry
# =============================================================================


@dataclass
class FeatureSetDefinition:
    name: str
    version: str
    schema: dict[str, type]  # field name → Python type

    def __post_init__(self) -> None:
        assert self.name, "feature set name required"
        assert self.version, "feature set version required"

    def validate(self, values: dict[str, Any]) -> None:
        """Raise ValueError if `values` doesn't match the declared schema."""
        missing = set(self.schema.keys()) - set(values.keys())
        if missing:
            msg = f"feature set {self.name} missing fields: {sorted(missing)}"
            raise ValueError(msg)
        for field_name, expected_type in self.schema.items():
            actual = values[field_name]
            if not isinstance(actual, expected_type):
                msg = (
                    f"feature {self.name}.{field_name} expected {expected_type.__name__}, "
                    f"got {type(actual).__name__}"
                )
                raise ValueError(msg)


class FeatureVersionRegistry:
    """Tracks the registered feature-set definitions and their current versions.

    Multiple versions can coexist in the registry — the live system uses
    `current_version()` to pin the production version while older snapshots
    remain replayable.
    """

    def __init__(self) -> None:
        # name → version → definition. The "current" version is the most
        # recently registered.
        self._defs: dict[str, dict[str, FeatureSetDefinition]] = {}
        self._current: dict[str, str] = {}

    def register(self, definition: FeatureSetDefinition) -> None:
        bucket = self._defs.setdefault(definition.name, {})
        bucket[definition.version] = definition
        self._current[definition.name] = definition.version
        logger.info(
            "Registered feature set %s v%s",
            definition.name, definition.version,
        )

    def current_version(self, name: str) -> str:
        if name not in self._current:
            msg = f"feature set {name!r} not registered"
            raise KeyError(msg)
        return self._current[name]

    def get_definition(
        self, name: str, version: str | None = None,
    ) -> FeatureSetDefinition:
        v = version or self.current_version(name)
        if name not in self._defs or v not in self._defs[name]:
            msg = f"feature set {name!r} version {v!r} not registered"
            raise KeyError(msg)
        return self._defs[name][v]


# =============================================================================
# Reconstruction
# =============================================================================


def reconstruct_features(
    journal: TradeJournal,
    store: FeatureSnapshotStore,
    intent_id: str,
) -> FeatureSnapshot | None:
    """Given an intent_id, find the FeatureSnapshot that produced it.

    Walks all journal events for the intent and looks for `snapshot_id` in
    each payload (set by the strategy when it emits the entry signal).
    Returns None if no snapshot reference is found.
    """
    events = journal.query_by_intent(intent_id)
    for ev in events:
        snap_id = ev.payload.get("snapshot_id") if isinstance(ev.payload, dict) else None
        if not snap_id:
            continue
        snap = store.fetch(snap_id)
        if snap is not None:
            return snap
    return None


def attach_snapshot_payload(
    snapshot: FeatureSnapshot,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the payload dict a strategy passes to journal.record() so the
    intent is reproducible.

    Strategies should call this helper rather than hand-rolling the payload
    keys, which keeps the schema consistent across strategies.
    """
    payload: dict[str, Any] = {
        "snapshot_id": snapshot.snapshot_id,
        "feature_set_name": snapshot.feature_set_name,
        "feature_set_version": snapshot.feature_set_version,
        "data_snapshot_id": snapshot.data_snapshot_id,
        "model_version": snapshot.model_version,
        "feature_ts": snapshot.ts.isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


__all__ = [
    "FeatureSetDefinition",
    "FeatureSnapshot",
    "FeatureSnapshotStore",
    "FeatureVersionRegistry",
    "attach_snapshot_payload",
    "reconstruct_features",
]


# Helper exported for tests that need to reproduce the canonical id without
# instantiating a FeatureSnapshot.
def canonical_snapshot_id(
    feature_set_name: str,
    feature_set_version: str,
    data_snapshot_id: str,
    model_version: str,
    ts: datetime,
    values: dict[str, Any],
) -> str:
    return _canonical_snapshot_id(
        feature_set_name=feature_set_name,
        feature_set_version=feature_set_version,
        data_snapshot_id=data_snapshot_id,
        model_version=model_version,
        ts=ts,
        values=values,
    )
