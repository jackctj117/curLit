"""Immutable trade audit log (CL-6mby) — append-only event journal with integrity chain.

Every meaningful execution event (intent submitted, order placed, order filled,
order rejected, reconciliation report) is appended to a journal table that is
NEVER updated or deleted. Each row carries:

    - row_hash: SHA-256 over the previous row's hash + this row's content
    - prev_hash: hash chained from the prior row

Tampering with any row breaks the chain — verify_chain() re-walks the table
recomputing hashes and reports the first divergence. This gives audit-grade
integrity against a compromised DB write path or operator tampering.

Wired into:
    OrderManager.submit_intent  → INTENT_SUBMITTED, ORDER_PLACED, ORDER_FILLED
    RejectionHandler.handle     → ORDER_REJECTED (per-attempt)
    PositionReconciler.reconcile → RECONCILIATION_REPORT (one per startup)

Tax export (src/execution/tax_export.py) reads the journal to group fills into
closed lots (FIFO) and produce the annual realized-P&L file used at tax time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# Sentinel used as the prev_hash for the very first row in an empty journal —
# distinguishes "first row" from "missing prev_hash due to tampering".
_GENESIS_HASH: str = "genesis"


class EventType(Enum):
    INTENT_SUBMITTED = "intent_submitted"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELED = "order_canceled"
    SWAP_CHARGED = "swap_charged"
    RECONCILIATION_REPORT = "reconciliation_report"


@dataclass(frozen=True)
class JournalEvent:
    """One row in the trade journal."""

    seq: int
    ts: datetime
    event_type: EventType
    intent_id: str | None
    strategy_id: str | None
    symbol: str | None
    payload: dict[str, Any]
    prev_hash: str
    row_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts.isoformat(),
            "event_type": self.event_type.value,
            "intent_id": self.intent_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "row_hash": self.row_hash,
        }


def _compute_row_hash(
    prev_hash: str,
    seq: int,
    ts: datetime,
    event_type: EventType,
    intent_id: str | None,
    strategy_id: str | None,
    symbol: str | None,
    payload: dict[str, Any],
) -> str:
    """Stable SHA-256 of (prev_hash || canonical row content).

    JSON serialization of payload uses sort_keys=True to make the hash
    independent of dict iteration order.
    """
    canonical = json.dumps(
        {
            "seq": seq,
            "ts": ts.isoformat(),
            "event_type": event_type.value,
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "payload": payload,
        },
        sort_keys=True,
        default=str,
    )
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(b"|")
    h.update(canonical.encode("utf-8"))
    return h.hexdigest()


class TradeJournal:
    """Append-only event journal with integrity chain.

    Persists to PostgreSQL via SQLAlchemy. Tables are created on init.
    Row-level INSERT only — UPDATE/DELETE are not exposed.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._create_tables()

    def _create_tables(self) -> None:
        # Dialect dispatch: PostgreSQL gets JSONB + TIMESTAMPTZ; SQLite (used in
        # tests) gets TEXT + TIMESTAMP. Sequence column is plain INTEGER —
        # callers (this class) supply seq explicitly to avoid AUTOINCREMENT
        # quirks across dialects.
        dialect = self.engine.dialect.name
        json_type = "JSONB" if dialect == "postgresql" else "TEXT"
        ts_type = "TIMESTAMPTZ" if dialect == "postgresql" else "TIMESTAMP"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS trade_journal_events (
                        seq          BIGINT PRIMARY KEY,
                        ts           {ts_type} NOT NULL,
                        event_type   TEXT NOT NULL,
                        intent_id    TEXT,
                        strategy_id  TEXT,
                        symbol       TEXT,
                        payload      {json_type} NOT NULL,
                        prev_hash    TEXT NOT NULL,
                        row_hash     TEXT NOT NULL UNIQUE
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS trade_journal_events_ts_idx
                    ON trade_journal_events (ts)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS trade_journal_events_intent_idx
                    ON trade_journal_events (intent_id)
                    """
                )
            )

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def record(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        intent_id: str | None = None,
        strategy_id: str | None = None,
        symbol: str | None = None,
        ts: datetime | None = None,
    ) -> JournalEvent:
        """Append one event. Returns the persisted JournalEvent (with seq + hash)."""
        ts = ts or datetime.now(UTC)
        with self.engine.begin() as conn:
            # Lock-free read of last hash + seq. PostgreSQL's BIGSERIAL gives us
            # monotonic seq; concurrent writers each see the latest committed
            # row at the time their transaction begins.
            row = conn.execute(
                text(
                    """
                    SELECT seq, row_hash
                    FROM trade_journal_events
                    ORDER BY seq DESC
                    LIMIT 1
                    """
                )
            ).fetchone()
            if row is None:
                prev_hash = _GENESIS_HASH
                next_seq = 1
            else:
                prev_hash = row[1]
                next_seq = row[0] + 1

            row_hash = _compute_row_hash(
                prev_hash=prev_hash,
                seq=next_seq,
                ts=ts,
                event_type=event_type,
                intent_id=intent_id,
                strategy_id=strategy_id,
                symbol=symbol,
                payload=payload,
            )

            conn.execute(
                text(
                    """
                    INSERT INTO trade_journal_events
                        (seq, ts, event_type, intent_id, strategy_id, symbol,
                         payload, prev_hash, row_hash)
                    VALUES (:seq, :ts, :etype, :iid, :sid, :sym, :pl, :ph, :rh)
                    """
                ),
                {
                    "seq": next_seq,
                    "ts": ts,
                    "etype": event_type.value,
                    "iid": intent_id,
                    "sid": strategy_id,
                    "sym": symbol,
                    "pl": json.dumps(payload, default=str),
                    "ph": prev_hash,
                    "rh": row_hash,
                },
            )

        event = JournalEvent(
            seq=next_seq,
            ts=ts,
            event_type=event_type,
            intent_id=intent_id,
            strategy_id=strategy_id,
            symbol=symbol,
            payload=payload,
            prev_hash=prev_hash,
            row_hash=row_hash,
        )
        logger.debug(
            "Trade journal: appended %s seq=%d intent=%s symbol=%s",
            event_type.value,
            next_seq,
            intent_id,
            symbol,
        )
        return event

    # ------------------------------------------------------------------
    # Read + verify
    # ------------------------------------------------------------------

    def all_events(self) -> list[JournalEvent]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT seq, ts, event_type, intent_id, strategy_id, symbol,
                           payload, prev_hash, row_hash
                    FROM trade_journal_events
                    ORDER BY seq ASC
                    """
                )
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def query_by_intent(self, intent_id: str) -> list[JournalEvent]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT seq, ts, event_type, intent_id, strategy_id, symbol,
                           payload, prev_hash, row_hash
                    FROM trade_journal_events
                    WHERE intent_id = :iid
                    ORDER BY seq ASC
                    """
                ),
                {"iid": intent_id},
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def query_fills_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[JournalEvent]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT seq, ts, event_type, intent_id, strategy_id, symbol,
                           payload, prev_hash, row_hash
                    FROM trade_journal_events
                    WHERE event_type IN ('order_filled', 'order_partial_fill')
                      AND ts >= :start AND ts < :end
                    ORDER BY ts ASC, seq ASC
                    """
                ),
                {"start": start, "end": end},
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def verify_chain(self) -> tuple[bool, int | None]:
        """Walk the chain in order; recompute hashes; report first divergence.

        Returns (True, None) if the chain is intact, otherwise (False, seq) where
        seq is the row whose recomputed hash does not match its stored row_hash
        OR whose prev_hash does not match the previous row's row_hash.
        """
        prev_hash = _GENESIS_HASH
        prev_seq = 0
        for event in self.all_events():
            if event.seq != prev_seq + 1:
                logger.error(
                    "Journal chain gap: expected seq %d got %d",
                    prev_seq + 1,
                    event.seq,
                )
                return False, event.seq
            if event.prev_hash != prev_hash:
                logger.error(
                    "Journal prev_hash mismatch at seq %d",
                    event.seq,
                )
                return False, event.seq
            recomputed = _compute_row_hash(
                prev_hash=event.prev_hash,
                seq=event.seq,
                ts=event.ts,
                event_type=event.event_type,
                intent_id=event.intent_id,
                strategy_id=event.strategy_id,
                symbol=event.symbol,
                payload=event.payload,
            )
            if recomputed != event.row_hash:
                logger.error(
                    "Journal row_hash mismatch at seq %d (tampering suspected)",
                    event.seq,
                )
                return False, event.seq
            prev_hash = event.row_hash
            prev_seq = event.seq
        return True, None

    @staticmethod
    def _row_to_event(row: Any) -> JournalEvent:
        seq, ts, etype, iid, sid, sym, payload, prev_hash, row_hash = row
        # PostgreSQL returns JSONB as dict; SQLite returns TEXT.
        payload_obj = json.loads(payload) if isinstance(payload, str) else payload
        # PostgreSQL returns TIMESTAMPTZ as datetime; SQLite returns ISO string.
        ts_obj = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        return JournalEvent(
            seq=int(seq),
            ts=ts_obj,
            event_type=EventType(etype),
            intent_id=iid,
            strategy_id=sid,
            symbol=sym,
            payload=payload_obj,
            prev_hash=prev_hash,
            row_hash=row_hash,
        )
