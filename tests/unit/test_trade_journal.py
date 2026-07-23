"""Unit tests — trade_journal: append, integrity chain, query, tampering detection (CL-6mby)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.execution.tax_export import (
    ClosedLot,
    export_annual,
    export_annual_to_parquet,
    match_fifo,
)
from src.execution.trade_journal import EventType, TradeJournal

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def journal() -> TradeJournal:
    engine = create_engine("sqlite:///:memory:")
    return TradeJournal(engine)


# =============================================================================
# Append + read
# =============================================================================


class TestAppend:
    def test_first_record_uses_genesis_prev_hash(self, journal: TradeJournal) -> None:
        ev = journal.record(
            EventType.INTENT_SUBMITTED,
            payload={"target_position": 1000.0},
            intent_id="i1",
            strategy_id="s1",
            symbol="EURUSD",
        )
        assert ev.seq == 1
        assert ev.prev_hash == "genesis"
        assert ev.row_hash != "genesis"

    def test_seq_monotonic(self, journal: TradeJournal) -> None:
        ev1 = journal.record(EventType.INTENT_SUBMITTED, {"x": 1})
        ev2 = journal.record(EventType.ORDER_PLACED, {"x": 2})
        ev3 = journal.record(EventType.ORDER_FILLED, {"x": 3})
        assert ev1.seq == 1
        assert ev2.seq == 2
        assert ev3.seq == 3

    def test_chain_links_prev_hash_to_prior_row(self, journal: TradeJournal) -> None:
        ev1 = journal.record(EventType.INTENT_SUBMITTED, {"x": 1})
        ev2 = journal.record(EventType.ORDER_PLACED, {"x": 2})
        assert ev2.prev_hash == ev1.row_hash

    def test_payload_with_complex_types_serializes(
        self,
        journal: TradeJournal,
    ) -> None:
        # Datetimes inside payloads are serialized via default=str.
        ts = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
        ev = journal.record(
            EventType.RECONCILIATION_REPORT,
            payload={"ts": ts, "entries": [{"symbol": "EURUSD", "qty": 1000.0}]},
        )
        all_events = journal.all_events()
        assert len(all_events) == 1
        assert all_events[0].seq == ev.seq


# =============================================================================
# Query helpers
# =============================================================================


class TestQuery:
    def test_query_by_intent(self, journal: TradeJournal) -> None:
        journal.record(EventType.INTENT_SUBMITTED, {"x": 1}, intent_id="i1")
        journal.record(EventType.ORDER_PLACED, {"x": 2}, intent_id="i1")
        journal.record(EventType.INTENT_SUBMITTED, {"x": 3}, intent_id="i2")
        events = journal.query_by_intent("i1")
        assert [e.event_type for e in events] == [
            EventType.INTENT_SUBMITTED,
            EventType.ORDER_PLACED,
        ]

    def test_query_fills_in_range(self, journal: TradeJournal) -> None:
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(3):
            journal.record(
                EventType.ORDER_FILLED,
                payload={"side": "buy", "quantity": 100, "fill_price": 1.10},
                intent_id=f"i{i}",
                symbol="EURUSD",
                ts=t0 + timedelta(days=i),
            )
        # Query middle window
        fills = journal.query_fills_in_range(
            t0 + timedelta(hours=12),
            t0 + timedelta(days=2, hours=12),
        )
        assert len(fills) == 2


# =============================================================================
# Chain integrity
# =============================================================================


class TestVerifyChain:
    def test_intact_chain_passes(self, journal: TradeJournal) -> None:
        for i in range(5):
            journal.record(EventType.ORDER_PLACED, {"x": i})
        ok, bad_seq = journal.verify_chain()
        assert ok is True
        assert bad_seq is None

    def test_tampered_payload_detected(self, journal: TradeJournal) -> None:
        for i in range(3):
            journal.record(EventType.ORDER_PLACED, {"x": i})
        # Tamper directly with the DB to simulate an attacker editing a row.
        with journal.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE trade_journal_events
                    SET payload = :tampered
                    WHERE seq = 2
                    """
                ),
                {"tampered": '{"x": 999}'},
            )
        ok, bad_seq = journal.verify_chain()
        assert ok is False
        assert bad_seq == 2

    def test_deleted_row_detected_as_gap(self, journal: TradeJournal) -> None:
        for i in range(3):
            journal.record(EventType.ORDER_PLACED, {"x": i})
        # Simulate row deletion (which breaks the seq chain).
        with journal.engine.begin() as conn:
            conn.execute(text("DELETE FROM trade_journal_events WHERE seq = 2"))
        ok, bad_seq = journal.verify_chain()
        assert ok is False
        # Row 3 has prev_hash pointing to row 2's hash; with row 2 gone, the
        # chain detects the gap at seq=3.
        assert bad_seq == 3


# =============================================================================
# Tax export FIFO matching
# =============================================================================


class TestFifoMatching:
    def test_simple_long_open_close(self) -> None:
        fills = [
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 1, tzinfo=UTC),
                "quantity": 1000.0,
                "price": 1.10,
                "intent_id": "open",
            },
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 5, tzinfo=UTC),
                "quantity": -1000.0,
                "price": 1.12,
                "intent_id": "close",
            },
        ]
        lots = match_fifo(fills)
        assert len(lots) == 1
        lot = lots[0]
        assert lot.symbol == "EURUSD"
        assert lot.side == "long"
        assert lot.quantity == 1000.0
        assert lot.realized_pnl_quote_ccy == pytest.approx(1000.0 * (1.12 - 1.10))

    def test_short_open_close(self) -> None:
        fills = [
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 1, tzinfo=UTC),
                "quantity": -1000.0,
                "price": 1.12,
                "intent_id": "open",
            },
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 5, tzinfo=UTC),
                "quantity": 1000.0,
                "price": 1.10,
                "intent_id": "close",
            },
        ]
        lots = match_fifo(fills)
        assert len(lots) == 1
        assert lots[0].side == "short"
        # Short profit: open 1.12, close 1.10 → +0.02 × 1000 = +20.
        assert lots[0].realized_pnl_quote_ccy == pytest.approx(20.0)

    def test_partial_close_leaves_open_lot(self) -> None:
        fills = [
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 1, tzinfo=UTC),
                "quantity": 1000.0,
                "price": 1.10,
                "intent_id": "open",
            },
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 5, tzinfo=UTC),
                "quantity": -400.0,
                "price": 1.12,
                "intent_id": "close1",
            },
        ]
        lots = match_fifo(fills)
        assert len(lots) == 1
        assert lots[0].quantity == 400.0
        assert lots[0].realized_pnl_quote_ccy == pytest.approx(400.0 * 0.02)

    def test_fifo_oldest_first_when_multiple_open(self) -> None:
        # Two long lots at different prices; close one's worth.
        fills = [
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 1, tzinfo=UTC),
                "quantity": 1000.0,
                "price": 1.10,
                "intent_id": "open1",
            },
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 2, tzinfo=UTC),
                "quantity": 1000.0,
                "price": 1.15,
                "intent_id": "open2",
            },
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 5, tzinfo=UTC),
                "quantity": -1000.0,
                "price": 1.20,
                "intent_id": "close1",
            },
        ]
        lots = match_fifo(fills)
        assert len(lots) == 1
        # Should match against the OLDEST lot first → open1 at 1.10.
        assert lots[0].open_intent_id == "open1"
        assert lots[0].open_price == 1.10
        assert lots[0].realized_pnl_quote_ccy == pytest.approx(1000.0 * 0.10)

    def test_reversal_creates_new_open_lot(self) -> None:
        # Long 1000 then sell 1500 = close 1000 + open short 500.
        fills = [
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 1, tzinfo=UTC),
                "quantity": 1000.0,
                "price": 1.10,
                "intent_id": "open",
            },
            {
                "symbol": "EURUSD",
                "ts": datetime(2026, 1, 5, tzinfo=UTC),
                "quantity": -1500.0,
                "price": 1.12,
                "intent_id": "reversal",
            },
        ]
        lots = match_fifo(fills)
        assert len(lots) == 1  # Only the close → 1000 lot is closed
        assert lots[0].quantity == 1000.0


# =============================================================================
# Annual export — end-to-end via journal
# =============================================================================


class TestAnnualExport:
    def test_export_annual_only_in_year(self, journal: TradeJournal) -> None:
        journal.record(
            EventType.ORDER_FILLED,
            payload={"side": "buy", "quantity": 1000, "fill_price": 1.10},
            intent_id="open26",
            symbol="EURUSD",
            ts=datetime(2026, 3, 1, tzinfo=UTC),
        )
        journal.record(
            EventType.ORDER_FILLED,
            payload={"side": "sell", "quantity": 1000, "fill_price": 1.12},
            intent_id="close26",
            symbol="EURUSD",
            ts=datetime(2026, 6, 1, tzinfo=UTC),
        )
        # Out-of-year fill should be excluded:
        journal.record(
            EventType.ORDER_FILLED,
            payload={"side": "buy", "quantity": 500, "fill_price": 1.05},
            intent_id="open27",
            symbol="EURUSD",
            ts=datetime(2027, 3, 1, tzinfo=UTC),
        )
        df = export_annual(journal, year=2026)
        assert len(df) == 1
        assert df.iloc[0]["realized_pnl_quote_ccy"] == pytest.approx(20.0)
        assert df.iloc[0]["symbol"] == "EURUSD"

    def test_export_to_parquet(
        self,
        journal: TradeJournal,
        tmp_path: Path,
    ) -> None:
        journal.record(
            EventType.ORDER_FILLED,
            payload={"side": "buy", "quantity": 1000, "fill_price": 1.10},
            intent_id="open",
            symbol="EURUSD",
            ts=datetime(2026, 3, 1, tzinfo=UTC),
        )
        journal.record(
            EventType.ORDER_FILLED,
            payload={"side": "sell", "quantity": 1000, "fill_price": 1.12},
            intent_id="close",
            symbol="EURUSD",
            ts=datetime(2026, 6, 1, tzinfo=UTC),
        )
        out = tmp_path / "tax_2026.parquet"
        n = export_annual_to_parquet(journal, year=2026, output_path=str(out))
        assert n == 1
        assert out.exists()


class TestClosedLotShape:
    def test_to_dict_keys(self) -> None:
        lot = ClosedLot(
            symbol="EURUSD",
            open_ts=datetime(2026, 1, 1, tzinfo=UTC),
            close_ts=datetime(2026, 6, 1, tzinfo=UTC),
            side="long",
            quantity=1000.0,
            open_price=1.10,
            close_price=1.12,
            open_intent_id="o1",
            close_intent_id="c1",
            realized_pnl_quote_ccy=20.0,
        )
        d = lot.to_dict()
        assert d["symbol"] == "EURUSD"
        assert d["realized_pnl_quote_ccy"] == 20.0
