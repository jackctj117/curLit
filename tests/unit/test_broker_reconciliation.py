"""Tests for src/execution/broker_reconciliation.py (CL-unlt).

Covers the matching logic and the four mismatch kinds. The OANDA HTTP
fetch + DB read are exercised via fakes — no live network or DB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.execution.broker_reconciliation import (
    FillRecord,
    _oanda_to_pair,
    reconcile_fills,
)


class TestDialectNormalization:
    """CL-2zt0 (P1): both the OANDA and journal sides canonicalize the symbol
    so an event leg (OANDA USDCAD vs journal USD_CAD) matches instead of
    generating permanent missing_internal/missing_broker noise."""

    def test_oanda_to_pair_canonicalizes(self) -> None:
        assert _oanda_to_pair("USD_CAD") == "USDCAD"
        assert _oanda_to_pair("EUR_USD") == "EURUSD"
        # Now also handles mixed case / other separators via canonical_symbol.
        assert _oanda_to_pair("eur/usd") == "EURUSD"
        assert _oanda_to_pair("xau-usd") == "XAUUSD"


def _o(ts: datetime, instrument: str, units: float, price: float, tid: str = "txn") -> FillRecord:
    return FillRecord(ts=ts, instrument=instrument, units=units,
                      price=price, transaction_id=tid, source="oanda")


def _i(ts: datetime, instrument: str, units: float, price: float, iid: str = "iid") -> FillRecord:
    return FillRecord(ts=ts, instrument=instrument, units=units,
                      price=price, transaction_id=iid, source="internal")


T0 = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


class TestExactMatch:
    def test_clean_pair(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000, 1.10, "tx1")]
        internal = [_i(T0, "EURUSD", 1000, 1.10, "i1")]
        rep = reconcile_fills(oanda, internal)
        assert rep.is_clean
        assert rep.matched == 1


class TestMissingInternal:
    def test_oanda_only_flagged(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000, 1.10)]
        internal: list[FillRecord] = []
        rep = reconcile_fills(oanda, internal)
        assert len(rep.mismatches) == 1
        assert rep.mismatches[0].kind == "missing_internal"


class TestMissingBroker:
    def test_internal_only_flagged(self) -> None:
        oanda: list[FillRecord] = []
        internal = [_i(T0, "EURUSD", 1000, 1.10)]
        rep = reconcile_fills(oanda, internal)
        assert len(rep.mismatches) == 1
        assert rep.mismatches[0].kind == "missing_broker"


class TestQuantityDrift:
    def test_drift_above_tolerance(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000, 1.10)]
        internal = [_i(T0, "EURUSD", 990, 1.10)]
        rep = reconcile_fills(oanda, internal)
        assert any(m.kind == "qty_drift" for m in rep.mismatches)

    def test_drift_within_tolerance(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000.0, 1.10)]
        internal = [_i(T0, "EURUSD", 1000.4, 1.10)]
        rep = reconcile_fills(oanda, internal)
        assert rep.is_clean


class TestPriceDrift:
    def test_drift_above_tolerance(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000, 1.1010)]
        internal = [_i(T0, "EURUSD", 1000, 1.1000)]  # 0.001 > 1e-4 tol
        rep = reconcile_fills(oanda, internal)
        assert any(m.kind == "price_drift" for m in rep.mismatches)

    def test_internal_zero_price_does_not_flag(self) -> None:
        # Pre-CL-mdle internal records sometimes have price=0.
        # That's a known gap, not a mismatch.
        oanda = [_o(T0, "EURUSD", 1000, 1.10)]
        internal = [_i(T0, "EURUSD", 1000, 0.0)]
        rep = reconcile_fills(oanda, internal)
        assert rep.is_clean


class TestTimeWindow:
    def test_within_window_matches(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000, 1.10)]
        internal = [_i(T0 + timedelta(seconds=15), "EURUSD", 1000, 1.10)]
        assert reconcile_fills(oanda, internal).is_clean

    def test_outside_window_does_not_match(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000, 1.10)]
        internal = [_i(T0 + timedelta(seconds=120), "EURUSD", 1000, 1.10)]
        rep = reconcile_fills(oanda, internal)
        # Each side now appears as a separate mismatch.
        kinds = {m.kind for m in rep.mismatches}
        assert kinds == {"missing_internal", "missing_broker"}


class TestSideSensitivity:
    def test_opposite_side_does_not_match(self) -> None:
        oanda = [_o(T0, "EURUSD", 1000, 1.10)]      # buy
        internal = [_i(T0, "EURUSD", -1000, 1.10)]  # sell
        rep = reconcile_fills(oanda, internal)
        # Different signs → matched as separate, both flagged
        assert any(m.kind == "missing_internal" for m in rep.mismatches)
        assert any(m.kind == "missing_broker" for m in rep.mismatches)


class TestReportShape:
    def test_to_dict_round_trip(self) -> None:
        rep = reconcile_fills(
            [_o(T0, "EURUSD", 1000, 1.10)],
            [_i(T0, "EURUSD", 1000, 1.10)],
        )
        d = rep.to_dict()
        assert d["matched"] == 1
        assert d["mismatches"] == []
