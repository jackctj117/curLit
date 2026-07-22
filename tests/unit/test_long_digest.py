"""Tests for the morning LONG-positions digest (CL-lpai)."""

from __future__ import annotations

from datetime import UTC, datetime

from src.execution.broker import Position
from src.monitoring.long_digest import (
    _describe_occ,
    build_long_digest,
    should_send,
)

NOW = datetime(2026, 7, 22, 15, 0, tzinfo=UTC)  # 11:00 ET


def _alpaca(occ="DHT260821C00020000", qty="1", avg="0.57", cur="0.30",
            plpc="-0.47"):
    return {"symbol": occ, "qty": qty, "avg_entry_price": avg,
            "current_price": cur, "unrealized_plpc": plpc}


def test_occ_description():
    assert _describe_occ("DHT260821C00020000") == "DHT $20 call exp 08-21"
    assert _describe_occ("ULCC260821P00006000") == "ULCC $6 put exp 08-21"
    assert _describe_occ("garbage") == "garbage"  # never hide a position


def test_longs_listed_shorts_counted():
    oanda = [
        Position(symbol="EURUSD", quantity=10_000, avg_price=1.0842,
                 unrealized_pnl=12.5),
        Position(symbol="USDCHF", quantity=-15_462, avg_price=0.86),
        Position(symbol="USDJPY", quantity=-76, avg_price=147.2),
    ]
    body = build_long_digest(oanda, [_alpaca()], now=NOW)
    assert "EURUSD +10,000" in body
    assert "USDCHF" not in body  # shorts not listed...
    assert "Shorts not listed: OANDA 2" in body  # ...but counted
    assert "DHT $20 call exp 08-21" in body
    assert "-47%" in body


def test_empty_and_unavailable_states():
    body = build_long_digest([], [], now=NOW)
    assert body.count("none") == 2
    body = build_long_digest(None, None, now=NOW)
    assert body.count("unavailable") == 2  # unreadable never looks flat


def test_should_send_gates():
    before_open = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)  # 08:00 ET
    after_time = datetime(2026, 7, 22, 13, 30, tzinfo=UTC)  # 09:30 ET
    assert not should_send(before_open, None)  # too early
    assert should_send(after_time, None)  # due
    assert not should_send(after_time, "2026-07-22")  # already sent today
    assert should_send(after_time, "2026-07-21")  # yesterday's send
    assert should_send(after_time, None, send_time_et="garbage")  # falls back
