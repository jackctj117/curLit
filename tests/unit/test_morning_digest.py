"""Tests for the morning digests (CL-ydp8): full positions + LONG ideas."""

from __future__ import annotations

from datetime import UTC, datetime

from src.execution.broker import Position
from src.monitoring.morning_digest import (
    _describe_occ,
    build_long_ideas_digest,
    build_position_digest,
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


# --------------------------------------------------------------------------- #
# position digest — full book
# --------------------------------------------------------------------------- #


def test_positions_lists_longs_and_shorts_with_reading():
    oanda = [
        Position(symbol="EURUSD", quantity=10_000, avg_price=1.0842,
                 unrealized_pnl=12.5),
        Position(symbol="USDCHF", quantity=-15_462, avg_price=0.86,
                 unrealized_pnl=-31.2),
    ]
    body = build_position_digest(oanda, [_alpaca()], now=NOW)
    assert "EURUSD +10,000" in body
    assert "long EUR / short USD" in body
    assert "USDCHF -15,462" in body  # shorts LISTED now
    assert "short USD / long CHF" in body  # economic reading
    assert "DHT $20 call exp 08-21" in body
    assert "-47%" in body


def test_positions_closed_and_balances_sections():
    closed = [
        {"venue": "OANDA", "desc": "USD_CAD short 8,916", "pl": "+2.00"},
        {"venue": "Alpaca", "desc": "LPG $47.5 call [stop_loss]",
         "pl": "$-234 (-73%)"},
    ]
    body = build_position_digest(
        [], [], closed_24h=closed,
        balances={"OANDA": "$100,002.58", "Alpaca": "$99,321.00"}, now=NOW,
    )
    assert "Closed last 24h" in body
    assert "USD_CAD short 8,916" in body and "+2.00" in body
    assert "OANDA: $100,002.58" in body and "Alpaca: $99,321.00" in body
    assert body.count("flat") == 2


def test_positions_unavailable_never_looks_flat():
    body = build_position_digest(None, None, now=NOW)
    assert body.count("unavailable") == 2
    assert "flat" not in body


# --------------------------------------------------------------------------- #
# LONG ideas digest — the shopping list
# --------------------------------------------------------------------------- #


def _idea(ticker="LPG", conf=0.78, pref="calls, 3-6 weeks",
          rat="Hormuz closure threat lifts LPG export rates"):
    return {"ticker": ticker, "action": "buy_calls", "confidence": conf,
            "preferred_instrument": pref, "rationale": rat}


def test_ideas_deduped_ranked_and_capped():
    ideas = [
        _idea("LPG", 0.78), _idea("LPG", 0.80),  # dupe: keep 0.80
        _idea("CENX", 0.79), _idea("BCO_USD", 0.71),
    ]
    body = build_long_ideas_digest(ideas, now=NOW, limit=2)
    assert "0.80" in body and body.index("LPG") < body.index("CENX")
    assert "BCO_USD" not in body  # capped...
    assert "Top 2 of 3 tickers" in body  # ...but honestly counted


def test_ideas_truncates_rationale_and_escapes():
    long_rat = "x" * 120 + "<b>"
    body = build_long_ideas_digest([_idea(rat=long_rat)], now=NOW)
    assert "…" in body
    assert "<b>x" not in body  # rationale html escaped (truncated anyway)


def test_ideas_empty_state():
    body = build_long_ideas_digest([], now=NOW)
    assert "no pending bullish ideas" in body


# --------------------------------------------------------------------------- #
# clock gate
# --------------------------------------------------------------------------- #


def test_should_send_gates():
    before_open = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)  # 08:00 ET
    after_time = datetime(2026, 7, 22, 13, 30, tzinfo=UTC)  # 09:30 ET
    assert not should_send(before_open, None)  # too early
    assert should_send(after_time, None)  # due
    assert not should_send(after_time, "2026-07-22")  # already sent today
    assert should_send(after_time, "2026-07-21")  # yesterday's send
    assert should_send(after_time, None, send_time_et="garbage")  # falls back
