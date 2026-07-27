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


def _alpaca(occ="DHT260821C00020000", qty="1", avg="0.57", cur="0.30", plpc="-0.47"):
    return {
        "symbol": occ,
        "qty": qty,
        "avg_entry_price": avg,
        "current_price": cur,
        "unrealized_plpc": plpc,
    }


def test_occ_description():
    assert _describe_occ("DHT260821C00020000") == "DHT $20 call exp 08-21"
    assert _describe_occ("ULCC260821P00006000") == "ULCC $6 put exp 08-21"
    assert _describe_occ("garbage") == "garbage"  # never hide a position


# --------------------------------------------------------------------------- #
# position digest — full book
# --------------------------------------------------------------------------- #


def test_positions_lists_longs_and_shorts_with_reading():
    oanda = [
        Position(symbol="EURUSD", quantity=10_000, avg_price=1.0842, unrealized_pnl=12.5),
        Position(symbol="USDCHF", quantity=-15_462, avg_price=0.86, unrealized_pnl=-31.2),
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
        {"venue": "Alpaca", "desc": "LPG $47.5 call [stop_loss]", "pl": "$-234 (-73%)"},
    ]
    body = build_position_digest(
        [],
        [],
        closed_24h=closed,
        balances={"OANDA": "$100,002.58", "Alpaca": "$99,321.00"},
        now=NOW,
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


def _idea(
    ticker="LPG",
    conf=0.78,
    pref="calls, 3-6 weeks",
    rat="Hormuz closure threat lifts LPG export rates",
):
    return {
        "ticker": ticker,
        "action": "buy_calls",
        "confidence": conf,
        "preferred_instrument": pref,
        "rationale": rat,
    }


def test_ideas_deduped_ranked_and_capped():
    ideas = [
        _idea("LPG", 0.78),
        _idea("LPG", 0.80),  # dupe: keep 0.80
        _idea("CENX", 0.79),
        _idea("BCO_USD", 0.71),
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
# ticker → company-name enrichment (CL-ikz2)
# --------------------------------------------------------------------------- #


def test_ideas_render_company_name_when_provided():
    body = build_long_ideas_digest(
        [_idea(ticker="VG", conf=0.78)],
        now=NOW,
        names={"VG": "Venture Global, Inc."},
    )
    assert "<b>VG</b> (Venture Global, Inc.) 0.78 via calls" in body


def test_ideas_degrade_to_bare_ticker_without_names():
    body = build_long_ideas_digest([_idea(ticker="VG", conf=0.78)], now=NOW)
    assert "<b>VG</b> 0.78" in body  # no parenthetical, exactly as before


def test_ideas_name_is_html_escaped_and_truncated():
    body = build_long_ideas_digest(
        [_idea(ticker="VG")],
        now=NOW,
        names={"VG": "Evil & Co <b>" + "x" * 40},
    )
    assert "<b>x" not in body.split("<b>VG</b>")[1]  # injected tag escaped
    assert "&amp;" in body  # ampersand escaped
    assert "…" in body  # long name truncated


def test_ideas_names_looked_up_by_upper_key():
    # Idea ticker is lower-case; the map is keyed UPPER (the dedup key).
    body = build_long_ideas_digest(
        [_idea(ticker="vg", conf=0.78)],
        now=NOW,
        names={"VG": "Venture Global, Inc."},
    )
    assert "(Venture Global, Inc.)" in body


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


# --------------------------------------------------------------------------- #
# exit-blocked (no bid) section — CL-p0pe / CL-hptt
# --------------------------------------------------------------------------- #


def _unsellable(
    occ="ASC260821C00017500",
    ticker="ASC",
    reason="stop_loss (no bid — market sell would be rejected)",
):
    return {"ticker": ticker, "occ_symbol": occ, "exit_reason": reason}


def test_unsellable_section_rendered():
    """An unsellable contract pages ONCE (CL-hptt) then goes quiet, so the
    digest is what keeps a days-stuck position visible."""
    body = build_position_digest([], [_alpaca()], now=NOW, unsellable=[_unsellable()])
    assert "Exit BLOCKED" in body
    assert "ASC $17.5 call exp 08-21" in body
    assert "wants stop_loss" in body  # the raw "(no bid — ...)" suffix is trimmed
    assert "sells when one returns" in body


def test_unsellable_section_absent_when_none():
    # A normal morning must gain no extra noise.
    for arg in (None, []):
        body = build_position_digest([], [_alpaca()], now=NOW, unsellable=arg)
        assert "Exit BLOCKED" not in body


def test_unsellable_lists_every_stuck_contract():
    body = build_position_digest(
        [],
        [_alpaca()],
        now=NOW,
        unsellable=[
            _unsellable(),
            _unsellable(occ="ASTL260821P00004000", ticker="ASTL"),
        ],
    )
    assert "ASC $17.5 call exp 08-21" in body
    assert "ASTL $4 put exp 08-21" in body


def test_unsellable_unparseable_occ_still_shown():
    # Never hide a position behind a formatting failure.
    body = build_position_digest([], [], now=NOW, unsellable=[_unsellable(occ="garbage")])
    assert "garbage" in body
