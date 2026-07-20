"""Unit tests — grounded trade cards (CL-jiqq).

The LLM emits PERCENTAGES (it has no live market data);
:func:`src.events.trade_card.build_trade_card` converts them to REAL
dollar levels off a live price. These tests pin the money math: stop /
target DIRECTION (long vs short), risk:reward, option moneyness, the
absent-price flag, and rounding. Getting the direction wrong here would
hand the operator a stop on the wrong side of the trade — so it is
checked exhaustively.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.events.trade_card import (
    STOP_PCT_STOCK_DEFAULT,
    STOP_PCT_UNDERLYING_DEFAULT,
    build_trade_card,
)


def _idea(**overrides: Any) -> dict[str, Any]:
    idea = {
        "ticker": "TSM",
        "action": "long",
        "direction": "bullish",
        "time_horizon": "short",
        "stop_loss_pct": 0.10,
        "target_pct": [0.10, 0.20],
        "entry_trigger": "on breakout",
        "invalidation": "loses support",
    }
    idea.update(overrides)
    return idea


class TestLongDirection:
    def test_long_stop_below_targets_above(self) -> None:
        card = build_trade_card(_idea(action="long"), 100.0)
        assert card["bullish"] is True
        assert card["stop_price"] == pytest.approx(90.0)   # 100 × (1 − 0.10)
        assert card["target_prices"] == [110.0, 120.0]      # 100 × (1 ± 0.10/0.20)
        assert card["priced"] is True
        assert card["no_live_price"] is False

    def test_long_risk_reward_first_target(self) -> None:
        # reward = |110 − 100| = 10 ; risk = |100 − 90| = 10 → 1.0
        card = build_trade_card(_idea(action="long"), 100.0)
        assert card["risk_reward"] == pytest.approx(1.0)


class TestShortDirection:
    def test_short_stop_above_targets_below(self) -> None:
        card = build_trade_card(
            _idea(action="short", direction="bearish", stop_loss_pct=0.08),
            100.0,
        )
        assert card["bullish"] is False
        assert card["stop_price"] == pytest.approx(108.0)   # 100 × (1 + 0.08)
        assert card["target_prices"] == [90.0, 80.0]         # 100 × (1 − 0.10/0.20)

    def test_short_risk_reward(self) -> None:
        # reward = |90 − 100| = 10 ; risk = |100 − 108| = 8 → 1.25 → 1.2
        card = build_trade_card(
            _idea(action="short", direction="bearish", stop_loss_pct=0.08),
            100.0,
        )
        assert card["risk_reward"] == pytest.approx(1.2)


class TestOptions:
    def test_buy_calls_strike_above_spot(self) -> None:
        card = build_trade_card(_idea(action="buy_calls"), 100.0)
        assert card["is_option"] is True
        assert card["bullish"] is True
        assert card["suggested_strike"] == pytest.approx(105.0)  # 5% OTM call
        assert "pick nearest listed strike" in card["suggested_strike_note"]

    def test_buy_puts_strike_below_spot(self) -> None:
        card = build_trade_card(
            _idea(action="buy_puts", direction="bearish"), 100.0,
        )
        assert card["is_option"] is True
        assert card["bullish"] is False
        assert card["suggested_strike"] == pytest.approx(95.0)   # 5% OTM put

    def test_option_stop_uses_underlying_default_not_premium_pct(self) -> None:
        # The idea's stop_loss_pct for an option is a % of PREMIUM, not a
        # share move — the dollar stop must use the underlying default so
        # a premium % never masquerades as a price.
        card = build_trade_card(
            _idea(action="buy_puts", direction="bearish", stop_loss_pct=0.40),
            100.0,
        )
        # 100 × (1 + STOP_PCT_UNDERLYING_DEFAULT), bearish → above spot.
        assert card["stop_price"] == pytest.approx(
            100.0 * (1 + STOP_PCT_UNDERLYING_DEFAULT),
        )
        # card["stop_loss_pct"] is the UNDERLYING move used for the dollar
        # stop (the premium fraction is not a share move and must not
        # produce a share-price stop).
        assert card["stop_loss_pct"] == pytest.approx(STOP_PCT_UNDERLYING_DEFAULT)

    def test_option_dte_window_from_horizon(self) -> None:
        short = build_trade_card(_idea(action="buy_puts", time_horizon="short"), 100.0)
        med = build_trade_card(_idea(action="buy_puts", time_horizon="medium"), 100.0)
        assert short["dte_window"] == "1-3 weeks"
        assert med["dte_window"] == "3-6 weeks"
        assert "days out" in med["dte_note"]

    def test_stock_has_no_strike_or_dte(self) -> None:
        card = build_trade_card(_idea(action="long"), 100.0)
        assert card["suggested_strike"] is None
        assert card["dte_window"] == ""


class TestAbsentPrice:
    def test_no_price_returns_pct_only_card_with_flag(self) -> None:
        card = build_trade_card(_idea(action="long"), None)
        assert card["priced"] is False
        assert card["no_live_price"] is True
        assert card["current_price"] is None
        assert card["stop_price"] is None
        assert card["target_prices"] == []
        assert card["risk_reward"] is None
        assert card["suggested_strike"] is None
        # Percentages survive — they are the LLM's honest input.
        assert card["stop_loss_pct"] == pytest.approx(0.10)
        assert card["target_pct"] == [0.10, 0.20]
        # Entry falls back to the trigger, never a fabricated price.
        assert card["entry_zone"] == "on breakout"

    def test_zero_or_negative_price_treated_as_absent(self) -> None:
        for bad in (0.0, -5.0):
            card = build_trade_card(_idea(), bad)
            assert card["priced"] is False
            assert card["stop_price"] is None

    def test_non_numeric_price_never_crashes(self) -> None:
        card = build_trade_card(_idea(), "not a number")  # type: ignore[arg-type]
        assert card["priced"] is False


class TestRoundingAndEdges:
    def test_big_name_rounds_to_dollar(self) -> None:
        card = build_trade_card(_idea(action="long", stop_loss_pct=0.10), 172.4)
        # >= 100 → 1 decimal
        assert card["current_price"] == pytest.approx(172.4)
        assert card["stop_price"] == pytest.approx(155.2)  # 172.4 × 0.90

    def test_penny_name_keeps_precision(self) -> None:
        card = build_trade_card(_idea(action="long", stop_loss_pct=0.10), 3.20)
        # < 10 → 3 decimals
        assert card["stop_price"] == pytest.approx(2.88)   # 3.20 × 0.90

    def test_missing_stop_uses_stock_default(self) -> None:
        idea = _idea(action="long")
        idea.pop("stop_loss_pct")
        card = build_trade_card(idea, 100.0)
        assert card["stop_price"] == pytest.approx(
            100.0 * (1 - STOP_PCT_STOCK_DEFAULT),
        )

    def test_no_targets_means_no_risk_reward(self) -> None:
        card = build_trade_card(_idea(action="long", target_pct=[]), 100.0)
        assert card["target_prices"] == []
        assert card["risk_reward"] is None

    def test_change_pct_carried_through(self) -> None:
        card = build_trade_card(_idea(action="long"), 100.0, change_pct=-1.83)
        assert card["change_pct"] == pytest.approx(-1.83)

    def test_entry_zone_anchors_to_real_price_with_trigger(self) -> None:
        card = build_trade_card(_idea(action="long"), 100.0)
        assert "on breakout" in card["entry_zone"]
        assert "$100" in card["entry_zone"]

    def test_entry_zone_no_trigger_uses_price(self) -> None:
        card = build_trade_card(_idea(action="long", entry_trigger=""), 100.0)
        assert card["entry_zone"].startswith("near $100")

    def test_malformed_targets_dropped_not_fatal(self) -> None:
        card = build_trade_card(
            _idea(action="long", target_pct=[0.10, "junk", -0.5, None]),
            100.0,
        )
        # only the one valid positive fraction survives
        assert card["target_prices"] == [110.0]
