"""Unit tests — horizon-based instrument selector (CL-mgcp).

Full decision matrix: horizon bands × direction × prefer_defined_risk
× iv_rank absent/present. The iv_rank=None cases pin the honesty
contract: NO IV logic runs until a caller supplies a real value.
"""

from __future__ import annotations

import pytest

from src.events.instrument_selector import (
    HORIZON_DEFAULT_DAYS,
    OPTION_STOP_MEDIUM,
    OPTION_STOP_SHORT,
    STOCK_STOP_LONG,
    STOCK_STOP_MEDIUM,
    choose_instrument,
    decision_for_idea,
    parse_holding_days,
)

# --------------------------------------------------------------------- #
# Band matrix
# --------------------------------------------------------------------- #


class TestShortHorizon:
    @pytest.mark.parametrize("days", [1, 3, 8])
    def test_bearish_prefers_puts(self, days: float) -> None:
        d = choose_instrument("bearish", days)
        assert d.action == "buy_puts"
        assert d.stop_loss_pct == OPTION_STOP_SHORT
        assert "1-3 weeks" in d.preferred_instrument
        assert d.horizon_band == "short"

    @pytest.mark.parametrize("days", [1, 8])
    def test_bullish_prefers_calls(self, days: float) -> None:
        d = choose_instrument("bullish", days)
        assert d.action == "buy_calls"
        assert d.stop_loss_pct == OPTION_STOP_SHORT
        assert "calls" in d.preferred_instrument

    def test_non_positive_days_clamps_to_short_band(self) -> None:
        assert choose_instrument("bearish", 0).horizon_band == "short"
        assert choose_instrument("bearish", -3).horizon_band == "short"

    def test_prefer_defined_risk_false_still_options_in_short_band(self) -> None:
        # Short-horizon event moves gap; defined risk is not optional.
        assert choose_instrument("bearish", 5, prefer_defined_risk=False).is_options


class TestMediumHorizon:
    @pytest.mark.parametrize("days", [9, 20, 25])
    def test_defined_risk_default_options(self, days: float) -> None:
        d = choose_instrument("bearish", days)
        assert d.action == "buy_puts"
        assert d.stop_loss_pct == OPTION_STOP_MEDIUM
        assert "3-6 weeks" in d.preferred_instrument
        assert d.horizon_band == "medium"

    def test_no_defined_risk_pref_gives_stock(self) -> None:
        d = choose_instrument("bearish", 15, prefer_defined_risk=False)
        assert d.action == "short"
        assert d.preferred_instrument == "stock"
        assert d.stop_loss_pct == STOCK_STOP_MEDIUM

    def test_bullish_stock_is_long(self) -> None:
        d = choose_instrument("bullish", 15, prefer_defined_risk=False)
        assert d.action == "long"


class TestStructuralHorizon:
    @pytest.mark.parametrize("days", [26, 60, 200])
    def test_stock_regardless_of_defined_risk_pref(self, days: float) -> None:
        for pref in (True, False):
            d = choose_instrument("bullish", days, prefer_defined_risk=pref)
            assert d.action == "long"
            assert d.preferred_instrument == "stock"
            assert d.stop_loss_pct == STOCK_STOP_LONG
            assert d.horizon_band == "structural"


class TestDirectionNormalisation:
    @pytest.mark.parametrize("direction", ["bearish", "short", "buy_puts", " BEARISH "])
    def test_bearish_synonyms(self, direction: str) -> None:
        assert choose_instrument(direction, 3).action == "buy_puts"

    @pytest.mark.parametrize("direction", ["bullish", "long", "buy_calls"])
    def test_bullish_synonyms(self, direction: str) -> None:
        assert choose_instrument(direction, 3).action == "buy_calls"

    def test_garbage_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            choose_instrument("sideways", 3)


# --------------------------------------------------------------------- #
# IV overlay — inert without data, active with it
# --------------------------------------------------------------------- #


class TestIvRank:
    def test_none_means_no_iv_logic(self) -> None:
        # The honesty contract: identical output with iv_rank omitted —
        # no fake IV default sneaks in anywhere.
        assert choose_instrument("bearish", 3) == choose_instrument(
            "bearish", 3, iv_rank=None,
        )

    def test_low_iv_keeps_options(self) -> None:
        d = choose_instrument("bearish", 3, iv_rank=30)
        assert d.action == "buy_puts"

    def test_high_iv_flips_options_to_stock(self) -> None:
        d = choose_instrument("bearish", 3, iv_rank=80)
        assert d.action == "short"
        assert d.preferred_instrument == "stock"
        assert d.stop_loss_pct == STOCK_STOP_MEDIUM
        assert "IV rank 80" in d.reason

    def test_high_iv_threshold_inclusive(self) -> None:
        assert choose_instrument("bullish", 3, iv_rank=74.9).is_options
        flipped = choose_instrument("bullish", 3, iv_rank=75.0)
        assert not flipped.is_options
        assert flipped.action == "long"

    def test_high_iv_on_stock_decision_is_noop(self) -> None:
        d = choose_instrument("bullish", 60, iv_rank=99)
        assert d.action == "long"
        assert d.stop_loss_pct == STOCK_STOP_LONG
        assert "IV rank" not in d.reason

    def test_high_iv_structural_band_boundary_stop(self) -> None:
        # 26d options never happen (structural band = stock), but a
        # medium-band flip at 25d uses the medium stock stop.
        d = choose_instrument("bearish", 25, iv_rank=90)
        assert d.stop_loss_pct == STOCK_STOP_MEDIUM


# --------------------------------------------------------------------- #
# Holding-period parsing + idea wrapper
# --------------------------------------------------------------------- #


class TestParseHoldingDays:
    def test_range_midpoint(self) -> None:
        assert parse_holding_days("2-7") == 4.5

    def test_single_number(self) -> None:
        assert parse_holding_days("5") == 5.0

    def test_decimals(self) -> None:
        assert parse_holding_days("1.5-2.5") == 2.0

    @pytest.mark.parametrize("raw", ["", None, "soon", "a few days"])
    def test_no_numbers_is_none(self, raw: str | None) -> None:
        assert parse_holding_days(raw) is None


class TestDecisionForIdea:
    def test_uses_holding_period_midpoint(self) -> None:
        idea = {"direction": "bearish", "holding_period_days": "2-6",
                "time_stop_days": 60, "time_horizon": "structural"}
        d = decision_for_idea(idea)
        assert d is not None
        assert d.horizon_band == "short"  # midpoint 4 wins over the rest

    def test_falls_back_to_time_stop_days(self) -> None:
        idea = {"direction": "bullish", "holding_period_days": "",
                "time_stop_days": 20}
        d = decision_for_idea(idea)
        assert d is not None
        assert d.horizon_band == "medium"

    def test_falls_back_to_horizon_default(self) -> None:
        idea = {"direction": "bullish", "time_horizon": "structural"}
        d = decision_for_idea(idea)
        assert d is not None
        assert d.horizon_band == "structural"
        assert HORIZON_DEFAULT_DAYS["structural"] > 25

    def test_direction_falls_back_to_action(self) -> None:
        d = decision_for_idea({"action": "buy_puts", "time_stop_days": 3})
        assert d is not None
        assert d.action == "buy_puts"

    def test_unusable_direction_returns_none(self) -> None:
        assert decision_for_idea({"action": "hedge", "time_stop_days": 3}) is None
