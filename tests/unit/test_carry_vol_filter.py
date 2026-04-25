"""Unit tests — strategies.carry_vol_filter (CL-c77 / A5).

Acceptance:
- Strategy generates intents on monthly rebalance
- Reduces exposure when vol spikes
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

from src.execution.broker import Account
from src.strategies.carry_vol_filter import (
    _RATE_SERIES_MAP,
    _USD_BASE_PAIRS,
    CarryVolFilterConfig,
    CarryVolFilterStrategy,
)

# =============================================================================
# Fakes
# =============================================================================


class _FakeDataProvider:
    """Test data provider with canned rate / vol series."""

    def __init__(
        self,
        rates: dict[str, float] | None = None,
        vol_series: list[float] | None = None,
    ) -> None:
        self.rates = rates or {}
        self.vol_series = vol_series

    def get_latest_value(self, series_id: str, as_of: datetime) -> float | None:
        # Map series_id back to currency, return matching rate.
        for ccy, sid in _RATE_SERIES_MAP.items():
            if sid == series_id:
                return self.rates.get(ccy)
        return None

    def get_series(
        self, name: str, start: datetime, end: datetime,
    ) -> list[float] | None:
        return self.vol_series


class _FakeBroker:
    def __init__(self, equity: float = 100_000) -> None:
        self._account = Account(balance=equity, equity=equity)

    def get_account(self) -> Account:
        return self._account


def _build_prices(quote_per_pair: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Build the price-tick dict the engine passes to generate_intents."""
    return {
        pair: {"bid": p - 0.0001, "ask": p + 0.0001}
        for pair, p in quote_per_pair.items()
    }


# Default G10-ish prices — bid/ask within 1 pip of mid.
_DEFAULT_PRICES = _build_prices({
    "EURUSD": 1.1000,
    "USDJPY": 150.00,
    "GBPUSD": 1.2500,
    "USDCHF": 0.9000,
    "USDCAD": 1.3500,
    "AUDUSD": 0.6700,
    "NZDUSD": 0.6100,
    "USDNOK": 10.50,
    "USDSEK": 10.80,
})


# =============================================================================
# Currency / pair conventions
# =============================================================================


class TestCurrencyMapping:
    def test_x_usd_pair(self) -> None:
        # EUR is X/USD → long EUR = buy EURUSD.
        pair, side = CarryVolFilterStrategy._currency_to_pair("EUR", 1)
        assert pair == "EURUSD"
        assert side == 1

    def test_usd_x_pair_inverts_side(self) -> None:
        # JPY is USD/JPY → long JPY = sell USDJPY (-1).
        pair, side = CarryVolFilterStrategy._currency_to_pair("JPY", 1)
        assert pair == "USDJPY"
        assert side == -1

    def test_short_x_usd_is_sell(self) -> None:
        pair, side = CarryVolFilterStrategy._currency_to_pair("EUR", -1)
        assert pair == "EURUSD"
        assert side == -1

    def test_short_usd_x_is_buy(self) -> None:
        pair, side = CarryVolFilterStrategy._currency_to_pair("JPY", -1)
        assert pair == "USDJPY"
        assert side == 1

    def test_usd_returns_empty(self) -> None:
        pair, side = CarryVolFilterStrategy._currency_to_pair("USD", 1)
        assert pair == ""
        assert side == 0

    def test_usd_base_pairs_set(self) -> None:
        assert "JPY" in _USD_BASE_PAIRS
        assert "CHF" in _USD_BASE_PAIRS
        assert "EUR" not in _USD_BASE_PAIRS


# =============================================================================
# Basket construction
# =============================================================================


class TestBasketConstruction:
    def test_top_k_bottom_k(self) -> None:
        rates = {
            "EUR": 0.04, "JPY": 0.005, "GBP": 0.045, "AUD": 0.045,
            "NZD": 0.05, "CHF": 0.01, "CAD": 0.04, "NOK": 0.04,
        }
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(top_k=3, bottom_k=3))
        b = strat._construct_baskets(rates)
        # Long basket: 3 highest yielders.
        assert len(b["long"]) == 3
        long_ccys = {c for c, _, _ in b["long"]}
        assert "NZD" in long_ccys
        # Short basket: 3 lowest yielders.
        assert len(b["short"]) == 3
        short_ccys = {c for c, _, _ in b["short"]}
        assert "JPY" in short_ccys

    def test_insufficient_currencies_returns_empty(self) -> None:
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(top_k=3, bottom_k=3))
        # Only 4 currencies, can't fit 3+3.
        b = strat._construct_baskets({"EUR": 0.04, "JPY": 0.005, "GBP": 0.05, "USD": 0.04})
        assert b["long"] == []
        assert b["short"] == []

    def test_min_spread_gate(self) -> None:
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(
            top_k=2, bottom_k=2, min_rate_spread=0.05,
        ))
        # Tiny spread between top and bottom — below gate.
        b = strat._construct_baskets({
            "EUR": 0.04, "JPY": 0.04, "GBP": 0.04, "AUD": 0.04, "NZD": 0.04, "CHF": 0.04,
        })
        assert b["long"] == []
        assert b["short"] == []

    def test_usd_excluded_from_basket(self) -> None:
        rates = {f"X{i}": 0.04 + i*0.001 for i in range(8)}
        rates["USD"] = 0.10  # very high, would dominate if included
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(top_k=2, bottom_k=2))
        # We're using made-up tickers — but USD must not appear in baskets.
        b = strat._construct_baskets(rates)
        ccys_in_baskets = (
            [c for c, _, _ in b["long"]] + [c for c, _, _ in b["short"]]
        )
        assert "USD" not in ccys_in_baskets


# =============================================================================
# Vol filter
# =============================================================================


class TestVolFilter:
    def test_zero_z_when_data_missing(self) -> None:
        strat = CarryVolFilterStrategy(data_provider=None)
        assert strat._compute_vol_z_score(datetime.now(UTC)) == 0.0

    def test_zero_z_on_short_history(self) -> None:
        provider = _FakeDataProvider(vol_series=[10.0] * 20)  # < 0.8 * 120
        strat = CarryVolFilterStrategy(
            CarryVolFilterConfig(vol_lookback_days=120),
            data_provider=provider,
        )
        assert strat._compute_vol_z_score(datetime.now(UTC)) == 0.0

    def test_positive_z_on_vol_spike(self) -> None:
        # Stable history then a spike — z should be strongly positive.
        history = [10.0] * 119 + [50.0]
        provider = _FakeDataProvider(vol_series=history)
        strat = CarryVolFilterStrategy(
            CarryVolFilterConfig(vol_lookback_days=120),
            data_provider=provider,
        )
        z = strat._compute_vol_z_score(datetime.now(UTC))
        assert z > 5.0  # extreme outlier

    def test_zero_when_zero_variance(self) -> None:
        provider = _FakeDataProvider(vol_series=[10.0] * 200)
        strat = CarryVolFilterStrategy(
            CarryVolFilterConfig(vol_lookback_days=120),
            data_provider=provider,
        )
        assert strat._compute_vol_z_score(datetime.now(UTC)) == 0.0


class TestExposureMultiplier:
    def setup_method(self) -> None:
        self.strat = CarryVolFilterStrategy()

    def test_normal_regime_full_exposure(self) -> None:
        assert self.strat._exposure_multiplier(0.0) == 1.00

    def test_mild_spike_75pct(self) -> None:
        assert self.strat._exposure_multiplier(1.0) == 0.75
        assert self.strat._exposure_multiplier(1.5) == 0.75

    def test_strong_spike_50pct(self) -> None:
        assert self.strat._exposure_multiplier(2.0) == 0.50
        assert self.strat._exposure_multiplier(2.99) == 0.50

    def test_extreme_zero_exposure(self) -> None:
        assert self.strat._exposure_multiplier(3.5) == 0.00

    def test_negative_z_uses_absolute(self) -> None:
        assert self.strat._exposure_multiplier(-2.5) == 0.50


# =============================================================================
# Rebalance scheduling
# =============================================================================


class TestRebalanceScheduling:
    def test_first_call_rebalances_if_past_day(self) -> None:
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(rebalance_day=1))
        # Day 5 of month, never rebalanced before → should rebalance.
        assert strat._should_rebalance(datetime(2026, 4, 5, tzinfo=UTC))

    def test_does_not_rebalance_again_in_same_month(self) -> None:
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(rebalance_day=1))
        strat._last_rebalance = date(2026, 4, 1)
        assert not strat._should_rebalance(datetime(2026, 4, 15, tzinfo=UTC))

    def test_rebalances_in_new_month(self) -> None:
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(rebalance_day=1))
        strat._last_rebalance = date(2026, 4, 1)
        assert strat._should_rebalance(datetime(2026, 5, 2, tzinfo=UTC))

    def test_does_not_rebalance_before_day(self) -> None:
        strat = CarryVolFilterStrategy(CarryVolFilterConfig(rebalance_day=15))
        # Earlier in the month than configured day.
        assert not strat._should_rebalance(datetime(2026, 4, 5, tzinfo=UTC))


# =============================================================================
# generate_intents end-to-end
# =============================================================================


class TestGenerateIntents:
    def _strategy(
        self, rates: dict[str, float], vol_series: list[float] | None = None,
    ) -> CarryVolFilterStrategy:
        return CarryVolFilterStrategy(
            CarryVolFilterConfig(top_k=3, bottom_k=3),
            data_provider=_FakeDataProvider(rates=rates, vol_series=vol_series),
            state_store=None,
        )

    def test_initial_rebalance_emits_long_short_intents(self) -> None:
        # All 9 non-USD G10 with distinct rates so baskets are well-defined.
        rates = {
            "EUR": 0.04, "JPY": 0.005, "GBP": 0.045, "AUD": 0.045,
            "NZD": 0.05, "CHF": 0.01, "CAD": 0.04, "NOK": 0.04, "SEK": 0.03,
            "USD": 0.045,
        }
        strat = self._strategy(rates)
        intents = asyncio.run(
            strat.generate_intents(_DEFAULT_PRICES, _FakeBroker()),
        )
        # 6 currencies in baskets → 6 intents (excluding any USD-only).
        assert len(intents) == 6
        # All must have nonzero target_position because we just opened.
        assert all(abs(i.target_position) > 0 for i in intents)

    def test_no_rebalance_when_spread_too_low(self) -> None:
        # All rates equal → spread = 0 → flat positions.
        rates = {ccy: 0.04 for ccy in _RATE_SERIES_MAP}
        strat = self._strategy(rates)
        intents = asyncio.run(
            strat.generate_intents(_DEFAULT_PRICES, _FakeBroker()),
        )
        # No new positions opened (any flatten intents would be for empty).
        assert all(i.target_position == 0 for i in intents)

    def test_high_vol_reduces_exposure_on_existing_positions(self) -> None:
        # Set up: first call with low vol → builds positions.
        # Second call with vol spike → should emit downsizing intents.
        rates = {
            "EUR": 0.04, "JPY": 0.005, "GBP": 0.045, "AUD": 0.045,
            "NZD": 0.05, "CHF": 0.01, "CAD": 0.04, "NOK": 0.04, "SEK": 0.03,
            "USD": 0.045,
        }
        # First call: stable history, no z spike.
        provider = _FakeDataProvider(rates=rates, vol_series=[10.0] * 200)
        strat = CarryVolFilterStrategy(
            CarryVolFilterConfig(top_k=3, bottom_k=3, rebalance_day=1),
            data_provider=provider,
        )
        # Push the rebalance "now" past the rebalance_day.
        asyncio.run(
            strat.generate_intents(_DEFAULT_PRICES, _FakeBroker()),
        )
        assert len(strat.current_positions) == 6
        baseline_exposure = strat._current_exposure
        assert baseline_exposure == 1.0

        # Second call: vol spikes → exposure drops to 0.50 (z>=2) or 0.0 (z>=3).
        provider.vol_series = [10.0] * 119 + [200.0]  # massive spike
        # Force still in same month so monthly rebalance does NOT fire — only
        # the vol-filter-triggered scaling path runs.
        strat._last_rebalance = datetime.now(UTC).date()
        scaled = asyncio.run(
            strat.generate_intents(_DEFAULT_PRICES, _FakeBroker()),
        )
        assert strat._current_exposure < baseline_exposure
        # All 6 currently-held positions should get scaled (some downsized).
        assert len(scaled) > 0


# =============================================================================
# Symbols + interface
# =============================================================================


class TestInterface:
    def test_id_and_signal_interval(self) -> None:
        strat = CarryVolFilterStrategy()
        assert strat.id == "carry_vol_filter"
        assert strat.signal_interval_seconds == 86_400

    def test_symbols_excludes_usd(self) -> None:
        strat = CarryVolFilterStrategy()
        assert "USD" not in strat.symbols
        # All G10 ex-USD pairs should appear.
        assert "EURUSD" in strat.symbols
        assert "USDJPY" in strat.symbols

    def test_fit_and_generate_signals_noop(self) -> None:
        strat = CarryVolFilterStrategy()
        # Don't raise.
        strat.fit(None)
        assert strat.generate_signals(None) is None
