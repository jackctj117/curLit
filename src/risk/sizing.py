"""Position sizing — volatility-targeted, Kelly fraction, risk-parity."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PositionSizer:
    @staticmethod
    def fixed_fractional(
        capital: float,
        risk_pct: float,
        stop_distance: float,
        price: float,
    ) -> float:
        """Size a position risking fixed % of capital on a stop-loss.

        Args:
            capital: total account equity in currency units
            risk_pct: fraction of capital to risk (e.g. 0.01 = 1%)
            stop_distance: absolute distance from entry to stop in price units
            price: current entry price

        Returns:
            number of units to trade
        """
        assert capital > 0, f"capital must be positive, got {capital}"
        assert 0 < risk_pct <= 1, f"risk_pct must be in (0, 1], got {risk_pct}"
        assert stop_distance > 0, f"stop_distance must be positive, got {stop_distance}"
        assert price > 0, f"price must be positive, got {price}"

        risk_amount = capital * risk_pct
        size = risk_amount / stop_distance
        # cap at 100% of capital for sanity
        max_size = capital / price
        return min(size, max_size)

    @staticmethod
    def volatility_target(
        capital: float,
        target_vol: float,
        realized_vol: float,
        price: float,
    ) -> float:
        """Size a position to contribute target_vol annualized vol to portfolio.

        Uses the relationship: position_vol = notional_vol / equity
        where notional_vol = realized_vol (annualized) of the pair.

        target_vol = 0.10 (10% annualized) is typical for a single strategy leg
        per Architecture doc Section 5.1. Max 20% per position per RiskManager.
        """
        assert capital > 0, f"capital must be positive, got {capital}"
        assert 0 < target_vol <= 0.50, f"target_vol implausible: {target_vol}"
        assert price > 0, f"price must be positive, got {price}"

        if realized_vol <= 0:
            logger.warning("volatility_target: realized_vol=%.4f, returning 0", realized_vol)
            return 0.0

        notional = capital * target_vol / realized_vol
        size = notional / price

        logger.debug(
            "vol_target: equity=%.0f target_vol=%.2f rv=%.2f price=%.4f -> size=%.0f",
            capital,
            target_vol,
            realized_vol,
            price,
            size,
        )
        return size

    @staticmethod
    def adjust_for_liquidity(
        base_size: float,
        symbol: str,
        ts: Any,
        observed_spread_bps: float,
        profile: Any,
    ) -> float:
        """CL-4wi5: scale ``base_size`` by the liquidity-window multiplier.

        ``profile`` is a ``LiquidityProfile`` (or anything with the same
        ``size_multiplier(symbol, ts, observed_spread_bps)`` shape).
        Returns the size-adjusted notional. The multiplier is in
        [0.0, 1.0] — 0 means refuse the entry; 0.5 is a thin window;
        1.0 is normal.

        Strategies wire this in front of every new entry:

            base = PositionSizer.fixed_fractional(...)
            adj  = PositionSizer.adjust_for_liquidity(
                base, symbol, ts, spread_bps, liquidity_profile,
            )
            if adj > 0:
                submit_intent(adj)
        """
        if profile is None:
            return base_size
        try:
            multiplier = profile.size_multiplier(
                symbol,
                ts,
                observed_spread_bps,
            )
        except Exception as exc:
            # Fail CLOSED: if we can't evaluate the liquidity window we
            # can't verify it's safe to enter — return 0 (no entry), same
            # missing-data posture as the rest of the money path.
            logger.warning(
                "liquidity_window: profile.size_multiplier raised for %s "
                "(%s: %s) — cannot verify liquidity, refusing entry",
                symbol,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return 0.0
        return float(base_size * multiplier)

    @staticmethod
    def kelly(edge: float, odds: float, kelly_fraction: float = 0.25) -> float:
        """Kelly-inspired position sizing fraction, clamped to [0, 1].

        Full Kelly: f* = edge - (1 - edge) / odds
        Half Kelly produces 75% of growth rate with 25% of drawdown risk (Thorp 1997).
        We default to 0.25 (quarter Kelly) for survival: higher retention of
        growth with dramatically lower ruin probability.

        Inputs are clamped, not asserted: a noisy estimator producing edge < 0
        or edge > 1 returns 0 size (no trade) rather than crashing the strategy
        mid-tick. Only kelly_fraction is asserted because it's a config knob
        the operator owns, not a runtime input.
        """
        assert 0 <= kelly_fraction <= 1, f"fraction must be in [0, 1], got {kelly_fraction}"

        if odds <= 0:
            logger.warning("kelly: odds=%.3f, returning 0", odds)
            return 0.0

        full_kelly = edge - (1.0 - edge) / odds
        result = max(0.0, min(1.0, full_kelly * kelly_fraction))

        logger.debug(
            "kelly: edge=%.3f odds=%.2f full=%.3f frac=%.3f", edge, odds, full_kelly, result
        )
        return result
