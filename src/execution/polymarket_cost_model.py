"""Polymarket cost model — fees, slippage proxy, gas (CL-poly-2).

OANDA cost = spread + per-unit commission. Polymarket has three different
axes per fill:

  * Maker fee — currently 0 bps (read from /markets/{id} per-market;
    DO NOT hardcode at strategy time)
  * Taker fee — currently 0 bps headline, but reflected in slippage
  * Gas — settlement on Polygon, ~$0.01/fill, but spikes on congestion
  * Spread / book impact — wide on illiquid markets, can dwarf modeled edge

Reference: Plan §5.

Cost-model contract: ``estimate(side, role, price, size, book_depth_at_price)``
returns Decimal USDC absolute cost. The risk sizer subtracts this from
expected return before deciding whether to enter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


# Default fee assumptions. Polymarket headline fees are 0 today (2026-05);
# markets sometimes carry per-market overrides, so the live broker should
# fetch ``/markets/{id}`` and pass per-market values into this model
# rather than relying on these defaults. Constants here exist so unit
# tests can exercise the math without a live API call.
_DEFAULT_MAKER_FEE_BPS: Decimal = Decimal("0")
_DEFAULT_TAKER_FEE_BPS: Decimal = Decimal("0")

# Polygon gas: an OrderFilled event at typical gwei prices costs ~0.0005
# MATIC. At MATIC ~$0.50 that's ~$0.0003, but congestion days can push
# fills to $0.05+. We set $0.01 as a single-fill estimate that's
# conservative on average and not insane on a spike day. Operator can
# raise via the constructor if Polygon gas is unusually high.
_DEFAULT_GAS_USDC_ESTIMATE: Decimal = Decimal("0.01")

# 50bps placeholder slippage when our order size exceeds half the book
# depth at our price level. Calibrate from observed fills (CL-poly-3
# follow-up).
_DEFAULT_SLIPPAGE_BPS_ON_DEEP_TAKER: Decimal = Decimal("50")
_DEEP_TAKER_BOOK_FRACTION: Decimal = Decimal("0.5")


@dataclass(frozen=True)
class PolymarketCostModel:
    """Per-fill cost estimator. All Decimal so the math composes cleanly
    with the sizer's bankroll-fraction arithmetic."""

    maker_fee_bps: Decimal = _DEFAULT_MAKER_FEE_BPS
    taker_fee_bps: Decimal = _DEFAULT_TAKER_FEE_BPS
    gas_usdc_estimate: Decimal = _DEFAULT_GAS_USDC_ESTIMATE
    slippage_bps_on_deep_taker: Decimal = _DEFAULT_SLIPPAGE_BPS_ON_DEEP_TAKER

    def estimate(
        self,
        *,
        side: str,
        role: str,
        price: Decimal,
        size: Decimal,
        book_depth_at_price: Decimal | None = None,
    ) -> Decimal:
        """Estimate USDC cost of a single fill.

        Args:
          side: "BUY"/"SELL" (or "YES"/"NO" — direction-agnostic for cost).
          role: "maker" or "taker". Determines which fee schedule applies.
          price: limit price in [0.01, 0.99].
          size: share count.
          book_depth_at_price: optional resting size at our price level.
            When provided, slippage adds when our order exceeds half the
            book — proxy for crossing the spread on a thin book.

        Returns:
          Decimal USDC. Floor 0 (a fill never has negative cost in this
          model — maker rebates would be negative, but headline is 0
          and we don't speculatively assume rebates we haven't observed).
        """
        del side  # kept for signature symmetry with FX cost models
        notional = price * size

        fee_bps = self.maker_fee_bps if role.lower() == "maker" else self.taker_fee_bps
        fee = notional * fee_bps / Decimal("10000")

        slippage = Decimal("0")
        if (
            role.lower() == "taker"
            and book_depth_at_price is not None
            and book_depth_at_price > 0
            and size > book_depth_at_price * _DEEP_TAKER_BOOK_FRACTION
        ):
            slippage = notional * self.slippage_bps_on_deep_taker / Decimal("10000")

        total = fee + slippage + self.gas_usdc_estimate
        logger.debug(
            "poly cost: notional=%s fee=%s slip=%s gas=%s total=%s",
            notional,
            fee,
            slippage,
            self.gas_usdc_estimate,
            total,
        )
        return max(Decimal("0"), total)
