"""Kelly fractional sizer for Polymarket binary outcomes (CL-poly-2).

Why a separate sizer: the FX vol-target sizer (src/risk/sizing.py) assumes
continuous P&L bounded by a stop-loss. Neither holds on Polymarket — a
share resolves to {0, 1} at expiry, so max loss = position cost
(no stop-loss exists). The right framework is Kelly fraction on edge
vs implied probability.

Reference: Plan §3.

Math (BUY YES at price p with model prob q):
  Expected return per dollar staked: q*(1/p - 1) + (1-q)*(-1) = q/p - 1
  Optimal Kelly fraction:             f* = (q - p) / (1 - p)

(Symmetric for BUY NO: edge = p - q, f* = (p - q) / p)

We apply fractional Kelly (default 0.25×) and a hard max-market-fraction
cap to bound drawdown variance and limit single-market blow-up.

Ignored cases (return None to caller):
  * edge below ``min_edge`` — model isn't confident enough
  * f* nonpositive — wrong-side edge
  * sizer's bankroll is zero or negative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


# Defaults — operator-tunable per call. Sourced from plan §3.

# 0.25 = quarter-Kelly. Half-Kelly captures most of the growth rate
# at much lower drawdown variance (Thorp 1997). Quarter-Kelly is even
# more conservative — appropriate for a binary venue where any single
# wrong call resolves to -100%, not the -2% you'd expect from FX.
_DEFAULT_KELLY_FRACTION: Decimal = Decimal("0.25")

# 5% per-market cap. Above this, a single market loss starts to dominate
# the bankroll's variance regardless of Kelly's optimality.
_DEFAULT_MAX_MARKET_FRACTION: Decimal = Decimal("0.05")

# 3% absolute edge floor (3 percentage points between model q and market
# p). Below this the spread + slippage + gas dominate the modeled edge.
_DEFAULT_MIN_EDGE: Decimal = Decimal("0.03")

# Polymarket prices live in [0.01, 0.99] in 0.01 ticks.
_PRICE_FLOOR: Decimal = Decimal("0.01")
_PRICE_CEIL: Decimal = Decimal("0.99")


@dataclass(frozen=True)
class PolySizeDecision:
    shares: Decimal
    cost_usdc: Decimal
    fraction_of_bankroll: Decimal
    rationale: str


def size_polymarket_order(
    *,
    bankroll: Decimal,
    market_price: Decimal,
    model_prob: Decimal,
    side: str,
    kelly_fraction: Decimal = _DEFAULT_KELLY_FRACTION,
    max_market_fraction: Decimal = _DEFAULT_MAX_MARKET_FRACTION,
    min_edge: Decimal = _DEFAULT_MIN_EDGE,
) -> PolySizeDecision | None:
    """Return a PolySizeDecision or None if the trade is below threshold.

    Args:
      bankroll: free USDC.e (or USDC) — total capital available for sizing.
      market_price: current limit price in [0.01, 0.99].
      model_prob: our model's estimated probability the YES side wins.
      side: "YES" or "NO". For NO, edge = p - q; for YES, edge = q - p.
      kelly_fraction: scale on the full Kelly. 0.25 default.
      max_market_fraction: hard cap on bankroll fraction per market.
      min_edge: minimum edge to bother trading. 0.03 default (3pp).

    Returns:
      PolySizeDecision when edge >= min_edge AND f* > 0.
      None when the call shouldn't be taken.

    Raises:
      ValueError on out-of-range inputs (price outside [0.01, 0.99],
      model_prob outside [0, 1], unknown side, non-positive bankroll).
    """
    side_u = side.upper()
    if side_u not in {"YES", "NO"}:
        msg = f"side must be 'YES' or 'NO', got {side!r}"
        raise ValueError(msg)
    if not (_PRICE_FLOOR <= market_price <= _PRICE_CEIL):
        msg = f"market_price {market_price} outside [{_PRICE_FLOOR}, {_PRICE_CEIL}]"
        raise ValueError(msg)
    if not (Decimal("0") <= model_prob <= Decimal("1")):
        msg = f"model_prob {model_prob} outside [0, 1]"
        raise ValueError(msg)
    if bankroll <= 0:
        msg = f"bankroll must be positive, got {bankroll}"
        raise ValueError(msg)
    if kelly_fraction <= 0 or kelly_fraction > Decimal("1"):
        msg = f"kelly_fraction must be in (0, 1], got {kelly_fraction}"
        raise ValueError(msg)
    if max_market_fraction <= 0 or max_market_fraction > Decimal("1"):
        msg = f"max_market_fraction must be in (0, 1], got {max_market_fraction}"
        raise ValueError(msg)

    p, q = market_price, model_prob

    if side_u == "YES":
        edge = q - p
        if edge < min_edge:
            return None
        # f* = (q - p) / (1 - p). Numerically stable when p in [0.01, 0.99].
        f_star = edge / (Decimal("1") - p)
    else:  # NO
        edge = p - q
        if edge < min_edge:
            return None
        f_star = edge / p

    if f_star <= 0:
        return None

    f = min(f_star * kelly_fraction, max_market_fraction)
    cost = (bankroll * f).quantize(Decimal("0.01"))
    if cost <= 0:
        return None

    # Buying at price p: shares = cost / p. Polymarket size is in shares
    # (each share has unit USDC face value at resolution = 1 if YES wins).
    shares = (cost / p).quantize(Decimal("0.01"))
    if shares <= 0:
        return None

    rationale = (
        f"side={side_u} p={p} q={q} edge={edge:.4f} "
        f"f*={f_star:.4f} kf={kelly_fraction} cap={max_market_fraction} "
        f"-> f={f:.4f}"
    )
    logger.debug("polymarket sizer: %s", rationale)

    return PolySizeDecision(
        shares=shares,
        cost_usdc=cost,
        fraction_of_bankroll=f,
        rationale=rationale,
    )
