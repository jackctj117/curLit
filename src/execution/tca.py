"""Per-fill Transaction Cost Analysis (CL-sl7x).

When a strategy underperforms in live vs paper, we need to decide: is it the
signal, the sizing, or execution? G4 (paper-live divergence) compares live
fills to a parallel paper-broker fill. TCA goes deeper: each live fill is
decomposed against the arrival-time mid into three additive cost components.

Components (all in bps, positive = cost to us):

    queue_bps      Half-spread at submission. The cheapest you could have
                   filled by being passive in the order book — a market
                   order pays the full half-spread; a limit order pays
                   less (or earns the rebate).

    impact_bps     Mid-price drift between submission and fill. Captures
                   our footprint — large orders move the market against
                   us before we complete.

    broker_bps     Fill price vs (arrival mid + queue + impact). What's
                   left is the broker's fill-quality miss against the
                   notional reference price — slippage attributable to
                   the venue, not our order behavior.

    total_bps      queue + impact + broker.

    implementation_shortfall_bps    Realized fill price vs arrival mid,
                                    independent of decomposition. This
                                    must equal total_bps under perfect
                                    accounting; we report both as a
                                    cross-check.

Pairs with G4: G4 measures paper-vs-live distribution; TCA decomposes the
per-trade source. Wire by recording TCAComponents into the trade_journal
ORDER_FILLED payload and emitting the per-component Histogram.

Reference: CL-sl7x.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.execution.broker import Fill
from src.monitoring.metrics import (
    tca_component_bps,
    tca_implementation_shortfall_bps,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillContext:
    """Inputs needed to decompose a fill into TCA components.

    arrival_mid:    mid-quote at the moment the order was decided/submitted.
    submit_spread:  bid-ask spread at submission, in price units (e.g. for
                    EURUSD ~0.0001 = 1 pip).
    fill_mid:       mid-quote at fill time. Captures the impact of the time
                    elapsed between submission and execution.
    """

    arrival_mid: float
    submit_spread: float
    fill_mid: float

    def __post_init__(self) -> None:
        assert self.arrival_mid > 0, f"arrival_mid must be > 0, got {self.arrival_mid}"
        assert self.fill_mid > 0, f"fill_mid must be > 0, got {self.fill_mid}"
        assert self.submit_spread >= 0, f"submit_spread must be >= 0, got {self.submit_spread}"


@dataclass
class TCAComponents:
    """Decomposed cost breakdown of one fill, in bps.

    All four values use the same sign convention: positive means cost to us.
    A favorable fill (e.g. queue-rebate) registers as negative.
    """

    queue_bps: float
    impact_bps: float
    broker_bps: float
    total_bps: float
    implementation_shortfall_bps: float
    pair: str
    side: str
    fill_id: str
    ts: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_bps": self.queue_bps,
            "impact_bps": self.impact_bps,
            "broker_bps": self.broker_bps,
            "total_bps": self.total_bps,
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "pair": self.pair,
            "side": self.side,
            "fill_id": self.fill_id,
            "ts": self.ts.isoformat(),
        }


@dataclass
class TCASummary:
    """Aggregated TCA across many fills (e.g. a week)."""

    n_fills: int
    avg_queue_bps: float
    avg_impact_bps: float
    avg_broker_bps: float
    avg_total_bps: float
    avg_implementation_shortfall_bps: float
    p50_total_bps: float
    p95_total_bps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_fills": self.n_fills,
            "avg_queue_bps": self.avg_queue_bps,
            "avg_impact_bps": self.avg_impact_bps,
            "avg_broker_bps": self.avg_broker_bps,
            "avg_total_bps": self.avg_total_bps,
            "avg_implementation_shortfall_bps": self.avg_implementation_shortfall_bps,
            "p50_total_bps": self.p50_total_bps,
            "p95_total_bps": self.p95_total_bps,
        }


# =============================================================================
# Computation
# =============================================================================


def _signed_bps(price_diff: float, reference: float, side: str) -> float:
    """Convert a price difference to bps, with sign = cost-to-us.

    For a buy: paying MORE than reference is cost (positive bps).
    For a sell: receiving LESS than reference is cost (positive bps).
    """
    if reference <= 0:
        return 0.0
    sign = 1.0 if side == "buy" else -1.0
    return sign * price_diff / reference * 10_000.0


def compute_tca(
    fill: Fill,
    context: FillContext,
) -> TCAComponents:
    """Decompose one fill into queue / impact / broker bps + total IS.

    Algorithm:
        queue_bps  = half-spread at submission, in cost-units
                     (we ALWAYS owe this on a market order — it's the
                     cheapest path that crosses the book)
        impact_bps = (fill_mid - arrival_mid) signed by side, in cost-units
                     (mid drift between submit and fill)
        broker_bps = (fill_price - fill_mid - half_submit_spread)
                     signed by side, in cost-units
                     (residual = how much worse than fill_mid + queue we got)
        total_bps  = queue + impact + broker
        IS_bps     = (fill_price - arrival_mid) signed by side
                     (independent cross-check; must equal total_bps)
    """
    half_spread = context.submit_spread / 2.0
    queue_bps = _signed_bps(half_spread, context.arrival_mid, "buy")  # always cost
    # Note: queue is by convention always positive (paying half-spread); we
    # compute via "buy" sign so it's always +bps regardless of trade side.

    impact_bps = _signed_bps(
        context.fill_mid - context.arrival_mid,
        context.arrival_mid,
        fill.side,
    )

    # Broker fill quality: how far is the broker's fill price from the
    # "expected" price (fill_mid + half-spread for our side)?
    expected_price_at_fill = context.fill_mid + (
        half_spread if fill.side == "buy" else -half_spread
    )
    broker_bps = _signed_bps(
        fill.price - expected_price_at_fill,
        context.arrival_mid,
        fill.side,
    )

    total_bps = queue_bps + impact_bps + broker_bps

    is_bps = _signed_bps(
        fill.price - context.arrival_mid,
        context.arrival_mid,
        fill.side,
    )

    return TCAComponents(
        queue_bps=queue_bps,
        impact_bps=impact_bps,
        broker_bps=broker_bps,
        total_bps=total_bps,
        implementation_shortfall_bps=is_bps,
        pair=fill.symbol,
        side=fill.side,
        fill_id=fill.fill_id,
        ts=fill.timestamp,
    )


def publish_tca_metrics(components: TCAComponents) -> None:
    """Emit the per-component + total IS Prometheus histograms."""
    try:
        tca_component_bps.labels(pair=components.pair, component="queue").observe(
            components.queue_bps,
        )
        tca_component_bps.labels(pair=components.pair, component="impact").observe(
            components.impact_bps,
        )
        tca_component_bps.labels(pair=components.pair, component="broker").observe(
            components.broker_bps,
        )
        tca_implementation_shortfall_bps.labels(
            pair=components.pair,
            side=components.side,
        ).observe(components.implementation_shortfall_bps)
    except Exception:
        logger.exception("TCA metric emit failed")


# =============================================================================
# Aggregation
# =============================================================================


def summarize(records: list[TCAComponents]) -> TCASummary:
    """Aggregate a list of TCA components into a TCASummary."""
    n = len(records)
    if n == 0:
        return TCASummary(
            n_fills=0,
            avg_queue_bps=0.0,
            avg_impact_bps=0.0,
            avg_broker_bps=0.0,
            avg_total_bps=0.0,
            avg_implementation_shortfall_bps=0.0,
            p50_total_bps=0.0,
            p95_total_bps=0.0,
        )

    queues = [r.queue_bps for r in records]
    impacts = [r.impact_bps for r in records]
    brokers = [r.broker_bps for r in records]
    totals = sorted([r.total_bps for r in records])
    iss = [r.implementation_shortfall_bps for r in records]

    p50_idx = n // 2
    # Inclusive p95 — index of the 95th percentile sample (rounded down).
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))

    return TCASummary(
        n_fills=n,
        avg_queue_bps=sum(queues) / n,
        avg_impact_bps=sum(impacts) / n,
        avg_broker_bps=sum(brokers) / n,
        avg_total_bps=sum(totals) / n,
        avg_implementation_shortfall_bps=sum(iss) / n,
        p50_total_bps=totals[p50_idx],
        p95_total_bps=totals[p95_idx],
    )


__all__ = [
    "FillContext",
    "TCAComponents",
    "TCASummary",
    "compute_tca",
    "publish_tca_metrics",
    "summarize",
]
