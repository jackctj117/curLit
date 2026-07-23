"""Aggressive NO-bias Polymarket strategy (CL-agg-poly-short).

Polymarket counterpart to AggressiveShortFXStrategy. Trades against
operator-curated markets resolving in a short window, with three rules:

  1. **Time gate.** Skip markets resolving more than 14 days out.
     Capital-locked horizon is the dominant cost on long-resolution
     markets; this strategy targets quick decisions.

  2. **NO-bias.** When both YES and NO sides pass the Kelly edge
     filter, prefer NO (treat the YES side as the over-priced one
     absent stronger signal). Config-flagged via the active
     ``RiskProfile.bias.prefer_polymarket_no``.

  3. **Hard exit 24h pre-resolution.** Resolution + UMA freeze risk
     (Plan §4) makes the last 24h before close uniquely dangerous:
     trading can halt before tokens redeem, and disputes lock
     capital for days. Default policy is "be out before that
     window opens." Per-market override available.

Sizing uses ``src.risk.polymarket_sizer.size_polymarket_order`` with
the active risk profile's kelly_fraction, max_polymarket_market_fraction,
and polymarket_min_edge. Aggressive_short cranks all three.

Implementation: this strategy doesn't directly place orders. Like
RateDiffMRStrategy, it generates ``OrderIntent``s consumed by the
PortfolioCoordinator + OMS. For paper backtests, the same
``generate_signals(data)`` shape produces position weights over the
panel's POLY: columns.

Reference: CL-yk4k. Brokers: polymarket-paper (development), polymarket-
amoy (testnet sign + submit), polymarket-mainnet (GATED — CL-poly-3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from src.risk.polymarket_sizer import (
    PolySizeDecision,
    size_polymarket_order,
)
from src.risk.risk_profile import RiskProfile, load_active_profile

logger = logging.getLogger(__name__)


# 14-day default horizon — past this, opportunity cost on locked
# capital dominates Kelly's expected edge. Operator can extend per
# strategy instance.
_DEFAULT_MAX_DAYS_TO_RESOLUTION: int = 14

# 24-hour pre-resolution exit. The UMA optimistic-oracle window
# can freeze trading + redemption for hours-to-days; this margin
# keeps us out of that window.
_DEFAULT_EXIT_BEFORE_RESOLUTION_HOURS: int = 24

# Default model-prob assumption when we have no informed view:
# 0.5 (no edge). Sub-strategies override this with their own
# model_prob source (sentiment-derived, NLP-derived, etc).
_DEFAULT_FALLBACK_MODEL_PROB: Decimal = Decimal("0.5")


@dataclass
class AggressiveShortPolymarketConfig:
    """Per-strategy config for the Polymarket NO-bias variant.

    The POLY symbols to watch + their resolution metadata are passed
    in via ``markets``, mirroring the operator-curated YAML pattern
    used by PolymarketHistoryIngester."""

    markets: list[dict[str, Any]] = field(default_factory=list)
    # Skip markets resolving past this horizon.
    max_days_to_resolution: int = _DEFAULT_MAX_DAYS_TO_RESOLUTION
    # Exit positions this many hours before market close.
    exit_before_resolution_hours: int = _DEFAULT_EXIT_BEFORE_RESOLUTION_HOURS
    # Function returning model probability per market. Required for
    # the strategy to do anything useful. Operator wires their model
    # at construction.
    model_prob_fn: Any | None = None
    id: str = "aggressive_short_polymarket"
    signal_interval_seconds: int = 300

    # Risk profile — defaults to whatever CURLIT_RISK_PROFILE selects.
    risk_profile: RiskProfile | None = field(default=None)


class AggressiveShortPolymarketStrategy:
    """Polymarket strategy that prefers NO positions and exits before
    resolution. Built to run under aggressive_short risk profile."""

    def __init__(
        self,
        config: AggressiveShortPolymarketConfig | None = None,
        data_provider: Any | None = None,
        state_store: Any | None = None,
    ) -> None:
        self.config = config or AggressiveShortPolymarketConfig()
        self.data = data_provider
        self.state = state_store
        self.risk = self.config.risk_profile or load_active_profile()
        # Cache profile reads
        self._kelly_fraction = Decimal(str(self.risk.sizing.kelly_fraction))
        self._max_market_frac = Decimal(
            str(self.risk.sizing.max_polymarket_market_fraction),
        )
        self._min_edge = Decimal(str(self.risk.sizing.polymarket_min_edge))
        self._prefer_no = self.risk.bias.prefer_polymarket_no
        logger.info(
            "AggressiveShortPolymarketStrategy init: profile=%s kf=%s "
            "max_market=%s min_edge=%s prefer_no=%s n_markets=%d",
            self.risk.name,
            self._kelly_fraction,
            self._max_market_frac,
            self._min_edge,
            self._prefer_no,
            len(self.config.markets),
        )

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        return [m["symbol"] for m in self.config.markets if "symbol" in m]

    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds

    # ----------------------------------------------------------------- #
    # Walk-forward harness contract
    # ----------------------------------------------------------------- #

    def fit(self, train_data: pd.DataFrame) -> None:
        """Polymarket strategy has no fit step — pricing is direct and
        model_prob_fn is operator-supplied. Stays for ABC compatibility."""
        del train_data

    def generate_signals(
        self,
        data: pd.DataFrame,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """Per-market position weight at each bar. Output is a DataFrame
        where columns are POLY: symbols and values are position weights
        in {-1, 0, +1} after applying:
          * time-gate filter (skip markets past horizon)
          * 24h pre-resolution exit
          * Kelly sizing decision
          * NO-bias when both sides pass the edge filter

        ``now`` is operator-overridable for backtests; defaults to
        wall-clock UTC. The walk-forward harness should pass the
        per-fold cutoff to keep the time-gate reproducible.
        """
        now = now or datetime.now(UTC)
        idx = data.index

        out = pd.DataFrame(0.0, index=idx, columns=self.symbols, dtype=float)

        for market in self.config.markets:
            sym = market.get("symbol")
            if not sym or sym not in data.columns:
                continue

            resolution_ts = self._parse_ts(market.get("resolution_ts"))
            if resolution_ts is None:
                # Unknown resolution → can't enforce time gate or
                # pre-resolution exit. Skip — operator should curate.
                logger.debug(
                    "skipping %s: no resolution_ts in market metadata",
                    sym,
                )
                continue

            days_to_resolution = (resolution_ts - now).total_seconds() / 86400
            if days_to_resolution < 0:
                # Already resolved — flatten everywhere.
                continue
            if days_to_resolution > self.config.max_days_to_resolution:
                continue

            # Walk bars; for each, decide entry/hold/exit.
            prices = data[sym].astype(float)
            for i, ts in enumerate(idx):
                bar_ts = self._parse_ts(ts) or now
                hours_to_resolution = (resolution_ts - bar_ts).total_seconds() / 3600
                # Exit (or refuse entry) within the pre-resolution window.
                if hours_to_resolution <= self.config.exit_before_resolution_hours:
                    out.iat[i, out.columns.get_loc(sym)] = 0.0
                    continue
                price = prices.iat[i]
                if pd.isna(price) or price <= 0 or price >= 1:
                    continue

                # Get the model's probability for this market at this bar.
                # If no model_prob_fn is wired, the strategy can't take
                # a position — it's all model-derived edge, not chart-
                # based. We return 0 in that case (graceful).
                if self.config.model_prob_fn is None:
                    continue
                try:
                    q = Decimal(
                        str(
                            self.config.model_prob_fn(sym, bar_ts, market),
                        )
                    )
                except Exception:
                    logger.debug(
                        "model_prob_fn failed for %s at %s",
                        sym,
                        bar_ts,
                        exc_info=True,
                    )
                    continue
                p = Decimal(str(price))

                # Probe both sides; pick per NO-bias if both pass.
                yes_dec = size_polymarket_order(
                    bankroll=Decimal("10000"),  # placeholder; sizer
                    # only uses fraction
                    market_price=p,
                    model_prob=q,
                    side="YES",
                    kelly_fraction=self._kelly_fraction,
                    max_market_fraction=self._max_market_frac,
                    min_edge=self._min_edge,
                )
                no_dec = size_polymarket_order(
                    bankroll=Decimal("10000"),
                    market_price=p,
                    model_prob=q,
                    side="NO",
                    kelly_fraction=self._kelly_fraction,
                    max_market_fraction=self._max_market_frac,
                    min_edge=self._min_edge,
                )
                position = self._choose_side(yes_dec, no_dec)
                out.iat[i, out.columns.get_loc(sym)] = position
        return out

    # ----------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------- #

    def _choose_side(
        self,
        yes_dec: PolySizeDecision | None,
        no_dec: PolySizeDecision | None,
    ) -> float:
        """Apply NO-bias when both pass. Returns the position weight
        (sign-of-side × fraction-of-bankroll)."""
        if yes_dec is None and no_dec is None:
            return 0.0
        if yes_dec is None:
            assert no_dec is not None
            return -float(no_dec.fraction_of_bankroll)
        if no_dec is None:
            return float(yes_dec.fraction_of_bankroll)
        # Both pass. NO-bias picks NO; else pick the bigger edge.
        if self._prefer_no:
            return -float(no_dec.fraction_of_bankroll)
        if no_dec.fraction_of_bankroll > yes_dec.fraction_of_bankroll:
            return -float(no_dec.fraction_of_bankroll)
        return float(yes_dec.fraction_of_bankroll)

    @staticmethod
    def _parse_ts(ts: Any) -> datetime | None:
        """Best-effort parse — accepts datetime, pandas Timestamp,
        ISO string, or None."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        if isinstance(ts, str):
            try:
                parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        # pandas Timestamp
        if hasattr(ts, "to_pydatetime"):
            d: datetime = ts.to_pydatetime()
            return d if d.tzinfo else d.replace(tzinfo=UTC)
        return None
