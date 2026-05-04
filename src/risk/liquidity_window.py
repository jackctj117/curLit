"""Liquidity-window sizing rules (CL-4wi5).

FX liquidity is non-uniform across the 168-hour week. This module
holds an hour-of-week × pair heatmap of expected spread (in bps) and
exposes a single function `size_multiplier_for_window` that the risk
sizer calls before each new entry.

The heatmap is derived from observed live + reference spreads. A
monthly cron job (scripts/refresh_liquidity_profile.py — TODO future
session) recomputes from realised spreads stored in trade_journal.

Sizing rules:
  - spread > liquidity_threshold × median  ⇒  multiplier 0.0 (block)
  - spread > 1.5 × median                  ⇒  multiplier 0.5
  - spread within 1.5 × median             ⇒  multiplier 1.0

Existing positions are NOT affected — exits run at any spread to avoid
the "stuck in a position because spread blew out" failure mode. Only
entries are gated.

Liquidity calendar exceptions (year-end, US holidays) compound on top
of the hour-of-week multiplier. The HolidayCalendar (existing) handles
those separately; this module is the intra-week dimension only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# 2.0 × median = "dead window" entry block. Above this multiple, expected
# round-trip cost dominates strategy edge for typical FX strategies
# (Sharpe < 1 net of costs at 2× spread).
_DEFAULT_BLOCK_THRESHOLD: float = 2.0
# 1.5 × median = "thin window" — half-size entries.
_DEFAULT_THIN_THRESHOLD: float = 1.5


@dataclass
class LiquidityProfile:
    """Hour-of-week × pair median spread (bps).

    Index keys are tuples ``(symbol, dow, hour_utc)`` where dow is
    0=Mon … 6=Sun and hour is 0-23 in UTC. Profile coverage is sparse
    by design — we only learn windows we've actually traded. Missing
    keys default to the global median (cheapest interpretation; if
    operator cares, add data).
    """

    median_spread_bps: dict[tuple[str, int, int], float] = field(
        default_factory=dict,
    )
    # Fallback global median per-pair when (dow, hour) bucket is empty.
    pair_median_bps: dict[str, float] = field(default_factory=dict)
    # Block / thin thresholds — overridable per environment.
    block_threshold: float = _DEFAULT_BLOCK_THRESHOLD
    thin_threshold: float = _DEFAULT_THIN_THRESHOLD

    def median_for(self, symbol: str, ts: datetime) -> float:
        bucket = self.median_spread_bps.get(
            (symbol, ts.weekday(), ts.hour),
        )
        if bucket is not None:
            return bucket
        return self.pair_median_bps.get(symbol, 1.0)

    def size_multiplier(
        self, symbol: str, ts: datetime, observed_spread_bps: float,
    ) -> float:
        """Map observed spread → multiplier in [0.0, 1.0].

        The threshold ratio is observed_spread / window_median, NOT
        observed_spread / pair_median. Compares within-window:
        a 5 bps spread at 22:00 UTC for EUR/USD might be totally
        normal even though it's 2× the daily median.
        """
        baseline = self.median_for(symbol, ts)
        if baseline <= 0:
            return 1.0
        ratio = observed_spread_bps / baseline
        if ratio >= self.block_threshold:
            return 0.0
        if ratio >= self.thin_threshold:
            return 0.5
        return 1.0


def size_multiplier_for_window(
    profile: LiquidityProfile,
    symbol: str,
    ts: datetime,
    observed_spread_bps: float,
) -> float:
    """Standalone helper for callers that don't want to hold a profile
    object directly. Risk sizer wires this in front of every new
    entry via PositionSizer.adjust_for_liquidity()."""
    return profile.size_multiplier(symbol, ts, observed_spread_bps)


# --------------------------------------------------------------------- #
# Profile construction from observed spreads
# --------------------------------------------------------------------- #


def build_profile_from_spreads(
    spreads: list[tuple[datetime, str, float]],
    block_threshold: float = _DEFAULT_BLOCK_THRESHOLD,
    thin_threshold: float = _DEFAULT_THIN_THRESHOLD,
) -> LiquidityProfile:
    """Group ``(ts, symbol, spread_bps)`` triples into a profile.

    Each (symbol, dow, hour) bucket gets the median of its samples.
    Buckets with fewer than 5 samples fall back to the pair-level
    median (signal too noisy below n=5).

    This is the construction path for the monthly refresh job. Pairs
    well with the trade_journal (CL-6mby) — pull last-30-days of
    INTENT_SUBMITTED events, join against PRICE quotes, derive spreads.
    """
    import statistics as _stats
    from collections import defaultdict

    bucketed: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    pair_all: dict[str, list[float]] = defaultdict(list)

    for ts, symbol, spread in spreads:
        if spread <= 0:
            continue
        bucketed[(symbol, ts.weekday(), ts.hour)].append(spread)
        pair_all[symbol].append(spread)

    median_spread_bps: dict[tuple[str, int, int], float] = {}
    for key, samples in bucketed.items():
        if len(samples) >= 5:
            median_spread_bps[key] = float(_stats.median(samples))

    pair_median_bps: dict[str, float] = {
        symbol: float(_stats.median(samples))
        for symbol, samples in pair_all.items()
        if samples
    }

    return LiquidityProfile(
        median_spread_bps=median_spread_bps,
        pair_median_bps=pair_median_bps,
        block_threshold=block_threshold,
        thin_threshold=thin_threshold,
    )
