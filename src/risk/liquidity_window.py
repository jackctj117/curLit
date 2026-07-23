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


def _canon(symbol: str) -> str:
    """Matching key: strip separators, upper-case (mirror of
    ``execution.broker.canonical_symbol`` — inlined to keep this module
    dependency-light). Lets a profile built from OANDA-underscore ids
    (``EUR_USD``) answer lookups from compact strategy symbols (``EURUSD``).
    """
    return symbol.replace("_", "").replace("/", "").replace("-", "").upper()


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
        key = _canon(symbol)
        bucket = self.median_spread_bps.get(
            (key, ts.weekday(), ts.hour),
        )
        if bucket is not None:
            return bucket
        return self.pair_median_bps.get(key, 1.0)

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


def spread_bps_from_tick(tick: object) -> float | None:
    """Derive the current bid/ask spread in bps from a live tick dict.

    Returns ``None`` when the tick has no usable two-sided quote (missing
    side, non-numeric, crossed, or non-positive mid). Callers treat a
    ``None`` spread as "cannot measure liquidity" and skip the window gate
    rather than fabricate a spread — the strategy still enters at full size,
    matching the None-profile passthrough posture. Entries are only ever
    *reduced* by a measured wide spread, never opened wider on a guess.
    """
    if not isinstance(tick, dict):
        return None
    bid = tick.get("bid")
    ask = tick.get("ask")
    if bid is None or ask is None:
        return None
    try:
        b = float(bid)
        a = float(ask)
    except (TypeError, ValueError):
        return None
    mid = (a + b) / 2.0
    if mid <= 0 or a < b:
        return None
    return (a - b) / mid * 10_000.0


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
        csym = _canon(symbol)
        bucketed[(csym, ts.weekday(), ts.hour)].append(spread)
        pair_all[csym].append(spread)

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


def merge_profiles(
    old: LiquidityProfile | None, new: LiquidityProfile,
) -> LiquidityProfile:
    """Overlay ``new`` onto ``old`` — fresh buckets win, stale buckets fill
    gaps. ``intraday_quotes`` only retains ~24h, so a single refresh sees at
    most one day's hours; merging successive daily runs accumulates the full
    168-hour week while always preferring the most recent sample for any
    bucket it just re-measured. Thresholds come from ``new``.
    """
    if old is None:
        return new
    median = {**old.median_spread_bps, **new.median_spread_bps}
    pair = {**old.pair_median_bps, **new.pair_median_bps}
    return LiquidityProfile(
        median_spread_bps=median,
        pair_median_bps=pair,
        block_threshold=new.block_threshold,
        thin_threshold=new.thin_threshold,
    )


# --------------------------------------------------------------------- #
# Persistence (monthly refresh writes; engine boot reads)
# --------------------------------------------------------------------- #
#
# The profile's bucket keys are ``(symbol, dow, hour)`` tuples, which JSON
# cannot encode as object keys — so we flatten them to "symbol|dow|hour"
# strings on write and re-split on read. Writes are atomic (tmp + replace)
# per the project's state-file convention.


def _bucket_key_to_str(key: tuple[str, int, int]) -> str:
    symbol, dow, hour = key
    return f"{symbol}|{dow}|{hour}"


def _bucket_key_from_str(s: str) -> tuple[str, int, int]:
    symbol, dow, hour = s.rsplit("|", 2)
    return (symbol, int(dow), int(hour))


def save_profile(profile: LiquidityProfile, path: str) -> None:
    """Atomically persist ``profile`` to ``path`` as JSON."""
    import contextlib
    import json
    import os
    import tempfile

    payload = {
        "median_spread_bps": {
            _bucket_key_to_str(k): v
            for k, v in profile.median_spread_bps.items()
        },
        "pair_median_bps": dict(profile.pair_median_bps),
        "block_threshold": profile.block_threshold,
        "thin_threshold": profile.thin_threshold,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_profile(path: str) -> LiquidityProfile | None:
    """Load a persisted profile, or ``None`` if the file is absent.

    A missing file is the normal cold-start state: no refresh has run yet,
    so the engine wires ``None`` and liquidity gating is inert (every entry
    passes at full size). A *corrupt* file raises — fail loud, per the
    project convention, rather than silently trading with a mangled gate.
    """
    import json
    import os

    if not os.path.exists(path):
        return None
    with open(path) as fh:
        payload = json.load(fh)
    median = {
        _bucket_key_from_str(k): float(v)
        for k, v in payload["median_spread_bps"].items()
    }
    pair_median = {
        str(k): float(v) for k, v in payload["pair_median_bps"].items()
    }
    return LiquidityProfile(
        median_spread_bps=median,
        pair_median_bps=pair_median,
        block_threshold=float(
            payload.get("block_threshold", _DEFAULT_BLOCK_THRESHOLD),
        ),
        thin_threshold=float(
            payload.get("thin_threshold", _DEFAULT_THIN_THRESHOLD),
        ),
    )
