"""Per-instrument tradability metadata for backtests (CL-nt0c).

Walk-forward (CL-gkk) prevents look-ahead within an instrument's
history. This module prevents the orthogonal failure: backtesting on
periods when the pair was illiquid, halted, or simply didn't exist on
the broker. Without this filter, results assume tradability that
never existed.

Two facts per instrument matter:
  - first_tradable_date: earliest date the broker quoted the pair.
  - liquidity_floor:     daily volume below which we consider the
                         pair illiquid (no entries; existing positions
                         can still exit).

The defaults below are conservative — operator should override per
their broker's actual instrument-availability history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InstrumentMetadata:
    """Tradability profile for one instrument."""

    symbol: str
    # Earliest date OANDA (or whoever is the broker-of-record) quoted
    # this pair. Backtests on dates before this are filtered out.
    first_tradable_date: date
    # Median daily volume floor in units. Below this, we don't enter.
    # 0 = no floor. Set per pair from observed live volume.
    liquidity_floor_units: float = 0.0
    # Optional last-tradable-date for pairs that have been delisted.
    # None = still available. CL-nt0c test asserts: a pair removed
    # mid-period yields no trades after its last_tradable_date.
    last_tradable_date: date | None = None


# Default registry — best-effort historical OANDA instrument
# availability. Pre-2010 dates are illustrative; verify before using
# for high-stakes backtests. Contributing to the registry > deferring
# to None, because None falls back to "unrestricted" and silently
# permits the bug we're trying to prevent.
DEFAULT_REGISTRY: dict[str, InstrumentMetadata] = {
    "EURUSD": InstrumentMetadata("EURUSD", date(2002, 1, 1), 1_000_000),
    "GBPUSD": InstrumentMetadata("GBPUSD", date(2002, 1, 1), 500_000),
    "USDJPY": InstrumentMetadata("USDJPY", date(2002, 1, 1), 1_000_000),
    "USDCHF": InstrumentMetadata("USDCHF", date(2002, 1, 1), 500_000),
    "USDCAD": InstrumentMetadata("USDCAD", date(2002, 1, 1), 500_000),
    "AUDUSD": InstrumentMetadata("AUDUSD", date(2002, 1, 1), 500_000),
    "NZDUSD": InstrumentMetadata("NZDUSD", date(2002, 1, 1), 250_000),
    # NOK/SEK quoted vs USD on OANDA from 2003 onwards.
    "USDNOK": InstrumentMetadata("USDNOK", date(2003, 1, 1), 100_000),
    "USDSEK": InstrumentMetadata("USDSEK", date(2003, 1, 1), 100_000),
    # Synthetic / FRED-only series (used as features, never traded).
    # last_tradable_date in the past so any backtest treating them as
    # tradable instruments fails the filter.
    "US_2Y": InstrumentMetadata(
        "US_2Y", date(1990, 1, 1), 0.0, last_tradable_date=date(1990, 1, 1),
    ),
    "US_10Y": InstrumentMetadata(
        "US_10Y", date(1990, 1, 1), 0.0, last_tradable_date=date(1990, 1, 1),
    ),
    "DE_10Y": InstrumentMetadata(
        "DE_10Y", date(1990, 1, 1), 0.0, last_tradable_date=date(1990, 1, 1),
    ),
}


@dataclass
class TradabilityFilter:
    """Wraps a metadata registry and applies the filter to backtest data.

    Used by WalkForwardRunner to mask out non-tradable bars before
    signal generation. Strategies receive only valid history; signals
    on filtered dates are silently dropped.
    """

    registry: dict[str, InstrumentMetadata] = field(
        default_factory=lambda: dict(DEFAULT_REGISTRY),
    )

    def is_tradable(
        self, symbol: str, on_date: date,
        observed_volume: float | None = None,
    ) -> bool:
        meta = self.registry.get(symbol)
        if meta is None:
            # Unknown instrument: default-deny in strict mode, but most
            # call sites can't know the registry exhaustively. Log a
            # WARN once and permit — operator will see and add it.
            logger.warning(
                "TradabilityFilter: %s not in registry — assuming tradable. "
                "Add to DEFAULT_REGISTRY to silence.", symbol,
            )
            return True
        if on_date < meta.first_tradable_date:
            return False
        if meta.last_tradable_date and on_date > meta.last_tradable_date:
            return False
        return not (
            observed_volume is not None
            and meta.liquidity_floor_units > 0
            and observed_volume < meta.liquidity_floor_units
        )

    def filter_dataframe(self, df: Any, symbol: str) -> Any:
        """Return df with rows outside the tradable window dropped."""
        meta = self.registry.get(symbol)
        if meta is None:
            return df
        # `df` is a pandas DataFrame indexed by Timestamp.
        import pandas as pd
        idx = pd.to_datetime(df.index)
        first = pd.Timestamp(meta.first_tradable_date)
        mask = idx >= first
        if meta.last_tradable_date:
            last = pd.Timestamp(meta.last_tradable_date)
            mask = mask & (idx <= last)
        return df.loc[mask]


def from_yaml(path: str) -> dict[str, InstrumentMetadata]:
    """Load registry from a YAML file. Schema:

        instruments:
          EURUSD:
            first_tradable_date: 2002-01-01
            liquidity_floor_units: 1000000
            last_tradable_date: null   # optional
    """
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    out: dict[str, InstrumentMetadata] = {}
    for sym, cfg in (doc.get("instruments") or {}).items():
        first = cfg.get("first_tradable_date")
        if isinstance(first, str):
            first = datetime.fromisoformat(first).date()
        last = cfg.get("last_tradable_date")
        if isinstance(last, str):
            last = datetime.fromisoformat(last).date()
        out[sym] = InstrumentMetadata(
            symbol=sym,
            first_tradable_date=first,
            liquidity_floor_units=float(cfg.get("liquidity_floor_units", 0)),
            last_tradable_date=last,
        )
    return out
