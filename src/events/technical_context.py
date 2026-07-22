"""Computed technical context for candidate tickers (CL-3xoj).

The missing timing pillar — done the HONEST way. Our LLMs never see a chart,
so feeding them pattern vocabulary ("bull flag", "neckline") invites
hallucinated pattern claims on names they can't see. Instead this module
COMPUTES real price-structure facts from actual daily bars and hands the model
(and the executor) grounded numbers:

  * trend state (close vs 20d vs 50d SMA),
  * distance from the 20d high/low,
  * nearest swing support/resistance in DOLLARS,
  * breakout state (at_highs / at_lows / range),
  * recent volume vs its 20d average,

plus :func:`alignment_score` — does the structure AGREE with an idea's
direction? A bullish idea in an uptrend near highs scores positive; a bullish
idea into a falling knife scores negative. Consumers:

  * :func:`format_context_block` → injected into the niche agent's grounding
    (ResearchTools) so entry triggers/invalidations reference real levels;
  * the Alpaca options executor skips ideas whose alignment is strongly
    AGAINST the thesis (its own env-tunable threshold).

Everything is fail-soft and the history fetch is injectable (no live yfinance
in unit tests). No pattern taxonomy on purpose — detection of H&S-style shapes
is subjective even for humans; trend + levels + breakout state capture most of
the timing value with none of the hallucination surface.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Injectable history fetch: ticker → pandas DataFrame with Close/High/Low/
#: Volume columns (yfinance daily shape), or None.
HistoryFn = Callable[[str], Any]

_TREND_UP, _TREND_DOWN, _TREND_SIDEWAYS = "uptrend", "downtrend", "sideways"
_AT_HIGHS, _AT_LOWS, _RANGE = "at_highs", "at_lows", "range"


@dataclass(frozen=True)
class TechnicalContext:
    ticker: str
    last_close: float
    sma20: float | None
    sma50: float | None
    trend: str                      # uptrend | downtrend | sideways
    pct_from_20d_high: float        # <= 0 (0 = at the high)
    pct_from_20d_low: float         # >= 0 (0 = at the low)
    support: float | None           # 20d swing low (excl. last bar)
    resistance: float | None        # 20d swing high (excl. last bar)
    breakout_state: str             # at_highs | at_lows | range
    volume_ratio: float | None      # mean(5d vol) / mean(20d vol)


def yfinance_history(ticker: str) -> Any:
    """Default fetch: ~3 months of daily bars. Fail-soft → None."""
    try:
        import yfinance as yf  # noqa: PLC0415 — deferred
        df = yf.Ticker(ticker).history(period="3mo", interval="1d")
        return df if df is not None and not df.empty else None
    except Exception:
        logger.debug("technical context: history unavailable for %s", ticker,
                     exc_info=True)
        return None


def compute_context(ticker: str, df: Any) -> TechnicalContext | None:
    """Daily-bars frame → :class:`TechnicalContext`. None when the frame is
    unusable (< 21 bars, missing columns). Never raises."""
    try:
        closes = df["Close"].dropna()
        if len(closes) < 21:
            return None
        last = float(closes.iloc[-1])
        sma20 = float(closes.tail(20).mean())
        sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else None

        if sma50 is not None and last > sma20 > sma50:
            trend = _TREND_UP
        elif sma50 is not None and last < sma20 < sma50:
            trend = _TREND_DOWN
        elif sma50 is None:
            trend = _TREND_UP if last > sma20 else _TREND_DOWN
        else:
            trend = _TREND_SIDEWAYS

        # Swing levels from the PRIOR 20 bars (exclude the last bar so
        # "resistance" isn't just today's own high).
        prior = df.iloc[-21:-1]
        highs = prior["High"].dropna() if "High" in df else prior["Close"].dropna()
        lows = prior["Low"].dropna() if "Low" in df else prior["Close"].dropna()
        resistance = float(highs.max()) if len(highs) else None
        support = float(lows.min()) if len(lows) else None

        hi20 = float(closes.tail(20).max())
        lo20 = float(closes.tail(20).min())
        pct_from_high = (last - hi20) / hi20 if hi20 > 0 else 0.0
        pct_from_low = (last - lo20) / lo20 if lo20 > 0 else 0.0

        if pct_from_high >= -0.01:
            breakout = _AT_HIGHS
        elif pct_from_low <= 0.01:
            breakout = _AT_LOWS
        else:
            breakout = _RANGE

        volume_ratio = None
        if "Volume" in df:
            vol = df["Volume"].dropna()
            if len(vol) >= 20:
                base = float(vol.tail(20).mean())
                if base > 0:
                    volume_ratio = float(vol.tail(5).mean()) / base

        return TechnicalContext(
            ticker=ticker, last_close=last, sma20=sma20, sma50=sma50,
            trend=trend, pct_from_20d_high=pct_from_high,
            pct_from_20d_low=pct_from_low, support=support,
            resistance=resistance, breakout_state=breakout,
            volume_ratio=volume_ratio,
        )
    except Exception:
        logger.debug("technical context: compute failed for %s", ticker,
                     exc_info=True)
        return None


def compute_for_ticker(
    ticker: str, history_fn: HistoryFn | None = None,
) -> TechnicalContext | None:
    """Fetch + compute in one step. Fail-soft → None."""
    fetch = history_fn or yfinance_history
    df = fetch(ticker)
    if df is None:
        return None
    return compute_context(ticker, df)


def alignment_score(ctx: TechnicalContext, direction: str) -> float:
    """Does the price structure AGREE with the idea's direction? [-1, 1].

    Trend contributes ±0.6, breakout state ±0.4. A bullish idea in an
    uptrend at the highs → +1.0; a bullish idea in a downtrend at the lows
    → -1.0; sideways/range → 0 contribution. This deliberately does NOT
    penalize "chasing" (extended moves) — the red-team critic owns the
    already-priced-in attack; alignment is purely structure-vs-thesis.
    """
    bullish = str(direction).strip().lower() == "bullish"
    score = 0.0
    if ctx.trend == _TREND_UP:
        score += 0.6 if bullish else -0.6
    elif ctx.trend == _TREND_DOWN:
        score += -0.6 if bullish else 0.6
    if ctx.breakout_state == _AT_HIGHS:
        score += 0.4 if bullish else -0.4
    elif ctx.breakout_state == _AT_LOWS:
        score += -0.4 if bullish else 0.4
    return max(-1.0, min(1.0, score))


def format_context_block(ctx: TechnicalContext) -> str:
    """One compact prompt block of COMPUTED facts (never pattern names)."""
    lines = [
        f"Technicals ({ctx.ticker}, computed from daily bars):",
        f"  last {ctx.last_close:.2f} | 20d SMA {ctx.sma20:.2f}"
        + (f" | 50d SMA {ctx.sma50:.2f}" if ctx.sma50 is not None else "")
        + f" | trend: {ctx.trend}",
        f"  {ctx.pct_from_20d_high * 100:+.1f}% from 20d high, "
        f"{ctx.pct_from_20d_low * 100:+.1f}% from 20d low "
        f"({ctx.breakout_state})",
    ]
    if ctx.support is not None and ctx.resistance is not None:
        lines.append(
            f"  swing support ~{ctx.support:.2f}, resistance "
            f"~{ctx.resistance:.2f} — use THESE for entry/invalidation levels",
        )
    if ctx.volume_ratio is not None:
        lines.append(f"  5d volume {ctx.volume_ratio:.1f}x its 20d average")
    return "\n".join(lines)
