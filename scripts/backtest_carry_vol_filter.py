#!/usr/bin/env python3
"""Carry + vol-filter backtest on real G10 yield + FX data (CL-lcxd).

The live strategy in src/strategies/carry_vol_filter.py is portfolio-level
(monthly rebalance, multi-pair basket) and uses the async generate_intents
protocol — doesn't fit WalkForwardRunner's single-pair (fit/generate_signals)
shape. So this script implements the same strategy logic as a vectorized
research backtest:

    1. Monthly rebalance: rank G10 long-term yields, long top_k / short
       bottom_k currencies vs USD.
    2. Apply VIX-z-score exposure scaling: high vol → smaller exposure.
    3. Equal-weight inside each basket. Hold to next rebalance.
    4. Daily P&L = FX move + carry differential − transaction costs at
       rebalance.

Reuses PerformanceAnalytics + stationary_bootstrap_sharpe_ci from the
existing backtest framework. Same metric definitions as CL-1sa for
direct comparability.

Usage:
    .venv/bin/python scripts/backtest_carry_vol_filter.py \\
        [--start 2016-06-01] [--end 2026-04-26] \\
        [--top-k 3] [--bottom-k 3] \\
        [--out reports/backtest_carry_vol_filter.json] [--bootstrap-n 5000]

Default start is 2016-06-01 (need ~120 trading days of VIX history before
the first rebalance for the vol-z window to be populated).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

# Make src.* imports work when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.bootstrap import stationary_bootstrap_sharpe_ci
from src.backtest.cost_model import CostModel
from src.runtime.run_engine import _build_db_engine

logger = logging.getLogger(__name__)


# Currency → FRED long-term-yield series id. We use 10Y proxies (FRED has
# 10Y series for each); the live strategy ranks on these the same way.
RATE_SERIES: dict[str, str] = {
    "USD": "DGS10",
    "EUR": "IRLTLT01DEM156N",  # German 10Y as EUR proxy
    "JPY": "IRLTLT01JPM156N",
    "GBP": "IRLTLT01GBM156N",
    "CAD": "IRLTLT01CAM156N",
    "AUD": "IRLTLT01AUM156N",
    "CHF": "IRLTLT01CHM156N",
    "NZD": "IRLTLT01NZM156N",
}

# Currency → (yfinance pair symbol, sign such that
#             "FX-pair return × sign" = "long-the-foreign-currency-vs-USD return").
# For pairs quoted FOREIGNUSD (EURUSD, GBPUSD, AUDUSD, NZDUSD): long foreign
# = +1 of the pair (pair goes up when foreign strengthens).
# For pairs quoted USDFOREIGN (USDJPY, USDCAD, USDCHF): long foreign = -1
# of the pair (pair goes up when USD strengthens, which is bad for foreign).
PAIR_DIRECTION: dict[str, tuple[str, int]] = {
    "EUR": ("EURUSD", +1),
    "GBP": ("GBPUSD", +1),
    "AUD": ("AUDUSD", +1),
    "NZD": ("NZDUSD", +1),
    "JPY": ("USDJPY", -1),
    "CAD": ("USDCAD", -1),
    "CHF": ("USDCHF", -1),
}


def _load_data(
    start: datetime, end: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load (rates, fx_returns_to_long_foreign, vix) — all daily, aligned.

    Returns:
        rates: DataFrame indexed by date, columns are currency codes,
               values are long-term yield in % (forward-filled monthly→daily).
        fx_returns: DataFrame indexed by date, columns are currency codes
               (excluding USD), values are daily "long-foreign-vs-USD" returns.
        vix: Series of daily VIX close.
    """
    engine = _build_db_engine()

    # --- Rates (FRED, monthly, ffill to daily) -------------------------
    rate_frames: list[pd.DataFrame] = []
    for ccy, series_id in RATE_SERIES.items():
        # DISTINCT ON because FREDIngester may have duplicates (CL-mht0).
        df = pd.read_sql(
            text(
                "SELECT DISTINCT ON (observation_date) "
                "  observation_date AS ts, value "
                "FROM macro_data "
                "WHERE series_id = :sid "
                "  AND observation_date BETWEEN :s AND :e "
                "ORDER BY observation_date, release_date DESC"
            ),
            engine, params={"sid": series_id, "s": start.date(), "e": end.date()},
        )
        if df.empty:
            logger.warning("No rate data for %s (%s)", ccy, series_id)
            continue
        df = df.set_index("ts")[["value"]].rename(columns={"value": ccy})
        df.index = pd.to_datetime(df.index)
        rate_frames.append(df)

    rates = pd.concat(rate_frames, axis=1).sort_index()
    # Resample to daily and forward-fill the monthly → daily.
    daily_idx = pd.date_range(rates.index.min(), rates.index.max(), freq="B")
    rates = rates.reindex(daily_idx, method="ffill").astype(float)

    # --- FX prices (yfinance, daily) -----------------------------------
    pairs = sorted({p for p, _ in PAIR_DIRECTION.values()})
    fx_df = pd.read_sql(
        text(
            "SELECT ts, symbol, close FROM prices "
            "WHERE symbol = ANY(:pairs) "
            "  AND ts BETWEEN :s AND :e "
            "ORDER BY ts"
        ),
        engine, params={"pairs": pairs, "s": start, "e": end},
    )
    fx_df["ts"] = pd.to_datetime(fx_df["ts"], utc=True).dt.tz_localize(None)
    fx_pivot = fx_df.pivot_table(
        index="ts", columns="symbol", values="close", aggfunc="last",
    ).astype(float)

    # Build a "long-foreign-vs-USD spot" series per currency by applying
    # the sign convention. Daily return is then a clean pct_change.
    fx_returns: dict[str, pd.Series] = {}
    for ccy, (pair, sign) in PAIR_DIRECTION.items():
        if pair not in fx_pivot.columns:
            logger.warning("No FX data for %s (%s)", ccy, pair)
            continue
        pct = fx_pivot[pair].pct_change()
        fx_returns[ccy] = (pct * sign).rename(ccy)
    fx_ret_df = pd.DataFrame(fx_returns).sort_index()

    # --- VIX -----------------------------------------------------------
    vix_df = pd.read_sql(
        text(
            "SELECT ts, close FROM prices WHERE symbol = 'VIX' "
            "AND ts BETWEEN :s AND :e ORDER BY ts"
        ),
        engine, params={"s": start, "e": end},
    )
    vix_df["ts"] = pd.to_datetime(vix_df["ts"], utc=True).dt.tz_localize(None)
    vix = vix_df.set_index("ts")["close"].astype(float).rename("VIX")

    return rates, fx_ret_df, vix


def _vol_z_exposure(vix: pd.Series, lookback: int = 120) -> pd.Series:
    """Daily exposure multiplier from rolling VIX z-score.

    Step function from the live strategy's default vol_z_thresholds:
        |z| < 1   → 1.00
        |z| < 2   → 0.75
        |z| < 3   → 0.50
        |z| ≥ 3   → 0.00
    """
    rolling_mean = vix.rolling(lookback).mean()
    rolling_std = vix.rolling(lookback).std()
    z = (vix - rolling_mean) / rolling_std
    abs_z = z.abs()

    exposure = pd.Series(1.0, index=vix.index)
    exposure[abs_z >= 1.0] = 0.75
    exposure[abs_z >= 2.0] = 0.50
    exposure[abs_z >= 3.0] = 0.00
    return exposure


def _rebalance_dates(
    index: pd.DatetimeIndex, rebalance_day: int = 1,
) -> list[pd.Timestamp]:
    """First business day on or after rebalance_day of each month."""
    out: list[pd.Timestamp] = []
    by_month = pd.Series(index, index=index).groupby(
        [index.year, index.month],
    )
    for _, group in by_month:
        # Pick the first day in this month that's >= rebalance_day.
        candidates = group[group.dt.day >= rebalance_day]
        if not candidates.empty:
            out.append(candidates.iloc[0])
    return out


def _build_weights(
    rates: pd.DataFrame, top_k: int, bottom_k: int,
    rebalance_dates: list[pd.Timestamp], tradable_ccys: list[str],
) -> pd.DataFrame:
    """Per-rebalance-date weights — equal-weight long top-k, equal-weight short bottom-k.

    Returns a DataFrame indexed by rebalance date, columns = tradable
    currencies, values = position weight (positive=long, negative=short),
    summing to 0 (zero-USD-exposure carry basket).
    """
    rows: list[dict[str, float]] = []
    idx: list[pd.Timestamp] = []
    for d in rebalance_dates:
        # Use the rates as of d (the rolling join means d picks up the
        # most recent monthly value via ffill).
        if d not in rates.index:
            continue
        snap = rates.loc[d, tradable_ccys].dropna()
        if len(snap) < (top_k + bottom_k):
            # Not enough data this month — skip rebalance, hold previous.
            continue
        sorted_ccys = snap.sort_values(ascending=False)
        long = list(sorted_ccys.head(top_k).index)
        short = list(sorted_ccys.tail(bottom_k).index)
        weights = {c: 0.0 for c in tradable_ccys}
        for c in long:
            weights[c] = +1.0 / top_k
        for c in short:
            weights[c] = -1.0 / bottom_k
        rows.append(weights)
        idx.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _compute_pnl(
    rates: pd.DataFrame, fx_returns: pd.DataFrame, vix: pd.Series,
    weights: pd.DataFrame, cost_per_turn: float,
) -> pd.Series:
    """Daily portfolio returns including carry, vol-scaling, and rebalance costs."""
    # Forward-fill weights from each rebalance date to the next.
    daily_weights = weights.reindex(fx_returns.index, method="ffill").fillna(0)
    exposure = _vol_z_exposure(vix).reindex(fx_returns.index, method="ffill").fillna(0)

    # FX P&L: sum(weight × fx_return) per day.
    fx_pnl = (daily_weights * fx_returns).sum(axis=1)

    # Carry P&L: holders of currency c earn rate_c × dt; USD-financed.
    # Daily carry per ccy = (rate_c - rate_USD) / 252 (annual yield → daily).
    rate_diff = rates.subtract(rates["USD"], axis=0) / 100.0  # rates were in %
    daily_carry_per_ccy = rate_diff / 252.0
    daily_carry = (daily_weights * daily_carry_per_ccy).sum(axis=1)

    # Apply vol-z exposure scaling.
    gross_pnl = (fx_pnl + daily_carry) * exposure

    # Transaction costs on rebalance days = sum of |Δweight| × cost_per_turn.
    weight_change = daily_weights.diff().abs().sum(axis=1)
    costs = weight_change * cost_per_turn

    return (gross_pnl - costs).fillna(0)


def _per_year_metrics(returns: pd.Series) -> list[dict[str, Any]]:
    """Yearly Sharpe + return + max DD for stability."""
    out: list[dict[str, Any]] = []
    for year, group in returns.groupby(returns.index.year):
        if len(group) < 20:
            continue
        sharpe = (group.mean() / group.std()) * np.sqrt(252) if group.std() > 0 else 0.0
        equity = (1 + group).cumprod()
        max_dd = float(((equity - equity.cummax()) / equity.cummax()).min())
        out.append({
            "year": int(year), "n_days": int(len(group)),
            "sharpe": float(sharpe),
            "total_return": float((1 + group).prod() - 1),
            "max_drawdown": max_dd,
        })
    return out


def _flag(value: float, lo: float, hi: float) -> str:
    return "✓" if lo <= value <= hi else "✗"


def _print_summary(report: dict[str, Any]) -> None:
    m = report["metrics"]
    ci = report["sharpe_ci_95"]
    cmp_ = report["architecture_comparison"]
    print()
    print("=" * 70)
    print(f"Carry+VolFilter — {report['data_window']['start']} → {report['data_window']['end']}")
    print(f"top_k={report['config']['top_k']} bottom_k={report['config']['bottom_k']}, "
          f"rebalances={report['n_rebalances']}, n_days={report['data_window']['n_days']}")
    print("=" * 70)
    print(f"  Sharpe:        {m['sharpe']:>7.3f}    95% CI: [{ci['low']:.3f}, {ci['high']:.3f}]   {cmp_['sharpe_in_expected_range']} (expect {cmp_['expected_sharpe_range']})")
    print(f"  Sortino:       {m['sortino']:>7.3f}")
    print(f"  Max drawdown:  {m['max_drawdown']:>7.1%}                                {cmp_['max_drawdown_in_expected_range']} (expect {cmp_['expected_max_drawdown_range']})")
    print(f"  Calmar:        {m['calmar']:>7.3f}")
    print(f"  Hit rate:      {m['hit_rate']:>7.1%}")
    print(f"  Profit factor: {m['profit_factor']:>7.3f}")
    print(f"  CAGR:          {m['cagr']:>7.1%}")
    print(f"  Vol (ann):     {m['volatility']:>7.1%}")
    print(f"  Total return:  {m['total_return']:>7.1%}")
    print(f"  Skew/Kurt:     {m['skewness']:.2f} / {m['kurtosis']:.2f}")
    print()
    print("  Per-year stability:")
    for y in report["per_year"]:
        print(f"    {y['year']}  sharpe={y['sharpe']:>6.2f}  return={y['total_return']:>+6.1%}  "
              f"maxDD={y['max_drawdown']:>+6.1%}  n={y['n_days']}")
    print("=" * 70)


# Architecture-doc benchmark; informational only — same shape as CL-1sa report.
_EXPECT_SHARPE_LO, _EXPECT_SHARPE_HI = 0.5, 1.0
_EXPECT_DD_LO, _EXPECT_DD_HI = -0.18, -0.10


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Carry+vol-filter walk-forward backtest on real data.",
    )
    parser.add_argument("--start", type=str, default="2016-06-01")
    parser.add_argument("--end", type=str, default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bottom-k", type=int, default=3)
    parser.add_argument(
        "--out", type=Path,
        default=Path("reports/backtest_carry_vol_filter.json"),
    )
    parser.add_argument("--bootstrap-n", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end <= start:
        logger.error("end must be after start")
        return 2

    logger.info("Loading data %s → %s", start.date(), end.date())
    rates, fx_returns, vix = _load_data(start, end)
    logger.info(
        "rates shape=%s ccys=%s; fx_returns shape=%s ccys=%s; vix len=%d",
        rates.shape, list(rates.columns), fx_returns.shape, list(fx_returns.columns),
        len(vix),
    )

    # Tradable currencies = those with both a rate AND an FX series. USD
    # is the financing currency (no FX exposure of its own).
    tradable = [c for c in rates.columns if c != "USD" and c in fx_returns.columns]
    if len(tradable) < args.top_k + args.bottom_k:
        logger.error(
            "Only %d tradable currencies but top_k+bottom_k=%d. Need to seed more.",
            len(tradable), args.top_k + args.bottom_k,
        )
        return 1
    logger.info("Tradable currencies: %s", tradable)

    rebalance_dates = _rebalance_dates(fx_returns.index, rebalance_day=1)
    logger.info("n rebalance dates: %d", len(rebalance_dates))

    weights = _build_weights(rates, args.top_k, args.bottom_k, rebalance_dates, tradable)
    logger.info("Built %d rebalance weight rows", len(weights))

    # Use EURUSD-equivalent cost as a representative; carry trades all
    # touch USD-pairs, so the average is reasonable for this analysis.
    cost_model = CostModel()
    cost_per_turn = cost_model.cost_per_turn

    daily_returns = _compute_pnl(rates, fx_returns, vix, weights, cost_per_turn)
    daily_returns = daily_returns.loc[start:end].dropna()

    if daily_returns.empty:
        logger.error("No daily returns computed")
        return 1

    metrics = PerformanceAnalytics.metrics(daily_returns)
    logger.info("Bootstrapping Sharpe 95%% CI (n=%d)", args.bootstrap_n)
    ci_low, ci_high = stationary_bootstrap_sharpe_ci(
        daily_returns, block_mean_len=20, n_bootstrap=args.bootstrap_n,
        confidence=0.95,
    )

    report = {
        "ran_at": datetime.now(UTC).isoformat(),
        "data_window": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "n_days": int(len(daily_returns)),
        },
        "config": {
            "top_k": args.top_k,
            "bottom_k": args.bottom_k,
            "vol_lookback_days": 120,
            "tradable_ccys": tradable,
            "rebalance_day": 1,
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "sharpe_ci_95": {"low": float(ci_low), "high": float(ci_high)},
        "n_rebalances": int(len(weights)),
        "per_year": _per_year_metrics(daily_returns),
        "architecture_comparison": {
            "sharpe_in_expected_range": _flag(metrics.get("sharpe", 0.0), _EXPECT_SHARPE_LO, _EXPECT_SHARPE_HI),
            "max_drawdown_in_expected_range": _flag(metrics.get("max_drawdown", 0.0), _EXPECT_DD_LO, _EXPECT_DD_HI),
            "expected_sharpe_range": [_EXPECT_SHARPE_LO, _EXPECT_SHARPE_HI],
            "expected_max_drawdown_range": [_EXPECT_DD_LO, _EXPECT_DD_HI],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report written to %s", args.out)

    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
