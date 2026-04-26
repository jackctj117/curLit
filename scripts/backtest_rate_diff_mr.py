#!/usr/bin/env python3
"""Rate-diff mean-reversion backtest on real 2015-onward data (CL-1sa).

Pure wiring of existing components — no new logic:
    src/data/{prices, macro_data}            — seeded by CL-4bu
    src/backtest/walkforward.py              — WalkForwardRunner + Config
    src/backtest/cost_model.py               — CostModel.cost_per_turn
    src/backtest/analytics.py                — PerformanceAnalytics.metrics
    src/backtest/bootstrap.py                — stationary_bootstrap_sharpe_ci
    src/strategies/rate_diff_mean_reversion  — RateDiffMRStrategy

Usage:
    .venv/bin/python scripts/backtest_rate_diff_mr.py \\
        [--start 2015-01-01] [--end 2025-12-31] \\
        [--out reports/backtest_rate_diff.json] [--bootstrap-n 5000]

Loads EURUSD + US_10Y from `prices` (yfinance) and DE_10Y from `macro_data`
(FRED IRLTLT01DEM156N — monthly, ffill'd to daily) directly. We don't go
through DataProvider.get_aligned_series because that helper drops symbols
that live in only one table (CL-5rtm — open follow-up bug).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

# Make src.* imports work when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.bootstrap import stationary_bootstrap_sharpe_ci
from src.backtest.cost_model import CostModel
from src.backtest.walkforward import WalkForwardConfig, WalkForwardRunner
from src.runtime.run_engine import _build_db_engine
from src.strategies.rate_diff_mean_reversion import (
    RateDiffMRConfig,
    RateDiffMRStrategy,
)

logger = logging.getLogger(__name__)


# Architecture-doc expectations for the rate-diff strategy. Used purely as
# an informational benchmark in the report — we don't fail the run on a
# miss.
_EXPECT_SHARPE_LO, _EXPECT_SHARPE_HI = 0.5, 1.0
_EXPECT_DD_LO, _EXPECT_DD_HI = -0.18, -0.10  # max DD between -18% and -10%
_EXPECT_HIT_LO, _EXPECT_HIT_HI = 0.55, 0.65


def _load_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Build the input DataFrame: EURUSD + US_10Y + DE_10Y aligned daily.

    EURUSD and US_10Y come from `prices` (daily from yfinance); DE_10Y is
    monthly from FRED IRLTLT01DEM156N — forward-filled to daily so the
    walk-forward sees a continuous spread series.
    """
    engine = _build_db_engine()

    eurusd = pd.read_sql(
        text(
            "SELECT ts, close FROM prices WHERE symbol='EURUSD' "
            "AND ts BETWEEN :s AND :e ORDER BY ts"
        ),
        engine, params={"s": start, "e": end},
    ).set_index("ts")["close"].rename("EURUSD")

    us10 = pd.read_sql(
        text(
            "SELECT ts, close FROM prices WHERE symbol='US_10Y' "
            "AND ts BETWEEN :s AND :e ORDER BY ts"
        ),
        engine, params={"s": start, "e": end},
    ).set_index("ts")["close"].rename("US_10Y")

    # DISTINCT ON because FREDIngester duplicates observations across reruns
    # (CL-mht0 — release_date=utcnow() makes the upsert key non-unique). We
    # take the most recent release_date per observation_date.
    de10 = pd.read_sql(
        text(
            "SELECT DISTINCT ON (observation_date) "
            "  observation_date AS ts, value "
            "FROM macro_data "
            "WHERE series_id='IRLTLT01DEM156N' "
            "  AND observation_date BETWEEN :s AND :e "
            "ORDER BY observation_date, release_date DESC"
        ),
        engine, params={"s": start.date(), "e": end.date()},
    ).set_index("ts")["value"].rename("DE_10Y")
    # Drop tz from the daily index so all three series align cleanly.
    eurusd.index = pd.to_datetime(eurusd.index).tz_localize(None)
    us10.index = pd.to_datetime(us10.index).tz_localize(None)
    de10.index = pd.to_datetime(de10.index)

    df = pd.concat([eurusd, us10, de10], axis=1)
    df["DE_10Y"] = df["DE_10Y"].ffill()
    df = df.dropna()
    df["US10Y_MINUS_DE10Y"] = df["US_10Y"].astype(float) - df["DE_10Y"].astype(float)
    df["close"] = df["EURUSD"].astype(float)
    return df


def _flag(value: float, lo: float, hi: float) -> str:
    """Comparison flag: ✓ if inside [lo, hi], ✗ otherwise."""
    return "✓" if lo <= value <= hi else "✗"


def _build_report(
    metrics: dict[str, float | int],
    ci_low: float,
    ci_high: float,
    fold_metrics_df: pd.DataFrame,
    n_obs: int,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Compose the JSON report."""
    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "data_window": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "n_obs_aligned": n_obs,
        },
        "config": {
            "pair": "EURUSD",
            "spread": "US10Y_MINUS_DE10Y",
            "is_window_days": 756,
            "oos_window_days": 63,
            "step_days": 63,
        },
        "oos_metrics": {k: float(v) for k, v in metrics.items()},
        "sharpe_ci_95": {"low": float(ci_low), "high": float(ci_high)},
        "n_folds": int(len(fold_metrics_df)),
        "folds": fold_metrics_df.to_dict(orient="records"),
        "architecture_comparison": {
            "sharpe_in_expected_range": _flag(
                metrics.get("sharpe", 0.0), _EXPECT_SHARPE_LO, _EXPECT_SHARPE_HI,
            ),
            "max_drawdown_in_expected_range": _flag(
                metrics.get("max_drawdown", 0.0), _EXPECT_DD_LO, _EXPECT_DD_HI,
            ),
            "hit_rate_in_expected_range": _flag(
                metrics.get("hit_rate", 0.0), _EXPECT_HIT_LO, _EXPECT_HIT_HI,
            ),
            "expected_sharpe_range": [_EXPECT_SHARPE_LO, _EXPECT_SHARPE_HI],
            "expected_max_drawdown_range": [_EXPECT_DD_LO, _EXPECT_DD_HI],
            "expected_hit_rate_range": [_EXPECT_HIT_LO, _EXPECT_HIT_HI],
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    """Human-readable console summary of the report."""
    m = report["oos_metrics"]
    ci = report["sharpe_ci_95"]
    cmp_ = report["architecture_comparison"]
    print()
    print("=" * 60)
    print(f"Rate-diff MR walk-forward — {report['data_window']['start']} → {report['data_window']['end']}")
    print(f"n_obs aligned: {report['data_window']['n_obs_aligned']}, n_folds: {report['n_folds']}")
    print("=" * 60)
    print(f"  Sharpe:        {m['sharpe']:>7.3f}    95% CI: [{ci['low']:.3f}, {ci['high']:.3f}]   {cmp_['sharpe_in_expected_range']} (expect {cmp_['expected_sharpe_range']})")
    print(f"  Sortino:       {m['sortino']:>7.3f}")
    print(f"  Max drawdown:  {m['max_drawdown']:>7.1%}                                {cmp_['max_drawdown_in_expected_range']} (expect {cmp_['expected_max_drawdown_range']})")
    print(f"  Calmar:        {m['calmar']:>7.3f}")
    print(f"  Hit rate:      {m['hit_rate']:>7.1%}                                {cmp_['hit_rate_in_expected_range']} (expect {cmp_['expected_hit_rate_range']})")
    print(f"  Profit factor: {m['profit_factor']:>7.3f}")
    print(f"  CAGR:          {m['cagr']:>7.1%}")
    print(f"  Vol (ann):     {m['volatility']:>7.1%}")
    print(f"  Total return:  {m['total_return']:>7.1%}")
    print(f"  Skew/Kurt:     {m['skewness']:.2f} / {m['kurtosis']:.2f}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rate-diff MR walk-forward backtest on real data.",
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("reports/backtest_rate_diff.json"),
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=5000,
        help="Bootstrap iterations for Sharpe CI (default 5000; 10k is more "
             "stable but slower).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end <= start:
        logger.error("end (%s) must be after start (%s)", end, start)
        return 2

    logger.info("Loading data %s → %s", start.date(), end.date())
    data = _load_data(start, end)
    if data.empty:
        logger.error("No data returned — has the seed script run? CL-4bu")
        return 1
    logger.info("Loaded %d aligned daily rows; columns=%s", len(data), list(data.columns))

    cfg = WalkForwardConfig(is_window_days=756, oos_window_days=63, step_days=63, min_history=756)
    runner = WalkForwardRunner(cfg)
    cost_model = CostModel()

    def _strategy_factory() -> RateDiffMRStrategy:
        # Each fold gets a fresh strategy (no state leaks across folds).
        return RateDiffMRStrategy(RateDiffMRConfig())

    logger.info("Running walk-forward (is=%dd, oos=%dd, step=%dd, min_history=%dd)",
                cfg.is_window_days, cfg.oos_window_days, cfg.step_days, cfg.min_history)
    result = runner.run(data, _strategy_factory, cost_model)

    if result.oos_returns.empty:
        logger.error("No OOS returns produced — data window may be too short for is_window_days=%d", cfg.is_window_days)
        return 1

    oos_returns = result.oos_returns
    metrics = PerformanceAnalytics.metrics(oos_returns)

    logger.info("Bootstrapping Sharpe 95%% CI (n=%d)", args.bootstrap_n)
    ci_low, ci_high = stationary_bootstrap_sharpe_ci(
        oos_returns, block_mean_len=20, n_bootstrap=args.bootstrap_n,
        confidence=0.95,
    )

    report = _build_report(
        metrics, ci_low, ci_high, result.fold_metrics,
        len(data), start, end,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report written to %s", args.out)

    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
