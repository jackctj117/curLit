#!/usr/bin/env python3
"""Optuna walk-forward hyperparameter search for the rate-diff MR strategy (CL-4nnr).

Searches RateDiffMRConfig hyperparameters against the REAL walk-forward
out-of-sample path (same components as scripts/backtest_rate_diff_mr.py):

    src/data/provider.DataProvider.get_aligned_series  — EURUSD + US_10Y + DE_10Y
    src/backtest/walkforward.WalkForwardRunner         — non-overlapping IS/OOS folds
    src/backtest/cost_model.CostModel                  — per-pair cost via get_cost_per_turn
    src/backtest/analytics.PerformanceAnalytics        — OOS metric computation

Objective (documented tradeoffs):
    score = sharpe - 1.5 * abs(max_drawdown)

  * Sharpe alone rewards low-vol configs that barely trade; the drawdown
    term penalizes tail pain the Sharpe hides (a -20% hole costs 0.30 of
    score — roughly the Sharpe gap between a mediocre and a good config).
    1.5 weights a 10% drawdown as 0.15 Sharpe-equivalents, matching the
    architecture doc's tolerance band (max DD -10%..-18% for Sharpe 0.5..1.0).
  * HARD PENALTY: configs producing fewer than MIN_TRADES position-change
    events over the whole OOS window return -10. Without it, the search
    converges on degenerate never-trade parameter corners (entry_z high,
    lookback so short the model never fits) whose flat return series
    scores sharpe=0, dd=0 → 0.0, beating every honest config that dares
    to lose money. -10 is far below any achievable real score, so TPE
    treats the whole degenerate region as uniformly terrible.

Param space is built by INTROSPECTING RateDiffMRConfig via
dataclasses.fields at runtime: base params are always searched; the newer
filter params (momentum_lookback_days, regime_max_vol_z, plus boolean
enable-flags) are searched only when the field actually exists on the
dataclass, so this script works both before and after the filter-param
change lands.

Note on lookback_days: in backtest mode the strategy's fit() uses the
whole train window, so we mirror live behavior (which refits on
tail(lookback_days)) via a thin adapter that trims the train slice.
Values < 100 produce no model (strategy's own minimum-rows guard) and
therefore no trades → they fall into the hard penalty automatically.

Usage:
    .venv/bin/python scripts/hyperopt_rate_model.py \\
        [--trials 60] [--start 2018-01-01] [--end 2026-07-17] \\
        [--pair EURUSD] [--study-db data/optuna/rate_diff_mr.db] [--jobs 1]

Output: configs/best_rate_params.json (best params + metadata) and a
top-10 trials table on stdout. Params are NOT auto-applied to the live
config — the diff the operator would make is printed instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Make src.* imports work when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.cost_model import CostModel
from src.backtest.walkforward import WalkForwardConfig, WalkForwardRunner
from src.strategies.rate_diff_mean_reversion import (
    RateDiffMRConfig,
    RateDiffMRStrategy,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Objective constants
# --------------------------------------------------------------------------- #

# Drawdown weight in the score — see module docstring for the tradeoff.
DD_WEIGHT = 1.5
# Minimum position-change events over the full OOS window. An entry and
# its exit are two events, so 8 events ≈ 4 round trips minimum across the
# multi-year OOS path — enough that the Sharpe isn't one lucky trade.
MIN_TRADES = 8
# Score returned for degenerate no-trade configs. Far below any real
# score (|sharpe| < ~3, |dd| <= 1 → real scores live in roughly [-4.5, 3]).
PENALTY_SCORE = -10.0

# --------------------------------------------------------------------------- #
# Search space — applied only for fields that exist on RateDiffMRConfig
# --------------------------------------------------------------------------- #

# name -> ("int"|"float", low, high)
NUMERIC_SPACE: dict[str, tuple[str, float, float]] = {
    "lookback_days": ("int", 60, 504),
    "entry_z_threshold": ("float", 1.0, 2.8),
    "exit_z_threshold": ("float", 0.1, 0.8),
    "stop_loss_z": ("float", 2.5, 4.5),
    "max_holding_days": ("int", 10, 45),
    # Newer filter params — searched only if the concurrent filter work
    # has added them to the dataclass.
    "momentum_lookback_days": ("int", 3, 10),
    "regime_max_vol_z": ("float", 1.0, 3.0),
}

# Boolean enable-flags for the new filters are discovered by name: any
# bool-typed dataclass field mentioning one of these stems is searched
# as a categorical True/False.
BOOL_FIELD_STEMS = ("filter", "momentum", "regime")


def _field_type_name(f: dataclasses.Field) -> str:
    """Field annotation as a string — works whether the strategy module
    uses postponed annotations (str) or real types."""
    t = f.type
    if isinstance(t, str):
        return t
    return getattr(t, "__name__", str(t))


def build_param_space(config_cls: type) -> dict[str, tuple]:
    """Introspect ``config_cls`` and return the searchable param space.

    Returns a dict of name -> spec, where spec is ("int", lo, hi),
    ("float", lo, hi) or ("bool",). Only fields that exist on the
    dataclass are included, so both pre- and post-filter-param versions
    of RateDiffMRConfig work.
    """
    space: dict[str, tuple] = {}
    cfg_fields = {f.name: f for f in dataclasses.fields(config_cls)}
    for name, spec in NUMERIC_SPACE.items():
        if name in cfg_fields:
            space[name] = spec
    for name, f in cfg_fields.items():
        if name in space:
            continue
        if _field_type_name(f) == "bool" and any(s in name for s in BOOL_FIELD_STEMS):
            space[name] = ("bool",)
    return space


def suggest_params(trial: Any, space: dict[str, tuple]) -> dict[str, Any]:
    """Draw one parameter set from ``space`` using an Optuna trial
    (anything with suggest_int / suggest_float / suggest_categorical)."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec[1]), int(spec[2]))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec[1]), float(spec[2]))
        elif kind == "bool":
            params[name] = trial.suggest_categorical(name, [False, True])
        else:  # pragma: no cover — spec table is module-local
            raise ValueError(f"unknown spec kind {kind!r} for {name}")
    return params


def score_result(metrics: dict[str, float], n_trades: int) -> float:
    """Objective score from OOS metrics — pure function, unit-tested.

    score = sharpe - DD_WEIGHT * |max_drawdown|, with the hard
    PENALTY_SCORE when fewer than MIN_TRADES position-change events
    occurred (degenerate no-trade configs must not win — see module
    docstring).
    """
    if n_trades < MIN_TRADES:
        return PENALTY_SCORE
    sharpe = float(metrics.get("sharpe", 0.0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    return sharpe - DD_WEIGHT * abs(max_dd)


# --------------------------------------------------------------------------- #
# Backtest plumbing
# --------------------------------------------------------------------------- #


class _LookbackTrimStrategy:
    """Adapter mirroring live refit behavior: fit on the tail
    ``lookback_days`` of the train window (live `_refit_if_stale` does
    ``df.tail(lookback_days)``; backtest ``fit()`` uses everything it is
    given). Makes lookback_days a real hyperparameter without touching
    src/."""

    def __init__(self, strategy: RateDiffMRStrategy) -> None:
        self._strategy = strategy

    def fit(self, train_data: pd.DataFrame) -> None:
        lookback = int(self._strategy.config.lookback_days)
        self._strategy.fit(train_data.tail(lookback))

    def generate_signals(self, test_data: pd.DataFrame) -> pd.Series:
        return self._strategy.generate_signals(test_data)


class _PairCost:
    """Minimal cost-model shim exposing the single attribute
    WalkForwardRunner reads (``cost_per_turn``) at the per-pair rate."""

    def __init__(self, cost_model: CostModel, pair: str) -> None:
        self.cost_per_turn = cost_model.get_cost_per_turn(pair)


# CL-x50g entry filters read named columns from the backtest frame. The
# config defaults (CVIX, USD_3M_OIS, EUR_3M_ESTR_OIS) are not in the DB,
# which would leave those filters permanently fail-open — the enable
# flags would be pure noise dimensions for TPE. Real proxies that ARE
# seeded (CL-4bu): VIX for the vol regime, US_2Y - DE_2Y for the
# short-rate carry spread. Overrides are applied only when the fields
# exist on RateDiffMRConfig, so pre-filter checkouts still work.
FILTER_SERIES_OVERRIDES: dict[str, str] = {
    "vol_index_series": "VIX",
    "carry_quote_rate_series": "US_2Y",
    "carry_base_rate_series": "DE_2Y",
}
_EXTRA_SERIES = ["VIX", "US_2Y", "DE_2Y"]


def load_data(pair: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Aligned daily frame: pair + US_10Y + DE_10Y (required), plus the
    filter proxy columns (VIX, US_2Y, DE_2Y) when available. Mirrors
    scripts/backtest_rate_diff_mr.py::_load_data."""
    from src.data.provider import DataProvider
    from src.runtime.run_engine import _build_db_engine

    dp = DataProvider(_build_db_engine())
    df = dp.get_aligned_series(
        [pair, "US_10Y", "DE_10Y", *_EXTRA_SERIES],
        start,
        end,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned from DataProvider for {pair}/US_10Y/DE_10Y "
            f"{start.date()}..{end.date()} — has the seed script run? CL-4bu"
        )
    df = df.copy()
    df["DE_10Y"] = df["DE_10Y"].ffill()
    for col in _EXTRA_SERIES:  # optional filter inputs — fail-open if absent
        if col in df.columns:
            df[col] = df[col].ffill()
    df = df.dropna(subset=[pair, "US_10Y", "DE_10Y"])
    df["US10Y_MINUS_DE10Y"] = df["US_10Y"].astype(float) - df["DE_10Y"].astype(float)
    df["close"] = df[pair].astype(float)
    return df


def make_objective(
    data: pd.DataFrame,
    space: dict[str, tuple],
    pair: str,
    wf_config: WalkForwardConfig,
    cost_model: CostModel,
):
    """Build the Optuna objective closure over pre-loaded data.

    Data is loaded ONCE and shared read-only across trials — each trial
    still constructs its own strategy/runner, so serial (default) and
    n_jobs>1 runs are both safe without a DB session per trial.
    """
    pair_cost = _PairCost(cost_model, pair)
    cfg_field_names = {f.name for f in dataclasses.fields(RateDiffMRConfig)}
    series_overrides = {k: v for k, v in FILTER_SERIES_OVERRIDES.items() if k in cfg_field_names}

    def objective(trial: Any) -> float:
        params = suggest_params(trial, space)
        cfg = RateDiffMRConfig(pair=pair, **series_overrides, **params)

        def factory() -> _LookbackTrimStrategy:
            return _LookbackTrimStrategy(RateDiffMRStrategy(cfg))

        result = WalkForwardRunner(wf_config).run(data, factory, pair_cost)
        if result.oos_returns.empty:
            trial.set_user_attr("n_trades", 0)
            return PENALTY_SCORE
        n_trades = int(result.trades["position_change"].astype(bool).sum())
        metrics = PerformanceAnalytics.metrics(result.oos_returns)
        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("sharpe", float(metrics.get("sharpe", 0.0)))
        trial.set_user_attr("max_drawdown", float(metrics.get("max_drawdown", 0.0)))
        trial.set_user_attr("hit_rate", float(metrics.get("hit_rate", 0.0)))
        trial.set_user_attr("profit_factor", float(metrics.get("profit_factor", 0.0)))
        return score_result(metrics, n_trades)

    return objective


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def config_diff(best_params: dict[str, Any]) -> list[str]:
    """Lines describing what the operator would change vs the current
    RateDiffMRConfig defaults. NOT applied automatically."""
    defaults = RateDiffMRConfig()
    lines: list[str] = []
    for name, value in sorted(best_params.items()):
        current = getattr(defaults, name, "<absent>")
        marker = " (unchanged)" if current == value else ""
        lines.append(f"  {name}: {current} -> {value}{marker}")
    return lines


def print_top_trials(study: Any, n: int = 10) -> None:
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)
    print()
    print(f"Top {min(n, len(done))} trials (of {len(done)} completed)")
    print("-" * 100)
    print(f"{'#':>4} {'score':>8} {'sharpe':>8} {'max_dd':>8} {'trades':>7}  params")
    for t in done[:n]:
        sharpe = t.user_attrs.get("sharpe", float("nan"))
        dd = t.user_attrs.get("max_drawdown", float("nan"))
        ntr = t.user_attrs.get("n_trades", 0)
        pstr = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(t.params.items())
        )
        print(f"{t.number:>4} {t.value:>8.3f} {sharpe:>8.3f} {dd:>8.1%} {ntr:>7}  {pstr}")
    print("-" * 100)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optuna walk-forward hyperparameter search for rate-diff MR (CL-4nnr).",
    )
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--pair", type=str, default="EURUSD")
    parser.add_argument(
        "--study-db",
        type=Path,
        default=Path("data/optuna/rate_diff_mr.db"),
        help="SQLite file backing the Optuna study (resumable across runs).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Optuna n_jobs. Default 1 (serial) — trials share one "
        "pre-loaded data frame, no DB session per trial needed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/best_rate_params.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    from src.dotenv_bootstrap import load_project_env

    load_project_env()

    import optuna

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end <= start:
        logger.error("end (%s) must be after start (%s)", end, start)
        return 2

    logger.info("Loading %s data %s -> %s", args.pair, start.date(), end.date())
    data = load_data(args.pair, start, end)
    logger.info("Loaded %d aligned daily rows; columns=%s", len(data), list(data.columns))

    wf_config = WalkForwardConfig(
        is_window_days=756,
        oos_window_days=63,
        step_days=63,
        min_history=756,
    )
    if len(data) < wf_config.min_history + wf_config.oos_window_days:
        logger.error(
            "Aligned frame too short for one walk-forward fold: %d rows, need >= %d. "
            "Check prices/macro_data coverage for %s + US_10Y + DE_10Y.",
            len(data),
            wf_config.min_history + wf_config.oos_window_days,
            args.pair,
        )
        return 1

    space = build_param_space(RateDiffMRConfig)
    logger.info("Search space (%d params): %s", len(space), sorted(space))

    args.study_db.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.study_db}"
    study = optuna.create_study(
        study_name=f"rate_diff_mr_{args.pair.lower()}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    logger.info(
        "Study %s @ %s (%d existing trials) — running %d new trials, jobs=%d",
        study.study_name,
        storage,
        len(study.trials),
        args.trials,
        args.jobs,
    )

    objective = make_objective(data, space, args.pair, wf_config, CostModel())
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)

    best = study.best_trial
    print_top_trials(study)

    payload = {
        "strategy": "rate_diff_mr",
        "pair": args.pair,
        "best_params": best.params,
        "best_score": float(best.value),
        "best_trial_number": best.number,
        "best_trial_attrs": dict(best.user_attrs),
        "objective": f"sharpe - {DD_WEIGHT}*abs(max_drawdown); "
        f"score={PENALTY_SCORE} if n_trades < {MIN_TRADES}",
        "metadata": {
            "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
            "n_obs_aligned": int(len(data)),
            "trials_in_study": len(study.trials),
            "trials_this_run": args.trials,
            "walkforward": {
                "is_window_days": wf_config.is_window_days,
                "oos_window_days": wf_config.oos_window_days,
                "step_days": wf_config.step_days,
            },
            "study_db": str(args.study_db),
            "study_name": study.study_name,
            "git_rev": _git_rev(),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info("Best params written to %s", args.out)

    print()
    print(
        f"Best trial #{best.number}: score={best.value:.3f} "
        f"(sharpe={best.user_attrs.get('sharpe', float('nan')):.3f}, "
        f"max_dd={best.user_attrs.get('max_drawdown', float('nan')):.1%}, "
        f"n_trades={best.user_attrs.get('n_trades', 0)})"
    )
    print()
    print(
        "NOT auto-applied. To adopt, the operator would edit "
        "src/strategies/rate_diff_mean_reversion.py::RateDiffMRConfig:"
    )
    for line in config_diff(best.params):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
