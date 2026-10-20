"""Real ``BacktestRunner`` for the research loop (CL-p9ix).

The Implementer agent writes a strategy file at
``src/strategies/_experimental/{slug}.py``; the loop's GATE 2 + verdict
engine want a candidate report whose top-level keys match the threshold
paths in ``docs/research/REVIEW_RULES.md`` (``oos_metrics.sharpe``,
``sharpe_ci_95.low``, ``edge_concentration``, etc.). Without a real
runner the loop hands ``backtest_runner=None`` to the Implementer, the
report ships with empty ``backtest_metrics``, and the verdict engine
ESCALATEs everything for missing-metric — no PROMOTE ever fires.

This module produces a callable the loop wires up:

  1. Imports the generated strategy file at ``code_path``. This is also
     the import-time check that catches missing-module / undefined-name
     bugs the syntax gate misses.
  2. Reads the class-level ``symbols`` attribute (the Implementer
     prompt requires it; we trust the contract here).
  3. Fetches a daily OHLCV series for each symbol via the
     ``DataProvider``, joins on the index, slices to the configured
     window.
  4. Runs ``WalkForwardRunner`` with a default ``CostModel``.
  5. Computes ``PerformanceAnalytics.metrics`` on the OOS returns +
     bootstrap 95% CI on Sharpe.
  6. Computes proxy values for the B-section rules (edge_concentration,
     regime_diversified, decay_severity) from the per-fold metrics.
     These are crude — operators reading the transcript can override
     via the dashboard. They're documented in the report's
     ``_metrics_provenance`` field so the audit trail is honest about
     what was measured vs estimated.

Returned dict has top-level keys that ``compute_verdict`` resolves:

  oos_metrics.{sharpe, n_trades, hit_rate, max_drawdown, profit_factor}
  sharpe_ci_95.{low, high}
  is_oos_sharpe_ratio
  edge_concentration
  regime_diversified
  decay_severity
  _metrics_provenance  (dict — what was computed vs estimated)

The runner returns ``raise``-on-failure semantics that the Implementer
then converts into a REJECTED-with-reason result: a strategy that
can't even import or fetch its data is correctly REJECTED, not
ESCALATEd.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import re
import sys
from collections.abc import Callable
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.bootstrap import stationary_bootstrap_sharpe_ci
from src.backtest.cost_model import CostModel
from src.backtest.instrument_metadata import DEFAULT_REGISTRY
from src.backtest.walkforward import WalkForwardConfig, WalkForwardRunner
from src.data.provider import DataProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Public type alias
# ---------------------------------------------------------------------- #


BacktestRunner = Callable[[Path], dict[str, Any]]


# ---------------------------------------------------------------------- #
# Default config
# ---------------------------------------------------------------------- #


DEFAULT_START: str = "2018-01-01"
DEFAULT_END: str = "2024-12-31"
# Bootstrap iterations — 2k is the sweet spot; 10k is closer to the
# academic default but adds wall-time linearly. Operator can override.
DEFAULT_BOOTSTRAP_N: int = 2000


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _import_strategy_module(code_path: Path) -> Any:
    """Load the strategy file as a unique module. The unique-name spec
    avoids collisions across reruns (each candidate gets its own
    namespace). Raises ``ImportError`` on dependency failures, which
    the loop translates into a REJECTED implementation."""
    if not code_path.exists():
        msg = f"strategy file not found at {code_path}"
        raise FileNotFoundError(msg)
    mod_name = f"_research_strategy_{code_path.stem}_{id(code_path)}"
    spec = importlib.util.spec_from_file_location(mod_name, str(code_path))
    if spec is None or spec.loader is None:
        msg = f"could not build import spec for {code_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Clean up the registration so a failed import doesn't poison
        # subsequent calls.
        sys.modules.pop(mod_name, None)
        raise
    return module


def _find_strategy_class(module: Any) -> type:
    """Pick the first top-level class that exposes the strategy
    protocol (has ``fit`` + ``generate_signals``). Falls back to the
    first top-level class if nothing matches the protocol — the
    walk-forward runner will then surface a clean error if it isn't
    actually a strategy."""
    candidates: list[type] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if not isinstance(obj, type):
            continue
        if obj.__module__ != module.__name__:
            continue
        candidates.append(obj)
    if not candidates:
        msg = f"no top-level class found in {module.__name__}"
        raise ValueError(msg)
    for cls in candidates:
        if hasattr(cls, "fit") and hasattr(cls, "generate_signals"):
            return cls
    return candidates[0]


def _read_symbols(strategy_cls: type) -> list[str]:
    """Read an explicit, nonempty symbol sequence; never invent EURUSD.

    Symbols are data identifiers, including signal-only inputs. Exact spelling
    matters: malformed identifiers are rejected, not silently normalized.
    """
    syms = getattr(strategy_cls, "symbols", None)
    if syms is None or isinstance(syms, property):
        try:
            inst = strategy_cls()
            syms = getattr(inst, "symbols", None)
        except Exception as exc:
            raise ValueError("strategy construction failed while reading symbols") from exc
    if not isinstance(syms, (list, tuple)) or not syms:
        raise ValueError("symbols must be an explicit nonempty list or tuple")
    if any(not isinstance(s, str) or not re.fullmatch(r"[A-Z][A-Z0-9_:.=-]*", s) for s in syms):
        raise ValueError("symbols contains a malformed data identifier")
    if len(set(syms)) != len(syms):
        raise ValueError("symbols contains duplicate identifiers")
    return list(syms)


def _read_execution_symbol(strategy_cls: type, symbols: list[str]) -> str:
    """The strategy's P&L is computed on this symbol's price returns.
    Strategies can declare ``execution_symbol`` to use cross-symbol
    signals to trade a single pair (e.g. ``symbols = ['DXY', 'EURUSD']``,
    ``execution_symbol = 'EURUSD'`` — DXY informs but EURUSD trades).
    An omitted target is inferred only for a single declared symbol. Unknown
    execution instruments cannot borrow a default instrument's metrics. The
    existing registry is a supported-instrument contract, not proof of actual
    venue availability or historical liquidity (those remain separate gates).
    """
    if not symbols:
        raise ValueError("execution requires declared symbols")
    if hasattr(strategy_cls, "execution_symbols"):
        raise ValueError("this runner supports one execution_symbol, not a basket")
    if hasattr(strategy_cls, "execution_symbol"):
        exec_sym = strategy_cls.execution_symbol
    elif len(symbols) == 1:
        exec_sym = symbols[0]
    else:
        raise ValueError("multiple input symbols require an explicit execution_symbol")
    if not isinstance(exec_sym, str) or not exec_sym or exec_sym not in symbols:
        raise ValueError("execution_symbol must name one of the declared symbols")
    meta = DEFAULT_REGISTRY.get(exec_sym)
    if meta is None or meta.last_tradable_date == meta.first_tradable_date:
        raise ValueError(f"execution_symbol {exec_sym!r} is not a supported trading instrument")
    return exec_sym


# ---------------------------------------------------------------------- #
# B-rule proxies (computed from walk-forward output)
# ---------------------------------------------------------------------- #


def _fold_sharpes(fold_metrics: pd.DataFrame, minimum: int) -> list[float] | None:
    """CL-7ys8: no implicit pandas NaN skipping or successful all-zero evidence."""
    if "test_sharpe" not in fold_metrics.columns or len(fold_metrics) < minimum:
        return None
    raw = list(fold_metrics["test_sharpe"])
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
        return None
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values) or not any(values):
        return None
    return values


def _compute_edge_concentration(fold_metrics: pd.DataFrame) -> float | None:
    """Crude proxy for rule B.1 (G5 feature attribution): how much of
    the realized edge sits in a single fold? max|fold_sharpe| / sum
    |fold_sharpe|. 1.0 = all edge in one fold; 1/n_folds = perfectly
    diversified across folds. The real G5 attribution looks at feature
    contributions, not folds — so this is a temporal proxy, never sufficient
    promotion evidence. At least two folds are needed to define concentration
    across folds; missing/invalid evidence remains unknown."""
    sharpes = _fold_sharpes(fold_metrics, minimum=2)
    if sharpes is None:
        return None
    # Scale first to avoid overflow in the sum for large finite diagnostics.
    largest = max(abs(value) for value in sharpes)
    scaled_sum = math.fsum(abs(value) / largest for value in sharpes)
    return 1.0 / scaled_sum


def _compute_regime_diversified(fold_metrics: pd.DataFrame) -> bool | None:
    """Crude proxy for rule B.2 (G6 regime decomposition): does the
    edge appear across multiple folds, or only one? True when at least
    2 folds have positive Sharpe out of at least 3 folds. This describes
    time folds, not measured regimes; all-positive folds aren't automatically
    evidence of a regime artifact. Insufficient evidence is unknown."""
    sharpes = _fold_sharpes(fold_metrics, minimum=3)
    if sharpes is None:
        return None
    n_pos = sum(value > 0 for value in sharpes)
    return n_pos >= 2


def _compute_decay_severity(fold_metrics: pd.DataFrame) -> str | None:
    """Crude proxy for rule B.3 (active decay): split fold metrics
    chronologically into thirds; compare oldest-third Sharpe to
    newest-third Sharpe. STRONG: newest < -50% of oldest;
    MODERATE: newest sign-flipped from oldest; NONE: otherwise."""
    # Six folds provide at least two observations per compared third; this is
    # a diagnostic minimum, not a claim of statistical power or measured decay.
    sharpes = _fold_sharpes(fold_metrics, minimum=6)
    if sharpes is None:
        return None
    third = len(sharpes) // 3
    scale = max(abs(value) for value in sharpes)
    oldest = math.fsum(value / scale / third for value in sharpes[:third])
    newest = math.fsum(value / scale / third for value in sharpes[-third:])
    if oldest <= 0:
        return None  # No positive baseline edge to evaluate decay against.
    if oldest > 0 and newest < -0.5 * oldest:
        return "STRONG"
    if oldest > 0 and newest < 0:
        return "MODERATE"
    return "NONE"


# ---------------------------------------------------------------------- #
# The runner
# ---------------------------------------------------------------------- #


def make_backtest_runner(
    data_provider: DataProvider,
    cost_model: CostModel | None = None,
    config: WalkForwardConfig | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
) -> BacktestRunner:
    """Build a backtest_runner callable to pass to
    ``Implementer.implement(backtest_runner=...)``.

    Each call:
      * imports the strategy at ``code_path``
      * fetches OHLCV via ``data_provider`` for each declared symbol
      * runs walk-forward
      * returns metrics matching the verdict engine's threshold paths

    Raises on import/data/walk-forward failure — Implementer turns the
    raise into a REJECTED result with the exception class + message,
    which is the right outcome (a strategy that can't even import is
    not "ambiguous", it's broken).
    """
    cost_model = cost_model or CostModel()
    config = config or WalkForwardConfig()

    def runner(code_path: Path) -> dict[str, Any]:
        # 1) Import. This is also the import-time check that catches
        #    bugs syntax_check missed.
        module = _import_strategy_module(code_path)
        strategy_cls = _find_strategy_class(module)
        symbols = _read_symbols(strategy_cls)
        execution_symbol = _read_execution_symbol(strategy_cls, symbols)
        logger.info(
            "research backtest: %s symbols=%s exec=%s window=%s..%s",
            strategy_cls.__name__,
            symbols,
            execution_symbol,
            start,
            end,
        )

        # 2) Pull data for ALL declared symbols so the strategy can
        #    use cross-symbol signals (CL-40n2 v1). DataProvider.
        #    get_aligned_series returns a wide DataFrame with one
        #    column per symbol; we add a 'close' alias for the
        #    execution_symbol so the walk-forward runner's existing
        #    return calc keeps working.
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        data = data_provider.get_aligned_series(
            symbols=symbols,
            start=start_dt,
            end=end_dt,
        )
        if data is None or data.empty:
            msg = (
                f"DataProvider returned no data for symbols={symbols} in "
                f"window {start}..{end}; cannot backtest"
            )
            raise ValueError(msg)
        if execution_symbol not in data.columns:
            msg = (
                f"backtest data missing execution_symbol column "
                f"{execution_symbol!r}; got {list(data.columns)}"
            )
            raise ValueError(msg)
        # Add 'close' alias pointing at execution_symbol's price series
        # so walk-forward's hard-coded data['close'] works.
        if "close" not in data.columns:
            data = data.assign(close=data[execution_symbol])
        # Forward-fill missing values across columns — different sources
        # (yfinance daily vs FRED monthly) have different timestamps,
        # so the merged frame is sparse. ffill makes signal computation
        # well-defined; missing-data risk lives on the strategy.
        data = data.ffill().dropna(subset=["close"])

        # 3) Walk-forward.
        wf_runner = WalkForwardRunner(config=config)
        result = wf_runner.run(
            data=data,
            strategy_factory=strategy_cls,  # zero-arg constructor
            cost_model=cost_model,
        )
        oos = result.oos_returns
        if oos.empty or oos.std() == 0:
            msg = (
                "walk-forward produced empty or constant OOS returns; "
                "strategy may not be generating signals"
            )
            raise ValueError(msg)

        # 4) Analytics + bootstrap CI.
        full = PerformanceAnalytics.metrics(oos)
        is_sharpe = (
            float(
                result.fold_metrics["train_sharpe"].mean(),
            )
            if not result.fold_metrics.empty
            else 0.0
        )
        oos_sharpe = float(full.get("sharpe", 0.0))
        is_oos_ratio = abs(is_sharpe / oos_sharpe) if oos_sharpe != 0 else 0.0
        ci_low: float | None
        ci_high: float | None
        ci_reason = ""
        try:
            ci_low, ci_high = stationary_bootstrap_sharpe_ci(
                oos,
                n_bootstrap=bootstrap_n,
            )
            if not math.isfinite(ci_low) or not math.isfinite(ci_high) or ci_low > ci_high:
                raise ValueError("bootstrap returned invalid confidence bounds")
        except Exception as exc:
            logger.warning(
                "bootstrap CI failed: %s: %s — confidence interval unavailable",
                type(exc).__name__,
                exc,
            )
            ci_low, ci_high = None, None
            ci_reason = "bootstrap_unavailable"

        # 5) B-rule proxies.
        edge_conc = _compute_edge_concentration(result.fold_metrics)
        regime_div = _compute_regime_diversified(result.fold_metrics)
        decay_sev = _compute_decay_severity(result.fold_metrics)

        # 6) Trade-count: number of position changes (entries+exits)
        n_trades = (
            int(result.trades["position_change"].astype(bool).sum())
            if "position_change" in result.trades.columns
            else 0
        )

        return {
            "oos_metrics": {
                "sharpe": oos_sharpe,
                "n_trades": n_trades,
                "hit_rate": float(full.get("hit_rate", 0.0)),
                "max_drawdown": float(full.get("max_drawdown", 0.0)),
                "profit_factor": float(full.get("profit_factor", 0.0)),
                "sortino": float(full.get("sortino", 0.0)),
                "total_return": float(full.get("total_return", 0.0)),
            },
            "sharpe_ci_95": {"low": ci_low, "high": ci_high},
            "is_oos_sharpe_ratio": is_oos_ratio,
            "edge_concentration": edge_conc,
            "regime_diversified": regime_div,
            "decay_severity": decay_sev,
            "fold_metrics": result.fold_metrics.to_dict(orient="records"),
            "_metric_evidence": {
                **{
                    name: {
                        "status": "insufficient" if value is None else "proxy",
                        "observation_count": len(result.fold_metrics),
                        "method": method,
                        "reason": "insufficient_or_invalid_folds"
                        if value is None
                        else "not_measured_gate_evidence",
                    }
                    for name, value, method in (
                        ("edge_concentration", edge_conc, "fold_sharpe_concentration"),
                        ("regime_diversified", regime_div, "positive_fold_count"),
                        ("decay_severity", decay_sev, "first_last_third_comparison"),
                    )
                },
                "sharpe_ci_95": {
                    "status": "insufficient" if ci_reason else "measured",
                    "observation_count": len(oos),
                    "method": f"stationary_bootstrap_n={bootstrap_n}",
                    "reason": ci_reason,
                },
            },
            "_metrics_provenance": {
                "oos_metrics": "computed from PerformanceAnalytics on walk-forward OOS returns",
                "sharpe_ci_95": ci_reason or f"stationary bootstrap, n={bootstrap_n}",
                "is_oos_sharpe_ratio": "mean(train_sharpe) / oos_sharpe across folds",
                "edge_concentration": "PROXY: max|fold_sharpe|/sum|fold_sharpe| across folds, not feature-attribution G5",
                "regime_diversified": "PROXY: ≥2 folds positive out of ≥3 — not regime-decomposed G6",
                "decay_severity": "PROXY: oldest-third vs newest-third fold-Sharpe comparison — not full decay test",
            },
        }

    return runner
