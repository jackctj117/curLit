"""Walk-forward backtesting framework — non-overlapping IS/OOS windows."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

logger = __import__("logging").getLogger(__name__)


class Strategy(Protocol):
    def fit(self, train_data: pd.DataFrame) -> None: ...
    def generate_signals(self, test_data: pd.DataFrame) -> pd.Series: ...


@dataclass
class WalkForwardConfig:
    # 756 trading days = ~3 years. Long enough to span at least one full
    # rate cycle so the in-sample fit doesn't bake in single-regime
    # parameters. Below ~2y, FX strategies overfit to a single Fed cycle.
    is_window_days: int = 756
    # 63 trading days = ~3 months out-of-sample. Short enough that we get
    # multiple OOS folds per year (4 quarterly checks); long enough that
    # the Sharpe of the OOS slice is statistically meaningful (n>=63).
    oos_window_days: int = 63
    # Step size = OOS window means non-overlapping OOS slices, which is
    # the canonical walk-forward design (Pardo 2008). Overlapping OOS
    # double-counts test data and inflates apparent significance.
    step_days: int = 63
    # Same 756-day floor as is_window — refuse to run if there isn't
    # enough history for even one fold.
    min_history: int = 756
    # CL-nt0c: optional TradabilityFilter — when set, columns of `data`
    # are filtered per-instrument to drop bars before first_tradable_date
    # and after last_tradable_date. Default None = no filter (legacy
    # behavior, treats every column as universally tradable).
    tradability_filter: Any = None
    # CL-x50g: cost realism. Default False = per-pair spread costs via
    # cost_model.get_cost_per_turn(pair) (pair inferred from the strategy
    # instance for Series signals, per-column for DataFrame signals) plus
    # overnight funding via cost_model.overnight_funding_daily, when the
    # cost model provides them. Set True to force the legacy flat
    # cost_model.cost_per_turn with no funding — keeps historical backtest
    # reports byte-comparable.
    legacy_flat_costs: bool = False


@dataclass
class WalkForwardResult:
    oos_signals: pd.Series
    oos_returns: pd.Series
    trades: pd.DataFrame
    fold_metrics: pd.DataFrame
    params_by_fold: list[dict[str, Any]] = field(default_factory=list)


class WalkForwardRunner:
    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def run(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable[[], Any],
        cost_model: Any,
    ) -> WalkForwardResult:
        cfg = self.config
        # CL-nt0c: when a tradability filter is configured, mask any
        # bar where any instrument is non-tradable BEFORE folding. The
        # mask is applied by setting the cell to NaN; downstream signal
        # generation already skips NaN rows. We don't drop the row
        # outright because dropping would shift OOS window indices.
        if cfg.tradability_filter is not None:
            data = self._apply_tradability(data, cfg.tradability_filter)

        folds: list[dict[str, Any]] = []
        oos_signals_all: list[pd.Series | pd.DataFrame] = []
        trades_all: list[pd.DataFrame] = []

        start = cfg.min_history
        while start + cfg.oos_window_days <= len(data):
            train_end = start
            train_start = max(0, train_end - cfg.is_window_days)
            test_end = min(len(data), train_end + cfg.oos_window_days)

            train = data.iloc[train_start:train_end]
            test = data.iloc[train_end:test_end]

            strategy = strategy_factory()
            strategy.fit(train)
            signals = strategy.generate_signals(test)
            oos_signals_all.append(signals)

            # CL-40n2 v2: signals can be either a single Series (single-
            # asset strategy) OR a DataFrame with one column per symbol
            # (joint multi-asset / cross-sectional). The trade-DataFrame
            # builder below handles both cases — _trades_for_signals
            # returns the same shape per fold so the OOS aggregation
            # stays uniform.
            # CL-x50g: thread the traded pair through so per-pair costs
            # apply (walk-forward is single-asset per run for Series
            # signals; DataFrame signals carry the pair per column).
            df = self._trades_for_signals(
                signals=signals, data=data, cost_model=cost_model,
                pair=self._infer_pair(strategy),
                legacy_flat_costs=cfg.legacy_flat_costs,
            )
            trades_all.append(df)

            # In-sample Sharpe must use STRATEGY RETURNS, not raw close
            # prices (CL-u9rn). The previous version computed Sharpe of
            # the price series itself, yielding mean(price)/std(price)
            # ≈ hundreds for FX pairs and breaking rule A.7. The fix
            # mirrors the OOS path: re-fit the strategy on this train
            # window's first 80% (in-sample), compute net returns on
            # the remaining 20% (still in-sample), take Sharpe of those.
            train_sharpe = self._train_sharpe(
                train, strategy_factory, cost_model,
                legacy_flat_costs=cfg.legacy_flat_costs,
            )
            folds.append({
                "fold_id": len(folds),
                "train_start": train.index[0], "train_end": train.index[-1],
                "test_start": test.index[0], "test_end": test.index[-1],
                "train_sharpe": train_sharpe,
                "test_sharpe": self._sharpe(df["net_return"]),
            })
            start += cfg.step_days

        return WalkForwardResult(
            oos_signals=pd.concat(oos_signals_all) if oos_signals_all else pd.Series(dtype=float),
            oos_returns=pd.concat([t["net_return"] for t in trades_all]) if trades_all else pd.Series(dtype=float),
            trades=pd.concat(trades_all) if trades_all else pd.DataFrame(),
            fold_metrics=pd.DataFrame(folds),
        )

    @staticmethod
    def _apply_tradability(
        data: pd.DataFrame, tf: Any,
    ) -> pd.DataFrame:
        """Mask out non-tradable cells per CL-nt0c.

        For each column treated as an instrument, set values to NaN on
        dates outside ``[first_tradable_date, last_tradable_date]``.
        Strategies see NaN and skip — no need to compress the index.
        """
        out = data.copy()
        for sym in out.columns:
            meta = tf.registry.get(str(sym))
            if meta is None:
                continue
            idx = pd.to_datetime(out.index)
            first = pd.Timestamp(meta.first_tradable_date)
            mask = idx >= first
            if meta.last_tradable_date is not None:
                last = pd.Timestamp(meta.last_tradable_date)
                mask = mask & (idx <= last)
            out.loc[~mask, sym] = float("nan")
        return out

    @staticmethod
    def _infer_pair(strategy: Any) -> str | None:
        """Best-effort single traded pair for per-pair costs (CL-x50g).

        Walk-forward runs are single-asset per run for Series-signal
        strategies; the pair comes from ``.symbols`` (live-engine
        contract, when exactly one) or ``.config.pair``. Multi-asset
        strategies return DataFrame signals and get per-column pairs in
        ``_trades_multi_asset`` instead — return None here.
        """
        try:
            symbols = getattr(strategy, "symbols", None)
            if symbols is not None and len(symbols) == 1:
                return str(symbols[0])
            pair = getattr(getattr(strategy, "config", None), "pair", None)
            if isinstance(pair, str) and pair:
                return pair
        except Exception:  # defensive: strategy properties may raise
            return None
        return None

    @staticmethod
    def _resolve_cost_per_turn(
        cost_model: Any, pair: str | None, legacy_flat_costs: bool,
    ) -> float:
        """Per-turn cost fraction: per-pair when known and supported,
        else the legacy flat ``cost_per_turn`` (CL-x50g)."""
        if not legacy_flat_costs and pair is not None:
            fn = getattr(cost_model, "get_cost_per_turn", None)
            if callable(fn):
                return float(fn(pair))
        return float(cost_model.cost_per_turn)

    @staticmethod
    def _funding_daily(cost_model: Any, legacy_flat_costs: bool) -> float:
        """Per-day funding cost fraction on the absolute held position
        (CL-x50g). 0.0 in legacy mode or when the cost model doesn't
        model funding — duck-typed so bare test doubles keep working."""
        if legacy_flat_costs:
            return 0.0
        try:
            return float(getattr(cost_model, "overnight_funding_daily", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _trades_for_signals(
        signals: pd.Series | pd.DataFrame,
        data: pd.DataFrame,
        cost_model: Any,
        pair: str | None = None,
        legacy_flat_costs: bool = False,
    ) -> pd.DataFrame:
        """Build the per-fold trades DataFrame from a strategy's signal
        output. Two shapes accepted (CL-40n2 v2):

          * Series: single-asset strategy. Use ``data['close']`` for
            return computation (the existing v1 path). Output frame
            has columns: signal, position, return, strategy_return,
            position_change, cost, funding, net_return.
          * DataFrame: joint multi-asset. Each column is one symbol's
            position weight. Returns are computed per column from the
            corresponding column in ``data``; portfolio
            ``strategy_return`` is the sum-product across symbols.
            Cost is summed across per-symbol position changes. Output
            frame has the same column names as the Series case so
            downstream aggregation (oos_returns, fold_metrics) is
            uniform.

        CL-x50g cost realism: ``pair`` routes per-pair spread costs via
        ``cost_model.get_cost_per_turn(pair)``; held positions pay
        ``cost_model.overnight_funding_daily`` per bar (crude flat swap
        model — see CostModel). ``legacy_flat_costs=True`` restores the
        flat ``cost_per_turn``-only behavior.
        """
        if isinstance(signals, pd.DataFrame):
            return WalkForwardRunner._trades_multi_asset(
                signals=signals, data=data, cost_model=cost_model,
                legacy_flat_costs=legacy_flat_costs,
            )
        # Single-asset (Series) path — preserves the v1 behavior.
        per_turn = WalkForwardRunner._resolve_cost_per_turn(
            cost_model, pair, legacy_flat_costs,
        )
        funding_daily = WalkForwardRunner._funding_daily(
            cost_model, legacy_flat_costs,
        )
        df = pd.DataFrame(index=signals.index)
        df["signal"] = signals
        df["position"] = signals.shift(1).fillna(0)
        df["return"] = (
            data.loc[df.index, "close"].pct_change().fillna(0)
            if "close" in data.columns
            else pd.Series(0, index=df.index)
        )
        df["strategy_return"] = df["position"] * df["return"]
        df["position_change"] = df["position"].diff().abs().fillna(0)
        # Funding accrues on the position held INTO each bar (position is
        # already the once-lagged signal, i.e. the overnight book).
        df["funding"] = df["position"].abs() * funding_daily
        df["cost"] = df["position_change"] * per_turn + df["funding"]
        df["net_return"] = df["strategy_return"] - df["cost"]
        return df

    @staticmethod
    def _trades_multi_asset(
        signals: pd.DataFrame,
        data: pd.DataFrame,
        cost_model: Any,
        legacy_flat_costs: bool = False,
    ) -> pd.DataFrame:
        """Build the trades frame for a multi-asset strategy. Each
        column of ``signals`` is one symbol's position weight; we
        compute per-symbol returns from ``data``'s same-named columns
        and aggregate to a portfolio return.

        Cost (CL-x50g): per-symbol position-change × that symbol's
        per-pair cost via ``cost_model.get_cost_per_turn(symbol)`` when
        available, plus per-symbol overnight funding on held positions;
        summed to portfolio level. ``legacy_flat_costs=True`` restores
        the flat same-cost-for-every-pair behavior with no funding.
        """
        # Cap to the symbols actually in the data (drop POLY: features
        # and any unrecognized columns from the position frame; they
        # can't carry P&L).
        tradeable = [c for c in signals.columns if c in data.columns]
        if not tradeable:
            # No tradeable columns — return an empty-shaped trades frame
            # with the expected columns so downstream concat works.
            return pd.DataFrame(
                {
                    "signal": pd.Series(0.0, index=signals.index),
                    "position": pd.Series(0.0, index=signals.index),
                    "return": pd.Series(0.0, index=signals.index),
                    "strategy_return": pd.Series(0.0, index=signals.index),
                    "position_change": pd.Series(0.0, index=signals.index),
                    "cost": pd.Series(0.0, index=signals.index),
                    "funding": pd.Series(0.0, index=signals.index),
                    "net_return": pd.Series(0.0, index=signals.index),
                },
            )

        # Build per-symbol position + return frames aligned to the
        # signal index.
        positions = signals[tradeable].shift(1).fillna(0)
        prices = data.loc[positions.index, tradeable]
        # fill_method=None so pct_change does NOT pad across NaN gaps
        # (e.g. instruments masked-out by TradabilityFilter post-CL-nt0c).
        # The default 'pad' would synthesize zero-return continuations,
        # which silently fakes tradability across delistings.
        returns = prices.pct_change(fill_method=None).fillna(0)

        # Per-symbol P&L contributions
        per_symbol_pnl = positions * returns
        per_symbol_change = positions.diff().abs().fillna(0)
        # CL-x50g: per-pair per-turn costs (column name IS the pair) +
        # overnight funding on the held book, per symbol.
        turn_costs = pd.Series({
            sym: WalkForwardRunner._resolve_cost_per_turn(
                cost_model, str(sym), legacy_flat_costs,
            )
            for sym in tradeable
        })
        funding_daily = WalkForwardRunner._funding_daily(
            cost_model, legacy_flat_costs,
        )
        per_symbol_cost = per_symbol_change.mul(turn_costs, axis=1)
        per_symbol_funding = positions.abs() * funding_daily

        # Aggregate to portfolio
        portfolio_return = per_symbol_pnl.sum(axis=1)
        portfolio_change = per_symbol_change.sum(axis=1)
        portfolio_funding = per_symbol_funding.sum(axis=1)
        portfolio_cost = per_symbol_cost.sum(axis=1) + portfolio_funding

        df = pd.DataFrame(index=positions.index)
        # "signal" carries the average raw signal across symbols so
        # the existing trades.position_change.astype(bool).sum()
        # n_trades-counter still works (any non-zero position move
        # counts as activity in some symbol).
        df["signal"] = signals[tradeable].mean(axis=1)
        df["position"] = positions.mean(axis=1)
        df["return"] = returns.mean(axis=1)
        df["strategy_return"] = portfolio_return
        df["position_change"] = portfolio_change
        df["cost"] = portfolio_cost
        df["funding"] = portfolio_funding
        df["net_return"] = portfolio_return - portfolio_cost
        return df

    @staticmethod
    def _sharpe(returns: pd.Series, periods: int = 252) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods))

    @staticmethod
    def _train_sharpe(
        train: pd.DataFrame,
        strategy_factory: Callable[[], Any],
        cost_model: Any,
        legacy_flat_costs: bool = False,
    ) -> float:
        """Compute in-sample Sharpe via an inner fit/test split (CL-u9rn).
        Splits ``train`` 80/20 — fit on the first 80%, generate signals
        on the last 20% (still in-sample), compute net strategy returns
        on that slice, return Sharpe.

        Handles both Series (single-asset) and DataFrame (multi-asset,
        CL-40n2 v2) signal shapes by reusing ``_trades_for_signals``.
        Falls back to 0.0 when the inner split is too short, the
        strategy errors, or produces empty signals — keeps rule A.7
        well-behaved by default."""
        n = len(train)
        if n < 50:
            return 0.0
        cut = int(n * 0.8)
        inner_train = train.iloc[:cut]
        inner_test = train.iloc[cut:]
        if len(inner_test) < 5:
            return 0.0
        try:
            inner_strategy = strategy_factory()
            inner_strategy.fit(inner_train)
            inner_signals = inner_strategy.generate_signals(inner_test)
        except Exception:
            return 0.0
        if inner_signals.empty:
            return 0.0
        # Reuse the same trades-builder the OOS path uses — handles
        # both Series (single-asset) and DataFrame (multi-asset)
        # signal shapes uniformly. CL-x50g: same cost realism as OOS.
        df = WalkForwardRunner._trades_for_signals(
            signals=inner_signals, data=train, cost_model=cost_model,
            pair=WalkForwardRunner._infer_pair(inner_strategy),
            legacy_flat_costs=legacy_flat_costs,
        )
        return WalkForwardRunner._sharpe(df["net_return"])
