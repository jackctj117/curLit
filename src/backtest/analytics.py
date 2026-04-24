"""Performance analytics — Sharpe, Sortino, drawdown, and trade-level metrics."""

import numpy as np
import pandas as pd


class PerformanceAnalytics:
    @staticmethod
    def metrics(
        returns: pd.Series, periods_per_year: int = 252,
    ) -> dict[str, float | int]:
        if len(returns) == 0:
            return {}

        ann_factor = periods_per_year
        total_return = (1 + returns).prod() - 1
        years = len(returns) / ann_factor
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        vol = returns.std() * np.sqrt(ann_factor)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor) if returns.std() > 0 else 0.0

        downside = returns[returns < 0]
        sortino = (
            (returns.mean() / downside.std()) * np.sqrt(ann_factor)
            if len(downside) > 0 and downside.std() > 0 else 0.0
        )

        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min())
        max_dd_idx = int(drawdown.idxmin()) if max_dd < 0 else -1
        dd_duration = (
            int(returns[max_dd_idx:][returns[max_dd_idx:] < 0].count())
            if max_dd_idx >= 0 else 0
        )
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

        trades = returns[returns != 0]
        if len(trades) == 0:
            hit_rate, avg_win, avg_loss, profit_factor = 0.0, 0.0, 0.0, float("inf")
        else:
            hit_rate = float((trades > 0).mean())
            wins = trades[trades > 0]
            losses = trades[trades < 0]
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            profit_factor = (
                float(wins.sum() / abs(losses.sum()))
                if len(losses) > 0 else float("inf")
            )

        var_95 = float(np.percentile(returns, 5))
        es_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": max_dd,
            "max_dd_duration_days": dd_duration,
            "calmar": float(calmar),
            "hit_rate": float(hit_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor) if profit_factor != float("inf") else 999.0,
            "var_95": var_95,
            "expected_shortfall_95": float(es_95),
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
        }

    @staticmethod
    def regime_metrics(
        returns: pd.Series, regime: pd.Series,
    ) -> pd.DataFrame:
        df = pd.DataFrame({"returns": returns, "regime": regime})
        grouped = df.groupby("regime")["returns"]
        sharpe_fn = lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0  # noqa: E731
        return pd.DataFrame({
            "mean_ret": grouped.mean() * 252,
            "sharpe": grouped.apply(sharpe_fn),
            "count": grouped.count(),
        })

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series, window: int = 63,
    ) -> pd.Series:
        return (
            returns.rolling(window).mean()
            / returns.rolling(window).std()
            * np.sqrt(252)
        )
