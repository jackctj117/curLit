"""Event-driven strategy backtester with expanding-window threshold calibration."""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EventBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict[str, Any]


class EventBacktester:
    def run(
        self,
        events: pd.DataFrame,
        prices: pd.DataFrame,
        holding_days: int = 10,
        hard_stop_pct: float = 0.015,
        trailing_trigger_pct: float = 0.02,
        trailing_distance_pct: float = 0.015,
        starting_equity: float = 100_000.0,
        cost_per_turn_bps: float = 1.0,
    ) -> EventBacktestResult:
        trades: list[dict[str, Any]] = []
        equity = starting_equity

        for idx, event in events.iterrows():
            pair = event.get("pair", "EURUSD")
            direction = event.get("direction", 1)
            entry_price = event.get(
                "entry_price", prices.loc[idx, pair] if idx in prices.index else None
            )
            if entry_price is None:
                continue

            # Find exit in subsequent prices
            exit_window = prices.loc[idx:].head(holding_days + 1)
            if len(exit_window) < 2:
                continue

            stop_loss = entry_price * (1.0 - direction * hard_stop_pct)
            trailing_stop: float | None = None
            peak = entry_price
            exit_price = exit_window[pair].iloc[-1]
            exit_reason = "time_exit"

            for ts in exit_window.index[1:]:
                current = exit_window.loc[ts, pair]
                pnl_pct = (current - entry_price) / entry_price * direction

                if pnl_pct >= trailing_trigger_pct:
                    if direction > 0:
                        peak = max(peak, current)
                        trailing_stop = peak * (1.0 - trailing_distance_pct)
                    else:
                        peak = min(peak, current)
                        trailing_stop = peak * (1.0 + trailing_distance_pct)

                if direction > 0:
                    if current <= stop_loss:
                        exit_price, exit_reason = stop_loss, "hard_stop"
                        break
                    if trailing_stop and current <= trailing_stop:
                        exit_price, exit_reason = trailing_stop, "trailing_stop"
                        break
                else:
                    if current >= stop_loss:
                        exit_price, exit_reason = stop_loss, "hard_stop"
                        break
                    if trailing_stop and current >= trailing_stop:
                        exit_price, exit_reason = trailing_stop, "trailing_stop"
                        break

            gross_pnl_pct = (exit_price - entry_price) / entry_price * direction
            net_pnl_pct = gross_pnl_pct - (cost_per_turn_bps / 10000.0 * 2)
            pnl_dollars = equity * net_pnl_pct
            equity += pnl_dollars

            trades.append(
                {
                    "entry_ts": idx,
                    "pair": pair,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "gross_pnl_pct": gross_pnl_pct,
                    "net_pnl_pct": net_pnl_pct,
                    "pnl_dollars": pnl_dollars,
                    "equity_after": equity,
                }
            )

        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(
            [e["equity_after"] for e in trades],
            index=[e["entry_ts"] for e in trades],
        )
        metrics = self._compute_metrics(trades_df, starting_equity, equity)
        return EventBacktestResult(trades_df, equity_curve, metrics)

    def _compute_metrics(
        self, trades: pd.DataFrame, start_eq: float, end_eq: float
    ) -> dict[str, Any]:
        if len(trades) == 0:
            return {"n_trades": 0}
        years = max((trades["entry_ts"].max() - trades["entry_ts"].min()).days / 365.25, 0.01)
        return {
            "n_trades": len(trades),
            "total_return": (end_eq / start_eq) - 1,
            "cagr": (end_eq / start_eq) ** (1 / years) - 1,
            "hit_rate": float((trades["net_pnl_pct"] > 0).mean()),
            "avg_win": float(trades[trades["net_pnl_pct"] > 0]["net_pnl_pct"].mean() or 0),
            "avg_loss": float(trades[trades["net_pnl_pct"] < 0]["net_pnl_pct"].mean() or 0),
        }
