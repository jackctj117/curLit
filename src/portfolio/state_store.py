"""Portfolio state store — persistence for reallocations, portfolio orders, returns history.

Implements PortfolioStateProtocol for the production live engine. Tables:
    portfolio_reallocations — every weight change with regime context
    portfolio_orders        — every aggregated, post-constraint order with contributions

Methods that depend on data not yet ingested (per-strategy daily returns, per-strategy
attributed positions) return safe defaults until the dependent work lands:

    get_positions_by_strategy        — returns [] (D3 / CL-8dq P&L attribution wires this)
    load_strategy_returns_history    — returns empty DataFrame (D6 / CL-5lq daily returns)

Coordinator behavior with empty results is deliberately safe: rebalance falls back
to equal weight; remove_strategy emits no liquidation orders (admin should liquidate
manually until per-strategy attribution lands).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class PortfolioStateStore:
    """SQLAlchemy-backed implementation of PortfolioStateProtocol."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._create_tables()

    def _create_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_reallocations (
                        ts        TIMESTAMPTZ NOT NULL PRIMARY KEY,
                        weights   JSONB NOT NULL,
                        regime    JSONB NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_orders (
                        ts                  TIMESTAMPTZ NOT NULL,
                        symbol              TEXT NOT NULL,
                        target_position     DOUBLE PRECISION NOT NULL,
                        contributions       JSONB NOT NULL,
                        PRIMARY KEY (ts, symbol)
                    )
                    """
                )
            )

    def record_reallocation(
        self,
        ts: datetime,
        weights: dict[str, float],
        regime: dict[str, Any],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO portfolio_reallocations (ts, weights, regime)
                    VALUES (:ts, :weights, :regime)
                    ON CONFLICT (ts) DO UPDATE
                    SET weights = EXCLUDED.weights,
                        regime  = EXCLUDED.regime
                    """
                ),
                {
                    "ts": ts,
                    "weights": json.dumps(weights),
                    "regime": json.dumps(regime, default=str),
                },
            )

    def record_portfolio_order(
        self,
        ts: datetime,
        symbol: str,
        target_position: float,
        strategy_contributions: dict[str, float],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO portfolio_orders (ts, symbol, target_position, contributions)
                    VALUES (:ts, :sym, :pos, :contribs)
                    ON CONFLICT (ts, symbol) DO UPDATE
                    SET target_position = EXCLUDED.target_position,
                        contributions   = EXCLUDED.contributions
                    """
                ),
                {
                    "ts": ts,
                    "sym": symbol,
                    "pos": float(target_position),
                    "contribs": json.dumps(strategy_contributions),
                },
            )

    def get_positions_by_strategy(self, strategy_id: str) -> list[Any]:
        """Return per-strategy attributed positions.

        Currently returns [] — per-strategy attribution requires D3 (CL-8dq).
        Until then, remove_strategy() will not auto-liquidate; admin must
        flatten positions manually before calling remove_strategy.
        """
        logger.warning(
            "get_positions_by_strategy(%s) returning [] — D3 attribution not yet wired",
            strategy_id,
        )
        return []

    def load_strategy_returns_history(
        self,
        strategy_ids: list[str],
        lookback_days: int,
    ) -> pd.DataFrame:
        """Return per-strategy daily returns history for risk parity.

        Currently returns empty DataFrame — daily returns ingestion lands with
        D6 (CL-5lq portfolio monitoring metrics). Until then the coordinator's
        rebalance falls back to equal weight, which is the correct conservative
        behavior in the absence of returns data.
        """
        logger.info(
            "load_strategy_returns_history(%s, lookback=%d) returning empty — D6 not yet wired",
            strategy_ids,
            lookback_days,
        )
        return pd.DataFrame()
