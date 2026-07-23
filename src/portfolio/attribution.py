"""P&L attribution per strategy (CL-8dq).

Records each fill against a strategy_id, then rolls up to realized +
unrealized P&L via FIFO matching. Pairs naturally with the trade journal
(audit trail) and TCA module (per-fill cost components).

Fills support proportional attribution across multiple strategies: a
single broker fill can be split via the ``proportions`` argument when an
intent metadata dict carries weights. Default is 100 % to the
intent.strategy_id.

Tables (created on first use, idempotent):

  strategy_fills              — append-only per-fill log per strategy
  strategy_daily_returns      — pre-aggregated daily returns for fast UI

Reads write through Postgres ON CONFLICT idempotency where applicable.

CL-mdle (P&L decomposition by cost component) extends this by recording
TCAComponents alongside each fill, then splitting net return into signal
alpha / spread / slippage / swap. The fields are reserved here; the
TCA wiring lives in execution/tca.py.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class FillAttribution:
    """One row in strategy_fills — a fill, attributed to one strategy.

    quantity is signed: positive = long-add (or short-cover), negative =
    long-sell (or short-add). This avoids a separate "side" field and
    makes FIFO matching one less branch to maintain.

    cost_components is optional and carries the four CL-mdle splits
    (signal_bps / spread_bps / slippage_bps / swap_bps). When None, the
    attribution still works — decomposition just isn't available for
    that fill.
    """

    strategy_id: str
    symbol: str
    quantity: float
    fill_price: float
    ts: datetime
    intent_id: str | None = None
    fill_id: str | None = None
    cost_components: dict[str, float] | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "fill_price": float(self.fill_price),
            "ts": self.ts,
            "intent_id": self.intent_id,
            "fill_id": self.fill_id,
            "cost_components": self.cost_components,
        }


@dataclass
class StrategyPnL:
    strategy_id: str
    realized: float
    unrealized: float
    open_quantity: float
    avg_open_price: float
    n_fills: int
    # CL-mdle decomposition — populated when cost_components are present
    # on the underlying fills, else zeros. realized + unrealized still
    # equals signal_alpha - (spread + slippage + swap).
    signal_alpha: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    swap_cost: float = 0.0


class PnLAttributor:
    """Per-strategy P&L attribution.

    Storage is dialect-agnostic — Postgres in production, sqlite in
    tests. cost_components is JSONB on Postgres, TEXT on sqlite (we
    JSON-encode at write time and parse on read).
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._create_tables()

    def _create_tables(self) -> None:
        dialect = self.engine.dialect.name
        json_type = "JSONB" if dialect == "postgresql" else "TEXT"
        ts_type = "TIMESTAMPTZ" if dialect == "postgresql" else "TIMESTAMP"
        with self.engine.begin() as conn:
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS strategy_fills (
                    fill_id          TEXT,
                    strategy_id      TEXT NOT NULL,
                    symbol           TEXT NOT NULL,
                    quantity         REAL NOT NULL,
                    fill_price       REAL NOT NULL,
                    ts               {ts_type} NOT NULL,
                    intent_id        TEXT,
                    cost_components  {json_type},
                    PRIMARY KEY (strategy_id, fill_id, ts)
                )
            """)
            )
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS strategy_fills_strategy_ts
                ON strategy_fills (strategy_id, ts)
            """)
            )
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS strategy_daily_returns (
                    date         DATE NOT NULL,
                    strategy_id  TEXT NOT NULL,
                    realized     REAL NOT NULL DEFAULT 0,
                    unrealized   REAL NOT NULL DEFAULT 0,
                    components   {json_type},
                    PRIMARY KEY (date, strategy_id)
                )
            """)
            )

    def attribute_fill(
        self,
        symbol: str,
        quantity: float,
        fill_price: float,
        ts: datetime | None = None,
        proportions: dict[str, float] | None = None,
        intent_id: str | None = None,
        fill_id: str | None = None,
        cost_components: dict[str, float] | None = None,
        # Convenience for the common 1:1 case where the caller knows the
        # strategy directly. If proportions is given, it wins.
        strategy_id: str | None = None,
    ) -> list[FillAttribution]:
        """Split one broker fill across one or more strategies.

        Args:
          proportions: ``{strategy_id: weight}`` summing to 1.0. Sub-fills
            are written as separate rows so per-strategy queries stay
            ergonomic. When None and ``strategy_id`` is given, it's
            equivalent to ``{strategy_id: 1.0}``.

        Returns the list of FillAttribution rows written. Idempotent on
        ``(strategy_id, fill_id, ts)`` — re-attributing the same fill
        is a no-op.
        """
        ts = ts or datetime.now(UTC)
        if proportions is None:
            if strategy_id is None:
                raise ValueError("Either proportions or strategy_id required")
            proportions = {strategy_id: 1.0}
        total_w = sum(proportions.values())
        assert abs(total_w - 1.0) < 1e-6, (
            f"Proportions must sum to 1.0, got {total_w} ({proportions})"
        )

        rows: list[FillAttribution] = []
        for sid, weight in proportions.items():
            qty = quantity * weight
            rows.append(
                FillAttribution(
                    strategy_id=sid,
                    symbol=symbol,
                    quantity=qty,
                    fill_price=fill_price,
                    ts=ts,
                    intent_id=intent_id,
                    fill_id=fill_id,
                    cost_components=cost_components,
                )
            )

        self._persist(rows)
        return rows

    def _persist(self, rows: list[FillAttribution]) -> None:
        import json as _json

        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            stmt = text("""
                INSERT INTO strategy_fills (fill_id, strategy_id, symbol,
                    quantity, fill_price, ts, intent_id, cost_components)
                VALUES (:fid, :sid, :sym, :qty, :px, :ts, :iid, CAST(:cc AS JSONB))
                ON CONFLICT (strategy_id, fill_id, ts) DO NOTHING
            """)
        else:
            stmt = text(
                "INSERT OR IGNORE INTO strategy_fills "
                "(fill_id, strategy_id, symbol, quantity, fill_price, "
                "ts, intent_id, cost_components) VALUES "
                "(:fid, :sid, :sym, :qty, :px, :ts, :iid, :cc)",
            )
        with self.engine.begin() as conn:
            for r in rows:
                conn.execute(
                    stmt,
                    {
                        "fid": r.fill_id,
                        "sid": r.strategy_id,
                        "sym": r.symbol,
                        "qty": float(r.quantity),
                        "px": float(r.fill_price),
                        "ts": r.ts,
                        "iid": r.intent_id,
                        "cc": _json.dumps(r.cost_components) if r.cost_components else None,
                    },
                )

    def compute_strategy_pnl(
        self,
        strategy_id: str,
        last_price: dict[str, float] | None = None,
        since: datetime | None = None,
    ) -> StrategyPnL:
        """Realized + unrealized P&L for one strategy via FIFO matching.

        Args:
          last_price: optional ``{symbol: mark_price}`` for the unrealized
            calculation. Symbols without a mark default to the avg_open
            price (zero unrealized contribution).
          since: only consider fills at or after this timestamp.

        FIFO is per-symbol — opening a long, closing it, then re-opening
        another long is two distinct realization events.
        """
        last_price = last_price or {}
        params: dict[str, Any] = {"sid": strategy_id}
        clauses = ["strategy_id = :sid"]
        if since is not None:
            params["since"] = since
            clauses.append("ts >= :since")
        where = " AND ".join(clauses)

        with self.engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                SELECT symbol, quantity, fill_price, cost_components
                FROM strategy_fills
                WHERE {where}
                ORDER BY ts, fill_id
            """),
                params,
            ).fetchall()

        # Per-symbol open lots: deque of (signed_qty, price). When a new
        # fill matches the existing inventory's sign, append a lot.
        # Opposite sign — peel lots FIFO and realize.
        inventory: dict[str, deque[tuple[float, float]]] = {}
        realized = 0.0
        signal = spread = slippage = swap = 0.0
        n_fills = 0
        import json as _json

        for sym, qty, px, cc in rows:
            n_fills += 1
            cc_dict: dict[str, float] = {}
            if cc:
                try:
                    cc_dict = _json.loads(cc) if isinstance(cc, str) else dict(cc)
                except (TypeError, ValueError):
                    cc_dict = {}
            signal += float(cc_dict.get("signal_bps", 0))
            spread += float(cc_dict.get("spread_bps", 0))
            slippage += float(cc_dict.get("slippage_bps", 0))
            swap += float(cc_dict.get("swap_bps", 0))

            lots = inventory.setdefault(sym, deque())
            qty_remaining = float(qty)
            while lots and qty_remaining != 0 and ((qty_remaining > 0) != (lots[0][0] > 0)):
                head_qty, head_px = lots[0]
                # Closing quantity is the smaller magnitude of the two.
                close_qty = min(abs(head_qty), abs(qty_remaining))
                # Sign of realized = same as the closing direction.
                # Long lot (head_qty > 0) closed by sell (qty_remaining < 0)
                # earns (sell_price - buy_price) * close_qty.
                pnl = (px - head_px) * close_qty * (1 if head_qty > 0 else -1)
                realized += pnl
                # Reduce or pop the lot
                new_head = head_qty + (close_qty if head_qty < 0 else -close_qty)
                if abs(new_head) < 1e-12:
                    lots.popleft()
                else:
                    lots[0] = (new_head, head_px)
                qty_remaining += close_qty if qty_remaining < 0 else -close_qty
            if abs(qty_remaining) > 1e-12:
                lots.append((qty_remaining, float(px)))

        # Unrealized = sum over open lots of (mark - lot_price) * lot_qty
        unrealized = 0.0
        open_qty = 0.0
        weighted_open_px = 0.0
        total_abs_open = 0.0
        for sym, lots in inventory.items():
            mark = last_price.get(sym)
            for lot_qty, lot_px in lots:
                open_qty += lot_qty
                total_abs_open += abs(lot_qty)
                weighted_open_px += abs(lot_qty) * lot_px
                if mark is not None:
                    unrealized += (mark - lot_px) * lot_qty

        avg_open = weighted_open_px / total_abs_open if total_abs_open > 0 else 0.0
        return StrategyPnL(
            strategy_id=strategy_id,
            realized=realized,
            unrealized=unrealized,
            open_quantity=open_qty,
            avg_open_price=avg_open,
            n_fills=n_fills,
            signal_alpha=signal,
            spread_cost=spread,
            slippage_cost=slippage,
            swap_cost=swap,
        )

    def list_strategies(self) -> list[str]:
        """All strategy_ids that have at least one attributed fill."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT strategy_id FROM strategy_fills ORDER BY strategy_id",
                )
            ).fetchall()
        return [r[0] for r in rows]


def emit_pnl_metrics(
    attributor: PnLAttributor,
    last_price: dict[str, float] | None = None,
) -> dict[str, StrategyPnL]:
    """Compute every strategy's P&L and emit Prometheus gauges.

    Returns the dict for callers (tests, dashboards) that want the same
    numbers as the gauge values. Failure is non-fatal — metrics emission
    is observability, not a control loop.
    """
    pnls: dict[str, StrategyPnL] = {}
    for sid in attributor.list_strategies():
        pnl = attributor.compute_strategy_pnl(sid, last_price=last_price)
        pnls[sid] = pnl
        try:
            from src.monitoring.metrics import strategy_attributed_pnl

            strategy_attributed_pnl.labels(
                strategy=sid,
                kind="realized",
            ).set(pnl.realized)
            strategy_attributed_pnl.labels(
                strategy=sid,
                kind="unrealized",
            ).set(pnl.unrealized)
            # CL-mdle decomposition gauges — set even when zero so dashboards
            # always have a series to render.
            for component, value in (
                ("signal_alpha", pnl.signal_alpha),
                ("spread_cost", pnl.spread_cost),
                ("slippage_cost", pnl.slippage_cost),
                ("swap_cost", pnl.swap_cost),
            ):
                strategy_attributed_pnl.labels(
                    strategy=sid,
                    kind=component,
                ).set(value)
        except Exception:
            logger.warning("Failed to emit pnl gauge for %s", sid)
    return pnls
