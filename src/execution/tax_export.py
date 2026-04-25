"""Tax export — group fills into closed lots (FIFO) for annual reporting (CL-6mby).

For US Section 988 reporting on FX, gains/losses are ordinary income — net
realized P&L by year. Lot matching uses FIFO (first-in-first-out): each closing
fill consumes the oldest opposite-side open lot, computing realized P&L per
matched chunk.

Currently implemented per-instrument, per-year. Multi-currency P&L conversion
to USD is NOT yet wired (requires FX-rate lookup per close — that's a follow-up
ticket once the feature/data versioning module lands and we have a clean
historical-rate query path).

Output: pandas DataFrame writable to Parquet via export_annual_to_parquet.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.execution.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


@dataclass
class ClosedLot:
    """One matched lot — open and close fills paired."""

    symbol: str
    open_ts: datetime
    close_ts: datetime
    side: str  # "long" or "short"
    quantity: float
    open_price: float
    close_price: float
    open_intent_id: str | None
    close_intent_id: str | None
    realized_pnl_quote_ccy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "open_ts": self.open_ts,
            "close_ts": self.close_ts,
            "side": self.side,
            "quantity": self.quantity,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "open_intent_id": self.open_intent_id,
            "close_intent_id": self.close_intent_id,
            "realized_pnl_quote_ccy": self.realized_pnl_quote_ccy,
        }


def _fill_quantity(payload: dict[str, Any]) -> float | None:
    """Extract signed quantity from a fill payload.

    Convention: positive for buy fills, negative for sell. Falls back to None
    if the payload is missing required fields.
    """
    qty = payload.get("quantity")
    side = payload.get("side")
    if qty is None or side is None:
        return None
    qty = float(qty)
    return -abs(qty) if side == "sell" else abs(qty)


def _fill_price(payload: dict[str, Any]) -> float | None:
    price = payload.get("fill_price") or payload.get("price")
    if price is None:
        return None
    return float(price)


def match_fifo(
    fills: list[dict[str, Any]],
) -> list[ClosedLot]:
    """Match fills into closed lots using FIFO accounting per symbol.

    fills: list of dicts with keys symbol, ts, quantity (signed), price, intent_id.
    Returns list of ClosedLot objects in the order trades closed.
    """
    open_lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    closed: list[ClosedLot] = []

    for fill in fills:
        symbol = fill["symbol"]
        qty = float(fill["quantity"])
        price = float(fill["price"])
        ts = fill["ts"]
        intent_id = fill.get("intent_id")

        if abs(qty) < 1e-12:
            continue

        lots = open_lots[symbol]

        if not lots or (lots[0]["qty"] > 0) == (qty > 0):
            # Same-direction (or no open lot) — extends open position.
            lots.append({
                "qty": qty,
                "price": price,
                "ts": ts,
                "intent_id": intent_id,
            })
            continue

        # Opposite direction — closes oldest open lots.
        remaining = abs(qty)
        closing_side = "long" if qty < 0 else "short"
        # If we're closing a long, the open lot's qty is positive; if closing a
        # short, the open lot's qty is negative. Side label refers to the lot
        # that was being closed (i.e., the original direction we held).
        while remaining > 1e-12 and lots and (lots[0]["qty"] > 0) != (qty > 0):
            lot = lots[0]
            lot_qty = abs(lot["qty"])
            close_qty = min(remaining, lot_qty)

            # Realized P&L per unit, in quote currency (e.g. USD for EURUSD).
            # For long closing (sell): pnl = (close_price - open_price) × qty.
            # For short closing (buy):  pnl = (open_price - close_price) × qty.
            if lot["qty"] > 0:
                pnl = (price - lot["price"]) * close_qty
                lot_side = "long"
            else:
                pnl = (lot["price"] - price) * close_qty
                lot_side = "short"

            closed.append(
                ClosedLot(
                    symbol=symbol,
                    open_ts=lot["ts"],
                    close_ts=ts,
                    side=lot_side,
                    quantity=close_qty,
                    open_price=lot["price"],
                    close_price=price,
                    open_intent_id=lot.get("intent_id"),
                    close_intent_id=intent_id,
                    realized_pnl_quote_ccy=pnl,
                )
            )

            remaining -= close_qty
            if abs(close_qty - lot_qty) < 1e-12:
                lots.popleft()
            else:
                # Partially consumed — reduce magnitude, preserve sign.
                sign = 1 if lot["qty"] > 0 else -1
                lot["qty"] = sign * (lot_qty - close_qty)

        # If we still have closing quantity but no matching lots, the fill
        # opens a new opposite-direction lot (reversal trade).
        if remaining > 1e-12:
            sign = 1 if qty > 0 else -1
            lots.append({
                "qty": sign * remaining,
                "price": price,
                "ts": ts,
                "intent_id": intent_id,
            })

        # Suppress the unused-variable warning while preserving the side label
        # for future reporting use (matched lots already carry side directly).
        _ = closing_side

    return closed


def export_annual(
    journal: TradeJournal,
    year: int,
) -> pd.DataFrame:
    """Build the annual realized-P&L lot table for tax reporting.

    Pulls all FILLED + PARTIAL_FILL events whose timestamp falls within the
    calendar year, FIFO-matches them into closed lots, returns a DataFrame
    one row per closed lot. Rows can be written to Parquet via
    export_annual_to_parquet.
    """
    start = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC)
    events = journal.query_fills_in_range(start, end)

    fills: list[dict[str, Any]] = []
    for ev in events:
        qty = _fill_quantity(ev.payload)
        price = _fill_price(ev.payload)
        if qty is None or price is None or ev.symbol is None:
            logger.warning(
                "Skipping malformed fill event seq=%d (qty=%s price=%s symbol=%s)",
                ev.seq, qty, price, ev.symbol,
            )
            continue
        fills.append({
            "symbol": ev.symbol,
            "ts": ev.ts,
            "quantity": qty,
            "price": price,
            "intent_id": ev.intent_id,
        })

    closed_lots = match_fifo(fills)
    if not closed_lots:
        return pd.DataFrame(columns=[
            "symbol", "open_ts", "close_ts", "side", "quantity",
            "open_price", "close_price", "open_intent_id", "close_intent_id",
            "realized_pnl_quote_ccy",
        ])

    df = pd.DataFrame([lot.to_dict() for lot in closed_lots])
    return df


def export_annual_to_parquet(
    journal: TradeJournal,
    year: int,
    output_path: str,
) -> int:
    """Run export_annual and write to Parquet. Returns number of closed lots."""
    df = export_annual(journal, year)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Wrote %d closed lots for %d to %s",
        len(df), year, output_path,
    )
    return len(df)
