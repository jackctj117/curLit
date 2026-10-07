"""Paper broker — simulated execution with cost model, full P&L tracking.

Equity is MARKED TO MARKET (CL-9ird): ``get_account().equity`` is cash
(initial capital + realized P&L - costs) plus the unrealized P&L of every
open position, marked at the latest ``set_price`` quote — the same quote
book ``place_order`` / ``get_price`` / ``stream_prices`` use, so there is
one authoritative price source. Before this, reported equity moved only on
realized deltas, so the drawdown / daily-loss kill switches could not see
an open position's losses on the paper venue.

A mark that cannot be computed honestly (no price, a non-finite price, or
a P&L currency that is not the account currency) — or any realized P&L /
cost booked in a non-account currency, even once flat — makes ``get_account``
RAISE ``AccountMarkUnavailableError`` — never a silent zero. The live-engine
health tick already counts account-read failures and halts after three
(fail closed), the same path a broker outage takes.
"""

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from .broker import Account, Broker, Order, OrderStatus, Position, currency_pair

logger = logging.getLogger(__name__)

#: Half-spread applied around the set_price mid by stream_prices, in basis
#: points (CL-8lv6): a deterministic ±0.5 bp synthetic book.
STREAM_HALF_SPREAD_BPS = 0.5

#: Home currency of the simulated paper account. USD matches the OANDA
#: practice account the paper venue stands in for.
DEFAULT_ACCOUNT_CURRENCY = "USD"


class AccountMarkUnavailableError(RuntimeError):
    """An open position cannot be marked in account currency (CL-9ird).

    Raised by ``PaperBroker.get_account`` instead of reporting an equity
    that silently omits the position: excluding an unmarkable position is
    the same as marking its P&L at zero, which hides losses from the
    drawdown / daily-loss kill switches.
    """


class PaperBroker(Broker):
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        stream_interval_sec: float = 1.0,
        account_currency: str = DEFAULT_ACCOUNT_CURRENCY,
    ) -> None:
        assert len(account_currency) == 3 and account_currency.isalpha(), (
            f"account_currency must be an ISO-4217 code, got {account_currency!r}"
        )
        self._capital = initial_capital
        self._stream_interval_sec = stream_interval_sec
        self._account_currency = account_currency.upper()
        # Cash = initial capital + realized P&L - costs (CL-9ird). Equity is
        # cash + the unrealized mark, computed on read — never stored, so it
        # cannot drift from the open book.
        self._cash = initial_capital
        self._realized_pnl = 0.0
        # Realized P&L + costs from fills whose P&L currency is NOT the
        # account currency, per currency, awaiting conversion (CL-vfw7).
        # Non-empty ⇒ account equity is UNKNOWN (get_account raises).
        self._unconverted: dict[str, float] = {}
        self._positions: dict[str, Position] = {}
        self._trade_log: list[dict[str, Any]] = []
        self._prices: dict[str, tuple[float, float]] = {}

    # -- Broker ABC ----------------------------------------------------

    def place_order(self, order: Order) -> Order:
        quote = self._prices.get(order.symbol)
        if quote is None:
            # CL-n3pt (P1): fail CLOSED, like get_price/stream. A symbol with
            # no set_price must NOT fill at a fabricated ~1.10 book — metals
            # and commodities trade in the 100s-1000s, so a 1.10 fill produces
            # nonsense fills, PnL and stops in a soak. REJECT so it flows
            # through the OMS's BrokerRejectedOrderError path exactly like a
            # venue reject, rather than silently filling at a made-up price.
            order.status = OrderStatus.REJECTED
            order.reject_reason = (
                f"NO_PRICE: PaperBroker has no price for {order.symbol!r} (set_price never called)"
            )
            logger.warning(
                "PaperBroker REJECTED %s %s x%s — no price set",
                order.symbol,
                order.side,
                order.quantity,
            )
            return order
        bid, ask = quote
        fill_price = ask if order.side == "buy" else bid

        # Slippage enforcement (CL-qyav) — simulated equivalent of OANDA's
        # FOK priceBound: reference is the current mid; a fill beyond
        # mid*(1±bps/1e4) is REJECTED (buy bound above, sell bound below).
        # The OMS raises REJECTED through BrokerRejectedOrderError, same as
        # a venue reject.
        if order.max_slippage_bps is not None and order.max_slippage_bps > 0:
            mid = (bid + ask) / 2.0
            frac = order.max_slippage_bps / 10_000.0
            bound = mid * (1 + frac) if order.side == "buy" else mid * (1 - frac)
            beyond = fill_price > bound if order.side == "buy" else fill_price < bound
            if beyond:
                order.status = OrderStatus.REJECTED
                order.reject_reason = (
                    f"SLIPPAGE_EXCEEDED: {order.side} fill {fill_price:.6f} "
                    f"beyond bound {bound:.6f} (mid {mid:.6f}, "
                    f"max {order.max_slippage_bps} bps)"
                )
                logger.warning(
                    "PaperBroker REJECTED %s %s x%s: %s",
                    order.symbol,
                    order.side,
                    order.quantity,
                    order.reject_reason,
                )
                return order

        cost = abs(order.quantity) * fill_price * 0.0001  # 1 bp round-trip

        current = self._positions.get(order.symbol)
        old_qty = current.quantity if current else 0.0
        old_avg = current.avg_price if current else 0.0
        realized = current.realized_pnl if current else 0.0
        delta = order.quantity if order.side == "buy" else -order.quantity
        new_qty = old_qty + delta

        # Average-entry accounting (CL-qyav P2 fix): the old formula blended
        # the fill into the basis on EVERY order, so a reduce corrupted
        # avg_price instead of realizing P&L, and a flip inherited a
        # nonsensical basis. Rules:
        #   open / add same-direction → volume-weighted average basis
        #   reduce / full close      → basis UNCHANGED, P&L realized on the
        #                              closed quantity
        #   flip through zero        → realize P&L on the whole old position,
        #                              basis resets to the fill price for the
        #                              residual quantity
        direction = 1.0 if old_qty > 0 else -1.0
        if old_qty == 0.0:
            avg_price = fill_price
        elif (old_qty > 0) == (delta > 0):
            # Adding to an existing position (delta == 0 degenerates to
            # one of the branches below with a no-op result).
            avg_price = (abs(old_qty) * old_avg + abs(delta) * fill_price) / abs(new_qty)
        elif abs(delta) <= abs(old_qty):
            # Partial reduce or full close: realize on the closed quantity.
            realized += abs(delta) * (fill_price - old_avg) * direction
            avg_price = old_avg
        else:
            # Flip through zero: close the whole old position, residual
            # opens at the fill price.
            realized += abs(old_qty) * (fill_price - old_avg) * direction
            avg_price = fill_price

        realized_delta = realized - (current.realized_pnl if current else 0.0)
        self._positions[order.symbol] = Position(
            symbol=order.symbol,
            quantity=new_qty,
            avg_price=avg_price,
            realized_pnl=realized,
        )
        # realized_delta and cost are denominated in the pair's QUOTE
        # currency. Only account-currency amounts may enter cash / realized
        # P&L (CL-9ird); anything else is held per currency until a
        # converter exists (TODO CL-vfw7) and makes equity UNKNOWN — booking
        # 1,000 JPY as 1,000 USD would misstate a realized result.
        pnl_ccy = self._pnl_currency(order.symbol)
        if pnl_ccy == self._account_currency:
            self._cash += realized_delta - cost
            self._realized_pnl += realized_delta
        else:
            key = pnl_ccy or f"unknown:{order.symbol}"
            self._unconverted[key] = self._unconverted.get(key, 0.0) + realized_delta - cost
            logger.warning(
                "PaperBroker: %s fill booked %.6f %s unconverted (account %s, CL-vfw7) — "
                "account equity is UNKNOWN until converted",
                order.symbol,
                realized_delta - cost,
                key,
                self._account_currency,
            )
        order.status = OrderStatus.FILLED
        # CL-pksi: report the executed quantity (the simulator fills fully).
        order.filled_quantity = float(order.quantity)
        self._trade_log.append(
            {
                "ts": datetime.now(UTC),
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "fill_price": fill_price,
                "cost": cost,
            }
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_order(self, order_id: str) -> Order:
        raise NotImplementedError

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> Account:
        """Account snapshot with equity MARKED TO MARKET (CL-9ird).

        ``equity = cash + unrealized``. ``realized_pnl`` and
        ``unrealized_pnl`` are reported separately so the mark (an
        estimate) is never presented as a realized result. Raises
        ``AccountMarkUnavailableError`` when any open position cannot be
        marked — equity is then UNKNOWN, not "cash only".
        """
        if self._unconverted:
            msg = (
                f"realized P&L/costs held in non-account currencies "
                f"{dict(self._unconverted)} cannot be converted to "
                f"{self._account_currency} (no converter wired, CL-vfw7) — equity UNKNOWN"
            )
            logger.error("PaperBroker: %s", msg)
            raise AccountMarkUnavailableError(msg)
        unrealized = self._unrealized_pnl()
        equity = self._cash + unrealized
        assert math.isfinite(equity), f"non-finite paper equity {equity}"
        logger.debug(
            "PaperBroker account: cash=%.2f realized=%.2f unrealized=%.2f equity=%.2f %s",
            self._cash,
            self._realized_pnl,
            unrealized,
            equity,
            self._account_currency,
        )
        return Account(
            balance=self._capital,
            equity=equity,
            margin_used=abs(sum(p.quantity * p.avg_price for p in self._positions.values())) * 0.02,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized,
        )

    @staticmethod
    def _pnl_currency(symbol: str) -> str | None:
        """Currency a position's P&L is denominated in: the pair's QUOTE
        currency, or None when the symbol is not a currency pair (e.g. an
        index CFD) and the currency cannot be determined."""
        pair = currency_pair(symbol)
        return pair[1] if pair is not None else None

    def _unrealized_pnl(self) -> float:
        """Open-position marks summed in ACCOUNT currency; raises if unknown.

        Each open position is marked at the EXIT side of its latest
        ``set_price`` quote (bid for a long, ask for a short) — what closing
        it now would realize, the convention OANDA uses for unrealizedPL.
        Flat positions (qty 0, retained for their realized history) carry
        no exposure and need no price.
        """
        total = 0.0
        for pos in self._positions.values():
            if pos.quantity == 0.0:
                continue
            quote = self._prices.get(pos.symbol)
            if quote is None:
                msg = (
                    f"cannot mark open position {pos.symbol} x{pos.quantity}: "
                    f"no price available — equity UNKNOWN"
                )
                logger.error("PaperBroker: %s", msg)
                raise AccountMarkUnavailableError(msg)
            bid, ask = quote
            mark = bid if pos.quantity > 0 else ask
            if not math.isfinite(mark) or mark <= 0.0:
                msg = (
                    f"cannot mark open position {pos.symbol} x{pos.quantity}: "
                    f"invalid exit-side price {mark!r} — equity UNKNOWN"
                )
                logger.error("PaperBroker: %s", msg)
                raise AccountMarkUnavailableError(msg)
            pnl_ccy = self._pnl_currency(pos.symbol)
            if pnl_ccy != self._account_currency:
                # TODO(CL-vfw7): convert quote-currency P&L to the account
                # currency via the shared AccountCurrencyConverter
                # (src/risk/currency.py) once it lands. Until then the mark is
                # UNKNOWN: adding JPY (or index points) to a USD equity is
                # wrong by orders of magnitude, and dropping it hides losses.
                ccy = pnl_ccy or "unknown"
                msg = (
                    f"cannot mark open position {pos.symbol} x{pos.quantity}: "
                    f"P&L currency {ccy} is not the account currency "
                    f"{self._account_currency} and no converter is wired "
                    f"(CL-vfw7) — equity UNKNOWN"
                )
                logger.error("PaperBroker: %s", msg)
                raise AccountMarkUnavailableError(msg)
            pnl = pos.quantity * (mark - pos.avg_price)
            logger.debug(
                "PaperBroker mark %s qty=%.4f avg=%.6f mark=%.6f unrealized=%.2f",
                pos.symbol,
                pos.quantity,
                pos.avg_price,
                mark,
                pnl,
            )
            total += pnl
        return total

    def get_price(self, symbol: str) -> tuple[float, float]:
        """Bid/ask for a configured symbol. RAISES on unknown symbols
        (ultrareview follow-up): the old silent (1.1000, 1.1002) default was
        the same fabricated-price fail-open as the coordinator's mid=1.0 —
        it priced gold/indices as if they were EURUSD in paper soaks and
        masked missing set_price wiring. Callers that can tolerate a missing
        price (coordinator._get_price) catch and fail closed."""
        try:
            return self._prices[symbol]
        except KeyError:
            msg = (
                f"PaperBroker has no price for {symbol!r} — call "
                f"set_price('{symbol}', bid, ask) first (known: "
                f"{sorted(self._prices)})"
            )
            raise KeyError(msg) from None

    async def stream_prices(
        self,
        symbols: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
        """Synthetic ticks for PRICED symbols only — fail closed (CL-8lv6 P0).

        The old implementation fabricated bid=1.1000/ask=1.1002 for EVERY
        requested symbol, ignoring set_price: the live engine's paper mode
        fills _last_prices from this stream, so all instruments were marked
        ~1.10 and paper-soak sizing/stops/PnL were meaningless. Same policy
        as get_price now: a symbol without a set_price value is ABSENT from
        the stream — never fabricated. Priced symbols tick around the
        CURRENT set_price mid with a deterministic ±STREAM_HALF_SPREAD_BPS
        synthetic spread, so set_price updates are reflected on the next
        pass. Tick shape matches OandaBroker.stream_prices:
        {symbol, bid, ask, ts}.
        """
        while True:
            for sym in symbols:
                quote = self._prices.get(sym)
                if quote is None:
                    continue  # fail closed: no set_price → no tick
                bid, ask = quote
                mid = (bid + ask) / 2.0
                half = mid * (STREAM_HALF_SPREAD_BPS / 10_000.0)
                yield {
                    "symbol": sym,
                    "bid": mid - half,
                    "ask": mid + half,
                    "ts": datetime.now(UTC).isoformat(),
                }
            await asyncio.sleep(self._stream_interval_sec)

    # -- Test helpers --------------------------------------------------

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        self._prices[symbol] = (bid, ask)

    @property
    def equity(self) -> float:
        """Marked equity (CL-9ird) — same value and failure mode as
        ``get_account().equity``."""
        return self.get_account().equity

    @property
    def cash(self) -> float:
        """Initial capital + realized P&L - costs (no unrealized mark)."""
        return self._cash
