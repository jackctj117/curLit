"""Alpaca paper EQUITY (shares) execution client (CL-ncbq).

The share-book twin of :mod:`src.execution.alpaca_options`. Same account, same
Trading API, same ``/v2/orders`` + ``/v2/positions`` endpoints — only the asset
class differs, so this client deliberately reuses the options client's HTTP
plumbing (``RequestFn`` transport shim, base URLs, non-2xx body logging)
instead of forking it.

WHY a share book at all (CL-4c7o): the desk's advisory equity ideas were right
on the UNDERLYING 80% of the time (41/51), while the short-dated OTM options
expressing them won 7% — the spread and theta consumed the 1-2% moves the
theses actually produced. Shares express the identical directional call with a
penny spread and no time decay, so this client needs NONE of the options
client's hard parts:

  * no contract SELECTION (no moneyness/DTE parse, no strike search) — the
    ticker IS the instrument;
  * no mid/spread machinery — an equity fill IS the honest basis.

What it does need beyond the options client is a stock quote endpoint (the
options feed is a different path) and ``get_order`` so the executor can read
back the real fill price rather than assume one.

Everything HTTP goes through the injectable ``RequestFn``, so the whole thing
is unit-tested without a live Alpaca account. Paper by default.
"""

from __future__ import annotations

import logging
from typing import Any

from src.execution.alpaca_exposure import validated_positions
from src.execution.alpaca_options import (
    DATA_BASE,
    LIVE_BASE,
    PAPER_BASE,
    RequestFn,
)

logger = logging.getLogger(__name__)


class AlpacaEquityClient:
    """Thin Alpaca Trading-API client scoped to US equities (paper by default).

    ``side`` on the wire is only ever ``buy`` or ``sell``: Alpaca has no
    "short" side — a ``sell`` on an account holding no shares OPENS a short,
    and a ``buy`` against a short position covers it. The executor's
    ``sell_short`` bookkeeping value is translated at the call site, not here.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        request_fn: RequestFn | None = None,
    ) -> None:
        self.base = PAPER_BASE if paper else LIVE_BASE
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        # Reuse the options client's transport (httpx + non-2xx body logging,
        # CL-hptt) so a rejection surfaces its real reason on both books.
        if request_fn is None:
            from src.execution.alpaca_options import AlpacaOptionsClient  # noqa: PLC0415

            request_fn = AlpacaOptionsClient._default_request
        self._request_fn = request_fn

    def _req(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        return self._request_fn(method, f"{self.base}{path}", self._headers, params, json_body)

    def get_account(self) -> dict[str, Any]:
        result: dict[str, Any] = self._req("GET", "/v2/account")
        return result

    def is_market_open(self) -> bool:
        """True iff the market is open now.

        Equity MARKET orders outside regular hours are rejected (Alpaca only
        accepts LIMIT + extended_hours), so the executor gates on this.
        Fail-safe: an unreadable clock returns False — never submit into an
        unknown state.
        """
        try:
            clock = self._req("GET", "/v2/clock")
            return bool(clock.get("is_open"))
        except Exception:
            logger.warning("alpaca equity: market clock unavailable", exc_info=True)
            return False

    def submit_equity_order(
        self,
        symbol: str,
        qty: int,
        side: str = "buy",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit a market order for whole shares.

        ``side`` must be Alpaca's own ``buy``/``sell``. Shorting is a plain
        ``sell`` with no position; covering is a plain ``buy``.

        ``client_order_id`` carries the idea id (same crash-safety trick as
        the options path): a crash between fill and DB record cannot double
        the position, because Alpaca enforces client_order_id uniqueness and
        422-rejects the resubmit, which the executor recovers as
        already-executed.
        """
        body = {
            "symbol": symbol.upper(),
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        result: dict[str, Any] = self._req("POST", "/v2/orders", json_body=body)
        return result

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """One order by Alpaca id, or None.

        Used to read back the REAL fill (``filled_avg_price``) rather than
        assume the last price was the fill, and to confirm an exit filled when
        the position list has not caught up yet. Fail-soft: any problem → None
        (the caller falls back to the price it already has).
        """
        try:
            result: dict[str, Any] = self._req("GET", f"/v2/orders/{order_id}")
            return result
        except Exception:
            logger.warning("alpaca equity: order lookup failed for %s", order_id, exc_info=True)
            return None

    def get_stock_quote(self, symbol: str) -> tuple[float | None, float | None]:
        """(bid, ask) for a stock from the market-data API, or (None, None).

        Separate host from the trading API (``data.alpaca.markets``), same
        auth headers. Unlike the options path this is NOT used to police a
        spread — liquid US equities quote in pennies — only to mark a position
        at the mid, which is a fairer read than a single-sided last trade.
        Fail-soft: (None, None) on any problem; the caller falls back to
        Alpaca's own position mark.
        """
        sym = symbol.upper()
        try:
            data = self._request_fn(
                "GET",
                f"{DATA_BASE}/v2/stocks/{sym}/quotes/latest",
                self._headers,
                None,
                None,
            )
            quote = (data or {}).get("quote")
            if not quote:
                return (None, None)
            bid, ask = quote.get("bp"), quote.get("ap")
            return (
                float(bid) if bid is not None else None,
                float(ask) if ask is not None else None,
            )
        except Exception:
            logger.warning("alpaca equity: stock quote failed for %s", sym, exc_info=True)
            return (None, None)

    def list_equity_positions(self) -> list[dict[str, Any]]:
        """Open US-equity positions only.

        The account is shared with the options book, so filtering on
        ``asset_class`` is what keeps the two books from managing each
        other's positions.
        """
        logger.info("alpaca equity: fetching position snapshot")
        positions = self._req("GET", "/v2/positions")
        return validated_positions(positions, asset_class="us_equity")
