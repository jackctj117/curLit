"""Alpaca paper options execution (CL-ldd2).

Turns the desk's ADVISORY options ideas (impact/niche agents emit
``buy_calls`` / ``buy_puts`` with a moneyness band + DTE window, not a specific
strike/expiry) into actual PAPER option orders on Alpaca — without disturbing
the Telegram advisory feed, which is a separate path.

The hard part is contract SELECTION: an idea says "slightly OTM calls (~3-5%
above spot), 3-5 weeks to expiry", so we
  1. parse a moneyness fraction + a DTE from the idea's ``preferred_instrument``
     (falling back to configured defaults for vaguer ideas),
  2. compute a target strike (OTM: above spot for calls, below for puts) and a
     target expiry,
  3. query Alpaca's options-contracts endpoint for live contracts near that
     target, and pick the closest listed one (nearest expiry, then strike).

Everything HTTP is injectable so the whole thing is unit-tested without a live
Alpaca account; the live path is a thin httpx shim. Paper by default.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

#: Injectable transport: (method, url, headers, params, json_body) -> parsed JSON.
RequestFn = Callable[
    [str, str, dict[str, str], dict[str, str] | None, dict[str, Any] | None],
    Any,
]

_MONEYNESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# "3-5 weeks", "3 weeks", "21 days", "3-6 week"
_DTE_RE = re.compile(r"(\d+)\s*(?:-\s*(\d+)\s*)?(week|day)s?", re.I)


@dataclass(frozen=True)
class ContractSelectionConfig:
    #: Fallback OTM fraction when the idea text has no explicit "%".
    default_moneyness: float = 0.04
    #: Fallback days-to-expiry when the idea text has no explicit weeks/days.
    default_dte_days: int = 28
    #: How far (± days) around the target expiry to search for a listed contract.
    exp_window_days: int = 10
    #: How far (± fraction) around the target strike to search.
    strike_window: float = 0.10


def parse_moneyness(text: str | None, default: float) -> float:
    m = _MONEYNESS_RE.search(text or "")
    if m:
        try:
            return max(0.0, float(m.group(1)) / 100.0)
        except ValueError:
            return default
    return default


def parse_dte_days(text: str | None, default: int) -> int:
    m = _DTE_RE.search(text or "")
    if not m:
        return default
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    mid = (lo + hi) / 2.0
    return int(round(mid * 7)) if m.group(3).lower().startswith("week") else int(round(mid))


def contract_target(
    idea: dict[str, Any], underlying_price: float, today: date,
    cfg: ContractSelectionConfig,
) -> tuple[str, float, date]:
    """(right, target_strike, target_expiry) for one options idea."""
    right = "call" if str(idea.get("action")) == "buy_calls" else "put"
    pref = str(idea.get("preferred_instrument") or "")
    moneyness = parse_moneyness(pref, cfg.default_moneyness)
    dte = parse_dte_days(pref, cfg.default_dte_days)
    # OTM by convention: calls above spot, puts below spot.
    strike = (underlying_price * (1 + moneyness) if right == "call"
              else underlying_price * (1 - moneyness))
    return right, strike, today + timedelta(days=dte)


class AlpacaOptionsClient:
    """Thin Alpaca Trading-API client scoped to options (paper by default)."""

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
        self._request_fn = request_fn or self._default_request

    @staticmethod
    def _default_request(
        method: str, url: str, headers: dict[str, str],
        params: dict[str, str] | None, json_body: dict[str, Any] | None,
    ) -> Any:
        import httpx  # noqa: PLC0415 — lazy
        resp = httpx.request(method, url, headers=headers, params=params,
                             json=json_body, timeout=20.0)
        resp.raise_for_status()
        return resp.json()

    def _req(
        self, method: str, path: str,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        return self._request_fn(method, f"{self.base}{path}", self._headers,
                                params, json_body)

    def get_account(self) -> dict[str, Any]:
        result: dict[str, Any] = self._req("GET", "/v2/account")
        return result

    def is_market_open(self) -> bool:
        """True iff the market is open now. Options MARKET orders are rejected
        (422) outside regular hours, so the executor gates on this. Fail-safe:
        an unreadable clock returns False (don't submit into an unknown state)."""
        try:
            clock = self._req("GET", "/v2/clock")
            return bool(clock.get("is_open"))
        except Exception:
            logger.warning("alpaca: market clock unavailable", exc_info=True)
            return False

    def find_contracts(
        self, underlying: str, right: str, exp_gte: date, exp_lte: date,
        strike_gte: float, strike_lte: float, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Live option contracts for ``underlying`` within an expiry+strike box."""
        params = {
            "underlying_symbols": underlying.upper(),
            "type": right,
            "expiration_date_gte": exp_gte.isoformat(),
            "expiration_date_lte": exp_lte.isoformat(),
            "strike_price_gte": f"{strike_gte:.2f}",
            "strike_price_lte": f"{strike_lte:.2f}",
            "status": "active",
            "limit": str(limit),
        }
        data = self._req("GET", "/v2/options/contracts", params=params)
        return list(data.get("option_contracts") or [])

    def submit_option_order(
        self, occ_symbol: str, qty: int, side: str = "buy",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit a market option order.

        ``client_order_id`` (review P1, double-buy): the executor passes the
        idea_id here so a crash between fill and DB record cannot re-buy —
        Alpaca enforces client_order_id uniqueness and rejects the duplicate,
        which the executor recovers as already-executed.
        """
        body = {
            "symbol": occ_symbol, "qty": str(int(qty)), "side": side,
            "type": "market", "time_in_force": "day",
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        result: dict[str, Any] = self._req("POST", "/v2/orders", json_body=body)
        return result

    def get_option_ask(self, occ_symbol: str) -> float | None:
        """Latest ask (per-share premium) for an option contract, or None.

        Used to enforce the premium cap before buying. Uses the indicative
        options feed (available without a paid market-data add-on). Fail-soft.
        """
        try:
            data = self._request_fn(
                "GET", f"{DATA_BASE}/v1beta1/options/quotes/latest",
                self._headers, {"symbols": occ_symbol, "feed": "indicative"},
                None,
            )
            quote = (data.get("quotes") or {}).get(occ_symbol) or {}
            ask = quote.get("ap")
            return float(ask) if ask else None
        except Exception:
            logger.warning("alpaca: option quote failed for %s", occ_symbol,
                           exc_info=True)
            return None

    def list_option_positions(self) -> list[dict[str, Any]]:
        positions = self._req("GET", "/v2/positions")
        return [
            p for p in (positions or [])
            if str(p.get("asset_class")) == "us_option"
        ]


def resolve_contract(
    client: AlpacaOptionsClient,
    idea: dict[str, Any],
    underlying_price: float,
    today: date,
    cfg: ContractSelectionConfig | None = None,
) -> dict[str, Any] | None:
    """Find the best-listed Alpaca contract for one options idea, or None.

    Picks the contract nearest the target expiry (then nearest the target
    strike). Fail-soft: any lookup error / no contracts → None.
    """
    cfg = cfg or ContractSelectionConfig()
    if underlying_price <= 0:
        return None
    right, target_strike, target_exp = contract_target(
        idea, underlying_price, today, cfg)
    try:
        contracts = client.find_contracts(
            str(idea.get("ticker")), right,
            exp_gte=target_exp - timedelta(days=cfg.exp_window_days),
            exp_lte=target_exp + timedelta(days=cfg.exp_window_days),
            strike_gte=target_strike * (1 - cfg.strike_window),
            strike_lte=target_strike * (1 + cfg.strike_window),
        )
    except Exception:
        logger.warning("alpaca: contract lookup failed for %s",
                       idea.get("ticker"), exc_info=True)
        return None
    if not contracts:
        return None

    def _dist(c: dict[str, Any]) -> tuple[int, float]:
        raw_exp = c.get("expiration_date")
        raw_strike = c.get("strike_price")
        if raw_exp is None or raw_strike is None:
            return (10**6, 10**6)
        try:
            exp = date.fromisoformat(str(raw_exp))
            strike = float(raw_strike)
        except (TypeError, ValueError):
            return (10**6, 10**6)
        return (abs((exp - target_exp).days), abs(strike - target_strike))

    return min(contracts, key=_dist)
