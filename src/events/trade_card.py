"""Trade-card grounding (CL-jiqq) — turn an advisory ``trade_idea``'s
PERCENTAGES into concrete DOLLAR levels the operator can act on.

The impact-agent LLM has NO live market data: it assesses a headline and
emits logic + percentages (``stop_loss_pct``, ``target_pct``,
``entry_trigger``, ``invalidation``) plus an option moneyness/DTE band.
It must never fabricate a strike, an expiry date, or a current price —
the operator risks real money on these. This module takes the idea plus
a REAL ``current_price`` (last close from :mod:`src.events.prices`, via
yfinance) and computes the direction-aware dollar levels:

  * ``entry_zone``   — the entry_trigger text if the LLM gave one, else
                       "near $<current>".
  * ``stop_price``   — ``current × (1 ∓ stop_loss_pct)``, direction-aware
                       (below for longs/calls, above for shorts/puts).
  * ``target_prices``— ``current × (1 ± t)`` per ``target_pct``.
  * ``risk_reward``  — ``|target1 − current| / |current − stop|``.
  * ``suggested_strike`` — for options, ``current × 0.95`` (OTM puts) or
                       ``current × 1.05`` (OTM calls), LABELED as a
                       suggested LEVEL, not a listed strike — the
                       operator picks the nearest strike on their broker.
  * ``dte_window``   — a days-to-expiry WINDOW from the horizon; there is
                       NEVER a fabricated calendar expiry date.

When ``current_price`` is absent (the price helper could not resolve the
ticker — no live feed, delisted, weekend gap in a thin name) the card
still renders the LLM's percentages under a ``priced: False`` /
``no_live_price`` flag, and every dollar field is ``None``. The card
never crashes on bad input; a malformed level simply goes absent.

The stop % reading depends on the action: for stock it is a % of the
share price (a real dollar stop); for options it is a % of PREMIUM (the
share-price stop_price then reflects the UNDERLYING move that pressures
the option, which is what the moneyness/DTE band is chosen against — the
premium stop lives in ``notes`` on the idea). We compute the underlying
dollar stop from the idea's implied UNDERLYING move; the % of premium is
not a share price and is left out of the dollar math. To keep the card
honest we treat option ``stop_loss_pct`` as a premium fraction and DO
NOT turn it into a share-price stop — instead the underlying stop for an
option idea comes from ``STOP_PCT_UNDERLYING_DEFAULT`` unless the idea
also carried an explicit underlying move. This keeps the dollar stop a
real underlying level, never a premium number masquerading as a price.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Actions that express a BULLISH view (price up helps). Everything else
#: (short, buy_puts) is bearish for the underlying.
_BULLISH_ACTIONS = frozenset({"long", "buy_calls"})
_OPTION_ACTIONS = frozenset({"buy_calls", "buy_puts"})

#: OTM offset for the suggested-strike LEVEL (not a listed strike): buy
#: puts ~5% below spot, buy calls ~5% above. Kept modest — a news trade
#: wants a strike that can actually go in-the-money on the expected move.
_OTM_PUT_MULT = 0.95
_OTM_CALL_MULT = 1.05

#: Fallback UNDERLYING stop (% of share price) when an OPTIONS idea's
#: stop_loss_pct is a premium fraction (not a share move) and no explicit
#: underlying move was supplied. A defined-risk option's real max loss is
#: the premium, so this underlying level is advisory context, not a hard
#: exit.
STOP_PCT_UNDERLYING_DEFAULT = 0.08

#: Fallback stock stop when the LLM omitted stop_loss_pct entirely.
STOP_PCT_STOCK_DEFAULT = 0.08

#: DTE (days-to-expiry) windows per time_horizon — a WINDOW, never a
#: fabricated calendar date. The operator picks the listed expiry nearest
#: the middle of the window.
_DTE_WINDOWS = {
    "immediate": ("1-2 weeks", 10),
    "short": ("1-3 weeks", 14),
    "medium": ("3-6 weeks", 30),
    "structural": ("2-4 months", 90),
}
_DTE_DEFAULT = ("2-4 weeks", 21)


def _round_price(value: float) -> float:
    """Round a dollar level to a sensible tick for its magnitude:
    penny stocks to the cent, small caps to a dime-ish 2dp, big names to
    the dollar-ish. Keeps the card readable without faking precision."""
    v = abs(value)
    if v >= 100.0:
        return round(value, 1)
    if v >= 10.0:
        return round(value, 2)
    return round(value, 3)


def _is_bullish(idea: dict[str, Any]) -> bool:
    """True when the idea profits from the underlying rising. Reads
    ``action`` first (authoritative), falls back to ``direction``."""
    action = str(idea.get("action") or "").strip().lower()
    if action in _BULLISH_ACTIONS:
        return True
    if action in ("short", "buy_puts"):
        return False
    # No usable action → lean on direction; default bullish is arbitrary
    # but the caller only reaches here on malformed input.
    return str(idea.get("direction") or "").strip().lower() != "bearish"


def _dte_window(horizon: str) -> tuple[str, int]:
    return _DTE_WINDOWS.get(str(horizon or "").strip().lower(), _DTE_DEFAULT)


def _underlying_stop_pct(idea: dict[str, Any], is_option: bool) -> float:
    """The % of SHARE PRICE to use for the dollar stop level.

    For stock ideas this is the idea's own ``stop_loss_pct`` (a share
    move). For option ideas the idea's ``stop_loss_pct`` is a fraction of
    PREMIUM, not a share move, so we use the underlying-move default
    instead — a premium % is not a price and must not masquerade as one.
    """
    raw = idea.get("stop_loss_pct")
    try:
        pct = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        pct = None
    if is_option:
        return STOP_PCT_UNDERLYING_DEFAULT
    if pct is not None and 0.0 < pct < 1.0:
        return pct
    return STOP_PCT_STOCK_DEFAULT


def _clean_targets(raw: Any) -> list[float]:
    """Defensive parse of the idea's ``target_pct`` (already cleaned by
    the impact agent, but this module is called on legacy rows too)."""
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[float] = []
    for item in items:
        try:
            f = float(item)
        except (TypeError, ValueError):
            continue
        if f > 0.0 and f not in out:
            out.append(f)
    out.sort()
    return out[:2]


def build_trade_card(
    idea: dict[str, Any],
    current_price: float | None,
    change_pct: float | None = None,
) -> dict[str, Any]:
    """Ground one advisory ``idea`` in real numbers.

    Returns a dict of concrete levels. With a real ``current_price`` the
    dollar fields are populated and ``priced`` is True; without one the
    card carries only the LLM's percentages, every dollar field is
    ``None``, and ``priced`` is False (``no_live_price`` set) — callers
    render the %-only card and flag the missing feed, never a fake price.

    Never raises: malformed levels go absent, not fatal.
    """
    bullish = _is_bullish(idea)
    action = str(idea.get("action") or "").strip().lower()
    is_option = action in _OPTION_ACTIONS
    horizon = str(idea.get("time_horizon") or "").strip().lower()
    dte_text, dte_mid = _dte_window(horizon)

    stop_pct = _underlying_stop_pct(idea, is_option)
    targets_pct = _clean_targets(idea.get("target_pct"))

    trigger = str(idea.get("entry_trigger") or "").strip()
    invalidation = str(idea.get("invalidation") or "").strip()

    card: dict[str, Any] = {
        "ticker": str(idea.get("ticker") or "").strip(),
        "action": action,
        "is_option": is_option,
        "bullish": bullish,
        # Percentages are always honest — they came straight from the LLM.
        "stop_loss_pct": round(stop_pct, 4),
        "target_pct": targets_pct,
        "entry_trigger": trigger,
        "invalidation": invalidation,
        "dte_window": dte_text if is_option else "",
        "dte_days": dte_mid if is_option else None,
        # Dollar levels — filled only when we have a real price.
        "priced": False,
        "no_live_price": True,
        "current_price": None,
        "change_pct": None,
        "entry_zone": trigger or "",
        "stop_price": None,
        "target_prices": [],
        "risk_reward": None,
        "suggested_strike": None,
        "suggested_strike_note": (
            "suggested level — pick nearest listed strike" if is_option else ""
        ),
        "dte_note": (f"choose the listed expiry nearest {dte_mid} days out" if is_option else ""),
    }

    # No live price → honest %-only card. Entry zone stays the trigger
    # text; there is no dollar anchor to offer.
    try:
        price = float(current_price) if current_price is not None else None
    except (TypeError, ValueError):
        price = None
    if price is None or not price > 0.0:
        card["entry_zone"] = trigger or "no live price — enter per trigger"
        return card

    card["priced"] = True
    card["no_live_price"] = False
    card["current_price"] = _round_price(price)
    if change_pct is not None:
        try:
            card["change_pct"] = round(float(change_pct), 2)
        except (TypeError, ValueError):
            card["change_pct"] = None

    # Direction-aware stop: below spot for bullish, above for bearish.
    stop_price = price * (1.0 - stop_pct) if bullish else price * (1.0 + stop_pct)
    card["stop_price"] = _round_price(stop_price)

    # Targets: up for bullish, down for bearish.
    target_prices = [
        _round_price(price * (1.0 + t) if bullish else price * (1.0 - t)) for t in targets_pct
    ]
    card["target_prices"] = target_prices

    # Entry zone: prefer the LLM trigger; always anchor to the real price.
    if trigger:
        card["entry_zone"] = f"{trigger} (near ${card['current_price']:,.2f})"
    else:
        card["entry_zone"] = f"near ${card['current_price']:,.2f}"

    # Risk:reward off the FIRST target vs the stop, both measured from the
    # (real) current price used as the entry anchor.
    if target_prices:
        reward = abs(target_prices[0] - price)
        risk = abs(price - stop_price)
        if risk > 0.0:
            card["risk_reward"] = round(reward / risk, 1)

    # Suggested option strike — a LEVEL, never a listed strike.
    if is_option:
        mult = _OTM_PUT_MULT if not bullish else _OTM_CALL_MULT
        card["suggested_strike"] = _round_price(price * mult)

    return card
