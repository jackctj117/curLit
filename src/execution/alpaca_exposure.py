"""Validated Alpaca position exposure (CL-0deu.1.1).

Only explicit empty positions AND working orders establish a flat account.
A missing response or an unparseable row is unknown exposure, never zero.
Working orders contribute remaining gross quantities. Atomic local reservations
and synchronized account snapshots remain CL-0deu.1; this is not a concurrency fence.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)

# Alpaca's orders endpoint caps a page at 500. A full page does not prove
# completeness; block rather than authorize from an accidentally truncated book.
OPEN_ORDER_LIMIT = 500
_WORKING_STATUSES = frozenset(
    {
        "new",
        "accepted",
        "pending_new",
        "partially_filled",
        "pending_cancel",
        "pending_replace",
        "accepted_for_bidding",
        "stopped",
        "suspended",
        "calculated",
    }
)


class ExposureUnavailableError(ValueError):
    """A position snapshot cannot safely authorize additional exposure."""


def working_order_quantities(payload: object, *, asset_class: str) -> dict[str, Decimal]:
    """Remaining gross quantities, including external and closing orders.

    Closing orders never release capacity before fills. Counting every working
    side conservatively avoids trusting an unverified open/close label. Validate
    the whole account response before filtering; notional/multileg orders need
    explicit quantity/leg support and therefore block instead of disappearing.
    """
    if not isinstance(payload, list) or len(payload) >= OPEN_ORDER_LIMIT:
        raise ExposureUnavailableError("open_orders_incomplete")
    totals: dict[str, Decimal] = {}
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ExposureUnavailableError("invalid_open_order")
        order_id, symbol, kind = row.get("id"), row.get("symbol"), row.get("asset_class")
        if not isinstance(order_id, str) or not order_id or order_id in seen:
            raise ExposureUnavailableError("invalid_open_order_identity")
        seen.add(order_id)
        if not isinstance(symbol, str) or not symbol or symbol != symbol.strip().upper():
            raise ExposureUnavailableError("invalid_open_order_symbol")
        if not isinstance(kind, str) or kind not in {"us_equity", "us_option", "crypto"}:
            raise ExposureUnavailableError("invalid_open_order_asset_class")
        status, side = row.get("status"), row.get("side")
        if (
            not isinstance(status, str)
            or status not in _WORKING_STATUSES
            or not isinstance(side, str)
            or side not in {"buy", "sell"}
        ):
            raise ExposureUnavailableError("invalid_open_order_state")
        if row.get("legs") or row.get("order_class") not in (None, "", "simple"):
            raise ExposureUnavailableError("unsupported_open_order_legs")
        qty = position_quantity(row.get("qty"), whole_contracts=kind == "us_option")
        filled = position_quantity(row.get("filled_qty"), whole_contracts=kind == "us_option")
        if qty <= 0 or filled < 0 or filled > qty:
            raise ExposureUnavailableError("invalid_open_order_quantity")
        if kind == "us_option":
            root = option_underlying(symbol)
            if any(char.isdigit() for char in root):
                raise ExposureUnavailableError("adjusted_option_underlying_unavailable")
        if kind == asset_class:
            totals[symbol] = totals.get(symbol, Decimal(0)) + qty - filled
    assert all(qty >= 0 for qty in totals.values())
    return totals


def pending_exposure(client: Any, *, asset_class: str) -> dict[str, Decimal]:
    """Require an explicit complete working-order snapshot before positions."""
    try:
        logger.info("Reading broker working-order exposure before entry")
        return working_order_quantities(client.list_open_orders(), asset_class=asset_class)
    except ExposureUnavailableError:
        raise
    except Exception as exc:
        raise ExposureUnavailableError("open_order_lookup_failed") from exc


# OCC's fixed suffix is expiry YYMMDD, call/put, and eight strike digits.
# The root can contain digits for adjusted contracts; don't match by prefix.
_OPTION_SYMBOL = re.compile(r"([A-Z][A-Z0-9.]*?)[0-9]{6}[CP][0-9]{8}")


def option_underlying(symbol: str) -> str:
    """Extract the exact OCC root or reject an unidentifiable contract."""
    match = _OPTION_SYMBOL.fullmatch(symbol)
    if match is None:
        raise ExposureUnavailableError("invalid_option_symbol")
    root = match.group(1)
    assert root
    return root


def position_quantity(value: object, *, whole_contracts: bool = False) -> Decimal:
    """Read signed broker quantities without truncating fractional holdings."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ExposureUnavailableError("invalid_position_quantity")
    try:
        quantity = Decimal(str(value))
    except InvalidOperation as exc:
        raise ExposureUnavailableError("invalid_position_quantity") from exc
    if not quantity.is_finite():
        raise ExposureUnavailableError("nonfinite_position_quantity")
    if whole_contracts and quantity != quantity.to_integral_value():
        raise ExposureUnavailableError("fractional_option_quantity")
    assert quantity.is_finite()
    return quantity


def validated_positions(payload: object, *, asset_class: str) -> list[dict[str, Any]]:
    """Validate the entire broker response BEFORE filtering an asset class.

    Any is confined to this JSON boundary: callers also need the broker's
    untouched price/P&L fields for exit management. Missing identity fields
    cannot be silently filtered away, as that could hide held exposure.
    """
    if not isinstance(payload, list):
        raise ExposureUnavailableError("positions_response_not_list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ExposureUnavailableError("position_row_not_object")
        symbol = row.get("symbol")
        kind = row.get("asset_class")
        if not isinstance(symbol, str) or not symbol or symbol != symbol.strip().upper():
            raise ExposureUnavailableError("invalid_position_symbol")
        # These are the asset classes used by Alpaca's shared account. A new
        # or misspelled class requires explicit handling, not silent omission.
        if kind not in ("us_equity", "us_option", "crypto"):
            raise ExposureUnavailableError("invalid_position_asset_class")
        if symbol in seen:
            raise ExposureUnavailableError("duplicate_position_symbol")
        seen.add(symbol)
        position_quantity(row.get("qty"), whole_contracts=kind == "us_option")
        if kind == "us_option":
            option_underlying(symbol)
        if kind == asset_class:
            result.append(row)
    return result


def require_positive_int(name: str, value: object, *, allow_zero: bool = False) -> None:
    """Reject unsafe direct config values, including bools masquerading as ints."""
    minimum = 0 if allow_zero else 1  # Zero caps deliberately disable entries.
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
