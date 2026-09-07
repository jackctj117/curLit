"""Validated Alpaca position exposure (CL-0deu.1.1).

Only an explicit empty position list establishes a flat account. A missing
response or an unparseable row is unknown exposure, never zero. This module
does not yet reconcile working orders or reserve capacity across workers;
those controls remain in CL-0deu.1.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


class ExposureUnavailableError(ValueError):
    """A position snapshot cannot safely authorize additional exposure."""


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
