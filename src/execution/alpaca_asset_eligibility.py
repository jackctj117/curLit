"""Broker-specific equity entry eligibility, separate from research identity.

Alpaca Asset fields: https://alpaca.markets/sdks/python/api_reference/trading/models.html
No symbol remapping, inferred booleans, or hard-to-borrow locate workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class AssetNotFoundError(RuntimeError):
    """Only an HTTP 404 from the exact asset lookup; never a submission error."""


class EligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    UNSUPPORTED = "unsupported_asset"
    INACTIVE = "inactive_asset"
    NOT_TRADABLE = "not_tradable"
    SHORT_NOT_SUPPORTED = "short_not_supported"
    BORROW_UNMET = "borrow_requirements_unmet"
    UNAVAILABLE = "eligibility_unavailable"


@dataclass(frozen=True)
class AssetEligibility:
    status: EligibilityStatus
    symbol: str
    checked_at: datetime
    asset_id: str | None = None
    exchange: str | None = None
    reason: str = ""

    @property
    def eligible(self) -> bool:
        return self.status == EligibilityStatus.ELIGIBLE


def assess_asset(asset: Any, symbol: str, *, short: bool, now: datetime) -> AssetEligibility:
    """Treat incomplete broker metadata as unknown, not a successful preflight."""
    assert now.tzinfo is not None
    if not isinstance(asset, dict):
        return AssetEligibility(
            EligibilityStatus.UNAVAILABLE, symbol, now, reason="invalid_metadata"
        )
    if (
        not isinstance(asset.get("id"), str)
        or not asset["id"]
        or asset.get("symbol") != symbol.upper()
        or not isinstance(asset.get("exchange"), str)
        or not asset["exchange"]
        or asset.get("status") not in {"active", "inactive"}
        or not isinstance(asset.get("tradable"), bool)
    ):
        return AssetEligibility(
            EligibilityStatus.UNAVAILABLE, symbol, now, reason="incomplete_or_mismatched_metadata"
        )
    result = AssetEligibility(
        EligibilityStatus.ELIGIBLE, symbol, now, asset["id"], asset["exchange"]
    )
    status = EligibilityStatus.ELIGIBLE
    if asset.get("class") != "us_equity":
        status = EligibilityStatus.UNAVAILABLE
    elif asset["status"] != "active":
        status = EligibilityStatus.INACTIVE
    elif not asset["tradable"]:
        status = EligibilityStatus.NOT_TRADABLE
    elif short:
        if not isinstance(asset.get("shortable"), bool) or not isinstance(
            asset.get("easy_to_borrow"), bool
        ):
            status = EligibilityStatus.UNAVAILABLE
        elif not asset["shortable"]:
            status = EligibilityStatus.SHORT_NOT_SUPPORTED
        elif not asset["easy_to_borrow"]:
            status = EligibilityStatus.BORROW_UNMET
    return AssetEligibility(status, symbol, now, result.asset_id, result.exchange, status.value)
