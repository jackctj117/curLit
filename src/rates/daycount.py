"""Day-count convention calculations for interest rate instruments."""

from datetime import date
from enum import Enum


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"


def year_fraction(d1: date, d2: date, convention: DayCountConvention) -> float:
    """Compute year fraction between two dates per convention."""
    days = (d2 - d1).days

    if convention == DayCountConvention.ACT_360:
        return days / 360.0

    if convention == DayCountConvention.ACT_365:
        return days / 365.0

    if convention == DayCountConvention.ACT_ACT:
        return days / 365.25

    if convention == DayCountConvention.THIRTY_360:
        d1_day = min(d1.day, 30)
        d2_day = min(d2.day, 30) if d1_day < 30 else d2.day
        return (
            (d2.year - d1.year) * 360
            + (d2.month - d1.month) * 30
            + (d2_day - d1_day)
        ) / 360.0

    raise ValueError(f"Unknown convention: {convention}")


OIS_CONVENTIONS: dict[str, DayCountConvention] = {
    "USD": DayCountConvention.ACT_360,
    "EUR": DayCountConvention.ACT_360,
    "GBP": DayCountConvention.ACT_365,
    "JPY": DayCountConvention.ACT_365,
    "CAD": DayCountConvention.ACT_365,
    "CHF": DayCountConvention.ACT_360,
    "AUD": DayCountConvention.ACT_365,
}
