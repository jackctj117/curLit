"""Business-day calendar with holiday-aware date adjustment."""

from datetime import date, timedelta
from enum import Enum

import holidays


class BusinessDayConvention(Enum):
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"


class Calendar:
    """Holiday-aware calendar for a single country."""

    def __init__(self, country: str) -> None:
        self.country = country
        self._holidays = holidays.country_holidays(country)

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._holidays

    def adjust(
        self,
        d: date,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    ) -> date:
        if self.is_business_day(d):
            return d

        if convention == BusinessDayConvention.FOLLOWING:
            while not self.is_business_day(d):
                d += timedelta(days=1)
            return d

        if convention == BusinessDayConvention.PRECEDING:
            while not self.is_business_day(d):
                d -= timedelta(days=1)
            return d

        # Modified following: roll forward, fallback to preceding if month changes.
        adj = d
        while not self.is_business_day(adj):
            adj += timedelta(days=1)
        if adj.month != d.month:
            adj = d
            while not self.is_business_day(adj):
                adj -= timedelta(days=1)
        return adj

    def add_business_days(self, d: date, n: int) -> date:
        result = d
        remaining = n
        while remaining > 0:
            result += timedelta(days=1)
            if self.is_business_day(result):
                remaining -= 1
        return result


CALENDARS: dict[str, Calendar] = {
    "USD": Calendar("US"),
    "EUR": Calendar("DE"),
    "GBP": Calendar("GB"),
    "JPY": Calendar("JP"),
    "CAD": Calendar("CA"),
    "CHF": Calendar("CH"),
    "AUD": Calendar("AU"),
}
