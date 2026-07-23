"""Unit tests — rates/OIS: day-count, calendars, curve bootstrapping."""

from datetime import date

from src.rates.calendar import CALENDARS, BusinessDayConvention
from src.rates.daycount import DayCountConvention, year_fraction


class TestDayCount:
    def test_act360_one_year(self) -> None:
        assert (
            abs(
                year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_360)
                - 366 / 360
            )
            < 0.01
        )

    def test_act365_one_year(self) -> None:
        assert (
            abs(
                year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_365)
                - 366 / 365
            )
            < 0.01
        )

    def test_thirty360_equal_months(self) -> None:
        assert (
            year_fraction(date(2024, 1, 1), date(2024, 2, 1), DayCountConvention.THIRTY_360)
            == 30 / 360
        )


class TestCalendar:
    def test_us_calendar_exists(self) -> None:
        assert "USD" in CALENDARS

    def test_july4_is_holiday(self) -> None:
        cal = CALENDARS["USD"]
        assert not cal.is_business_day(date(2025, 7, 4))

    def test_adjust_following(self) -> None:
        cal = CALENDARS["USD"]
        sat = date(2025, 7, 5)
        adjusted = cal.adjust(sat, BusinessDayConvention.FOLLOWING)
        assert adjusted.weekday() < 5
