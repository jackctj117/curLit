"""OIS curve bootstrapper — from par quotes to discount factors and forward rates."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from .calendar import CALENDARS, BusinessDayConvention
from .daycount import OIS_CONVENTIONS, DayCountConvention, year_fraction


@dataclass
class OISQuote:
    tenor_label: str
    tenor_days: int
    maturity_date: date
    par_rate: float


@dataclass
class OISCurve:
    currency: str
    valuation_date: date
    quotes: list[OISQuote]
    day_count: DayCountConvention
    discount_factors: dict[date, float] = field(default_factory=dict)
    _interpolator: Any = field(default=None, init=False)

    # -- factory --------------------------------------------------------

    @classmethod
    def from_quotes(
        cls, currency: str, valuation_date: date, raw_quotes: dict[str, float],
    ) -> "OISCurve":
        calendar = CALENDARS[currency]
        dc = OIS_CONVENTIONS[currency]

        quotes: list[OISQuote] = []
        for label, rate in raw_quotes.items():
            maturity = cls._tenor_to_date(valuation_date, label, calendar)
            quotes.append(OISQuote(
                tenor_label=label,
                tenor_days=(maturity - valuation_date).days,
                maturity_date=maturity,
                par_rate=rate,
            ))
        quotes.sort(key=lambda q: q.tenor_days)

        curve = cls(currency, valuation_date, quotes, dc)
        curve._bootstrap()
        return curve

    @staticmethod
    def _tenor_to_date(start: date, tenor: str, calendar: Any) -> date:
        num = int("".join(c for c in tenor if c.isdigit()))
        unit = "".join(c for c in tenor if c.isalpha()).upper()

        if unit == "D":
            raw = start + timedelta(days=num)
        elif unit == "W":
            raw = start + timedelta(weeks=num)
        elif unit == "M":
            month = start.month + num
            year = start.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            try:
                raw = date(year, month, start.day)
            except ValueError:
                if month == 12:
                    raw = date(year + 1, 1, 1) - timedelta(days=1)
                else:
                    raw = date(year, month + 1, 1) - timedelta(days=1)
        elif unit == "Y":
            try:
                raw = date(start.year + num, start.month, start.day)
            except ValueError:
                raw = date(start.year + num, start.month, 28)
        else:
            raise ValueError(f"Unknown tenor unit: {unit}")

        adjusted: date = calendar.adjust(raw, BusinessDayConvention.MODIFIED_FOLLOWING)
        return adjusted

    # -- bootstrapping --------------------------------------------------

    def _bootstrap(self) -> None:
        self.discount_factors = {self.valuation_date: 1.0}

        for quote in self.quotes:
            df = self._solve_df_for_quote(quote)
            self.discount_factors[quote.maturity_date] = df

        dates = sorted(self.discount_factors.keys())
        tenors = np.array([(d - self.valuation_date).days for d in dates], dtype=float)
        dfs = np.array([float(self.discount_factors[d]) for d in dates])
        log_dfs = np.log(dfs)
        # interp1d requires kind-specific minimum point counts: cubic >= 4,
        # quadratic >= 3, linear >= 2. Production curves typically have 5+
        # quotes (cubic stays the common path); the degree-adaptive fallback
        # is purely defensive against sparse curves and tests.
        n = len(tenors)
        assert n >= 2, "OIS bootstrap needs at least one quote (got zero)"
        if n >= 4:
            kind = "cubic"
        elif n == 3:
            kind = "quadratic"
        else:
            kind = "linear"
        self._interpolator = interp1d(
            tenors, log_dfs, kind=kind, bounds_error=False, fill_value="extrapolate",
        )

    def _solve_df_for_quote(self, quote: OISQuote) -> float:
        T = quote.maturity_date
        tau_total = year_fraction(self.valuation_date, T, self.day_count)

        # OIS swap convention: instruments with tenor <2Y pay a single coupon
        # at maturity (money-market formula). Annual coupons start at 2Y. The
        # 730-day cutoff covers calendar-adjustment slack near 1Y boundaries —
        # a 365D quote can be bumped to ~368 days by holiday adjustment, and
        # treating it as multi-coupon generates phantom intermediate coupons.
        if quote.tenor_days < 730:
            return 1.0 / (1.0 + quote.par_rate * tau_total)

        coupon_dates = self._generate_coupon_dates(T)
        prev_dates = coupon_dates[:-1]
        known_sum = 0.0
        prev_date = self.valuation_date
        for cd in prev_dates:
            tau_i = year_fraction(prev_date, cd, self.day_count)
            # discount_factor() may need to interpolate if cd is not yet in
            # the bootstrapped set — that requires earlier (shorter-tenor)
            # quotes to have already been solved. Raise a clear error if
            # the schedule isn't dense enough rather than crashing on None.
            if cd not in self.discount_factors and self._interpolator is None:
                msg = (
                    f"OIS bootstrap gap: solving {quote.tenor_label} swap "
                    f"requires intermediate DF at {cd}, but no shorter quote "
                    f"covers it. Provide annual quotes (1Y, 2Y, …) up to the "
                    f"longest tenor."
                )
                raise ValueError(msg)
            known_sum += tau_i * self.discount_factor(cd)
            prev_date = cd
        tau_final = year_fraction(prev_date, T, self.day_count)
        S = quote.par_rate
        return (1.0 - S * known_sum) / (1.0 + S * tau_final)

    def _generate_coupon_dates(self, maturity: date) -> list[date]:
        dates: list[date] = []
        d = maturity
        while d > self.valuation_date:
            dates.append(d)
            try:
                d = date(d.year - 1, d.month, d.day)
            except ValueError:
                d = date(d.year - 1, d.month, 28)
            d = CALENDARS[self.currency].adjust(d, BusinessDayConvention.MODIFIED_FOLLOWING)
        return sorted(dates)

    # -- query methods --------------------------------------------------

    def discount_factor(self, d: date) -> float:
        if d in self.discount_factors:
            return float(self.discount_factors[d])
        if self._interpolator is None:
            msg = (
                f"Cannot interpolate DF at {d}: bootstrap is incomplete "
                f"(no interpolator). Curve may have been constructed with "
                f"insufficient quotes."
            )
            raise ValueError(msg)
        tenor = float((d - self.valuation_date).days)
        return float(np.exp(self._interpolator(tenor)))

    def zero_rate(self, d: date, dc: DayCountConvention | None = None) -> float:
        dc = dc or self.day_count
        df = self.discount_factor(d)
        tau = year_fraction(self.valuation_date, d, dc)
        if tau <= 0:
            return 0.0
        return -np.log(df) / tau  # type: ignore[no-any-return]

    def forward_rate(self, d1: date, d2: date, dc: DayCountConvention | None = None) -> float:
        dc = dc or self.day_count
        df1 = self.discount_factor(d1)
        df2 = self.discount_factor(d2)
        tau = year_fraction(d1, d2, dc)
        if tau <= 0:
            return 0.0
        return (df1 / df2 - 1) / tau

    def implied_rate_at_meeting(self, meeting_date: date, meeting_gap_days: int = 42) -> float:
        calendar = CALENDARS[self.currency]
        start = calendar.adjust(meeting_date - timedelta(days=1), BusinessDayConvention.PRECEDING)
        end = calendar.adjust(meeting_date + timedelta(days=meeting_gap_days), BusinessDayConvention.FOLLOWING)
        return self.forward_rate(start, end)

    def policy_path(self, meeting_dates: list[date]) -> dict[date, float]:
        return {mtg: self.implied_rate_at_meeting(mtg) for mtg in meeting_dates}

    def meeting_probability(
        self, meeting_date: date, current_rate: float, move_size: float = 0.0025,
    ) -> dict[str, float]:
        implied = self.implied_rate_at_meeting(meeting_date)
        delta = implied - current_rate
        num_moves = delta / move_size

        if abs(delta) > move_size * 3:
            return {
                "hike_prob": 1.0 if delta > 0 else 0.0,
                "hold_prob": 0.0,
                "cut_prob": 1.0 if delta < 0 else 0.0,
                "moves_implied": num_moves,
            }

        if delta >= 0:
            hike_prob = min(max(delta / move_size, 0.0), 1.0)
            return {"hike_prob": hike_prob, "hold_prob": 1.0 - hike_prob,
                    "cut_prob": 0.0, "moves_implied": num_moves}
        cut_prob = min(max(-delta / move_size, 0.0), 1.0)
        return {"hike_prob": 0.0, "hold_prob": 1.0 - cut_prob,
                "cut_prob": cut_prob, "moves_implied": num_moves}
