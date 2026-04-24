# 02 — Features and Models

Feature engineering, rate differential model, OIS curve construction, and central bank reaction functions.

## Feature Store

**Location:** `src/features/store.py`
**Purpose:** Register and compute features with dependency tracking.

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FeatureSpec:
    name: str
    dependencies: list[str]
    compute_fn: callable
    lookback_days: int


class FeatureStore:
    def __init__(self, engine):
        self.engine = engine
        self.specs = {}
    
    def register(self, spec: FeatureSpec):
        self.specs[spec.name] = spec
    
    def compute(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        order = self._topo_sort()
        features = {}
        for name in order:
            spec = self.specs[name]
            inputs = {dep: features[dep] for dep in spec.dependencies 
                      if dep in features}
            if not inputs:
                inputs = self._load_raw(symbol, spec.dependencies, start, end)
            features[name] = spec.compute_fn(**inputs)
        return pd.DataFrame(features)
```

## Standard Feature Computations

**Location:** `src/features/computations.py`
**Purpose:** Reusable feature computation functions.

```python
import pandas as pd
import numpy as np


def rate_differential_zscore(us_2y: pd.Series, de_2y: pd.Series, 
                              window: int = 252) -> pd.Series:
    diff = us_2y - de_2y
    rolling_mean = diff.rolling(window).mean()
    rolling_std = diff.rolling(window).std()
    return (diff - rolling_mean) / rolling_std


def realized_vol(price: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(price / price.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def cot_zscore(net_position: pd.Series, window: int = 156) -> pd.Series:
    # 3-year Z-score (156 weeks)
    rolling_mean = net_position.rolling(window).mean()
    rolling_std = net_position.rolling(window).std()
    return (net_position - rolling_mean) / rolling_std


def carry(target_rate: pd.Series, base_rate: pd.Series) -> pd.Series:
    return target_rate - base_rate


def momentum_signal(price: pd.Series, lookback: int = 63) -> pd.Series:
    return price / price.shift(lookback) - 1
```

## Rate Differential Model

**Location:** `src/models/rate_diff.py`
**Purpose:** Rolling regression of FX pair on rate spread. Used by rate-diff mean reversion strategy.

```python
import statsmodels.api as sm
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class RateDiffModelResult:
    coefficients: pd.Series
    residual_std: float
    r_squared: float
    fit_date: datetime
    training_start: datetime
    training_end: datetime


class RateDiffModel:
    def __init__(self, pair: str, factors: list[str], window_days: int = 756):
        self.pair = pair
        self.factors = factors
        self.window = window_days
        self.result: RateDiffModelResult | None = None
    
    def fit(self, df: pd.DataFrame) -> RateDiffModelResult:
        """df should have columns: target, *self.factors"""
        y = df['target']
        X = sm.add_constant(df[self.factors])
        model = sm.OLS(y, X).fit()
        self.result = RateDiffModelResult(
            coefficients=model.params,
            residual_std=model.resid.std(),
            r_squared=model.rsquared,
            fit_date=datetime.now(),
            training_start=df.index[0],
            training_end=df.index[-1],
        )
        return self.result
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = sm.add_constant(df[self.factors])
        return X @ self.result.coefficients
    
    def deviation_zscore(self, df: pd.DataFrame) -> pd.Series:
        fair_value = self.predict(df)
        deviation = df['target'] - fair_value
        return deviation / self.result.residual_std
```

## Day Count Conventions

**Location:** `src/rates/daycount.py`
**Purpose:** Proper day count conventions for OIS curve math.

```python
from datetime import date, timedelta
from enum import Enum


class DayCountConvention(Enum):
    ACT_360 = 'ACT/360'
    ACT_365 = 'ACT/365'
    ACT_ACT = 'ACT/ACT'
    THIRTY_360 = '30/360'


def year_fraction(d1: date, d2: date, convention: DayCountConvention) -> float:
    days = (d2 - d1).days
    if convention == DayCountConvention.ACT_360:
        return days / 360.0
    elif convention == DayCountConvention.ACT_365:
        return days / 365.0
    elif convention == DayCountConvention.ACT_ACT:
        return days / 365.25
    elif convention == DayCountConvention.THIRTY_360:
        d1_day = min(d1.day, 30)
        d2_day = min(d2.day, 30) if d1_day < 30 else d2.day
        return ((d2.year - d1.year) * 360 + 
                (d2.month - d1.month) * 30 + 
                (d2_day - d1_day)) / 360.0
    else:
        raise ValueError(f"Unknown convention: {convention}")


OIS_CONVENTIONS = {
    'USD': DayCountConvention.ACT_360,
    'EUR': DayCountConvention.ACT_360,
    'GBP': DayCountConvention.ACT_365,
    'JPY': DayCountConvention.ACT_365,
    'CAD': DayCountConvention.ACT_365,
    'CHF': DayCountConvention.ACT_360,
    'AUD': DayCountConvention.ACT_365,
}
```

## Business Day Calendar

**Location:** `src/rates/calendar.py`
**Purpose:** Holiday handling for OIS date adjustments.

```python
from datetime import date, timedelta
from enum import Enum
import holidays


class BusinessDayConvention(Enum):
    FOLLOWING = 'following'
    MODIFIED_FOLLOWING = 'modified_following'
    PRECEDING = 'preceding'


class Calendar:
    def __init__(self, country: str):
        self.country = country
        self.holidays = holidays.country_holidays(country)
    
    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays
    
    def adjust(self, d: date, 
               convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING) -> date:
        if self.is_business_day(d):
            return d
        
        if convention == BusinessDayConvention.FOLLOWING:
            while not self.is_business_day(d):
                d += timedelta(days=1)
            return d
        elif convention == BusinessDayConvention.PRECEDING:
            while not self.is_business_day(d):
                d -= timedelta(days=1)
            return d
        elif convention == BusinessDayConvention.MODIFIED_FOLLOWING:
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
        while n > 0:
            result += timedelta(days=1)
            if self.is_business_day(result):
                n -= 1
        return result


CALENDARS = {
    'USD': Calendar('US'),
    'EUR': Calendar('DE'),  # Approximation for TARGET2
    'GBP': Calendar('GB'),
    'JPY': Calendar('JP'),
    'CAD': Calendar('CA'),
    'CHF': Calendar('CH'),
    'AUD': Calendar('AU'),
}
```

## OIS Curve

**Location:** `src/rates/ois_curve.py`
**Purpose:** Production-grade OIS curve bootstrapping with proper day counts and business day adjustment.

```python
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq

from src.rates.daycount import DayCountConvention, year_fraction, OIS_CONVENTIONS
from src.rates.calendar import Calendar, CALENDARS, BusinessDayConvention


@dataclass
class OISQuote:
    tenor_label: str
    tenor_days: int
    maturity_date: date
    par_rate: float    # Decimal (0.0450 for 4.50%)


@dataclass
class OISCurve:
    """Bootstrapped OIS curve from par quotes."""
    currency: str
    valuation_date: date
    quotes: list[OISQuote]
    day_count: DayCountConvention
    calendar: Calendar
    discount_factors: dict[date, float] = field(default_factory=dict)
    interpolator: Callable = None
    
    @classmethod
    def from_quotes(cls, currency: str, valuation_date: date, 
                     raw_quotes: dict[str, float]) -> 'OISCurve':
        """Build from {tenor_label: par_rate} dict."""
        calendar = CALENDARS[currency]
        day_count = OIS_CONVENTIONS[currency]
        
        quotes = []
        for tenor_label, par_rate in raw_quotes.items():
            maturity = cls._tenor_to_date(valuation_date, tenor_label, calendar)
            tenor_days = (maturity - valuation_date).days
            quotes.append(OISQuote(tenor_label, tenor_days, maturity, par_rate))
        
        quotes.sort(key=lambda q: q.tenor_days)
        curve = cls(currency, valuation_date, quotes, day_count, calendar)
        curve._bootstrap()
        return curve
    
    @staticmethod
    def _tenor_to_date(start: date, tenor: str, calendar: Calendar) -> date:
        num = int(''.join(c for c in tenor if c.isdigit()))
        unit = ''.join(c for c in tenor if c.isalpha()).upper()
        
        if unit == 'D':
            raw = start + timedelta(days=num)
        elif unit == 'W':
            raw = start + timedelta(weeks=num)
        elif unit == 'M':
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
        elif unit == 'Y':
            try:
                raw = date(start.year + num, start.month, start.day)
            except ValueError:
                raw = date(start.year + num, start.month, 28)
        else:
            raise ValueError(f"Unknown tenor unit: {unit}")
        
        return calendar.adjust(raw, BusinessDayConvention.MODIFIED_FOLLOWING)
    
    def _bootstrap(self):
        """Iteratively solve for discount factors at each quote tenor."""
        self.discount_factors = {self.valuation_date: 1.0}
        
        for quote in self.quotes:
            df = self._solve_df_for_quote(quote)
            self.discount_factors[quote.maturity_date] = df
        
        dates = sorted(self.discount_factors.keys())
        tenors = np.array([(d - self.valuation_date).days for d in dates])
        dfs = np.array([self.discount_factors[d] for d in dates])
        
        log_dfs = np.log(dfs)
        self.interpolator = interp1d(
            tenors, log_dfs, kind='cubic', 
            bounds_error=False, fill_value='extrapolate'
        )
    
    def _solve_df_for_quote(self, quote: OISQuote) -> float:
        """Solve for discount factor at quote maturity."""
        T = quote.maturity_date
        tau_total = year_fraction(self.valuation_date, T, self.day_count)
        
        if quote.tenor_days <= 365:
            # Single payment
            return 1.0 / (1.0 + quote.par_rate * tau_total)
        
        # Multi-payment: annual coupons
        coupon_dates = self._generate_coupon_dates(T)
        prev_dates = coupon_dates[:-1]
        
        known_sum = 0.0
        prev_date = self.valuation_date
        for cd in prev_dates:
            tau_i = year_fraction(prev_date, cd, self.day_count)
            df_i = self.discount_factor(cd)
            known_sum += tau_i * df_i
            prev_date = cd
        
        tau_final = year_fraction(prev_date, T, self.day_count)
        S = quote.par_rate
        df_T = (1 - S * known_sum) / (1 + S * tau_final)
        
        return df_T
    
    def _generate_coupon_dates(self, maturity: date) -> list[date]:
        """Generate annual coupon dates from maturity backward."""
        dates = []
        d = maturity
        while d > self.valuation_date:
            dates.append(d)
            try:
                d = date(d.year - 1, d.month, d.day)
            except ValueError:
                d = date(d.year - 1, d.month, 28)
            d = self.calendar.adjust(d, BusinessDayConvention.MODIFIED_FOLLOWING)
        return sorted(dates)
    
    def discount_factor(self, d: date) -> float:
        if d in self.discount_factors:
            return self.discount_factors[d]
        tenor = (d - self.valuation_date).days
        return float(np.exp(self.interpolator(tenor)))
    
    def zero_rate(self, d: date, 
                   day_count: DayCountConvention = None) -> float:
        dc = day_count or self.day_count
        df = self.discount_factor(d)
        tau = year_fraction(self.valuation_date, d, dc)
        if tau <= 0:
            return 0.0
        return -np.log(df) / tau
    
    def forward_rate(self, d1: date, d2: date, 
                      day_count: DayCountConvention = None) -> float:
        dc = day_count or self.day_count
        df1 = self.discount_factor(d1)
        df2 = self.discount_factor(d2)
        tau = year_fraction(d1, d2, dc)
        if tau <= 0:
            return 0.0
        return (df1 / df2 - 1) / tau
    
    def implied_rate_at_meeting(self, meeting_date: date, 
                                  meeting_gap_days: int = 42) -> float:
        """Implied policy rate set at a CB meeting."""
        start = self.calendar.adjust(
            meeting_date - timedelta(days=1),
            BusinessDayConvention.PRECEDING
        )
        end = self.calendar.adjust(
            meeting_date + timedelta(days=meeting_gap_days),
            BusinessDayConvention.FOLLOWING
        )
        return self.forward_rate(start, end)
    
    def policy_path(self, meeting_dates: list[date]) -> dict[date, float]:
        result = {}
        for mtg in meeting_dates:
            result[mtg] = self.implied_rate_at_meeting(mtg)
        return result
    
    def meeting_probability(self, meeting_date: date, 
                             current_rate: float, 
                             move_size: float = 0.0025) -> dict[str, float]:
        """Probability distribution over {hike, hold, cut}."""
        implied = self.implied_rate_at_meeting(meeting_date)
        delta = implied - current_rate
        
        if abs(delta) > move_size * 3:
            return {
                'hike_prob': 1.0 if delta > 0 else 0.0,
                'hold_prob': 0.0,
                'cut_prob': 1.0 if delta < 0 else 0.0,
                'moves_implied': delta / move_size,
            }
        
        if delta >= 0:
            hike_prob = min(max(delta / move_size, 0), 1)
            return {
                'hike_prob': hike_prob,
                'hold_prob': 1 - hike_prob,
                'cut_prob': 0.0,
                'moves_implied': delta / move_size,
            }
        else:
            cut_prob = min(max(-delta / move_size, 0), 1)
            return {
                'hike_prob': 0.0,
                'hold_prob': 1 - cut_prob,
                'cut_prob': cut_prob,
                'moves_implied': delta / move_size,
            }
```

## OIS Curve Tests

**Location:** `tests/test_ois_curve.py`
**Purpose:** Sanity checks for curve math — roundtrip pricing must balance.

```python
from datetime import date
import pytest
from src.rates.ois_curve import OISCurve


def test_par_rate_roundtrip():
    """Curve pricing OIS at par should give zero NPV."""
    quotes = {
        '1M': 0.0525, '3M': 0.0520, '6M': 0.0510,
        '1Y': 0.0485, '2Y': 0.0445, '5Y': 0.0420, '10Y': 0.0410
    }
    curve = OISCurve.from_quotes('USD', date(2026, 4, 24), quotes)
    
    for tenor_label, par_rate in quotes.items():
        quote = next(q for q in curve.quotes if q.tenor_label == tenor_label)
        from src.rates.daycount import year_fraction
        tau = year_fraction(curve.valuation_date, quote.maturity_date, 
                           curve.day_count)
        df_T = curve.discount_factor(quote.maturity_date)
        
        if quote.tenor_days <= 365:
            implied_par = (1 - df_T) / (tau * df_T)
            assert abs(implied_par - par_rate) < 1e-6


def test_discount_factors_monotonic():
    quotes = {'1M': 0.05, '3M': 0.049, '6M': 0.048, '1Y': 0.045, 
              '2Y': 0.043, '5Y': 0.042}
    curve = OISCurve.from_quotes('USD', date(2026, 4, 24), quotes)
    
    sorted_dates = sorted(curve.discount_factors.keys())
    dfs = [curve.discount_factors[d] for d in sorted_dates]
    for i in range(1, len(dfs)):
        assert dfs[i] < dfs[i-1]
        assert dfs[i] > 0


def test_forward_rate_consistency():
    """With flat curve, all forwards should equal par rate."""
    quotes = {'1M': 0.05, '3M': 0.05, '6M': 0.05, '1Y': 0.05}
    curve = OISCurve.from_quotes('USD', date(2026, 4, 24), quotes)
    
    d_future = date(2026, 10, 24)
    fwd = curve.forward_rate(curve.valuation_date, d_future)
    assert abs(fwd - 0.05) < 1e-4
```

## Build Curve from SOFR Futures

**Location:** `src/rates/sofr_futures_curve.py`
**Purpose:** Retail-accessible OIS curve construction from free CME SOFR futures data.

```python
import pandas as pd
from datetime import date

from src.rates.ois_curve import OISCurve


def build_sofr_curve_from_futures(sr1_data: pd.DataFrame, 
                                    sr3_data: pd.DataFrame,
                                    valuation_date: date) -> OISCurve:
    """
    Construct SOFR OIS curve from SR1 and SR3 futures.
    
    SR1 for first ~12 months (higher resolution).
    SR3 beyond (deeper liquidity at longer tenors).
    """
    sr1_data = sr1_data[sr1_data['expiry'] > valuation_date].sort_values('expiry')
    sr3_data = sr3_data[sr3_data['expiry'] > valuation_date].sort_values('expiry')
    
    tenor_rates = {}
    
    for _, row in sr1_data.head(12).iterrows():
        days_to_expiry = (row['expiry'] - valuation_date).days
        tenor_rates[f'{days_to_expiry}D'] = row['implied_rate']
    
    for _, row in sr3_data.iloc[4:].head(8).iterrows():
        days_to_expiry = (row['expiry'] - valuation_date).days
        tenor_rates[f'{days_to_expiry}D'] = row['implied_rate']
    
    return OISCurve.from_quotes('USD', valuation_date, tenor_rates)
```

## Fed Reaction Function

**Location:** `src/models/fed_reaction_function.py`
**Purpose:** Quantitative model of Fed policy response to inflation, employment, and financial conditions.

```python
from dataclasses import dataclass
from scipy.optimize import minimize
import pandas as pd


@dataclass
class FedReactionFunction:
    # Weights on different variables (fit historically)
    w_core_pce: float = 0.6
    w_unemployment: float = 0.3
    w_fci: float = 0.1
    
    # Targets
    pce_target: float = 2.0
    u_star: float = 4.2         # Estimated natural rate of unemployment
    fci_neutral: float = 0.0    # Goldman FCI neutral level
    r_star: float = 0.5         # Real neutral rate estimate
    
    def implied_rate(self, core_pce: float, unemployment: float, 
                     fci: float) -> float:
        """Compute policy rate per reaction function (Taylor-like)."""
        inflation_gap = core_pce - self.pce_target
        unemployment_gap = self.u_star - unemployment  # Positive = tight labor
        fci_gap = self.fci_neutral - fci
        
        rate = (
            self.r_star + core_pce  # Nominal neutral
            + self.w_core_pce * 1.5 * inflation_gap  # Taylor principle (>1)
            + self.w_unemployment * unemployment_gap
            + self.w_fci * fci_gap
        )
        return max(0, rate)  # Zero lower bound
    
    def fit(self, historical_data: pd.DataFrame):
        """Calibrate weights from historical Fed actions.
        historical_data columns: core_pce, unemployment, fci, fed_funds"""
        def loss(params):
            w_pce, w_u, w_fci = params
            self.w_core_pce, self.w_unemployment, self.w_fci = w_pce, w_u, w_fci
            predicted = historical_data.apply(
                lambda row: self.implied_rate(
                    row['core_pce'], row['unemployment'], row['fci']
                ), axis=1
            )
            return ((predicted - historical_data['fed_funds']) ** 2).sum()
        
        result = minimize(loss, x0=[0.6, 0.3, 0.1], 
                         bounds=[(0, 2), (0, 2), (0, 1)])
        self.w_core_pce, self.w_unemployment, self.w_fci = result.x
```
