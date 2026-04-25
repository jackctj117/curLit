"""Strategy 3: Carry + Volatility Filter (CL-c77 / A5).

G10 carry trade with vol-regime-based exposure scaling. Long the highest-yielding
currencies, short the lowest, scale total exposure by a volatility z-score.

References:
    - Menkhoff et al. (2012), "Carry Trades and Global Foreign Exchange Volatility"
    - reference/05_strategies.md (Strategy 3)

Cadence:
    Monthly rebalance on the first business day of the month — chooses the
    long/short baskets by current short-rate ranking. Daily vol filter trims
    exposure when realized vol spikes above its rolling baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np

from src.execution.oms import OrderIntent

logger = logging.getLogger(__name__)


# G10 yield series identifiers — what data_provider.get_latest_value expects.
_RATE_SERIES_MAP: dict[str, str] = {
    "USD": "USD_3M_OIS",
    "EUR": "EUR_3M_ESTR_OIS",
    "JPY": "JPY_3M_TONA_OIS",
    "GBP": "GBP_3M_SONIA_OIS",
    "CHF": "CHF_3M_SARON_OIS",
    "CAD": "CAD_3M_CORRA_OIS",
    "AUD": "AUD_3M_BBSW",
    "NZD": "NZD_3M_BKBM",
    "NOK": "NOK_3M_NIBOR",
    "SEK": "SEK_3M_STIBOR",
}

# Pairs where USD is the BASE currency (USD/X). Holding "long X" in carry means
# selling these pairs. All other G10 pairs are quoted X/USD.
_USD_BASE_PAIRS: frozenset[str] = frozenset(
    {"JPY", "CHF", "CAD", "NOK", "SEK"},
)


@dataclass
class CarryVolFilterConfig:
    """Configuration for the carry + vol filter strategy."""

    currencies: list[str] = field(default_factory=lambda: [
        "USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "NOK", "SEK",
    ])
    top_k: int = 3
    bottom_k: int = 3

    # Calendar day-of-month to rebalance on (1 = first business day).
    rebalance_day: int = 1

    # Vol index (e.g. CVIX, JPMVXY, GVZ) and lookback for the z-score.
    vol_index_series: str = "CVIX"
    vol_lookback_days: int = 120

    # Mapping vol-z thresholds → exposure multiplier. The applied multiplier
    # is the value at the largest threshold key still ≤ |z|. So |z| 0 → 100%,
    # |z| 1 → 75%, |z| 2 → 50%, |z| 3 → 0%. Stepwise (not interpolated) for
    # operational clarity in alerts.
    vol_z_thresholds: dict[float, float] = field(default_factory=lambda: {
        0.0: 1.00,
        1.0: 0.75,
        2.0: 0.50,
        3.0: 0.00,
    })

    # Equal-weight per-pair sizing fraction relative to equity, capped here.
    max_position_pct: float = 0.15

    # Minimum rate spread (long basket - short basket) below which the
    # strategy goes flat. Prevents flat-yield-curve regimes from forcing trades.
    min_rate_spread: float = 0.005

    # Strategy emits intents at most every this often. Daily cadence (86400s)
    # is sufficient — the rebalance is monthly, vol-filter is daily.
    signal_interval_seconds: int = 86_400

    id: str = "carry_vol_filter"


@dataclass
class CarryPosition:
    """Per-currency target position in the carry basket."""

    currency: str
    side: int  # +1 long, -1 short
    weight: float
    entry_ts: datetime
    reference_rate: float


class CarryVolFilterStrategy:
    """Monthly-rebalance G10 carry trade with daily vol-regime exposure scaling.

    Wired into LiveEngine the same way as the other strategies (rate_diff,
    cb_sentiment): exposes id + symbols + signal_interval_seconds, returns
    OrderIntent list from generate_intents. Scaling and constraints are
    applied downstream by the PortfolioCoordinator.
    """

    def __init__(
        self,
        config: CarryVolFilterConfig | None = None,
        data_provider: Any | None = None,
        state_store: Any | None = None,
    ) -> None:
        self.config = config or CarryVolFilterConfig()
        self.data = data_provider
        self.state = state_store
        self.current_positions: dict[str, CarryPosition] = {}
        self._last_rebalance: date | None = None
        self._current_exposure: float = 1.0

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        return [
            self._currency_to_pair(ccy, 1)[0]
            for ccy in self.config.currencies
            if ccy != "USD"
        ]

    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds

    def fit(self, train_data: Any) -> None:
        """No-op fit — strategy state lives entirely in config + live state."""
        return None

    def generate_signals(self, data: Any) -> None:
        return None

    # ------------------------------------------------------------------
    # Pair / currency conventions
    # ------------------------------------------------------------------

    @staticmethod
    def _currency_to_pair(ccy: str, side: int) -> tuple[str, int]:
        """Map a currency + carry-side to (pair, broker_side).

        Carry "long X" means we want to be holding currency X. The broker side
        depends on the FX pair convention:
            - For X/USD pairs (EUR, GBP, AUD, NZD): long X = buy X/USD (+1).
            - For USD/X pairs (JPY, CHF, CAD, NOK, SEK): long X = sell USD/X
              (-1) so we earn X-denominated returns.
        """
        if ccy == "USD":
            # USD-side carry must be expressed as the inverse of another pair;
            # we rely on the basket being non-USD to avoid this.
            return ("", 0)
        if ccy in _USD_BASE_PAIRS:
            return (f"USD{ccy}", -side)
        return (f"{ccy}USD", side)

    # ------------------------------------------------------------------
    # Rates + baskets
    # ------------------------------------------------------------------

    def _get_current_rates(self, as_of: datetime) -> dict[str, float]:
        """Pull each currency's short rate from the data provider."""
        if self.data is None:
            return {}
        rates: dict[str, float] = {}
        for ccy in self.config.currencies:
            series_id = _RATE_SERIES_MAP.get(ccy)
            if series_id is None:
                continue
            try:
                rate = self.data.get_latest_value(series_id, as_of)
            except Exception:
                logger.exception("get_latest_value failed for %s", series_id)
                continue
            if rate is not None:
                rates[ccy] = float(rate)
        return rates

    def _construct_baskets(
        self,
        rates: dict[str, float],
    ) -> dict[str, Any]:
        """Sort by rate, take top_k as long basket, bottom_k as short basket."""
        # Exclude USD from the basket — we trade FX pairs, not direct USD carry.
        non_usd = {c: r for c, r in rates.items() if c != "USD"}
        if len(non_usd) < self.config.top_k + self.config.bottom_k:
            return {"long": [], "short": [], "spread": 0.0}

        sorted_ccys = sorted(non_usd.items(), key=lambda x: x[1], reverse=True)
        long_basket = sorted_ccys[: self.config.top_k]
        short_basket = sorted_ccys[-self.config.bottom_k :]

        rate_spread = long_basket[-1][1] - short_basket[0][1]
        # 1e-9 tolerance avoids spurious failures from float subtraction
        # (e.g. 0.045 - 0.04 = 0.004999999...) below the configured threshold.
        if rate_spread < self.config.min_rate_spread - 1e-9:
            return {"long": [], "short": [], "spread": rate_spread}

        return {
            "long": [
                (c, r, 1.0 / self.config.top_k) for c, r in long_basket
            ],
            "short": [
                (c, r, 1.0 / self.config.bottom_k) for c, r in short_basket
            ],
            "spread": rate_spread,
        }

    # ------------------------------------------------------------------
    # Vol filter
    # ------------------------------------------------------------------

    def _compute_vol_z_score(self, as_of: datetime) -> float:
        """Compute the z-score of the latest vol-index value vs its rolling mean.

        Falls back to 0 (no scaling) when the data provider is missing or
        returns insufficient history. This makes the strategy safely degrade
        rather than fail when the fx_volatility table isn't yet populated.
        """
        if self.data is None:
            return 0.0
        end = as_of
        # Pull 2× lookback to allow the rolling stats to warm up.
        start = end - timedelta(days=self.config.vol_lookback_days * 2)
        try:
            vol_series = self.data.get_series(
                self.config.vol_index_series, start, end,
            )
        except Exception:
            logger.exception("get_series failed for %s", self.config.vol_index_series)
            return 0.0
        if vol_series is None or len(vol_series) < self.config.vol_lookback_days * 0.8:
            return 0.0
        # Use the last lookback_days as the rolling window.
        recent = np.asarray(vol_series, dtype=float)[-self.config.vol_lookback_days :]
        if len(recent) < 2:
            return 0.0
        mean = float(np.mean(recent[:-1]))
        sd = float(np.std(recent[:-1], ddof=1))
        if sd <= 0:
            # Flat baseline — any deviation is "infinitely surprising". Return
            # a synthetic large z so the exposure filter still kicks in. Sign
            # matches the direction of the deviation; magnitude (10) is enough
            # to push past every configured threshold and zero out exposure.
            delta = float(recent[-1] - mean)
            if abs(delta) < 1e-9:
                return 0.0
            return 10.0 if delta > 0 else -10.0
        return float((recent[-1] - mean) / sd)

    def _exposure_multiplier(self, vol_z: float) -> float:
        """Step function from |vol_z| to exposure scaler per config thresholds."""
        z = abs(vol_z)
        thresholds = sorted(self.config.vol_z_thresholds.items())
        applied = thresholds[0][1]
        for threshold, mult in thresholds:
            if z >= threshold:
                applied = mult
        return float(applied)

    # ------------------------------------------------------------------
    # Rebalance scheduling
    # ------------------------------------------------------------------

    def _should_rebalance(self, now: datetime) -> bool:
        """Rebalance once per month on or after rebalance_day's first business day.

        We check that today is in the configured day-of-month range AND the
        last_rebalance was in a different month. Operationally this means:
        first time we get called this month past day N, we rebalance.
        """
        today = now.date()
        if self._last_rebalance is None:
            return today.day >= self.config.rebalance_day
        # Already rebalanced in this calendar month?
        if (
            self._last_rebalance.year == today.year
            and self._last_rebalance.month == today.month
        ):
            return False
        return today.day >= self.config.rebalance_day

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _compute_position_size(
        self,
        weight: float,
        equity: float,
        exposure: float,
    ) -> float:
        """Per-pair notional given basket weight, equity, current exposure scale.

        Capped at max_position_pct of equity per pair to prevent
        single-currency outsizing under extreme rate spreads.
        """
        notional = equity * weight * exposure
        cap = equity * self.config.max_position_pct
        return float(min(notional, cap))

    # ------------------------------------------------------------------
    # Intent generation
    # ------------------------------------------------------------------

    async def generate_intents(
        self,
        prices: dict[str, dict[str, Any]],
        broker: Any,
    ) -> list[OrderIntent]:
        """Daily call: apply vol filter, monthly rebalance produces baskets."""
        now = datetime.now(UTC)
        vol_z = self._compute_vol_z_score(now)
        new_exposure = self._exposure_multiplier(vol_z)

        intents: list[OrderIntent] = []

        # If exposure has been cut sharply (e.g. crisis regime), scale current
        # positions BEFORE waiting for the next monthly rebalance.
        if (
            new_exposure < self._current_exposure * 0.8
            and self.current_positions
        ):
            intents.extend(self._scale_positions(prices, broker, new_exposure))
            self._current_exposure = new_exposure

        if not self._should_rebalance(now):
            return intents

        rates = self._get_current_rates(now)
        baskets = self._construct_baskets(rates)

        if not baskets["long"]:
            # No tradable spread or insufficient rate data — flatten and exit.
            intents.extend(self._flatten_all())
            self._last_rebalance = now.date()
            self._current_exposure = new_exposure
            self._record_rebalance(now, rates, baskets, vol_z, new_exposure)
            return intents

        account = broker.get_account()

        new_positions: dict[str, CarryPosition] = {}
        for ccy, rate, weight in baskets["long"]:
            new_positions[ccy] = CarryPosition(
                currency=ccy, side=1, weight=weight,
                entry_ts=now, reference_rate=rate,
            )
        for ccy, rate, weight in baskets["short"]:
            new_positions[ccy] = CarryPosition(
                currency=ccy, side=-1, weight=weight,
                entry_ts=now, reference_rate=rate,
            )

        all_ccys = set(self.current_positions.keys()) | set(new_positions.keys())
        for ccy in all_ccys:
            if ccy == "USD":
                continue
            pair, broker_side = self._currency_to_pair(ccy, 1)
            if not pair:
                continue
            if ccy in new_positions:
                pos = new_positions[ccy]
                notional = self._compute_position_size(
                    weight=pos.weight,
                    equity=account.equity,
                    exposure=new_exposure,
                )
                price = self._lookup_price(prices, pair)
                if price <= 0:
                    continue
                target_qty = pos.side * broker_side * notional / price
            else:
                target_qty = 0.0
            intents.append(
                OrderIntent(
                    strategy_id=self.id,
                    symbol=pair,
                    target_position=target_qty,
                    urgency="normal",
                ),
            )

        self.current_positions = new_positions
        self._current_exposure = new_exposure
        self._last_rebalance = now.date()
        self._record_rebalance(now, rates, baskets, vol_z, new_exposure)
        return intents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_price(prices: dict[str, dict[str, Any]], pair: str) -> float:
        tick = prices.get(pair)
        if not tick:
            return 0.0
        bid = float(tick.get("bid", 0.0) or 0.0)
        ask = float(tick.get("ask", 0.0) or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return bid or ask

    def _scale_positions(
        self,
        prices: dict[str, dict[str, Any]],
        broker: Any,
        new_exposure: float,
    ) -> list[OrderIntent]:
        """Scale all currently-held positions to the new exposure level."""
        if self._current_exposure <= 0:
            return []
        scale = new_exposure / self._current_exposure
        if abs(scale - 1.0) < 1e-9:
            return []
        out: list[OrderIntent] = []
        account = broker.get_account()
        for ccy, pos in self.current_positions.items():
            pair, broker_side = self._currency_to_pair(ccy, 1)
            if not pair:
                continue
            price = self._lookup_price(prices, pair)
            if price <= 0:
                continue
            old_notional = (
                self._compute_position_size(
                    weight=pos.weight, equity=account.equity,
                    exposure=self._current_exposure,
                )
            )
            new_notional = old_notional * scale
            target_qty = pos.side * broker_side * new_notional / price
            out.append(
                OrderIntent(
                    strategy_id=self.id,
                    symbol=pair,
                    target_position=target_qty,
                    urgency="normal",
                ),
            )
        return out

    def _flatten_all(self) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for ccy in self.current_positions:
            pair, _ = self._currency_to_pair(ccy, 1)
            if not pair:
                continue
            intents.append(
                OrderIntent(
                    strategy_id=self.id,
                    symbol=pair,
                    target_position=0.0,
                    urgency="normal",
                ),
            )
        self.current_positions.clear()
        return intents

    def _record_rebalance(
        self,
        now: datetime,
        rates: dict[str, float],
        baskets: dict[str, Any],
        vol_z: float,
        exposure: float,
    ) -> None:
        if self.state is None or not hasattr(self.state, "record_signal"):
            return
        try:
            self.state.record_signal(
                strategy_id=self.id,
                ts=now,
                signal={
                    "rates": rates,
                    "long_basket": [c for c, _, _ in baskets.get("long", [])],
                    "short_basket": [c for c, _, _ in baskets.get("short", [])],
                    "spread": float(baskets.get("spread", 0.0)),
                    "vol_z": float(vol_z),
                    "exposure": float(exposure),
                },
            )
        except Exception:
            logger.exception("record_signal failed for %s rebalance", self.id)
