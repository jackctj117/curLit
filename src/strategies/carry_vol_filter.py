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
from datetime import UTC, date, datetime
from typing import Any

from src.execution.oms import OrderIntent
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
)
from src.risk.liquidity_window import LiquidityProfile, spread_bps_from_tick
from src.risk.sizing import PositionSizer
from src.strategies.vol_regime import compute_vol_z_score

_FEATURE_SET_NAME = "carry_vol_filter"
_FEATURE_SET_VERSION = "v1"

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
        snapshot_store: FeatureSnapshotStore | None = None,
        liquidity_profile: LiquidityProfile | None = None,
    ) -> None:
        self.config = config or CarryVolFilterConfig()
        self.data = data_provider
        self.state = state_store
        self.snapshot_store = snapshot_store
        # CL-y412: liquidity-window gate applied to NEW basket legs only at
        # rebalance (dropped/retained legs pass through so an exit or trim is
        # never blocked). None = inert until the monthly refresh writes one.
        self._liquidity_profile = liquidity_profile
        self.current_positions: dict[str, CarryPosition] = {}
        self._last_rebalance: date | None = None
        self._current_exposure: float = 1.0

    def _emit_snapshot(self, values: dict[str, Any]) -> dict[str, Any]:
        """Build, persist, and return a snapshot-reference payload (or {})."""
        if self.snapshot_store is None:
            return {}
        snapshot = FeatureSnapshot.create(
            feature_set_name=_FEATURE_SET_NAME,
            feature_set_version=_FEATURE_SET_VERSION,
            data_snapshot_id="live",
            model_version=(
                f"top_k={self.config.top_k},bot_k={self.config.bottom_k},"
                f"min_spread={self.config.min_rate_spread}"
            ),
            ts=datetime.now(UTC),
            values=values,
        )
        try:
            self.snapshot_store.store(snapshot)
        except Exception:
            logger.exception(
                "Failed to store feature snapshot for %s — intent will lack "
                "snapshot reference",
                self.id,
            )
            return {}
        return attach_snapshot_payload(snapshot)

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

    # NOTE: no fit()/generate_signals() — those belong to the walk-forward
    # backtest protocol (src/backtest/walkforward.Strategy); live-only
    # strategies are driven exclusively via generate_intents() and the
    # dead no-op stubs were removed (CL-e6lx).

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
            except Exception as exc:
                # Use logger.warning (not exception) in tight loops —
                # logger.exception retains traceback frames + locals,
                # which can leak slowly under repeated errors (CL-2yta).
                logger.warning(
                    "get_latest_value failed for %s: %s: %s",
                    series_id, type(exc).__name__, exc,
                )
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

        CL-x50g: logic extracted to src/strategies/vol_regime.py so the
        rate-diff entry filters share the identical z-score definition.
        """
        return compute_vol_z_score(
            self.data,
            self.config.vol_index_series,
            self.config.vol_lookback_days,
            as_of,
        )

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
            self._tag_intents(intents, vol_z, new_exposure, trigger="vol_scale")
            return intents

        rates = self._get_current_rates(now)
        baskets = self._construct_baskets(rates)

        if not baskets["long"]:
            # No tradable spread or insufficient rate data — flatten and exit.
            intents.extend(self._flatten_all())
            self._last_rebalance = now.date()
            self._current_exposure = new_exposure
            self._record_rebalance(now, rates, baskets, vol_z, new_exposure)
            self._tag_intents(
                intents, vol_z, new_exposure, trigger="rebalance_flatten",
                baskets=baskets,
            )
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
        blocked_new: list[str] = []
        for ccy in all_ccys:
            if ccy == "USD":
                continue
            pair, broker_side = self._currency_to_pair(ccy, 1)
            if not pair:
                continue
            if ccy in new_positions:
                pos = new_positions[ccy]
                is_new_leg = ccy not in self.current_positions
                notional = self._compute_position_size(
                    weight=pos.weight,
                    equity=account.equity,
                    exposure=new_exposure,
                )
                price = self._lookup_price(prices, pair)
                if price <= 0:
                    # CL-uorm (P1): no price → can't size this leg. A NEW leg
                    # must NOT be recorded as held — leaving it in
                    # new_positions was a phantom hold that drove later
                    # vol-scale/rebalance as if the risk were open. A RETAINED
                    # leg (real broker position) stays and just skips this
                    # rebalance tick.
                    if is_new_leg:
                        blocked_new.append(ccy)
                    continue
                target_qty = pos.side * broker_side * notional / price
                # CL-y412: gate GENUINELY NEW legs only — a leg carried over
                # from the prior basket (retained) or being dropped (target 0)
                # is a rebalance/exit and must run at any spread. A new leg in
                # a dead window is blocked (skip open, drop from state so no
                # phantom hold) or trimmed (0.5×).
                if is_new_leg and self._liquidity_profile is not None:
                    spread_bps = spread_bps_from_tick(prices.get(pair))
                    if spread_bps is not None:
                        liq_qty = PositionSizer.adjust_for_liquidity(
                            target_qty, pair, now, spread_bps,
                            self._liquidity_profile,
                        )
                        if liq_qty == 0.0:
                            logger.info(
                                "Carry leg %s (%s) not opened — dead liquidity "
                                "window (spread=%.1fbps)",
                                ccy, pair, spread_bps,
                            )
                            blocked_new.append(ccy)
                            continue
                        target_qty = liq_qty
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

        # Blocked new legs never opened — drop them so state stays truthful
        # (the reconciler and next rebalance must not believe we hold them).
        for ccy in blocked_new:
            new_positions.pop(ccy, None)

        self.current_positions = new_positions
        self._current_exposure = new_exposure
        self._last_rebalance = now.date()
        self._record_rebalance(now, rates, baskets, vol_z, new_exposure)
        self._tag_intents(
            intents, vol_z, new_exposure, trigger="rebalance",
            baskets=baskets,
        )
        return intents

    def _tag_intents(
        self,
        intents: list[OrderIntent],
        vol_z: float,
        exposure: float,
        trigger: str,
        baskets: dict[str, Any] | None = None,
    ) -> None:
        """Build one snapshot per generate_intents call; tag every intent.

        All intents in a single tick share the same feature evaluation
        (vol_z + basket composition), so one snapshot covers them all.
        """
        if not intents:
            return
        values: dict[str, Any] = {
            "trigger": trigger,
            "vol_z": float(vol_z),
            "exposure": float(exposure),
            "n_intents": len(intents),
        }
        if baskets is not None:
            values["long_basket"] = [
                {"ccy": ccy, "rate": float(rate), "weight": float(weight)}
                for ccy, rate, weight in baskets.get("long", [])
            ]
            values["short_basket"] = [
                {"ccy": ccy, "rate": float(rate), "weight": float(weight)}
                for ccy, rate, weight in baskets.get("short", [])
            ]
        meta = self._emit_snapshot(values)
        if not meta:
            return
        for intent in intents:
            intent.metadata = dict(meta)

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
