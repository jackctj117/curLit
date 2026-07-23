"""Strategy 1: Rate differential mean reversion on EUR/USD.

CL-x50g: carry / momentum / vol-regime entry filters. Each filter is
individually toggleable and gates NEW entries only — exits and stops are
never blocked. Disabling all three reproduces the pre-filter behavior
exactly, in both the backtest path (generate_signals) and the live path
(generate_intents).
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.execution.oms import OrderIntent
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
)
from src.risk.liquidity_window import LiquidityProfile, spread_bps_from_tick
from src.risk.sizing import PositionSizer
from src.strategies.vol_regime import compute_vol_z_score, rolling_vol_z

logger = logging.getLogger(__name__)


# Strategy identity for the feature_versioning registry.
_FEATURE_SET_NAME = "rate_diff_mr"
_FEATURE_SET_VERSION = "v1"


@dataclass
class RateDiffMRConfig:
    pair: str = "EURUSD"
    rate_spread_series: str = "US10Y_MINUS_DE10Y"
    # 252 trading days ≈ 1 year — window long enough to capture multiple
    # rate cycles, short enough to adapt to regime changes (Arch §4.1)
    lookback_days: int = 252
    # 1.5 std deviations — balances signal frequency against false positives.
    # At 1.5σ, expect signal ~13% of days (two-tailed normal CDF).
    entry_z_threshold: float = 1.5
    # 0.3σ — tight enough to capture mean reversion profit, loose enough
    # to avoid whipsaw on noise near zero. (Arch §7.1)
    exit_z_threshold: float = 0.3
    # 3.5σ — approximately 1-in-5000 occurrence under normality.
    # If Z extends this far, the model assumptions have broken and the
    # position should be stopped out regardless of conviction.
    stop_loss_z: float = 3.5
    # 30 days — mean reversion in FX typically runs its course within
    # 2-6 weeks (Lo and MacKinlay 1988). Beyond 30 days, the thesis
    # is stale and costs (swap, opportunity) accumulate. (Arch §7.1)
    max_holding_days: int = 30
    # 10% annualized — typical for a single strategy leg in a multi-strategy
    # portfolio. At 3-4 strategies with ~0.3 correlation, portfolio vol
    # targets 8-12%. (Arch §5.1)
    volatility_target: float = 0.10
    # 20% — prevents any single position from dominating portfolio.
    # Consistent with RiskManager per-symbol cap. (Arch §5.2)
    max_position_pct: float = 0.20
    # 0.10 — lowered from 0.25 because current live data has sparse
    # German yield data (27 rows). At R² < 0.10, the rate relationship
    # is effectively noise and trading would be gambling.
    min_r_squared: float = 0.10
    # 3600s = 1 hour — daily signals suffice for a mean reversion strategy
    # with 30-day holding period. Hourly check catches regime break early
    # without unnecessary compute. (Arch §7.1)
    signal_interval_seconds: int = 3600

    # ---- CL-x50g entry filters ------------------------------------------
    # Each filter is individually toggleable and gates NEW entries only —
    # exits/stops are never blocked. Defaults ON for new deployments;
    # disabling all three reproduces pre-CL-x50g behavior exactly.
    carry_filter_enabled: bool = True
    momentum_filter_enabled: bool = True
    regime_filter_enabled: bool = True
    # Carry proxy = carry_quote_rate_series MINUS carry_base_rate_series
    # (USD_3M_OIS - EUR_3M_ESTR_OIS for EURUSD). SIGN CONVENTION: a POSITIVE
    # US-minus-EUR short-rate spread favors USD strength → EURUSD downside,
    # so SHORT entries require carry >= 0 (carry must not oppose the short).
    # Mirror for longs: LONG entries require carry <= 0 (negative spread
    # favors EUR strength / EURUSD upside). Missing OIS data fails OPEN:
    # the filter passes with a WARNING logged once per gap, so a data
    # outage never silently disables the strategy.
    carry_quote_rate_series: str = "USD_3M_OIS"
    carry_base_rate_series: str = "EUR_3M_ESTR_OIS"
    # Short-horizon momentum from daily closes: p_t / p_{t-k} - 1. The
    # entry must not fade a strong move against it: entering SHORT requires
    # k-day momentum <= momentum_max_opposing; entering LONG requires
    # momentum >= -momentum_max_opposing. Default 0.0 = momentum must not
    # oppose the trade direction at all.
    momentum_lookback_days: int = 5
    momentum_max_opposing: float = 0.0
    # Regime filter: block entries when the vol-index z-score (vs its
    # rolling baseline, same definition as carry_vol_filter / CL-c77)
    # exceeds this. Mean-reversion entries in vol blowouts are the classic
    # whipsaw — 2.0σ keeps entries out of crisis regimes only.
    regime_max_vol_z: float = 2.0
    vol_index_series: str = "CVIX"
    vol_lookback_days: int = 120

    id: str = "eurusd_rate_diff_mr"


class RateDiffMRStrategy:
    def __init__(
        self,
        config: RateDiffMRConfig | None = None,
        data_provider: Any = None,
        state_store: Any = None,
        snapshot_store: FeatureSnapshotStore | None = None,
        liquidity_profile: LiquidityProfile | None = None,
    ) -> None:
        self.config = config or RateDiffMRConfig()
        self.data = data_provider
        self.state = state_store
        self.snapshot_store = snapshot_store
        # CL-y412: hour-of-week liquidity gate for NEW entries only. None =
        # inert passthrough until the monthly refresh writes a profile.
        self._liquidity_profile = liquidity_profile
        self._model: dict[str, Any] | None = None
        self._last_fit: datetime | None = None
        self._position_size: float = 0.0
        self._entry_z: float | None = None
        self._entry_ts: datetime | None = None
        # CL-z1: last known-good value of the CONFIGURED rate-spread series
        # (the one the model was fit on). Reused across a transient data gap
        # so a brief outage does not fabricate a spread the model never saw.
        self._last_spread: float | None = None
        # CL-x50g: True while an OIS data gap is active — the missing-carry
        # warning is logged once per gap (reset when data returns), not per
        # tick, so an outage doesn't flood the logs.
        self._carry_gap_active: bool = False

    def _emit_snapshot(self, values: dict[str, Any]) -> dict[str, Any]:
        """Build, persist, and return a snapshot-reference payload.

        Returns a dict suitable to attach as OrderIntent.metadata so the
        trade journal records the snapshot_id alongside the intent. Empty
        dict if no snapshot store is configured (graceful degradation:
        engine still trades, just without per-intent snapshot capture).
        """
        if self.snapshot_store is None:
            return {}
        model_version = (
            f"alpha={self._model['alpha']:.6f},beta={self._model['beta']:.6f},"
            f"std={self._model['residual_std']:.6f}"
            if self._model else "no_model"
        )
        snapshot = FeatureSnapshot.create(
            feature_set_name=_FEATURE_SET_NAME,
            feature_set_version=_FEATURE_SET_VERSION,
            data_snapshot_id="live",
            model_version=model_version,
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
        return [self.config.pair]

    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds

    def _refit_if_stale(self) -> None:
        now = datetime.now(UTC)
        if self._last_fit and (now - self._last_fit).days < 7:
            return
        if self.data is None:
            logger.debug("Model refit skipped — no DataProvider")
            return
        logger.info("Model refit starting (last=%s)", self._last_fit)
        try:
            # CL-gr8o follow-up: fit against the CONFIGURED spread series
            # (US2Y_MINUS_DE2Y in live config — now populated daily by
            # scripts/refresh_rates.py). The old hardcoded US_10Y/DE_10Y
            # legs were never ingested, so dropna() collapsed the fit to a
            # handful of stray rows (n=9, R²=0.235 < the 0.25 gate) and the
            # strategy stayed dormant even after the data gap was fixed.
            df = self.data.get_aligned_series(
                [self.config.pair, self.config.rate_spread_series],
                now - timedelta(days=self.config.lookback_days * 2), now,
            )
            if df is None or len(df) < 100:
                logger.warning("Model refit skipped — insufficient data (%d rows)", len(df) if df is not None else 0)
                return
            if self.config.rate_spread_series not in df.columns:
                logger.warning(
                    "Model refit skipped — spread series %s absent from "
                    "aligned data", self.config.rate_spread_series,
                )
                return
            import statsmodels.api as sm
            df = df.dropna().tail(self.config.lookback_days)
            df["spread"] = df[self.config.rate_spread_series]
            X = sm.add_constant(df[["spread"]])
            y = df[self.config.pair]
            ols = sm.OLS(y, X).fit()
            self._model = {
                "alpha": float(ols.params.iloc[0]),
                "beta": float(ols.params.iloc[1]),
                "r_squared": float(ols.rsquared),
                "residual_std": float(ols.resid.std()),
            }
            self._last_fit = now
            assert self._model["residual_std"] > 0, "zero residual std — degenerate model"
            logger.info("Model refit: R²=%.3f β=%.4f σ=%.5f n=%d",
                         self._model["r_squared"], self._model["beta"],
                         self._model["residual_std"], len(df))
        except Exception:
            # Broad by design: a failed refit keeps the previous model; the
            # tick must survive. Logged loudly with strategy context (CL-gmr1).
            logger.exception(
                "Model refit failed for %s (%s) — keeping previous model",
                self.id, self.config.pair,
            )

    def _compute_z_score(self, price: float, spread: float) -> float | None:
        if self._model is None:
            return None
        fair = self._model["alpha"] + self._model["beta"] * spread
        return float((price - fair) / self._model["residual_std"])

    def fit(self, train_data: pd.DataFrame) -> None:
        import statsmodels.api as sm
        df = train_data.dropna()
        if len(df) < 100:
            return
        df["spread"] = df.get(self.config.rate_spread_series, 0)
        X = sm.add_constant(df[["spread"]])
        y = df[self.config.pair]
        ols = sm.OLS(y, X).fit()
        self._model = {"alpha": float(ols.params.iloc[0]), "beta": float(ols.params.iloc[1]),
                        "r_squared": float(ols.rsquared), "residual_std": float(ols.resid.std())}
        self._last_fit = datetime.now(UTC)

    def _any_filter_enabled(self) -> bool:
        """CL-x50g: True when at least one entry filter is switched on."""
        c = self.config
        return bool(
            c.carry_filter_enabled
            or c.momentum_filter_enabled
            or c.regime_filter_enabled,
        )

    def _entry_filter_masks(
        self, df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """CL-x50g: per-row entry-permission masks for the backtest path.

        Returns ``(allow_long, allow_short)`` boolean arrays aligned to
        ``df``'s rows, or None when every filter is disabled (caller then
        skips filtering entirely — byte-identical to pre-filter behavior).

        NO LOOKAHEAD: every row's filter value uses data up to and
        including that row only — carry is the row's own OIS spread,
        momentum is ``p_t / p_{t-k} - 1``, and the vol z-score baseline is
        strictly prior rows (see :func:`rolling_vol_z`).

        Fail-open contract: rows where a filter's inputs are missing/NaN
        (data outage, warmup) PASS that filter. Missing OIS data logs a
        WARNING once per generate_signals call (one summary per gap, not
        one line per row).
        """
        if not self._any_filter_enabled():
            return None
        c = self.config
        n = len(df)
        allow_long = np.ones(n, dtype=bool)
        allow_short = np.ones(n, dtype=bool)

        if c.carry_filter_enabled:
            quote = df.get(c.carry_quote_rate_series)
            base = df.get(c.carry_base_rate_series)
            if quote is None or base is None:
                logger.warning(
                    "Carry filter: OIS series missing from backtest frame "
                    "(%s present=%s, %s present=%s) — filter passes (fail-open)",
                    c.carry_quote_rate_series, quote is not None,
                    c.carry_base_rate_series, base is not None,
                )
            else:
                carry = (
                    pd.to_numeric(quote, errors="coerce")
                    - pd.to_numeric(base, errors="coerce")
                )
                known = carry.notna().to_numpy()
                if not known.all():
                    logger.warning(
                        "Carry filter: %d/%d rows missing OIS data — those "
                        "rows pass (fail-open)",
                        int((~known).sum()), n,
                    )
                vals = carry.to_numpy(dtype=float)
                # Positive US-minus-EUR spread favors USD strength → pair
                # downside: shorts need carry >= 0, longs need carry <= 0.
                allow_short &= ~known | (vals >= 0.0)
                allow_long &= ~known | (vals <= 0.0)

        if c.momentum_filter_enabled:
            if c.pair not in df.columns:
                logger.warning(
                    "Momentum filter: close column %s missing from backtest "
                    "frame — filter passes (fail-open)", c.pair,
                )
            else:
                closes = pd.to_numeric(df[c.pair], errors="coerce")
                mom = closes.pct_change(
                    periods=c.momentum_lookback_days, fill_method=None,
                )
                known = mom.notna().to_numpy()
                vals = mom.to_numpy(dtype=float)
                # Don't fade a strong move against the trade: shorts need
                # momentum <= max_opposing, longs need >= -max_opposing.
                allow_short &= ~known | (vals <= c.momentum_max_opposing)
                allow_long &= ~known | (vals >= -c.momentum_max_opposing)

        if c.regime_filter_enabled:
            vol = df.get(c.vol_index_series)
            if vol is None:
                logger.debug(
                    "Regime filter: vol series %s missing from backtest "
                    "frame — filter passes (fail-open)", c.vol_index_series,
                )
            else:
                vol_z = rolling_vol_z(
                    pd.to_numeric(vol, errors="coerce"), c.vol_lookback_days,
                )
                known = vol_z.notna().to_numpy()
                vals = vol_z.to_numpy(dtype=float)
                # Direction-agnostic: vol blowouts whipsaw both sides.
                ok = ~known | (vals <= c.regime_max_vol_z)
                allow_short &= ok
                allow_long &= ok

        return allow_long, allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self._model is None:
            self.fit(data)
        if self._model is None:
            return pd.Series(0.0, index=data.index)
        df = data.copy()
        df["spread"] = df.get(self.config.rate_spread_series, 0)
        df["fair"] = self._model["alpha"] + self._model["beta"] * df["spread"]
        df["z"] = (df[self.config.pair] - df["fair"]) / self._model["residual_std"]
        # CL-x50g: entry filters gate NEW entries only; None = all disabled.
        masks = self._entry_filter_masks(df)
        positions = []
        pos = 0.0
        entry_i = 0
        for i, z in enumerate(df["z"]):
            if abs(pos) < 0.001:
                if z < -self.config.entry_z_threshold:
                    if masks is None or masks[0][i]:
                        pos = 1.0
                        entry_i = i
                elif z > self.config.entry_z_threshold and (
                    masks is None or masks[1][i]
                ):
                    pos = -1.0
                    entry_i = i
            else:
                days = i - entry_i
                stop = (pos > 0 and z < -self.config.stop_loss_z) or (pos < 0 and z > self.config.stop_loss_z)
                exit_ok = (pos > 0 and z >= -self.config.exit_z_threshold) or (pos < 0 and z <= self.config.exit_z_threshold)
                if stop or exit_ok or days > self.config.max_holding_days:
                    pos = 0.0
            positions.append(pos)
        return pd.Series(positions, index=df.index, dtype=float)

    def _compute_live_momentum(self, now: datetime) -> float | None:
        """CL-x50g: point-in-time k-day close momentum for the live path.

        ``p_t / p_{t-k} - 1`` over the last ``momentum_lookback_days``
        trading rows from get_aligned_series. None when the provider is
        missing or history is too short (caller fails open).
        """
        if self.data is None:
            return None
        lookback = self.config.momentum_lookback_days
        # 3× calendar buffer + slack covers weekends/holidays for k trading rows.
        start = now - timedelta(days=lookback * 3 + 5)
        try:
            df = self.data.get_aligned_series([self.config.pair], start, now)
        except Exception as exc:
            logger.warning(
                "Momentum filter: get_aligned_series failed for %s: %s: %s",
                self.config.pair, type(exc).__name__, exc,
            )
            return None
        if df is None or self.config.pair not in df.columns:
            return None
        closes = df[self.config.pair].dropna()
        if len(closes) < lookback + 1:
            return None
        return float(closes.iloc[-1] / closes.iloc[-(lookback + 1)] - 1.0)

    def _evaluate_entry_filters_live(
        self, direction: int, now: datetime,
    ) -> tuple[bool, dict[str, Any]]:
        """CL-x50g: point-in-time entry-filter evaluation for the live path.

        ``direction`` is +1 (long the pair) or -1 (short). Returns
        ``(allowed, diagnostics)``; diagnostics records per-filter
        enabled/passed/value and is attached to the intent snapshot so the
        journal shows exactly which filters blocked or passed.

        Fail-open contract: missing data passes the filter. Missing OIS
        data logs a WARNING once per gap (``_carry_gap_active``), so a
        data outage never silently disables the strategy — nor floods logs.
        """
        c = self.config
        diag: dict[str, Any] = {}
        allowed = True

        # Carry: positive quote-minus-base (US-minus-EUR) short-rate spread
        # favors USD strength → EURUSD downside. Shorts need carry >= 0,
        # longs need carry <= 0 (carry must not oppose the trade).
        if c.carry_filter_enabled:
            quote = base = None
            if self.data is not None:
                try:
                    quote = self.data.get_latest_value(c.carry_quote_rate_series, now)
                    base = self.data.get_latest_value(c.carry_base_rate_series, now)
                except Exception as exc:
                    # Broad by design: a rate-lookup failure is handled as a
                    # data gap below (which warns once, latched) — but keep
                    # the underlying error visible for diagnosis (CL-gmr1).
                    logger.debug(
                        "%s: carry rate lookup failed: %s: %s",
                        self.id, type(exc).__name__, exc,
                    )
                    quote = base = None  # treated as a data gap below
            carry: float | None = None
            passed = True
            if quote is None or base is None:
                if not self._carry_gap_active:
                    logger.warning(
                        "Carry filter: OIS data unavailable (%s=%s, %s=%s) — "
                        "filter passes (fail-open); suppressing repeat "
                        "warnings until data returns",
                        c.carry_quote_rate_series, quote,
                        c.carry_base_rate_series, base,
                    )
                    self._carry_gap_active = True
            else:
                if self._carry_gap_active:
                    logger.info("Carry filter: OIS data restored")
                    self._carry_gap_active = False
                carry = float(quote) - float(base)
                passed = carry >= 0.0 if direction < 0 else carry <= 0.0
            diag["carry"] = {"enabled": True, "passed": passed, "value": carry}
            allowed = allowed and passed
        else:
            diag["carry"] = {"enabled": False, "passed": True, "value": None}

        # Momentum: don't fade a strong move against the trade.
        if c.momentum_filter_enabled:
            mom = self._compute_live_momentum(now)
            if mom is None:
                passed = True  # fail-open on missing history
            elif direction < 0:
                passed = mom <= c.momentum_max_opposing
            else:
                passed = mom >= -c.momentum_max_opposing
            diag["momentum"] = {"enabled": True, "passed": passed, "value": mom}
            allowed = allowed and passed
        else:
            diag["momentum"] = {"enabled": False, "passed": True, "value": None}

        # Regime: no fresh mean-reversion entries into a vol blowout.
        if c.regime_filter_enabled:
            vol_z = compute_vol_z_score(
                self.data, c.vol_index_series, c.vol_lookback_days, now,
            )
            passed = vol_z <= c.regime_max_vol_z
            diag["regime"] = {
                "enabled": True, "passed": passed, "value": float(vol_z),
            }
            allowed = allowed and passed
        else:
            diag["regime"] = {"enabled": False, "passed": True, "value": None}

        return allowed, diag

    async def generate_intents(
        self, prices: dict[str, Any], broker: Any,
    ) -> list[OrderIntent]:
        self._refit_if_stale()
        if self._model is None or self._model["r_squared"] < self.config.min_r_squared:
            r2 = self._model["r_squared"] if self._model else 0
            logger.debug("Quality gate: R²=%.3f below threshold %.3f, sitting out",
                          r2, self.config.min_r_squared)
            if self._position_size != 0:
                logger.info("Flattening — model quality degraded below gate")
                self._position_size = 0
                meta = self._emit_snapshot({
                    "trigger": "model_quality_degraded",
                    "r_squared": float(r2),
                    "min_r_squared": self.config.min_r_squared,
                })
                return [OrderIntent(
                    strategy_id=self.id, symbol=self.config.pair,
                    target_position=0, metadata=meta,
                )]
            return []

        tick = prices.get(self.config.pair)
        if tick is None:
            logger.debug("No tick for %s", self.config.pair)
            return []

        current_price = (tick["bid"] + tick["ask"]) / 2
        assert current_price > 0, f"non-positive price {current_price}"

        # CL-z1 (P0): the live spread MUST be the SAME series the model was
        # fit on (config.rate_spread_series). The old code hardcoded
        # US_10Y-DE_10Y — a series that is not even ingested — so it always
        # fell through to a constant 0.5, and z was computed against a
        # fabricated spread the model never learned: every entry/exit/stop
        # fired on a residual disconnected from the actual rate differential.
        series_id = self.config.rate_spread_series
        now = datetime.now(UTC)
        current_spread: float | None = None
        if self.data:
            try:
                spread_df = self.data.get_aligned_series(
                    [series_id], now - timedelta(days=5), now,
                )
                if spread_df is not None and series_id in spread_df.columns:
                    col = spread_df[series_id].dropna()
                    if len(col) > 0:
                        current_spread = float(col.iloc[-1])
            except Exception as exc:
                # Broad by design: the tick must survive a data-provider
                # failure. warning w/o traceback per CL-2yta.
                logger.warning(
                    "%s: spread query for %s failed (%s: %s)",
                    self.id, series_id, type(exc).__name__, exc,
                )
        if current_spread is None:
            # Data gap: reuse the last known-good value rather than fabricate.
            current_spread = self._last_spread
        if current_spread is None:
            # No live value AND no cache (cold start / persistent outage).
            # Fail CLOSED — skip the tick rather than trade on a made-up
            # spread. There is no open position to strand on a true cold
            # start (this strategy does not persist size across restart).
            logger.warning(
                "%s: no live value for %s and no cached spread — skipping "
                "tick (refusing to fabricate the model's input)",
                self.id, series_id,
            )
            return []
        self._last_spread = current_spread

        z = self._compute_z_score(current_price, current_spread)
        if z is None:
            logger.debug("Z-score unavailable")
            return []

        from src.monitoring.metrics import signal_z_score, signals_generated

        signal_z_score.labels(strategy_id=self.id, pair=self.config.pair).set(z)
        signals_generated.labels(strategy_id=self.id, action="evaluate").inc()

        logger.debug("Z=%.2f price=%.5f spread=%.4f pos=%.0f", z, current_price, current_spread, self._position_size)

        # Exit check
        if abs(self._position_size) > 0.001:
            should_exit = False
            exit_reason = ""
            if self._position_size > 0:
                if z >= -self.config.exit_z_threshold:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z < -self.config.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:
                if z <= self.config.exit_z_threshold:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z > self.config.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
            if not should_exit and self._entry_ts and \
               (datetime.now(UTC) - self._entry_ts).days > self.config.max_holding_days:
                should_exit = True
                exit_reason = "timeout"
            if should_exit:
                assert self._entry_z is not None
                logger.info("Exit %s: entry_z=%.2f exit_z=%.2f pos=%.0f reason=%s",
                            self.config.pair, self._entry_z, z, self._position_size, exit_reason)
                meta = self._emit_snapshot({
                    "trigger": "exit",
                    "exit_reason": exit_reason,
                    "z": float(z),
                    "entry_z": float(self._entry_z),
                    "price": float(current_price),
                    "spread": float(current_spread),
                })
                self._position_size = 0.0
                self._entry_z = None
                self._entry_ts = None
                signals_generated.labels(strategy_id=self.id, action="exit").inc()
                return [OrderIntent(
                    strategy_id=self.id, symbol=self.config.pair,
                    target_position=0, metadata=meta,
                )]
            return []

        # Entry check
        if abs(z) >= self.config.entry_z_threshold:
            assert self._model is not None
            assert self._position_size == 0.0, f"entry with existing pos {self._position_size}"
            direction = -1 if z > 0 else 1
            # CL-x50g: entry filters gate NEW entries only — exits above are
            # never blocked. Snapshot records which filters blocked/passed.
            filters_ok, filter_diag = self._evaluate_entry_filters_live(
                direction, datetime.now(UTC),
            )
            if not filters_ok:
                blocked = [
                    name for name, d in filter_diag.items()
                    if d["enabled"] and not d["passed"]
                ]
                logger.info(
                    "Entry blocked by filters %s: z=%.2f dir=%d",
                    blocked, z, direction,
                )
                signals_generated.labels(
                    strategy_id=self.id, action="entry_blocked",
                ).inc()
                self._emit_snapshot({
                    "trigger": "entry_blocked",
                    "z": float(z),
                    "direction": int(direction),
                    "price": float(current_price),
                    "spread": float(current_spread),
                    "filters": filter_diag,
                })
                return []
            account = broker.get_account()
            equity = account.equity
            assert equity > 0, f"non-positive equity {equity}"
            # 0.08 is the default realized vol placeholder when feature data is
            # unavailable. In production, this should be replaced with
            # DataProvider.get_realized_vol(pair, window=20). (Arch §5.1)
            vol = 0.08
            notional = equity * self.config.volatility_target / max(vol, 0.01)
            notional = min(notional, equity * self.config.max_position_pct)
            size = direction * notional / current_price

            # CL-y412: liquidity-window gate — NEW entries only (the exit
            # branch above returns before reaching here). A wide spread in a
            # dead window blocks the entry (0.0×) or halves it (0.5×). The
            # sign is preserved because adjust_for_liquidity scales magnitude.
            if self._liquidity_profile is not None:
                spread_bps = spread_bps_from_tick(tick)
                if spread_bps is not None:
                    liq_size = PositionSizer.adjust_for_liquidity(
                        size, self.config.pair, datetime.now(UTC),
                        spread_bps, self._liquidity_profile,
                    )
                    if liq_size == 0.0:
                        logger.info(
                            "Entry blocked by dead liquidity window: "
                            "z=%.2f dir=%d spread=%.1fbps",
                            z, direction, spread_bps,
                        )
                        signals_generated.labels(
                            strategy_id=self.id, action="entry_blocked",
                        ).inc()
                        self._emit_snapshot({
                            "trigger": "entry_blocked",
                            "z": float(z),
                            "direction": int(direction),
                            "price": float(current_price),
                            "spread": float(current_spread),
                            "liquidity_spread_bps": float(spread_bps),
                        })
                        return []
                    size = liq_size

            self._position_size = size
            self._entry_z = z
            self._entry_ts = datetime.now(UTC)
            logger.info("Entry: z=%.2f dir=%d size=%.0f", z, direction, size)
            signals_generated.labels(strategy_id=self.id, action="entry").inc()
            meta = self._emit_snapshot({
                "trigger": "entry",
                "z": float(z),
                "direction": int(direction),
                "price": float(current_price),
                "spread": float(current_spread),
                "vol": float(vol),
                "size": float(size),
                "entry_z_threshold": self.config.entry_z_threshold,
                # CL-x50g: per-filter enabled/passed/value at entry time.
                "filters": filter_diag,
            })
            return [OrderIntent(
                strategy_id=self.id, symbol=self.config.pair,
                target_position=size, metadata=meta,
            )]

        return []
