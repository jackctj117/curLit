"""Strategy 1: Rate differential mean reversion on EUR/USD."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.execution.oms import OrderIntent
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
)

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
    id: str = "eurusd_rate_diff_mr"


class RateDiffMRStrategy:
    def __init__(
        self,
        config: RateDiffMRConfig = None,
        data_provider=None,
        state_store=None,
        snapshot_store: FeatureSnapshotStore | None = None,
    ) -> None:
        self.config = config or RateDiffMRConfig()
        self.data = data_provider
        self.state = state_store
        self.snapshot_store = snapshot_store
        self._model: dict | None = None
        self._last_fit: datetime | None = None
        self._position_size: float = 0.0
        self._entry_z: float | None = None
        self._entry_ts: datetime | None = None

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
            df = self.data.get_aligned_series(
                [self.config.pair, "US_10Y", "DE_10Y"],
                now - timedelta(days=self.config.lookback_days * 2), now,
            )
            if df is None or len(df) < 100:
                logger.warning("Model refit skipped — insufficient data (%d rows)", len(df) if df is not None else 0)
                return
            import statsmodels.api as sm
            df = df.dropna().tail(self.config.lookback_days)
            df["spread"] = df.get("US_10Y", 0) - df.get("DE_10Y", 0)
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
            logger.exception("Model refit failed")

    def _compute_z_score(self, price: float, spread: float) -> float | None:
        if self._model is None:
            return None
        fair = self._model["alpha"] + self._model["beta"] * spread
        return (price - fair) / self._model["residual_std"]

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

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self._model is None:
            self.fit(data)
        if self._model is None:
            return pd.Series(0.0, index=data.index)
        df = data.copy()
        df["spread"] = df.get(self.config.rate_spread_series, 0)
        df["fair"] = self._model["alpha"] + self._model["beta"] * df["spread"]
        df["z"] = (df[self.config.pair] - df["fair"]) / self._model["residual_std"]
        positions = []
        pos = 0.0
        entry_i = 0
        for i, z in enumerate(df["z"]):
            if abs(pos) < 0.001:
                if z < -self.config.entry_z_threshold:
                    pos = 1.0
                    entry_i = i
                elif z > self.config.entry_z_threshold:
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

    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
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

        current_spread = 0.5  # fallback placeholder
        if self.data:
            try:
                spread_data = self.data.get_aligned_series(
                    ["US_10Y", "DE_10Y"],
                    datetime.now(UTC) - timedelta(days=5), datetime.now(UTC),
                )
                if spread_data is not None and len(spread_data) > 0:
                    current_spread = float(spread_data["US_10Y"].iloc[-1] - spread_data["DE_10Y"].iloc[-1])
            except Exception:
                logger.debug("Spread query failed — using fallback 0.5")

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
            })
            return [OrderIntent(
                strategy_id=self.id, symbol=self.config.pair,
                target_position=size, metadata=meta,
            )]

        return []
