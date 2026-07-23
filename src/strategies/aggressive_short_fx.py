"""Aggressive short-bias FX strategy (CL-agg-fx-short).

Conviction-weighted multi-signal strategy. Each bar, we compute three
independent direction signals:

  1. Rate-diff Z-score  — reuses the existing RateDiffMRStrategy math.
                          A positive Z (FX above fair value) → SHORT
                          signal. The conservative threshold is 1.5σ;
                          aggressive_short reads from the risk_profile
                          and uses whatever ``entry_z_threshold`` is
                          configured (default 1.0 under aggressive).
  2. Sentiment delta    — central-bank statement net-shift. Hawkish
                          shift on the foreign CB → that currency
                          strengthens → SHORT the pair (USDxxx) or
                          LONG (xxxUSD); we generate {-1, +1} signed
                          to the pair direction.
  3. POLY implied-prob  — Polymarket market on a binary outcome
                          relevant to the pair (e.g. "Fed cuts by Y"
                          → DXY/EURUSD direction). Deviation from
                          the recent mean ± 3pp triggers a signal.

A position only fires when **at least 2 of 3 signals agree on direction**.
Per the bias config (aggressive_short ⇒ ``long_signal_multiplier=0.0``,
``short_signal_multiplier=1.0``), long positions are filtered to zero —
only short signals make it through.

Holding is capped at ``risk_profile.holding.max_holding_days`` (5 under
aggressive). Stop-loss + exit are standard mean-reversion: exit when
z reverts toward 0, hard stop at ``stop_loss_z``.

Reference: CL-eft9. Reads risk profile via
``src.risk.risk_profile.load_active_profile()`` at construction; no
strategy code changes when the profile flips.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.risk.risk_profile import RiskProfile, load_active_profile

logger = logging.getLogger(__name__)


# Default signal-agreement quorum. 2-of-3 = strong conviction without
# requiring perfect consensus. 3-of-3 would be too restrictive (signals
# rarely all align); 1-of-3 isn't conviction-weighted at all.
_DEFAULT_AGREEMENT_QUORUM: int = 2


@dataclass
class AggressiveShortFXConfig:
    """Per-strategy config. Most values flow through from the active
    risk profile (entry_z_threshold, max_holding_days, sizing); only
    strategy-specific knobs live here."""

    pair: str = "EURUSD"
    rate_spread_series: str = "US10Y_MINUS_DE10Y"
    # Z-score window for FX-vs-spread fair value. 252 daily = 1y.
    lookback_days: int = 252
    # 2-of-3 quorum. 3-of-3 too restrictive, 1-of-3 not conviction-weighted.
    agreement_quorum: int = _DEFAULT_AGREEMENT_QUORUM
    # Sentiment delta polarity: which CB shift means SHORT the pair?
    # For EURUSD, hawkish ECB (foreign) → EUR strengthens → LONG EURUSD;
    # hawkish Fed (domestic USD) → USD strengthens → SHORT EURUSD.
    # The operator picks which CB's hawkish shift implies SHORT.
    sentiment_short_on: str = "fed"  # hawkish fed shift → short EURUSD
    # POLY: symbol providing the implied-probability feature. None
    # disables the third signal (strategy falls back to 2-signal
    # quorum, requiring both to agree).
    poly_symbol: str | None = None
    # Deviation threshold in probability units (3pp = 0.03).
    poly_deviation_threshold: float = 0.03
    poly_lookback_days: int = 14
    # Identity for the strategy state store + journal.
    id: str = "aggressive_short_fx"
    signal_interval_seconds: int = 3600

    # Strategy-level risk profile — defaults to whatever
    # CURLIT_RISK_PROFILE selects at boot. Caller can override at
    # construction for backtest sweeps.
    risk_profile: RiskProfile | None = field(default=None)


@dataclass
class _SignalTriplet:
    """One bar's three signal readings, all in {-1, 0, +1}.
    -1 = short, 0 = neutral, +1 = long. The strategy sums these and
    applies the bias multiplier before deciding to enter."""

    rate_diff: int = 0
    sentiment: int = 0
    poly: int = 0

    def agreement(self) -> int:
        """Returns the dominant direction: -1, 0, or +1.
        Tie or no signal → 0. 2-of-3 short → -1. 2-of-3 long → +1."""
        s = self.rate_diff + self.sentiment + self.poly
        # Treat poly_disabled (=0) as silent. With a 0 contribution,
        # 2-of-2 still counts: rate_diff + sentiment summing to ±2 wins.
        if s <= -2:
            return -1
        if s >= 2:
            return 1
        return 0


class AggressiveShortFXStrategy:
    """Multi-signal short-bias FX strategy.

    Strategy lifecycle:
      construct → fit(train_data) → generate_signals(test_data)

    fit() refits the linear rate-diff model on the train window.
    generate_signals() walks the test bars, computes each signal, and
    emits positions that respect the risk profile's bias multipliers.
    """

    def __init__(
        self,
        config: AggressiveShortFXConfig | None = None,
        data_provider: Any | None = None,
        nlp_provider: Any | None = None,
        state_store: Any | None = None,
    ) -> None:
        self.config = config or AggressiveShortFXConfig()
        self.data = data_provider
        self.nlp = nlp_provider
        self.state = state_store
        # Risk profile: explicit > config > load active.
        self.risk = self.config.risk_profile or load_active_profile()
        self._model: dict[str, float] | None = None
        # Cache the entry threshold + max holding from the profile so
        # we don't re-read it on every bar.
        self._z_entry: float = self.risk.strategy_gates.entry_z_threshold
        self._z_stop: float = self._z_entry * 2.5  # 2.5x entry = stop
        self._z_exit: float = 0.3
        self._max_hold: int = self.risk.holding.max_holding_days
        logger.info(
            "AggressiveShortFXStrategy init: profile=%s z_entry=%.2f max_hold=%d quorum=%d/3",
            self.risk.name,
            self._z_entry,
            self._max_hold,
            self.config.agreement_quorum,
        )

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        out = [self.config.pair]
        if self.config.poly_symbol:
            out.append(self.config.poly_symbol)
        return out

    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds

    # ----------------------------------------------------------------- #
    # Walk-forward harness contract: fit + generate_signals
    # ----------------------------------------------------------------- #

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the rate-diff linear model on the train window. Re-uses
        the same statsmodels OLS as RateDiffMRStrategy — partial pooling
        across G10 (CL-l0x) is a strict generalization we don't reach
        for here because aggressive_short is a single-pair runner."""
        try:
            import statsmodels.api as sm
        except ImportError:
            logger.warning("statsmodels not installed — skipping fit")
            return
        df = train_data.dropna()
        if len(df) < 100:
            logger.warning(
                "fit skipped: only %d rows, need >=100",
                len(df),
            )
            return
        df = df.tail(self.config.lookback_days).copy()
        df["spread"] = df.get(self.config.rate_spread_series, 0)
        x = sm.add_constant(df[["spread"]])
        y = df[self.config.pair]
        ols = sm.OLS(y, x).fit()
        self._model = {
            "alpha": float(ols.params.iloc[0]),
            "beta": float(ols.params.iloc[1]),
            "r_squared": float(ols.rsquared),
            "residual_std": float(ols.resid.std()),
        }
        logger.debug(
            "fit: R²=%.3f β=%.4f σ=%.4f n=%d",
            self._model["r_squared"],
            self._model["beta"],
            self._model["residual_std"],
            len(df),
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Per-bar position series in {-1, 0}. Long signals are
        filtered to zero by the aggressive_short bias multipliers; only
        shorts make it through. Other profiles let long signals through
        (long_signal_multiplier=1.0 in conservative/aggressive)."""
        if self._model is None:
            self.fit(data)
        if self._model is None:
            return pd.Series(0.0, index=data.index)

        # Cache the bias multipliers locally for the hot loop.
        long_mult = self.risk.bias.long_signal_multiplier
        short_mult = self.risk.bias.short_signal_multiplier
        gate_r2 = self.risk.strategy_gates.min_r_squared

        if self._model["r_squared"] < gate_r2:
            logger.info(
                "R²=%.3f below profile gate %.3f — sitting out",
                self._model["r_squared"],
                gate_r2,
            )
            return pd.Series(0.0, index=data.index)

        df = data.copy()
        df["spread"] = df.get(self.config.rate_spread_series, 0)
        df["fair"] = self._model["alpha"] + self._model["beta"] * df["spread"]
        df["z"] = (df[self.config.pair] - df["fair"]) / self._model["residual_std"]

        # Precompute the three signals as int Series.
        rd_signal = self._rate_diff_signal(df["z"])
        sn_signal = self._sentiment_signal(df.index)
        poly_signal = self._poly_signal(df)

        positions: list[float] = []
        pos = 0.0
        entry_idx = 0
        for i, z in enumerate(df["z"]):
            triplet = _SignalTriplet(
                rate_diff=int(rd_signal.iat[i]) if i < len(rd_signal) else 0,
                sentiment=int(sn_signal.iat[i]) if i < len(sn_signal) else 0,
                poly=int(poly_signal.iat[i]) if i < len(poly_signal) else 0,
            )
            agree = triplet.agreement()

            # Apply bias multiplier. Long is filtered to 0 under
            # aggressive_short. Short is passed through scaled.
            if agree > 0:
                desired = long_mult
            elif agree < 0:
                desired = -short_mult
            else:
                desired = 0.0

            # Manage open position separately from new-entry logic so
            # that an aggressive_short flatten-of-long doesn't crash
            # mid-fold during a profile flip.
            if abs(pos) < 0.001:
                # No position — open one if the quorum + bias agree.
                if desired != 0:
                    pos = desired
                    entry_idx = i
            else:
                days_held = i - entry_idx
                # Stop-loss: z runs further against us than z_stop.
                stop = (pos > 0 and z < -self._z_stop) or (pos < 0 and z > self._z_stop)
                # Exit: z reverts toward zero past the exit threshold.
                exit_ok = (pos > 0 and z >= -self._z_exit) or (pos < 0 and z <= self._z_exit)
                if stop or exit_ok or days_held >= self._max_hold:
                    pos = 0.0
            positions.append(pos)

        return pd.Series(positions, index=df.index, dtype=float)

    # ----------------------------------------------------------------- #
    # Per-signal helpers
    # ----------------------------------------------------------------- #

    def _rate_diff_signal(self, z: pd.Series) -> pd.Series:
        """+1 when FX is below fair value (z < -entry → long),
        -1 when above (z > +entry → short), 0 otherwise."""
        out = pd.Series(0, index=z.index, dtype=int)
        out.loc[z < -self._z_entry] = 1
        out.loc[z > self._z_entry] = -1
        return out

    def _sentiment_signal(self, idx: pd.Index) -> pd.Series:
        """Read recent CB-statement diff events from NLPDataProvider.

        A hawkish shift on ``config.sentiment_short_on`` central bank
        means SHORT the pair (the domestic currency strengthens against
        the foreign).

        Returns 0 across the index when no NLP provider is wired —
        graceful degradation. The strategy still functions on just the
        rate-diff + POLY signals if 2-of-2 agree.
        """
        out = pd.Series(0, index=idx, dtype=int)
        if self.nlp is None:
            return out
        try:
            target_cb = self.config.sentiment_short_on
            # Pull recent diff events from the NLP provider — convention
            # established in CL-cb-sentiment.
            events = self.nlp.get_recent_diff_events(
                since=idx[0].to_pydatetime() if hasattr(idx[0], "to_pydatetime") else idx[0],
                cbs=[target_cb],
            )
        except Exception as exc:
            # Broad by design: signal generation must survive a provider
            # failure, but a dead NLP feed is not "no sentiment" — log it
            # visibly (CL-gmr1). warning without traceback per CL-2yta.
            logger.warning(
                "%s: sentiment signal NLP provider failed (%s: %s) — sentiment leg contributes 0",
                self.id,
                type(exc).__name__,
                exc,
            )
            return out
        if not events:
            return out
        # Net shift > 0.5 = hawkish; < -0.5 = dovish. Pin signals to
        # the bar nearest each event timestamp.
        for ev in events:
            ev_ts = ev.get("ts")
            if ev_ts is None:
                continue
            net_shift = float(ev.get("net_shift", 0))
            if abs(net_shift) < 0.5:
                continue
            # Find the closest index point to ev_ts.
            try:
                ev_ts_pd = pd.Timestamp(ev_ts)
                pos = out.index.get_indexer([ev_ts_pd], method="nearest")[0]
            except (ValueError, TypeError):
                # Narrowed from bare `except Exception` (CL-gmr1): the intent
                # here is strictly "unparsable event timestamp / non-monotonic
                # index" — anything else should surface.
                logger.debug(
                    "%s: skipping sentiment event with unusable ts %r",
                    self.id,
                    ev_ts,
                )
                continue
            if 0 <= pos < len(out):
                # Hawkish CB ⇒ short pair (-1); dovish ⇒ long (+1).
                out.iat[pos] = -1 if net_shift > 0 else 1
        return out

    def _poly_signal(self, df: pd.DataFrame) -> pd.Series:
        """Polymarket implied-probability deviation. Reads the POLY:
        column from the panel and emits a signal when it moves more
        than ``poly_deviation_threshold`` away from its trailing mean.

        Direction convention: rising implied prob on a "Fed cuts"
        market ⇒ USD weakens (Fed cut = dovish) ⇒ LONG EURUSD. We
        encode that in the sign here; operator overrides per-pair
        via config if the polarity reverses.
        """
        out = pd.Series(0, index=df.index, dtype=int)
        sym = self.config.poly_symbol
        if not sym or sym not in df.columns:
            return out
        series = df[sym].astype(float)
        rolling_mean = series.rolling(self.config.poly_lookback_days).mean()
        deviation = series - rolling_mean
        threshold = self.config.poly_deviation_threshold
        out.loc[deviation > threshold] = 1
        out.loc[deviation < -threshold] = -1
        return out
