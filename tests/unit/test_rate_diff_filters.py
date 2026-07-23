"""Tests — CL-x50g carry/momentum/regime entry filters on RateDiffMRStrategy.

Coverage:
    - disabled filters reproduce byte-identical signals to the legacy loop
    - each filter individually blocks / allows entries (both directions)
    - filters gate NEW entries only — exits are never blocked
    - no lookahead: signals at row t are unchanged by rows > t
    - missing OIS data fails OPEN with a warning (once per gap, live path)
    - live path (generate_intents) blocks entries and records per-filter
      pass/block diagnostics in the snapshot metadata

Fixture trick: with ``_model = {alpha: 0, beta: 1, residual_std: 1}`` the
z-score is ``price - spread_series``, so the spread column steers z to any
target while the price column independently steers momentum.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.execution.broker import Position
from src.risk.liquidity_window import LiquidityProfile
from src.strategies.rate_diff_mean_reversion import (
    RateDiffMRConfig,
    RateDiffMRStrategy,
)


class _PosBroker:
    """Broker double exposing get_positions for the CL-0h30 sync."""

    def __init__(self, positions: list[Position]) -> None:
        self._positions = positions

    def get_positions(self) -> list[Position]:
        return self._positions

    def get_account(self) -> Any:
        return SimpleNamespace(equity=100_000.0)


class _LogStore:
    """Mirrors StrategyStateStore's append-log semantics: last row wins,
    'entry' = open, 'exit' = flat."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, dict[str, Any]]] = []

    def record_entry(self, sid: str, ts: Any, signal: dict[str, Any], size: float) -> None:
        self._rows.append(("entry", {**signal, "size": size}))

    def record_exit(self, sid: str, ts: Any, reason: str) -> None:
        self._rows.append(("exit", {"reason": reason}))

    def get_current_position(self, sid: str) -> dict[str, Any] | None:
        if not self._rows or self._rows[-1][0] == "exit":
            return None
        return self._rows[-1][1]


class TestPositionPersistence:
    """CL-bccy (P0): rate-diff persists its position to the durable store so
    the cold-start reconciler sees it after a restart and doesn't flatten the
    live broker leg as an orphan."""

    def test_position_persisted_for_cold_start(self) -> None:
        store = _LogStore()
        s = RateDiffMRStrategy(_config(), state_store=store)
        s._position_size = -1000.0
        s._persist_position("entry")
        pos = store.get_current_position(s.id)
        assert pos is not None
        assert pos["symbol"] == "EURUSD"
        assert pos["size"] == -1000.0  # reconciler reads symbol + size

    def test_flat_persisted_as_exit(self) -> None:
        store = _LogStore()
        s = RateDiffMRStrategy(_config(), state_store=store)
        s._position_size = 1000.0
        s._persist_position("entry")
        s._position_size = 0.0
        s._persist_position("mean_reversion")
        assert store.get_current_position(s.id) is None  # reconciler sees flat

    def test_persist_is_best_effort_without_store(self) -> None:
        s = RateDiffMRStrategy(_config(), state_store=None)
        s._position_size = 1000.0
        s._persist_position("entry")  # must not raise


class TestBrokerPositionSync:
    """CL-0h30 (P0): the strategy's belief is reconciled with the broker at the
    start of each tick, so a rejected entry/exit self-heals instead of leaving
    a phantom (or unstopped risk)."""

    def test_rejected_entry_self_heals_to_flat(self) -> None:
        s = RateDiffMRStrategy(_config())
        s._position_size = 1000.0   # thinks it's long (entry was rejected)
        s._entry_z = 5.0
        s._entry_ts = datetime.now(UTC)
        s._sync_position_from_broker(_PosBroker([]))  # broker actually flat
        assert s._position_size == 0.0
        assert s._entry_z is None and s._entry_ts is None

    def test_rejected_exit_adopts_live_broker_position(self) -> None:
        s = RateDiffMRStrategy(_config())
        s._position_size = 0.0      # thinks it's flat (exit was rejected)
        s._sync_position_from_broker(
            _PosBroker([Position("EURUSD", -500.0, 1.10)]),
        )
        assert s._position_size == -500.0     # adopts the live leg
        assert s._entry_ts is not None        # stamped so the time stop works
        assert s._entry_z == 0.0

    def test_sync_noop_when_in_agreement(self) -> None:
        s = RateDiffMRStrategy(_config())
        s._position_size = -500.0
        s._entry_z = 3.0
        s._sync_position_from_broker(
            _PosBroker([Position("EURUSD", -500.0, 1.10)]),
        )
        assert s._position_size == -500.0
        assert s._entry_z == 3.0  # untouched — nothing to reconcile

    def test_sync_skipped_when_broker_has_no_get_positions(self) -> None:
        # Best-effort: a broker read failure must not crash the tick.
        class _NoPos:
            pass
        s = RateDiffMRStrategy(_config())
        s._position_size = 1000.0
        s._sync_position_from_broker(_NoPos())  # AttributeError → swallowed
        assert s._position_size == 1000.0  # belief unchanged

PAIR = "EURUSD"
SPREAD = "US10Y_MINUS_DE10Y"
USD_OIS = "USD_3M_OIS"
EUR_OIS = "EUR_3M_ESTR_OIS"

_MODEL = {"alpha": 0.0, "beta": 1.0, "r_squared": 0.9, "residual_std": 1.0}


def _config(**overrides: Any) -> RateDiffMRConfig:
    """Config with all CL-x50g filters OFF unless overridden."""
    defaults: dict[str, Any] = {
        "carry_filter_enabled": False,
        "momentum_filter_enabled": False,
        "regime_filter_enabled": False,
    }
    defaults.update(overrides)
    return RateDiffMRConfig(**defaults)


def _strategy(cfg: RateDiffMRConfig, **kwargs: Any) -> RateDiffMRStrategy:
    s = RateDiffMRStrategy(cfg, **kwargs)
    s._model = dict(_MODEL)  # skip fit; z = price - spread column
    return s


def _frame(
    z_path: list[float],
    prices: list[float] | None = None,
    **extra_cols: list[float] | float,
) -> pd.DataFrame:
    """Frame where row i's z-score equals ``z_path[i]`` exactly."""
    n = len(z_path)
    px = prices if prices is not None else [1.10] * n
    assert len(px) == n
    df = pd.DataFrame(
        {
            PAIR: px,
            SPREAD: [p - z for p, z in zip(px, z_path, strict=True)],
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    for col, vals in extra_cols.items():
        df[col] = vals
    return df


def _legacy_reference_signals(
    df: pd.DataFrame, cfg: RateDiffMRConfig,
) -> pd.Series:
    """Reimplementation of the pre-CL-x50g signal loop (fixture oracle)."""
    z_series = (df[PAIR] - (_MODEL["alpha"] + _MODEL["beta"] * df[SPREAD]))
    positions = []
    pos, entry_i = 0.0, 0
    for i, z in enumerate(z_series / _MODEL["residual_std"]):
        if abs(pos) < 0.001:
            if z < -cfg.entry_z_threshold:
                pos, entry_i = 1.0, i
            elif z > cfg.entry_z_threshold:
                pos, entry_i = -1.0, i
        else:
            days = i - entry_i
            stop = (pos > 0 and z < -cfg.stop_loss_z) or (
                pos < 0 and z > cfg.stop_loss_z
            )
            exit_ok = (pos > 0 and z >= -cfg.exit_z_threshold) or (
                pos < 0 and z <= cfg.exit_z_threshold
            )
            if stop or exit_ok or days > cfg.max_holding_days:
                pos = 0.0
        positions.append(pos)
    return pd.Series(positions, index=df.index, dtype=float)


# A z-path exercising short entry/exit, long entry/exit, and idle periods.
_Z_MIXED = [0.0, 0.5, 2.0, 1.0, 0.2, 0.0, -2.0, -1.0, -0.2, 0.0, 2.0, 0.1]


# =============================================================================
# Disabled filters == legacy behavior
# =============================================================================


class TestDisabledFiltersBaseline:
    def test_disabled_matches_legacy_reference(self) -> None:
        cfg = _config()
        df = _frame(_Z_MIXED)
        out = _strategy(cfg).generate_signals(df)
        pd.testing.assert_series_equal(
            out, _legacy_reference_signals(df, cfg), check_exact=True,
        )

    def test_disabled_ignores_filter_columns(self) -> None:
        """Adversarial OIS/CVIX columns must not change disabled output."""
        cfg = _config()
        plain = _frame(_Z_MIXED)
        hostile = _frame(
            _Z_MIXED,
            **{
                USD_OIS: [0.01] * len(_Z_MIXED),   # carry opposes shorts
                EUR_OIS: [0.05] * len(_Z_MIXED),
                "CVIX": [50.0] * len(_Z_MIXED),    # vol blowout
            },
        )
        out_plain = _strategy(cfg).generate_signals(plain)
        out_hostile = _strategy(cfg).generate_signals(hostile)
        pd.testing.assert_series_equal(out_plain, out_hostile, check_exact=True)


# =============================================================================
# Carry filter
# =============================================================================


def _carry_frame(z_path: list[float], usd: float, eur: float) -> pd.DataFrame:
    n = len(z_path)
    return _frame(z_path, **{USD_OIS: [usd] * n, EUR_OIS: [eur] * n})


class TestCarryFilter:
    CFG = {"carry_filter_enabled": True}

    def test_short_blocked_when_carry_negative(self) -> None:
        # USD < EUR rate → carry = -0.01 opposes a EURUSD short → no entry.
        df = _carry_frame([0.0, 0.0, 2.0, 2.0, 0.0], usd=0.02, eur=0.03)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert (out == 0.0).all()

    def test_short_allowed_when_carry_positive(self) -> None:
        # Positive US-minus-EUR spread favors USD strength → short passes.
        df = _carry_frame([0.0, 0.0, 2.0, 2.0, 0.0], usd=0.03, eur=0.02)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[2] == -1.0

    def test_long_blocked_when_carry_positive(self) -> None:
        df = _carry_frame([0.0, 0.0, -2.0, -2.0, 0.0], usd=0.03, eur=0.02)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert (out == 0.0).all()

    def test_long_allowed_when_carry_negative(self) -> None:
        df = _carry_frame([0.0, 0.0, -2.0, -2.0, 0.0], usd=0.02, eur=0.03)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[2] == 1.0

    def test_zero_carry_allows_both_directions(self) -> None:
        # Neutral carry opposes neither side (>=0 for shorts, <=0 for longs).
        df = _carry_frame([0.0, 2.0, 0.0, -2.0, 0.0], usd=0.03, eur=0.03)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[1] == -1.0
        assert out.iloc[3] == 1.0

    def test_missing_ois_passes_with_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # No OIS columns at all → fail-open: entry still happens + WARNING.
        df = _frame([0.0, 0.0, 2.0, 2.0, 0.0])
        with caplog.at_level(logging.WARNING):
            out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[2] == -1.0
        carry_warnings = [
            r for r in caplog.records
            if "Carry filter" in r.getMessage() and r.levelno == logging.WARNING
        ]
        assert len(carry_warnings) == 1

    def test_nan_ois_rows_pass_with_single_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        n = 5
        df = _frame(
            [0.0, 0.0, 2.0, 2.0, 0.0],
            **{USD_OIS: [np.nan] * n, EUR_OIS: [0.03] * n},
        )
        with caplog.at_level(logging.WARNING):
            out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[2] == -1.0  # NaN carry rows fail open
        carry_warnings = [
            r for r in caplog.records
            if "Carry filter" in r.getMessage() and r.levelno == logging.WARNING
        ]
        assert len(carry_warnings) == 1  # one summary, not one per row

    def test_exits_never_blocked(self) -> None:
        # Enter short at row 2 (carry favorable), carry flips against the
        # position afterwards — exit at z<=0.3 must still happen.
        z = [0.0, 0.0, 2.0, 1.5, 0.1, 0.0]
        usd = [0.03, 0.03, 0.03, 0.01, 0.01, 0.01]
        eur = [0.02] * 6
        df = _frame(z, **{USD_OIS: usd, EUR_OIS: eur})
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[2] == -1.0
        assert out.iloc[3] == -1.0
        assert out.iloc[4] == 0.0  # exit fired despite opposing carry


# =============================================================================
# Momentum filter
# =============================================================================


class TestMomentumFilter:
    CFG = {"momentum_filter_enabled": True, "momentum_lookback_days": 3}

    def test_short_blocked_by_positive_momentum(self) -> None:
        # Price grinding up → 3d momentum > 0 at the would-be short entry.
        prices = [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
        z = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0]
        df = _frame(z, prices=prices)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert (out == 0.0).all()

    def test_short_allowed_when_momentum_nonpositive(self) -> None:
        # Flat price → momentum exactly 0 → passes (<= 0 tolerated).
        df = _frame([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0])
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[4] == -1.0

    def test_long_blocked_by_negative_momentum(self) -> None:
        prices = [1.16, 1.15, 1.14, 1.13, 1.12, 1.11, 1.10]
        z = [0.0, 0.0, 0.0, 0.0, -2.0, -2.0, 0.0]
        df = _frame(z, prices=prices)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert (out == 0.0).all()

    def test_long_allowed_by_positive_momentum(self) -> None:
        prices = [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
        z = [0.0, 0.0, 0.0, 0.0, -2.0, -2.0, 0.0]
        df = _frame(z, prices=prices)
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[4] == 1.0

    def test_max_opposing_tolerance(self) -> None:
        # ~0.9% 3d up-move is under the 5% tolerance → short still allowed.
        cfg = _config(
            momentum_filter_enabled=True,
            momentum_lookback_days=3,
            momentum_max_opposing=0.05,
        )
        prices = [1.100, 1.101, 1.102, 1.103, 1.110, 1.110, 1.110]
        z = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0]
        out = _strategy(cfg).generate_signals(_frame(z, prices=prices))
        assert out.iloc[4] == -1.0

    def test_warmup_rows_fail_open(self) -> None:
        # Entry inside the lookback warmup (momentum NaN) → filter passes.
        df = _frame([0.0, 2.0, 2.0, 0.0])
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[1] == -1.0


# =============================================================================
# Regime filter
# =============================================================================


def _cvix(baseline_n: int, entry_vals: list[float]) -> list[float]:
    """Noisy-but-stable CVIX baseline followed by explicit values."""
    base = [10.0 + (0.2 if i % 2 else -0.2) for i in range(baseline_n)]
    return base + entry_vals


class TestRegimeFilter:
    CFG = {"regime_filter_enabled": True, "vol_lookback_days": 10}

    def test_entry_blocked_in_vol_blowout(self) -> None:
        n = 20
        z = [0.0] * (n - 2) + [2.0, 2.0]
        cvix = _cvix(n - 2, [50.0, 50.0])  # z-score >> 2 at entry rows
        df = _frame(z, **{"CVIX": cvix})
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert (out == 0.0).all()

    def test_entry_allowed_in_calm_regime(self) -> None:
        n = 20
        z = [0.0] * (n - 2) + [2.0, 2.0]
        cvix = _cvix(n - 2, [10.0, 10.0])  # at baseline mean → z ≈ 0
        df = _frame(z, **{"CVIX": cvix})
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[n - 2] == -1.0

    def test_blocks_both_directions(self) -> None:
        # Fresh spike per frame: a persistent spike gets absorbed into the
        # rolling baseline after a few bars (by design), so test each
        # direction against its own fresh blowout.
        n = 20
        for z_entry in (2.0, -2.0):
            z = [0.0] * (n - 2) + [z_entry, 0.0]
            cvix = _cvix(n - 2, [50.0, 50.0])
            df = _frame(z, **{"CVIX": cvix})
            out = _strategy(_config(**self.CFG)).generate_signals(df)
            assert (out == 0.0).all(), f"entry z={z_entry} not blocked"

    def test_warmup_fails_open(self) -> None:
        # Too little CVIX history for a z-score → filter passes.
        df = _frame([0.0, 2.0, 2.0, 0.0], **{"CVIX": [10.0, 50.0, 50.0, 50.0]})
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[1] == -1.0

    def test_missing_cvix_column_fails_open(self) -> None:
        df = _frame([0.0] * 18 + [2.0, 2.0])
        out = _strategy(_config(**self.CFG)).generate_signals(df)
        assert out.iloc[18] == -1.0


# =============================================================================
# No lookahead
# =============================================================================


class TestNoLookahead:
    def test_prefix_signals_unchanged_by_future_rows(self) -> None:
        """Filter values at row t must not use rows > t: generating on any
        prefix of the frame yields the identical prefix of signals."""
        rng = np.random.default_rng(7)
        n = 60
        z = list(rng.normal(0.0, 1.6, n))
        prices = list(1.10 + np.cumsum(rng.normal(0, 0.004, n)))
        cvix = list(10.0 + rng.normal(0, 1.0, n))
        usd = list(0.03 + rng.normal(0, 0.005, n))
        eur = list(0.02 + rng.normal(0, 0.005, n))
        df = _frame(
            z, prices=prices,
            **{"CVIX": cvix, USD_OIS: usd, EUR_OIS: eur},
        )
        cfg = _config(
            carry_filter_enabled=True,
            momentum_filter_enabled=True,
            regime_filter_enabled=True,
            vol_lookback_days=10,
        )
        full = _strategy(cfg).generate_signals(df)
        for t in (5, 15, 30, 45, n - 1):
            prefix = _strategy(cfg).generate_signals(df.iloc[:t])
            pd.testing.assert_series_equal(
                prefix, full.iloc[:t], check_exact=True,
            )


# =============================================================================
# Live path — generate_intents
# =============================================================================


class _FakeProvider:
    """Minimal DataProvider double for the live filter path."""

    def __init__(
        self,
        latest: dict[str, float | None] | None = None,
        series: dict[str, pd.Series] | None = None,
        pair_closes: list[float] | None = None,
    ) -> None:
        self.latest = latest or {}
        self.series = series or {}
        self.pair_closes = pair_closes

    def get_latest_value(self, series_id: str, as_of: datetime) -> float | None:
        return self.latest.get(series_id)

    def get_series(self, series_id: str, start: datetime, end: datetime) -> pd.Series:
        return self.series.get(series_id, pd.Series(dtype=float))

    def get_aligned_series(
        self, symbols: list[str], start: datetime, end: datetime,
    ) -> pd.DataFrame | None:
        if symbols == [PAIR] and self.pair_closes is not None:
            idx = pd.date_range(end=end, periods=len(self.pair_closes), freq="D")
            return pd.DataFrame({PAIR: self.pair_closes}, index=idx)
        # CL-z1: the live tick now fetches the CONFIGURED spread series (the
        # one the model was fit on), not a hardcoded US_10Y/DE_10Y pair.
        if symbols == [SPREAD]:
            idx = pd.date_range(end=end, periods=5, freq="D")
            return pd.DataFrame({SPREAD: [1.5] * 5}, index=idx)
        if set(symbols) == {"US_10Y", "DE_10Y"}:
            idx = pd.date_range(end=end, periods=3, freq="D")
            return pd.DataFrame({"US_10Y": 4.0, "DE_10Y": 2.5}, index=idx)
        return None


class _FakeBroker:
    def get_account(self) -> Any:
        return SimpleNamespace(equity=100_000.0)


class _CapturingStore:
    def __init__(self) -> None:
        self.snapshots: list[Any] = []

    def store(self, snapshot: Any) -> None:
        self.snapshots.append(snapshot)


_PRICES = {PAIR: {"bid": 1.0999, "ask": 1.1001}}  # mid 1.1 → z = +10 → short


def _live_strategy(cfg: RateDiffMRConfig, provider: _FakeProvider) -> tuple[
    RateDiffMRStrategy, _CapturingStore,
]:
    store = _CapturingStore()
    s = RateDiffMRStrategy(cfg, data_provider=provider, snapshot_store=store)
    # z = (1.1 - 1.0) / 0.01 = +10 → entry-short candidate every tick.
    s._model = {"alpha": 1.0, "beta": 0.0, "r_squared": 0.9, "residual_std": 0.01}
    s._last_fit = datetime.now(UTC)  # skip refit
    return s, store


class TestLivePathFilters:
    def test_entry_blocked_by_momentum_records_snapshot(self) -> None:
        cfg = _config(momentum_filter_enabled=True, momentum_lookback_days=5)
        provider = _FakeProvider(
            pair_closes=[1.10 + 0.005 * i for i in range(10)],  # rising → mom>0
        )
        strat, store = _live_strategy(cfg, provider)
        intents = asyncio.run(strat.generate_intents(_PRICES, _FakeBroker()))
        assert intents == []
        assert strat._position_size == 0.0
        assert len(store.snapshots) == 1
        values = store.snapshots[0].values
        assert values["trigger"] == "entry_blocked"
        assert values["filters"]["momentum"]["enabled"] is True
        assert values["filters"]["momentum"]["passed"] is False

    def test_entry_allowed_records_filter_metadata(self) -> None:
        cfg = _config(
            carry_filter_enabled=True,
            momentum_filter_enabled=True,
            regime_filter_enabled=True,
            momentum_lookback_days=5,
        )
        provider = _FakeProvider(
            latest={USD_OIS: 0.045, EUR_OIS: 0.02},  # carry > 0 → short OK
            series={"CVIX": pd.Series([10.0] * 240)},  # flat → vol z 0
            pair_closes=[1.12 - 0.003 * i for i in range(10)],  # falling → mom<0
        )
        strat, store = _live_strategy(cfg, provider)
        intents = asyncio.run(strat.generate_intents(_PRICES, _FakeBroker()))
        assert len(intents) == 1
        assert intents[0].target_position < 0  # short entry
        assert "snapshot_id" in intents[0].metadata
        values = store.snapshots[0].values
        assert values["trigger"] == "entry"
        for name in ("carry", "momentum", "regime"):
            assert values["filters"][name]["enabled"] is True
            assert values["filters"][name]["passed"] is True
        assert values["filters"]["carry"]["value"] == pytest.approx(0.025)

    def test_missing_ois_warns_once_per_gap(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        cfg = _config(
            carry_filter_enabled=True,
            momentum_filter_enabled=True,
            momentum_lookback_days=5,
        )
        # OIS missing entirely; momentum blocks entries so we stay flat and
        # the carry filter is re-evaluated every tick.
        provider = _FakeProvider(
            latest={},
            pair_closes=[1.10 + 0.005 * i for i in range(10)],
        )
        strat, store = _live_strategy(cfg, provider)
        with caplog.at_level(logging.WARNING):
            asyncio.run(strat.generate_intents(_PRICES, _FakeBroker()))
            asyncio.run(strat.generate_intents(_PRICES, _FakeBroker()))
        gap_warnings = [
            r for r in caplog.records
            if "OIS data unavailable" in r.getMessage()
        ]
        assert len(gap_warnings) == 1  # once per gap, not per tick
        # Carry failed OPEN both ticks (blocked entries came from momentum).
        for snap in store.snapshots:
            assert snap.values["filters"]["carry"]["passed"] is True
            assert snap.values["filters"]["momentum"]["passed"] is False

    def test_all_filters_disabled_enters_like_legacy(self) -> None:
        strat, store = _live_strategy(_config(), _FakeProvider())
        intents = asyncio.run(strat.generate_intents(_PRICES, _FakeBroker()))
        assert len(intents) == 1
        assert intents[0].target_position < 0
        values = store.snapshots[0].values
        assert values["trigger"] == "entry"
        for name in ("carry", "momentum", "regime"):
            assert values["filters"][name]["enabled"] is False
            assert values["filters"][name]["passed"] is True


class TestLiveSpreadSeries:
    """CL-z1 (P0): the live z-score must use config.rate_spread_series (the
    series the model was fit on), NOT a hardcoded US_10Y-DE_10Y (not ingested)
    or a constant 0.5 fallback."""

    def _strat(self, provider: _FakeProvider) -> tuple[
        RateDiffMRStrategy, _CapturingStore,
    ]:
        store = _CapturingStore()
        s = RateDiffMRStrategy(
            _config(), data_provider=provider, snapshot_store=store,
        )
        # beta=1 so the spread VALUE actually reaches z (unlike the beta=0
        # fixture) — proves which series feeds the model at tick time.
        s._model = {
            "alpha": 0.0, "beta": 1.0, "r_squared": 0.9, "residual_std": 0.01,
        }
        s._last_fit = datetime.now(UTC)
        return s, store

    def test_uses_configured_series_not_placeholder(self) -> None:
        s, store = self._strat(_FakeProvider())  # serves SPREAD=1.5
        asyncio.run(s.generate_intents(_PRICES, _FakeBroker()))
        # The recorded spread is the configured series (1.5), never 0.5.
        assert store.snapshots[-1].values["spread"] == 1.5
        assert s._last_spread == 1.5

    def test_no_series_and_no_cache_skips_tick(self) -> None:
        class _EmptyProv(_FakeProvider):
            def get_aligned_series(self, symbols, start, end):  # type: ignore[override]
                return None
        s, _ = self._strat(_EmptyProv())
        # Fail closed: no fabricated 0.5 spread, no trade.
        assert asyncio.run(s.generate_intents(_PRICES, _FakeBroker())) == []
        assert s._position_size == 0.0

    def test_cache_bridges_transient_gap(self) -> None:
        # Tick 1 populates the cache from SPREAD=1.5; tick 2's provider
        # returns nothing → reuse 1.5 rather than skip or fabricate.
        s, store = self._strat(_FakeProvider())
        asyncio.run(s.generate_intents(_PRICES, _FakeBroker()))
        assert s._last_spread == 1.5
        s.data = None  # simulate the data provider going away
        asyncio.run(s.generate_intents(_PRICES, _FakeBroker()))
        assert s._last_spread == 1.5  # bridged, not skipped


class TestLivePathLiquidity:
    """CL-y412: liquidity-window gate on the live entry path.

    _PRICES spread = 0.0002 / 1.1 * 1e4 ≈ 1.818 bps; a pair_median steers the
    ratio across the thin (1.5×) and block (2.0×) thresholds.
    """

    def _strat(self, profile: LiquidityProfile | None) -> tuple[
        RateDiffMRStrategy, _CapturingStore,
    ]:
        store = _CapturingStore()
        s = RateDiffMRStrategy(
            _config(), data_provider=_FakeProvider(), snapshot_store=store,
            liquidity_profile=profile,
        )
        s._model = {
            "alpha": 1.0, "beta": 0.0, "r_squared": 0.9, "residual_std": 0.01,
        }
        s._last_fit = datetime.now(UTC)
        return s, store

    def test_dead_window_blocks_entry(self) -> None:
        # median 0.5 → ratio 3.6 ≥ 2.0 → block; entry suppressed, snapshot
        # records the liquidity spread.
        s, store = self._strat(LiquidityProfile(pair_median_bps={"EURUSD": 0.5}))
        intents = asyncio.run(s.generate_intents(_PRICES, _FakeBroker()))
        assert intents == []
        assert s._position_size == 0.0
        assert store.snapshots[-1].values["trigger"] == "entry_blocked"
        assert "liquidity_spread_bps" in store.snapshots[-1].values

    def test_no_profile_enters_full(self) -> None:
        s, _ = self._strat(None)
        intents = asyncio.run(s.generate_intents(_PRICES, _FakeBroker()))
        assert len(intents) == 1

    def test_normal_window_enters_full(self) -> None:
        # median 2.0 → ratio 0.91 < 1.5 → multiplier 1.0 → full size.
        s_full, _ = self._strat(None)
        full = asyncio.run(s_full.generate_intents(_PRICES, _FakeBroker()))
        s, _ = self._strat(LiquidityProfile(pair_median_bps={"EURUSD": 2.0}))
        intents = asyncio.run(s.generate_intents(_PRICES, _FakeBroker()))
        assert len(intents) == 1
        assert intents[0].target_position == pytest.approx(
            full[0].target_position,
        )

    def test_thin_window_halves_entry(self) -> None:
        # median 1.0 → ratio 1.818 in [1.5, 2.0) → 0.5× — enters at half size.
        s_full, _ = self._strat(None)
        full = asyncio.run(s_full.generate_intents(_PRICES, _FakeBroker()))
        s, _ = self._strat(LiquidityProfile(pair_median_bps={"EURUSD": 1.0}))
        intents = asyncio.run(s.generate_intents(_PRICES, _FakeBroker()))
        assert len(intents) == 1
        assert intents[0].target_position == pytest.approx(
            full[0].target_position * 0.5,
        )
