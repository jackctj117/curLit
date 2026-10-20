"""Tests for the aggressive short-bias FX strategy (CL-eft9)."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.risk.risk_profile import (
    BiasConfig,
    HoldingConfig,
    KillSwitchConfig,
    RiskProfile,
    SizingConfig,
    StrategyGatesConfig,
)
from src.strategies.aggressive_short_fx import (
    AggressiveShortFXConfig,
    AggressiveShortFXStrategy,
    _SignalTriplet,
)


def _profile(
    name: str = "aggressive_short",
    long_mult: float = 0.0,
    short_mult: float = 1.0,
    entry_z: float = 1.0,
    min_r2: float = 0.05,
    max_hold: int = 5,
) -> RiskProfile:
    return RiskProfile(
        name=name,
        sizing=SizingConfig(kelly_fraction=0.5, max_position_pct=0.4),
        kill_switches=KillSwitchConfig(daily_loss_limit_pct=-0.10),
        strategy_gates=StrategyGatesConfig(
            min_r_squared=min_r2,
            entry_z_threshold=entry_z,
        ),
        holding=HoldingConfig(max_holding_days=max_hold),
        bias=BiasConfig(
            long_signal_multiplier=long_mult,
            short_signal_multiplier=short_mult,
        ),
    )


class TestSignalTriplet:
    def test_two_of_three_short_wins(self) -> None:
        # Two shorts + one neutral → -1
        assert _SignalTriplet(-1, -1, 0).agreement() == -1

    def test_two_of_three_long_wins(self) -> None:
        assert _SignalTriplet(1, 1, 0).agreement() == 1

    def test_one_each_no_quorum(self) -> None:
        # One long, one short, one neutral → 0 (no quorum)
        assert _SignalTriplet(1, -1, 0).agreement() == 0

    def test_one_signal_only_no_quorum(self) -> None:
        # Single short signal → 0 (not 2-of-3)
        assert _SignalTriplet(-1, 0, 0).agreement() == 0

    def test_three_of_three_strong(self) -> None:
        assert _SignalTriplet(-1, -1, -1).agreement() == -1
        assert _SignalTriplet(1, 1, 1).agreement() == 1


def _entry_fixture(
    profile: RiskProfile, *, direction: int, bars: int,
) -> tuple[AggressiveShortFXStrategy, pd.DataFrame]:
    """CL-qwrv: fit a known line, then evaluate separate daily observations.

    A constant offset in the training set gets absorbed by the OLS intercept;
    it cannot force an out-of-sample signal. This residual pattern instead has
    zero sum and zero dot product with the linear spread, so the fitted fair
    value is 1.10 + .05 * spread. 120 rows exceed the 100-row fit minimum.
    """
    spread = np.linspace(-1, 1, 120)
    residual = np.tile([0.005, -0.005, -0.005, 0.005], 30)
    train = pd.DataFrame({
        "EURUSD": 1.10 + 0.05 * spread + residual,
        "US10Y_MINUS_DE10Y": spread,
    }, index=pd.date_range("2024-01-01", periods=120, freq="D"))
    strategy = AggressiveShortFXStrategy(AggressiveShortFXConfig(
        risk_profile=profile, poly_symbol=None, agreement_quorum=1,
    ))
    strategy.fit(train)
    # Residual magnitude ~1 sigma: above the .5 entry threshold but below
    # its 1.25-sigma stop and outside the .3 exit threshold.
    holdout = pd.DataFrame({
        "EURUSD": np.full(bars, 1.10 - direction * 0.005),
        "US10Y_MINUS_DE10Y": np.zeros(bars),
    }, index=pd.date_range("2024-05-01", periods=bars, freq="D"))
    return strategy, holdout


@pytest.mark.parametrize(("votes", "expected"), [
    ((-1, -1, -1), (-1, -1, -1)),
    ((-1, -1, 0), (-1, -1, 0)),
    ((-1, -1, 1), (-1, -1, 0)),
    ((-1, 0, 0), (-1, 0, 0)),
    ((-1, 0, 1), (0, 0, 0)),
    ((-1, 1, 1), (1, 1, 0)),
    ((0, 0, 0), (0, 0, 0)),
    ((0, 0, 1), (1, 0, 0)),
    ((0, 1, 1), (1, 1, 0)),
    ((1, 1, 1), (1, 1, 1)),
])
def test_quorum_truth_table(
    votes: tuple[int, int, int], expected: tuple[int, int, int],
) -> None:
    """Explicit oracle for all 27 signal triples and all three quorums."""
    for ordering in set(permutations(votes)):
        for quorum, decision in enumerate(expected, start=1):
            assert _SignalTriplet(*ordering).agreement(quorum) == decision


@given(
    st.tuples(*(st.integers(-1, 1) for _ in range(3))),
    st.integers(1, 3),
)
def test_quorum_sign_symmetry(votes: tuple[int, int, int], quorum: int) -> None:
    result = _SignalTriplet(*votes).agreement(quorum)
    assert result in (-1, 0, 1)
    assert _SignalTriplet(*(-v for v in votes)).agreement(quorum) == -result
    assert _SignalTriplet(*reversed(votes)).agreement(quorum) == result


@pytest.mark.parametrize("quorum", [0, 4, -1, True, 1.5])
def test_invalid_quorum_rejected(quorum: int) -> None:
    with pytest.raises(ValueError, match="quorum"):
        AggressiveShortFXConfig(agreement_quorum=quorum)
    with pytest.raises(ValueError, match="quorum"):
        _SignalTriplet(-1, 0, 0).agreement(quorum)


@pytest.mark.parametrize("signal", [2, -2, True, 0.5])
def test_invalid_direction_rejected(signal: int) -> None:
    with pytest.raises(ValueError, match="signal"):
        _SignalTriplet(signal, 0, 0).agreement()


@pytest.mark.parametrize("long_multiplier", [0.0, 1.0])
def test_long_bias_applies_to_an_actual_entry(long_multiplier: float) -> None:
    strategy, panel = _entry_fixture(
        _profile(long_mult=long_multiplier, entry_z=0.5),
        direction=1, bars=4,
    )
    assert strategy.generate_signals(panel).tolist() == [long_multiplier] * 4


@pytest.mark.parametrize("max_hold", [1, 3, 5])
def test_position_closes_at_max_holding(max_hold: int) -> None:
    strategy, panel = _entry_fixture(
        _profile(entry_z=0.5, max_hold=max_hold),
        direction=-1, bars=2 * (max_hold + 1),
    )
    # Entry at day zero; flatten exactly max_hold days later. A still-valid
    # signal may open a NEW position on the following day, not on the close bar.
    expected = ([-1.0] * max_hold + [0.0]) * 2
    assert strategy.generate_signals(panel).tolist() == expected


class TestRSquaredGate:
    def test_low_r_squared_sits_out(self) -> None:
        # Pure-noise panel — r² will be near zero.
        n = 200
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        panel = pd.DataFrame(
            {
                "EURUSD": 1.10 + rng.normal(0, 0.005, n),
                "US10Y_MINUS_DE10Y": rng.normal(0, 0.1, n),
            },
            index=idx,
        )
        cfg = AggressiveShortFXConfig(
            risk_profile=_profile(min_r2=0.30),  # high R² gate
            poly_symbol=None,
        )
        strategy = AggressiveShortFXStrategy(config=cfg)
        positions = strategy.generate_signals(panel)
        # All zero — model failed the R² gate.
        assert (positions != 0).sum() == 0


class TestRiskProfileLoad:
    def test_uses_active_profile_when_no_override(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Don't pass risk_profile in the config — should load the
        # active one. Set env var so we control which.
        monkeypatch.setenv("CURLIT_RISK_PROFILE", "aggressive_short")
        from src.risk.risk_profile import load_active_profile

        load_active_profile.cache_clear() if hasattr(
            load_active_profile,
            "cache_clear",
        ) else None
        strategy = AggressiveShortFXStrategy()
        # aggressive_short → long_signal_multiplier=0.0.
        assert strategy.risk.bias.long_signal_multiplier == 0.0
        assert strategy.risk.holding.max_holding_days == 5
