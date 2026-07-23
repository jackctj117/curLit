"""Test PositionSizer.adjust_for_liquidity wires through to LiquidityProfile (CL-4wi5)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.risk.liquidity_window import LiquidityProfile
from src.risk.sizing import PositionSizer


def _ts(dow: int, hour: int) -> datetime:
    base = datetime(2026, 4, 6, hour, 0)
    from datetime import timedelta as _td

    return base + _td(days=dow)


class TestAdjustForLiquidity:
    def test_normal_window_passes_full_size(self) -> None:
        prof = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 12): 1.0},
            pair_median_bps={"EURUSD": 1.0},
        )
        out = PositionSizer.adjust_for_liquidity(
            base_size=10_000.0,
            symbol="EURUSD",
            ts=_ts(0, 12),
            observed_spread_bps=1.2,
            profile=prof,
        )
        assert out == pytest.approx(10_000.0)

    def test_dead_window_blocks_entry(self) -> None:
        prof = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 22): 1.0},
            pair_median_bps={"EURUSD": 1.0},
        )
        out = PositionSizer.adjust_for_liquidity(
            base_size=10_000.0,
            symbol="EURUSD",
            ts=_ts(0, 22),
            observed_spread_bps=3.0,
            profile=prof,
        )
        assert out == 0.0

    def test_thin_window_halves(self) -> None:
        prof = LiquidityProfile(
            median_spread_bps={("EURUSD", 4, 22): 1.0},
            pair_median_bps={"EURUSD": 1.0},
        )
        out = PositionSizer.adjust_for_liquidity(
            base_size=10_000.0,
            symbol="EURUSD",
            ts=_ts(4, 22),
            observed_spread_bps=1.7,
            profile=prof,
        )
        assert out == 5_000.0

    def test_none_profile_passthrough(self) -> None:
        # Conservative behavior — strategies that don't have a profile
        # configured still trade at full base size.
        out = PositionSizer.adjust_for_liquidity(
            base_size=10_000.0,
            symbol="EURUSD",
            ts=_ts(0, 12),
            observed_spread_bps=1.0,
            profile=None,
        )
        assert out == 10_000.0

    def test_profile_exception_fails_closed(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A raising profile means liquidity can't be verified — the entry
        is refused (0.0), never sized at full (the old fail-open)."""

        class _BoomProfile:
            def size_multiplier(self, *_a: object, **_k: object) -> float:
                raise RuntimeError("boom")

        with caplog.at_level("WARNING", logger="src.risk.sizing"):
            out = PositionSizer.adjust_for_liquidity(
                base_size=10_000.0,
                symbol="EURUSD",
                ts=_ts(0, 12),
                observed_spread_bps=1.0,
                profile=_BoomProfile(),
            )
        assert out == 0.0
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "EURUSD" in msg  # which symbol was refused
        assert "RuntimeError" in msg  # exception summary
        assert "boom" in msg
