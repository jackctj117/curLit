"""Tests for the aggressive NO-bias Polymarket strategy (CL-yk4k)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd

from src.risk.polymarket_sizer import PolySizeDecision
from src.risk.risk_profile import (
    BiasConfig,
    HoldingConfig,
    KillSwitchConfig,
    RiskProfile,
    SizingConfig,
    StrategyGatesConfig,
)
from src.strategies.aggressive_short_polymarket import (
    AggressiveShortPolymarketConfig,
    AggressiveShortPolymarketStrategy,
)


def _profile(prefer_no: bool = True, max_market: float = 0.15) -> RiskProfile:
    return RiskProfile(
        name="aggressive_short",
        sizing=SizingConfig(
            kelly_fraction=0.5,
            max_polymarket_market_fraction=max_market,
            polymarket_min_edge=0.015,
        ),
        kill_switches=KillSwitchConfig(),
        strategy_gates=StrategyGatesConfig(),
        holding=HoldingConfig(max_holding_days=5),
        bias=BiasConfig(prefer_polymarket_no=prefer_no),
    )


def _market(
    symbol: str = "POLY:demo",
    resolution_offset_days: int = 7,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    return {
        "symbol": symbol,
        "resolution_ts": (now + timedelta(days=resolution_offset_days)).isoformat(),
    }


def _panel(symbols: list[str], prices: dict, n: int = 10) -> pd.DataFrame:
    """Build a price panel: index is daily, columns are symbols.
    ``prices`` is ``{symbol: scalar}`` — constant price across all bars."""
    idx = pd.date_range("2026-05-01", periods=n, freq="D", tz="UTC")
    data = {s: [prices.get(s, 0.5)] * n for s in symbols}
    return pd.DataFrame(data, index=idx)


class TestTimeGate:
    def test_far_resolution_skipped(self) -> None:
        now = datetime(2026, 5, 14, tzinfo=UTC)
        # Market resolves 60 days from now — past the 14-day horizon.
        market = _market(resolution_offset_days=60, now=now)
        cfg = AggressiveShortPolymarketConfig(
            markets=[market],
            model_prob_fn=lambda *a, **kw: 0.30,  # strong YES edge
            risk_profile=_profile(),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)
        panel = _panel(["POLY:demo"], {"POLY:demo": 0.50}, n=5)
        out = s.generate_signals(panel, now=now)
        # All zero — beyond horizon.
        assert (out["POLY:demo"] != 0).sum() == 0

    def test_within_horizon_can_trade(self) -> None:
        now = datetime(2026, 5, 14, tzinfo=UTC)
        market = _market(resolution_offset_days=7, now=now)
        cfg = AggressiveShortPolymarketConfig(
            markets=[market],
            # Strong NO edge: market at 0.70, model says 0.45.
            model_prob_fn=lambda *a, **kw: 0.45,
            risk_profile=_profile(prefer_no=True),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)
        # Build panel with current ts so bar_ts is well within horizon.
        idx = pd.date_range(
            "2026-05-14", periods=5, freq="D", tz="UTC",
        )
        panel = pd.DataFrame({"POLY:demo": [0.70] * 5}, index=idx)
        out = s.generate_signals(panel, now=now)
        # Should produce NO (negative) positions on bars where the
        # 24h exit window isn't open yet (i.e., resolution is >24h away).
        # First 5 bars: 5/6/7d, 4d, 3d to resolution — last bar is
        # ~7d-4d = 3d away, well outside 24h. Should fire NO.
        assert (out["POLY:demo"] < 0).sum() > 0


class TestExitWindow:
    def test_within_24h_of_resolution_exits(self) -> None:
        # Market resolves in 12h from "now" — within the 24h exit window.
        now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        resolution = now + timedelta(hours=12)
        market = {
            "symbol": "POLY:demo",
            "resolution_ts": resolution.isoformat(),
        }
        cfg = AggressiveShortPolymarketConfig(
            markets=[market],
            model_prob_fn=lambda *a, **kw: 0.30,  # huge edge
            risk_profile=_profile(),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)
        # Single bar at "now".
        panel = pd.DataFrame(
            {"POLY:demo": [0.50]},
            index=pd.DatetimeIndex([now]),
        )
        out = s.generate_signals(panel, now=now)
        # Inside the exit window: position must be 0.
        assert out["POLY:demo"].iat[0] == 0.0


class TestNOBias:
    def test_prefers_no_when_both_pass(self) -> None:
        # Price at 0.50, model_prob 0.30 means:
        #   YES edge = 0.30 - 0.50 = -0.20 < 0 → YES rejected
        #   NO edge = 0.50 - 0.30 = +0.20 → NO accepted
        # NO is the only one passing, so position is short.
        now = datetime(2026, 5, 14, tzinfo=UTC)
        market = _market(resolution_offset_days=5, now=now)
        cfg = AggressiveShortPolymarketConfig(
            markets=[market],
            model_prob_fn=lambda *a, **kw: 0.30,
            risk_profile=_profile(prefer_no=True),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)
        idx = pd.date_range("2026-05-14", periods=1, freq="D", tz="UTC")
        panel = pd.DataFrame({"POLY:demo": [0.50]}, index=idx)
        out = s.generate_signals(panel, now=now)
        assert out["POLY:demo"].iat[0] < 0

    def test_choose_side_no_bias_with_both_passing(self) -> None:
        """Direct unit test of the side-chooser when both YES and NO
        pass the edge filter. Build _choose_side manually."""
        cfg = AggressiveShortPolymarketConfig(
            risk_profile=_profile(prefer_no=True),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)

        yes = PolySizeDecision(
            shares=Decimal("100"), cost_usdc=Decimal("50"),
            fraction_of_bankroll=Decimal("0.04"), rationale="...",
        )
        no = PolySizeDecision(
            shares=Decimal("100"), cost_usdc=Decimal("50"),
            fraction_of_bankroll=Decimal("0.03"), rationale="...",
        )
        # NO has smaller edge but the bias wins → NO chosen.
        result = s._choose_side(yes, no)
        assert result < 0
        assert abs(result) == 0.03

    def test_choose_side_picks_larger_edge_without_bias(self) -> None:
        cfg = AggressiveShortPolymarketConfig(
            risk_profile=_profile(prefer_no=False),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)

        yes = PolySizeDecision(
            shares=Decimal("100"), cost_usdc=Decimal("50"),
            fraction_of_bankroll=Decimal("0.04"),  # bigger
            rationale="...",
        )
        no = PolySizeDecision(
            shares=Decimal("100"), cost_usdc=Decimal("50"),
            fraction_of_bankroll=Decimal("0.03"),
            rationale="...",
        )
        # Both pass, no bias → YES wins (bigger).
        result = s._choose_side(yes, no)
        assert result > 0
        assert result == 0.04


class TestNoModelProb:
    def test_strategy_no_ops_without_model_prob_fn(self) -> None:
        """Without a model_prob_fn, the strategy can't take positions —
        no chart-based edge here. Should return all-zero positions."""
        now = datetime(2026, 5, 14, tzinfo=UTC)
        cfg = AggressiveShortPolymarketConfig(
            markets=[_market(now=now)],
            model_prob_fn=None,
            risk_profile=_profile(),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)
        idx = pd.date_range("2026-05-14", periods=3, freq="D", tz="UTC")
        panel = pd.DataFrame({"POLY:demo": [0.50, 0.55, 0.60]}, index=idx)
        out = s.generate_signals(panel, now=now)
        assert (out["POLY:demo"] != 0).sum() == 0


class TestInvalidPrice:
    def test_price_outside_unit_skipped(self) -> None:
        """Polymarket prices live in (0, 1); 0 or 1 is degenerate."""
        now = datetime(2026, 5, 14, tzinfo=UTC)
        cfg = AggressiveShortPolymarketConfig(
            markets=[_market(now=now)],
            model_prob_fn=lambda *a, **kw: 0.30,
            risk_profile=_profile(),
        )
        s = AggressiveShortPolymarketStrategy(config=cfg)
        idx = pd.date_range("2026-05-14", periods=2, freq="D", tz="UTC")
        # 0 and 1 — both degenerate. Strategy should sit out.
        panel = pd.DataFrame({"POLY:demo": [0.0, 1.0]}, index=idx)
        out = s.generate_signals(panel, now=now)
        assert (out["POLY:demo"] != 0).sum() == 0
