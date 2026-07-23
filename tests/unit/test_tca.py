"""Unit tests — execution.tca (CL-sl7x)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.execution.broker import Fill
from src.execution.tca import (
    FillContext,
    TCAComponents,
    compute_tca,
    summarize,
)

T0 = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)


def _fill(
    symbol: str = "EURUSD",
    side: str = "buy",
    price: float = 1.10,
    fill_id: str = "f1",
) -> Fill:
    return Fill(
        order_id="o1",
        fill_id=fill_id,
        symbol=symbol,
        side=side,
        quantity=1000,
        price=price,
        timestamp=T0,
    )


# =============================================================================
# Validation
# =============================================================================


class TestFillContext:
    def test_valid_context(self) -> None:
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0001, fill_mid=1.1001)
        assert ctx.arrival_mid == 1.10

    def test_invalid_arrival_mid_rejected(self) -> None:
        with pytest.raises(AssertionError):
            FillContext(arrival_mid=0.0, submit_spread=0.0001, fill_mid=1.10)

    def test_negative_spread_rejected(self) -> None:
        with pytest.raises(AssertionError):
            FillContext(arrival_mid=1.10, submit_spread=-0.0001, fill_mid=1.10)


# =============================================================================
# Buy-side decomposition
# =============================================================================


class TestBuyDecomposition:
    def test_perfect_fill_on_arrival_mid(self) -> None:
        # arrival mid 1.10, no spread, fills at exactly 1.10.
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0, fill_mid=1.10)
        f = _fill(side="buy", price=1.10)
        c = compute_tca(f, ctx)
        assert c.queue_bps == pytest.approx(0.0)
        assert c.impact_bps == pytest.approx(0.0)
        assert c.broker_bps == pytest.approx(0.0)
        assert c.total_bps == pytest.approx(0.0)
        assert c.implementation_shortfall_bps == pytest.approx(0.0)

    def test_pure_queue_cost(self) -> None:
        # arrival 1.10, spread 0.0002 (1 bp half-spread), fill at ask = 1.1001.
        # Mid hasn't moved, broker priced exactly at half-spread.
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0002, fill_mid=1.10)
        f = _fill(side="buy", price=1.1001)
        c = compute_tca(f, ctx)
        # Queue = half-spread = 0.0001/1.10 * 10000 ≈ 0.91 bps.
        assert c.queue_bps == pytest.approx(0.909, abs=0.01)
        assert c.impact_bps == pytest.approx(0.0, abs=0.01)
        assert c.broker_bps == pytest.approx(0.0, abs=0.01)
        # Total ≈ implementation shortfall.
        assert c.total_bps == pytest.approx(c.implementation_shortfall_bps, abs=0.01)

    def test_impact_only(self) -> None:
        # No spread, but mid moves up between submit and fill, fill at new mid.
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0, fill_mid=1.1001)
        f = _fill(side="buy", price=1.1001)
        c = compute_tca(f, ctx)
        # Impact ≈ (1.1001 - 1.10)/1.10 * 10000 = ~0.91 bp adverse for buy.
        assert c.queue_bps == pytest.approx(0.0, abs=0.01)
        assert c.impact_bps == pytest.approx(0.909, abs=0.01)
        assert c.broker_bps == pytest.approx(0.0, abs=0.01)
        assert c.implementation_shortfall_bps == pytest.approx(0.909, abs=0.01)

    def test_broker_fill_quality_miss(self) -> None:
        # No spread, no impact — but broker fills 1bp HIGHER than fill_mid.
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0, fill_mid=1.10)
        f = _fill(side="buy", price=1.1001)
        c = compute_tca(f, ctx)
        assert c.queue_bps == pytest.approx(0.0, abs=0.01)
        assert c.impact_bps == pytest.approx(0.0, abs=0.01)
        # Broker is the whole 0.91 bp adverse delta.
        assert c.broker_bps == pytest.approx(0.909, abs=0.01)

    def test_combined_components_sum_to_is(self) -> None:
        # Spread + impact + broker miss all simultaneously.
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0002, fill_mid=1.1001)
        f = _fill(side="buy", price=1.1003)
        c = compute_tca(f, ctx)
        # total_bps must equal implementation_shortfall_bps within float noise.
        assert c.total_bps == pytest.approx(
            c.implementation_shortfall_bps,
            abs=1e-6,
        )


# =============================================================================
# Sell-side decomposition (sign flips)
# =============================================================================


class TestSellDecomposition:
    def test_pure_queue_cost_on_sell(self) -> None:
        # Spread 2 bp, fill at bid (1.0999).
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0002, fill_mid=1.10)
        f = _fill(side="sell", price=1.0999)
        c = compute_tca(f, ctx)
        # Queue is the half-spread, always positive cost regardless of side.
        assert c.queue_bps == pytest.approx(0.909, abs=0.01)
        assert c.impact_bps == pytest.approx(0.0, abs=0.01)
        assert c.broker_bps == pytest.approx(0.0, abs=0.01)

    def test_impact_adverse_on_sell_when_mid_drops(self) -> None:
        # Selling when mid moves DOWN is adverse to us.
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0, fill_mid=1.0999)
        f = _fill(side="sell", price=1.0999)
        c = compute_tca(f, ctx)
        # Mid moved -0.0001 (1bp lower). For a sell, that's +0.91 bps cost.
        assert c.impact_bps == pytest.approx(0.909, abs=0.01)

    def test_total_equals_is_on_sell(self) -> None:
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0002, fill_mid=1.0999)
        f = _fill(side="sell", price=1.0998)
        c = compute_tca(f, ctx)
        assert c.total_bps == pytest.approx(c.implementation_shortfall_bps, abs=1e-6)


# =============================================================================
# Favorable fills (negative bps)
# =============================================================================


class TestFavorableFill:
    def test_buy_at_better_than_mid(self) -> None:
        # Mid 1.10, fill 1.0999 — we BOUGHT lower than mid, favorable.
        # No queue (zero spread), mid didn't move.
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0, fill_mid=1.10)
        f = _fill(side="buy", price=1.0999)
        c = compute_tca(f, ctx)
        # Total cost should be negative (we got money back).
        assert c.implementation_shortfall_bps < 0
        # Broker delivered the favorable price.
        assert c.broker_bps < 0


# =============================================================================
# Aggregation
# =============================================================================


class TestSummarize:
    def test_empty_summary(self) -> None:
        summary = summarize([])
        assert summary.n_fills == 0
        assert summary.avg_total_bps == 0.0

    def test_summary_averages_correctly(self) -> None:
        records = [
            TCAComponents(
                queue_bps=1.0,
                impact_bps=0.5,
                broker_bps=0.5,
                total_bps=2.0,
                implementation_shortfall_bps=2.0,
                pair="EURUSD",
                side="buy",
                fill_id=f"f{i}",
                ts=T0,
            )
            for i in range(10)
        ]
        summary = summarize(records)
        assert summary.n_fills == 10
        assert summary.avg_queue_bps == pytest.approx(1.0)
        assert summary.avg_impact_bps == pytest.approx(0.5)
        assert summary.avg_broker_bps == pytest.approx(0.5)
        assert summary.avg_total_bps == pytest.approx(2.0)
        assert summary.p50_total_bps == pytest.approx(2.0)

    def test_p95_picks_high_tail(self) -> None:
        records = [
            TCAComponents(
                queue_bps=0.0,
                impact_bps=0.0,
                broker_bps=0.0,
                total_bps=float(i),
                implementation_shortfall_bps=float(i),
                pair="EURUSD",
                side="buy",
                fill_id=f"f{i}",
                ts=T0,
            )
            for i in range(100)
        ]
        summary = summarize(records)
        # p95 over [0..99] → ~94 (round(0.95 * 99) = 94).
        assert 90 <= summary.p95_total_bps <= 99
        # p50 ≈ 50.
        assert 40 <= summary.p50_total_bps <= 60


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_components_to_dict(self) -> None:
        ctx = FillContext(arrival_mid=1.10, submit_spread=0.0001, fill_mid=1.10)
        f = _fill()
        d = compute_tca(f, ctx).to_dict()
        for key in (
            "queue_bps",
            "impact_bps",
            "broker_bps",
            "total_bps",
            "implementation_shortfall_bps",
            "pair",
            "side",
            "fill_id",
            "ts",
        ):
            assert key in d

    def test_summary_to_dict(self) -> None:
        d = summarize([]).to_dict()
        assert d["n_fills"] == 0
        assert "avg_total_bps" in d
