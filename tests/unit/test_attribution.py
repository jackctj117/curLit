"""Tests for portfolio.attribution.PnLAttributor (CL-8dq + CL-mdle).

Covers:
  - Single-strategy fills, simple long-then-flat realization
  - Long → partial close → fully-close
  - Short cycle (sell-open, buy-cover)
  - FIFO across multiple lots
  - Multi-strategy proportional split
  - Idempotency on re-attribution
  - CL-mdle cost-component decomposition: signal/spread/slippage/swap
  - emit_pnl_metrics happy-path
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine

from src.portfolio.attribution import (
    PnLAttributor,
    emit_pnl_metrics,
)


@pytest.fixture
def attr_engine(tmp_path):  # type: ignore[no-untyped-def]
    return create_engine(f"sqlite:///{tmp_path / 'attr.db'}")


@pytest.fixture
def attr(attr_engine):  # type: ignore[no-untyped-def]
    return PnLAttributor(attr_engine)


_T0 = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


def _ts(seconds: int) -> datetime:
    from datetime import timedelta as _td

    return _T0 + _td(seconds=seconds)


class TestRealization:
    def test_long_round_trip(self, attr: PnLAttributor) -> None:
        attr.attribute_fill("EURUSD", 1000.0, 1.10, ts=_ts(0), strategy_id="s1", fill_id="f1")
        attr.attribute_fill("EURUSD", -1000.0, 1.12, ts=_ts(60), strategy_id="s1", fill_id="f2")
        pnl = attr.compute_strategy_pnl("s1")
        assert pnl.realized == pytest.approx(20.0)  # 0.02 * 1000
        assert pnl.open_quantity == pytest.approx(0.0)
        assert pnl.unrealized == pytest.approx(0.0)

    def test_short_round_trip(self, attr: PnLAttributor) -> None:
        attr.attribute_fill("EURUSD", -1000.0, 1.12, ts=_ts(0), strategy_id="s1", fill_id="f1")
        attr.attribute_fill("EURUSD", 1000.0, 1.10, ts=_ts(60), strategy_id="s1", fill_id="f2")
        pnl = attr.compute_strategy_pnl("s1")
        # Sold high (1.12), bought back low (1.10) → +0.02 * 1000.
        assert pnl.realized == pytest.approx(20.0)

    def test_partial_close(self, attr: PnLAttributor) -> None:
        attr.attribute_fill("EURUSD", 1000.0, 1.10, ts=_ts(0), strategy_id="s1", fill_id="f1")
        attr.attribute_fill("EURUSD", -400.0, 1.13, ts=_ts(60), strategy_id="s1", fill_id="f2")
        pnl = attr.compute_strategy_pnl("s1", last_price={"EURUSD": 1.15})
        # Realized: 0.03 * 400 = 12
        assert pnl.realized == pytest.approx(12.0)
        assert pnl.open_quantity == pytest.approx(600.0)
        # Unrealized: (1.15 - 1.10) * 600 = 30
        assert pnl.unrealized == pytest.approx(30.0)

    def test_fifo_across_lots(self, attr: PnLAttributor) -> None:
        attr.attribute_fill("EURUSD", 500.0, 1.10, ts=_ts(0), strategy_id="s1", fill_id="a")
        attr.attribute_fill("EURUSD", 500.0, 1.15, ts=_ts(60), strategy_id="s1", fill_id="b")
        # Sell 700: closes all of lot1 (500@1.10) and 200 of lot2 (200@1.15)
        attr.attribute_fill("EURUSD", -700.0, 1.20, ts=_ts(120), strategy_id="s1", fill_id="c")
        pnl = attr.compute_strategy_pnl("s1")
        # 500*(1.20-1.10) + 200*(1.20-1.15) = 50 + 10 = 60
        assert pnl.realized == pytest.approx(60.0)
        assert pnl.open_quantity == pytest.approx(300.0)


class TestMultiStrategySplit:
    def test_proportional_split_writes_two_rows(
        self,
        attr: PnLAttributor,
    ) -> None:
        attr.attribute_fill(
            "EURUSD",
            1000.0,
            1.10,
            ts=_ts(0),
            proportions={"s1": 0.6, "s2": 0.4},
            fill_id="shared",
        )
        s1 = attr.compute_strategy_pnl("s1")
        s2 = attr.compute_strategy_pnl("s2")
        assert s1.open_quantity == pytest.approx(600.0)
        assert s2.open_quantity == pytest.approx(400.0)

    def test_proportions_must_sum_to_one(self, attr: PnLAttributor) -> None:
        with pytest.raises(AssertionError):
            attr.attribute_fill(
                "EURUSD",
                1000.0,
                1.10,
                ts=_ts(0),
                proportions={"s1": 0.6, "s2": 0.5},
                fill_id="bad",
            )


class TestIdempotency:
    def test_same_fill_id_does_not_duplicate(self, attr: PnLAttributor) -> None:
        for _ in range(3):
            attr.attribute_fill(
                "EURUSD", 1000.0, 1.10, ts=_ts(0), strategy_id="s1", fill_id="dedup"
            )
        pnl = attr.compute_strategy_pnl("s1")
        assert pnl.n_fills == 1
        assert pnl.open_quantity == pytest.approx(1000.0)


class TestCostDecomposition:
    def test_components_sum_into_pnl(self, attr: PnLAttributor) -> None:
        attr.attribute_fill(
            "EURUSD",
            1000.0,
            1.10,
            ts=_ts(0),
            strategy_id="s1",
            fill_id="f1",
            cost_components={
                "signal_bps": 5.0,
                "spread_bps": 1.0,
                "slippage_bps": 0.5,
                "swap_bps": 0.0,
            },
        )
        attr.attribute_fill(
            "EURUSD",
            -1000.0,
            1.12,
            ts=_ts(60),
            strategy_id="s1",
            fill_id="f2",
            cost_components={
                "signal_bps": 5.0,
                "spread_bps": 1.0,
                "slippage_bps": 0.5,
                "swap_bps": 0.0,
            },
        )
        pnl = attr.compute_strategy_pnl("s1")
        assert pnl.signal_alpha == pytest.approx(10.0)
        assert pnl.spread_cost == pytest.approx(2.0)
        assert pnl.slippage_cost == pytest.approx(1.0)
        assert pnl.swap_cost == pytest.approx(0.0)

    def test_no_components_means_zero_decomposition(
        self,
        attr: PnLAttributor,
    ) -> None:
        attr.attribute_fill("EURUSD", 1000.0, 1.10, ts=_ts(0), strategy_id="s1", fill_id="f1")
        pnl = attr.compute_strategy_pnl("s1")
        assert pnl.signal_alpha == 0
        assert pnl.spread_cost == 0


class TestListAndEmit:
    def test_list_strategies(self, attr: PnLAttributor) -> None:
        attr.attribute_fill("EURUSD", 1000, 1.10, strategy_id="s1", fill_id="x")
        attr.attribute_fill("GBPUSD", 1000, 1.30, strategy_id="s2", fill_id="y")
        assert sorted(attr.list_strategies()) == ["s1", "s2"]

    def test_emit_pnl_metrics_returns_dict(self, attr: PnLAttributor) -> None:
        attr.attribute_fill("EURUSD", 1000, 1.10, strategy_id="s1", fill_id="x")
        out = emit_pnl_metrics(attr, last_price={"EURUSD": 1.12})
        assert "s1" in out
        assert out["s1"].open_quantity == pytest.approx(1000.0)
        # Unrealized = (1.12 - 1.10) * 1000 = 20
        assert out["s1"].unrealized == pytest.approx(20.0)


class TestSinceFilter:
    def test_since_excludes_earlier_fills(self, attr: PnLAttributor) -> None:
        attr.attribute_fill("EURUSD", 1000, 1.10, ts=_ts(0), strategy_id="s1", fill_id="old")
        attr.attribute_fill("EURUSD", -1000, 1.12, ts=_ts(60), strategy_id="s1", fill_id="recent")
        # Only the recent fill is in scope — appears as a -1000 short open.
        pnl = attr.compute_strategy_pnl("s1", since=_ts(30))
        assert pnl.realized == 0
        assert pnl.open_quantity == pytest.approx(-1000.0)
