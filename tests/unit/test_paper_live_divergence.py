"""Unit tests — edge_testing.paper_live_divergence (G4 / CL-yq5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.edge_testing.paper_live_divergence import (
    PaperLiveDivergence,
    _price_diff_bps,
)
from src.execution.broker import Fill

T0 = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)


def _fill(
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    ts: datetime,
    fill_id: str | None = None,
) -> Fill:
    return Fill(
        order_id=order_id,
        fill_id=fill_id or f"{order_id}-fill",
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=ts,
    )


# =============================================================================
# Price-diff helper
# =============================================================================


class TestPriceDiffBps:
    def test_buy_at_higher_live_is_positive_bps(self) -> None:
        # Buy at 1.10 paper, 1.10011 live → 1 bp adverse for us.
        bps = _price_diff_bps(1.10, 1.10011, "buy")
        assert bps == pytest.approx(1.0, abs=0.05)

    def test_sell_at_lower_live_is_positive_bps(self) -> None:
        # Sell at 1.10 paper, 1.09989 live → 1 bp adverse for us.
        bps = _price_diff_bps(1.10, 1.09989, "sell")
        assert bps == pytest.approx(1.0, abs=0.05)

    def test_buy_at_lower_live_is_negative_bps(self) -> None:
        # Live filled cheaper than paper → favorable to us → negative bps.
        bps = _price_diff_bps(1.10, 1.09989, "buy")
        assert bps == pytest.approx(-1.0, abs=0.05)

    def test_zero_paper_price_returns_zero(self) -> None:
        assert _price_diff_bps(0.0, 1.0, "buy") == 0.0


# =============================================================================
# Matching
# =============================================================================


class TestMatching:
    def test_simple_one_to_one_match(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [
            _fill(
                "o1",
                "EURUSD",
                "buy",
                1000,
                1.10011,
                T0 + timedelta(milliseconds=200),
            ),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_matched == 1
        assert report.n_paper_fills == 1
        assert report.n_live_fills == 1
        assert report.avg_price_diff_bps == pytest.approx(1.0, abs=0.05)
        assert report.avg_latency_ms == pytest.approx(200.0)

    def test_match_within_tolerance_only(self) -> None:
        # Live fill 90 seconds later — outside default 60s tolerance.
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [
            _fill(
                "o1",
                "EURUSD",
                "buy",
                1000,
                1.10,
                T0 + timedelta(seconds=90),
            ),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_matched == 0
        assert report.unmatched_paper_intents == ["o1-fill"]

    def test_different_symbols_dont_match(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [_fill("o1", "USDJPY", "buy", 1000, 150.0, T0)]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_matched == 0

    def test_different_sides_dont_match(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [_fill("o1", "EURUSD", "sell", 1000, 1.10, T0)]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_matched == 0

    def test_each_live_fill_used_at_most_once(self) -> None:
        # 2 paper fills, 1 live fill — only 1 should match.
        paper = [
            _fill("o1", "EURUSD", "buy", 1000, 1.10, T0),
            _fill("o2", "EURUSD", "buy", 1000, 1.10, T0 + timedelta(seconds=1)),
        ]
        live = [
            _fill("oL", "EURUSD", "buy", 1000, 1.10011, T0 + timedelta(milliseconds=100)),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_matched == 1
        assert len(report.unmatched_paper_intents) == 1

    def test_signal_match_rate(self) -> None:
        # 4 paper, 2 live → match_rate 0.5.
        paper = [
            _fill(f"o{i}", "EURUSD", "buy", 1000, 1.10, T0 + timedelta(seconds=i * 5))
            for i in range(4)
        ]
        live = [
            _fill(f"oL{i}", "EURUSD", "buy", 1000, 1.10011, T0 + timedelta(seconds=i * 5))
            for i in range(2)
        ]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.signal_match_rate == pytest.approx(0.5)


# =============================================================================
# Flagging
# =============================================================================


class TestFlagging:
    def test_diff_above_threshold_flagged(self) -> None:
        # 5 bps live drift on a buy → flagged at default 2 bp threshold.
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [
            _fill("oL", "EURUSD", "buy", 1000, 1.10055, T0 + timedelta(milliseconds=50)),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_flagged == 1
        assert report.matched_pairs[0].flagged is True

    def test_diff_below_threshold_not_flagged(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [
            _fill(
                "oL",
                "EURUSD",
                "buy",
                1000,
                1.1001,  # ~0.9 bp
                T0 + timedelta(milliseconds=50),
            ),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_flagged == 0
        assert report.matched_pairs[0].flagged is False

    def test_custom_threshold_changes_flag_count(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [
            _fill(
                "oL",
                "EURUSD",
                "buy",
                1000,
                1.1001,  # ~0.9 bp
                T0 + timedelta(milliseconds=50),
            ),
        ]
        report = PaperLiveDivergence(flag_threshold_bps=0.5).compare(paper, live)
        # At 0.5 bp threshold, ~0.9 bp diff is flagged.
        assert report.n_flagged == 1


# =============================================================================
# Aggregation
# =============================================================================


class TestAggregation:
    def test_avg_diff_signed(self) -> None:
        # One favorable (-2 bp), one adverse (+4 bp) → avg +1 bp.
        paper = [
            _fill("o1", "EURUSD", "buy", 1000, 1.10, T0),
            _fill("o2", "EURUSD", "buy", 1000, 1.10, T0 + timedelta(seconds=10)),
        ]
        live = [
            _fill("oL1", "EURUSD", "buy", 1000, 1.09978, T0 + timedelta(milliseconds=100)),
            _fill(
                "oL2", "EURUSD", "buy", 1000, 1.10044, T0 + timedelta(seconds=10, milliseconds=100)
            ),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_matched == 2
        assert report.avg_price_diff_bps == pytest.approx(1.0, abs=0.1)
        # Absolute average = 3 bps (averaging |2| and |4|).
        assert report.avg_abs_price_diff_bps == pytest.approx(3.0, abs=0.1)

    def test_no_matches_yields_zero_metrics(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [_fill("oL", "USDJPY", "buy", 1000, 150.0, T0)]
        report = PaperLiveDivergence().compare(paper, live)
        assert report.n_matched == 0
        assert report.avg_price_diff_bps == 0.0
        assert report.avg_latency_ms == 0.0
        assert report.signal_match_rate == 0.0

    def test_empty_inputs(self) -> None:
        report = PaperLiveDivergence().compare([], [])
        assert report.n_matched == 0
        assert report.signal_match_rate == 0.0


# =============================================================================
# Construction validation
# =============================================================================


class TestConstruction:
    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(AssertionError):
            PaperLiveDivergence(flag_threshold_bps=0.0)
        with pytest.raises(AssertionError):
            PaperLiveDivergence(flag_threshold_bps=-1.0)

    def test_invalid_tolerance_rejected(self) -> None:
        with pytest.raises(AssertionError):
            PaperLiveDivergence(match_tolerance_seconds=0.0)


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_to_dict_keys(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [
            _fill("oL", "EURUSD", "buy", 1000, 1.10011, T0 + timedelta(milliseconds=100)),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        d = report.to_dict()
        for key in (
            "n_paper_fills",
            "n_live_fills",
            "n_matched",
            "avg_price_diff_bps",
            "avg_abs_price_diff_bps",
            "avg_latency_ms",
            "signal_match_rate",
            "n_flagged",
            "flag_threshold_bps",
        ):
            assert key in d, f"to_dict missing {key}"

    def test_matched_pair_to_dict(self) -> None:
        paper = [_fill("o1", "EURUSD", "buy", 1000, 1.10, T0)]
        live = [
            _fill("oL", "EURUSD", "buy", 1000, 1.10011, T0 + timedelta(milliseconds=100)),
        ]
        report = PaperLiveDivergence().compare(paper, live)
        pair_d = report.matched_pairs[0].to_dict()
        assert pair_d["symbol"] == "EURUSD"
        assert pair_d["flagged"] is False
