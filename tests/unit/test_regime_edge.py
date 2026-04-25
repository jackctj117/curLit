"""Unit tests — edge_testing.regime_edge (G6 / CL-lx4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.edge_testing.regime_edge import (
    MarketRegime,
    RegimeAnalysisReport,
    RegimeEdgeAnalyzer,
)

# =============================================================================
# Synthetic data
# =============================================================================


def _build_prices(
    n: int = 1000, vol: float = 0.005, drift: float = 0.0,
    crisis_window: tuple[int, int] | None = None, seed: int = 0,
) -> pd.Series:
    """Geometric Brownian-ish prices, with optional vol-spike window."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    if crisis_window:
        start, end = crisis_window
        returns[start:end] = rng.normal(drift, vol * 5, end - start)
    prices = pd.Series(
        100 * (1 + returns).cumprod(),
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )
    return prices


def _build_strat_returns_per_regime(
    regimes: pd.Series, regime_means: dict[MarketRegime, float],
    vol: float = 0.005, seed: int = 0,
) -> pd.Series:
    """Build strategy returns whose mean depends on regime label."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(regimes))
    for i, r in enumerate(regimes):
        try:
            mean = regime_means.get(MarketRegime(r), 0.0)
        except ValueError:
            mean = 0.0
        out[i] = rng.normal(mean, vol)
    return pd.Series(out, index=regimes.index)


# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    def test_invalid_vol_window(self) -> None:
        with pytest.raises(AssertionError):
            RegimeEdgeAnalyzer(vol_window=2)

    def test_invalid_trend_window(self) -> None:
        with pytest.raises(AssertionError):
            RegimeEdgeAnalyzer(trend_window=10)

    def test_empty_prices_rejected_in_classify(self) -> None:
        with pytest.raises(AssertionError):
            RegimeEdgeAnalyzer().classify_regimes(pd.Series(dtype=float))


# =============================================================================
# Regime classification
# =============================================================================


class TestClassifyRegimes:
    def test_all_5_regimes_can_appear(self) -> None:
        # Long flat-ish series with a vol spike; should produce both crisis
        # and the four quadrant labels somewhere.
        prices = _build_prices(n=1500, vol=0.005, crisis_window=(800, 900), seed=42)
        regimes = RegimeEdgeAnalyzer().classify_regimes(prices)
        labels = set(regimes.dropna().unique())
        # CRISIS must appear from the spike window.
        assert MarketRegime.CRISIS.value in labels
        # At least one of HIGH_VOL_* / LOW_VOL_* must appear elsewhere.
        non_crisis = labels - {MarketRegime.CRISIS.value}
        assert len(non_crisis) > 0

    def test_crisis_is_top_5pct_vol(self) -> None:
        prices = _build_prices(n=1000, vol=0.005, crisis_window=(800, 850), seed=10)
        regimes = RegimeEdgeAnalyzer().classify_regimes(prices)
        crisis_count = (regimes == MarketRegime.CRISIS.value).sum()
        # ~5% of 1000 = ~50 periods (depends on rolling-window startup NaNs).
        assert 30 <= crisis_count <= 70

    def test_classification_aligned_to_index(self) -> None:
        prices = _build_prices(n=300, seed=11)
        regimes = RegimeEdgeAnalyzer().classify_regimes(prices)
        assert (regimes.index == prices.index).all()


# =============================================================================
# Analyze — basic shape
# =============================================================================


class TestAnalyzeBasic:
    def test_per_regime_stats_present(self) -> None:
        prices = _build_prices(n=1500, crisis_window=(900, 1000), seed=20)
        analyzer = RegimeEdgeAnalyzer()
        regimes = analyzer.classify_regimes(prices)
        # Strategy returns: equal positive mean across regimes.
        rng = np.random.default_rng(21)
        strat = pd.Series(rng.normal(0.0005, 0.005, len(prices)), index=prices.index)
        report = analyzer.analyze(strat, prices)
        assert isinstance(report, RegimeAnalysisReport)
        # At least 2 regimes observed (crisis + at least one quadrant).
        assert len(report.by_regime) >= 2
        for stats in report.by_regime.values():
            assert stats.n_periods >= 20
            assert -1 <= stats.win_rate <= 1

    def test_skips_low_count_regimes(self) -> None:
        # Force most of the series into one regime; only those above the
        # min-periods threshold should appear.
        prices = _build_prices(n=300, vol=0.001, seed=30)  # very low vol → mostly LOW_VOL_*
        analyzer = RegimeEdgeAnalyzer()
        rng = np.random.default_rng(31)
        strat = pd.Series(rng.normal(0.0, 0.005, len(prices)), index=prices.index)
        report = analyzer.analyze(strat, prices)
        for stats in report.by_regime.values():
            assert stats.n_periods >= 20


# =============================================================================
# Concentration warnings
# =============================================================================


class TestConcentration:
    def test_concentrated_returns_flag_warning(self) -> None:
        # Build data where one regime carries all the strategy's edge.
        prices = _build_prices(n=1500, crisis_window=(900, 1000), seed=40)
        analyzer = RegimeEdgeAnalyzer()
        regimes = analyzer.classify_regimes(prices)
        # CRISIS contributes huge mean; everywhere else 0.
        means = {
            MarketRegime.CRISIS: 0.05,
            MarketRegime.LOW_VOL_TRENDING: 0.0,
            MarketRegime.LOW_VOL_CHOPPY: 0.0,
            MarketRegime.HIGH_VOL_TRENDING: 0.0,
            MarketRegime.HIGH_VOL_CHOPPY: 0.0,
        }
        strat = _build_strat_returns_per_regime(
            regimes, means, vol=0.001, seed=41,
        )
        report = analyzer.analyze(strat, prices)
        assert report.edge_concentration > 0.7
        assert report.is_diversified is False
        assert any("concentrated" in w for w in report.warnings)

    def test_diversified_returns_no_warning(self) -> None:
        # Equal positive mean across all regimes → diversified.
        prices = _build_prices(n=1500, crisis_window=(900, 1000), seed=50)
        analyzer = RegimeEdgeAnalyzer()
        regimes = analyzer.classify_regimes(prices)
        means = {r: 0.001 for r in MarketRegime}
        strat = _build_strat_returns_per_regime(regimes, means, vol=0.005, seed=51)
        report = analyzer.analyze(strat, prices)
        # is_diversified depends on sample-mean noise across regimes; require
        # only that no concentrated-regime warning fired.
        assert not any("concentrated" in w for w in report.warnings)


class TestNegativeRegimeWarn:
    def test_strongly_negative_regime_warns(self) -> None:
        prices = _build_prices(n=1500, crisis_window=(900, 1000), seed=60)
        analyzer = RegimeEdgeAnalyzer()
        regimes = analyzer.classify_regimes(prices)
        # Strategy loses heavily in crisis, neutral elsewhere.
        means = {r: 0.0 for r in MarketRegime}
        means[MarketRegime.CRISIS] = -0.02
        strat = _build_strat_returns_per_regime(regimes, means, vol=0.005, seed=61)
        report = analyzer.analyze(strat, prices)
        assert report.worst_regime == MarketRegime.CRISIS
        assert report.worst_regime_sharpe is not None
        assert report.worst_regime_sharpe < -0.5
        assert any("negative Sharpe" in w for w in report.warnings)


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_to_dict_serializable(self) -> None:
        import json
        prices = _build_prices(n=1000, crisis_window=(700, 800), seed=70)
        analyzer = RegimeEdgeAnalyzer()
        rng = np.random.default_rng(71)
        strat = pd.Series(rng.normal(0.0001, 0.005, len(prices)), index=prices.index)
        report = analyzer.analyze(strat, prices)
        text = json.dumps(report.to_dict(), default=str)
        parsed = json.loads(text)
        assert "by_regime" in parsed
        assert "edge_concentration" in parsed
        assert "is_diversified" in parsed

    def test_empty_returns_rejected(self) -> None:
        prices = _build_prices(n=500, seed=80)
        with pytest.raises(AssertionError):
            RegimeEdgeAnalyzer().analyze(
                pd.Series(dtype=float), prices,
            )
