"""Tests for AllocationPolicy (CL-15r4) — the promotion + ramp logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.portfolio.allocation_policy import (
    AllocationPolicy,
    PolicyConfig,
    PromotionVerdict,
)


def _returns_with_sharpe(target_sharpe: float, n: int = 100) -> pd.Series:
    """Build a constant-return series with the target annualized Sharpe.

    Constant returns => zero std => undefined Sharpe; instead seed with
    a deterministic mean and std that produce the target. Sharpe =
    (mean / std) * sqrt(252).
    """
    if target_sharpe == 0:
        # Use random-walk-like noise centered on zero.
        return pd.Series([0.001 if i % 2 else -0.001 for i in range(n)])
    # mean=target_sharpe/sqrt(252) * std  (pick std=0.01)
    std = 0.01
    mean = target_sharpe / (252.0**0.5) * std
    # Build a series with this exact mean and std.
    base = [-std, std] * (n // 2)  # mean=0, std≈std
    return pd.Series([x + mean for x in base[:n]])


class TestEvaluate:
    def test_holds_when_paper_too_short(self) -> None:
        policy = AllocationPolicy()
        paper_start = datetime.now(UTC) - timedelta(days=30)
        d = policy.evaluate(
            "s1",
            paper_start,
            _returns_with_sharpe(1.0, 30),
        )
        assert d.verdict is PromotionVerdict.HOLD
        assert "Paper days" in d.reason

    def test_rejects_when_sharpe_below_floor(self) -> None:
        policy = AllocationPolicy()
        paper_start = datetime.now(UTC) - timedelta(days=120)
        # Sharpe below floor (default 0.5) → REJECT
        d = policy.evaluate(
            "s1",
            paper_start,
            _returns_with_sharpe(0.1, 100),
        )
        assert d.verdict is PromotionVerdict.REJECT

    def test_promotes_at_default_pct(self) -> None:
        policy = AllocationPolicy()
        paper_start = datetime.now(UTC) - timedelta(days=120)
        d = policy.evaluate(
            "s1",
            paper_start,
            _returns_with_sharpe(1.5, 100),
        )
        assert d.verdict is PromotionVerdict.PROMOTE
        assert d.initial_weight == pytest.approx(0.05)

    def test_per_strategy_override(self) -> None:
        cfg = PolicyConfig(
            overrides={"flagship": {"min_paper_days": 0, "initial_pct": 0.10}},
        )
        policy = AllocationPolicy(cfg)
        paper_start = datetime.now(UTC) - timedelta(days=5)
        d = policy.evaluate(
            "flagship",
            paper_start,
            _returns_with_sharpe(1.5, 30),
        )
        assert d.verdict is PromotionVerdict.PROMOTE
        assert d.initial_weight == pytest.approx(0.10)


class TestRampWeight:
    def test_no_history_keeps_current(self) -> None:
        policy = AllocationPolicy()
        assert policy.ramp_weight("s1", 0.05, [], 0) == 0.05

    def test_first_profitable_month_keeps_initial(self) -> None:
        policy = AllocationPolicy()
        # streak=1 → initial_pct + ramp*(1-1) = initial_pct
        assert policy.ramp_weight("s1", 0.05, [0.01], 1) == pytest.approx(0.05)

    def test_two_profitable_months_adds_one_ramp(self) -> None:
        policy = AllocationPolicy()
        new = policy.ramp_weight("s1", 0.05, [0.01, 0.01], 2)
        assert new == pytest.approx(0.10)  # 0.05 + 0.05

    def test_loss_month_resets_to_initial(self) -> None:
        policy = AllocationPolicy()
        # Three good months then a loss → streak=0 → reset.
        new = policy.ramp_weight("s1", 0.20, [0.01, 0.01, 0.01, -0.005], 4)
        assert new == pytest.approx(0.05)

    def test_caps_at_max_pct(self) -> None:
        cfg = PolicyConfig(initial_pct=0.05, ramp_per_month=0.10, max_pct=0.30)
        policy = AllocationPolicy(cfg)
        # 6 profitable months: 0.05 + 0.10 * 5 = 0.55, capped to 0.30
        new = policy.ramp_weight("s1", 0.20, [0.01] * 6, 6)
        assert new == pytest.approx(0.30)

    def test_partial_loss_streak(self) -> None:
        policy = AllocationPolicy()
        # Trailing 2 wins after a loss: streak=2 → 0.05 + 0.05 = 0.10
        new = policy.ramp_weight("s1", 0.20, [-0.005, 0.01, 0.01], 3)
        assert new == pytest.approx(0.10)
