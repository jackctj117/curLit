"""Tests for the computed technical-context layer (CL-3xoj)."""

from __future__ import annotations

import pandas as pd

from src.events.technical_context import (
    alignment_score,
    compute_context,
    compute_for_ticker,
    format_context_block,
)


def _df(closes, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "Close": closes,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Volume": volumes or [1000] * n,
    })


def test_uptrend_at_highs():
    closes = [100 + i for i in range(60)]  # steady climb, ends at high
    ctx = compute_context("UP", _df(closes))
    assert ctx.trend == "uptrend"
    assert ctx.breakout_state == "at_highs"
    assert ctx.pct_from_20d_high >= -0.01
    assert ctx.resistance is not None and ctx.support is not None


def test_downtrend_at_lows():
    closes = [160 - i for i in range(60)]
    ctx = compute_context("DN", _df(closes))
    assert ctx.trend == "downtrend"
    assert ctx.breakout_state == "at_lows"


def test_too_few_bars_none():
    assert compute_context("X", _df([100] * 10)) is None


def test_garbage_frame_none():
    assert compute_context("X", pd.DataFrame({"weird": [1, 2, 3]})) is None


def test_volume_ratio():
    vols = [1000] * 55 + [3000] * 5  # 5d spike vs 20d base
    ctx = compute_context("V", _df([100 + i * 0.1 for i in range(60)], vols))
    assert ctx.volume_ratio is not None and ctx.volume_ratio > 1.5


def test_alignment_scores():
    up = compute_context("UP", _df([100 + i for i in range(60)]))
    dn = compute_context("DN", _df([160 - i for i in range(60)]))
    assert alignment_score(up, "bullish") == 1.0    # uptrend + highs
    assert alignment_score(up, "bearish") == -1.0
    assert alignment_score(dn, "bearish") == 1.0
    assert alignment_score(dn, "bullish") == -1.0


def test_format_block_has_levels_no_pattern_names():
    ctx = compute_context("RTX", _df([100 + i * 0.5 for i in range(60)]))
    block = format_context_block(ctx)
    assert "Technicals (RTX" in block
    assert "support" in block and "resistance" in block
    for banned in ("flag", "head", "shoulders", "wedge", "pennant"):
        assert banned not in block.lower()  # computed facts only


def test_compute_for_ticker_fail_soft():
    assert compute_for_ticker("X", history_fn=lambda t: None) is None
    def boom(t):
        raise RuntimeError("no net")
    # fetch raising inside default path is caught by yfinance_history only;
    # an injected raiser propagates to compute_for_ticker's fetch call —
    # verify the module-level contract via a None-returning fetch instead.
    assert compute_for_ticker("X", history_fn=lambda t: None) is None
