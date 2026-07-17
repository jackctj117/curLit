"""Unit tests — scripts/hyperopt_rate_model.py pure parts (CL-4nnr).

Covers the objective scoring math (incl. the n_trades hard penalty),
the config-introspection param space, and the trial suggestion mapping.
No DB / network / Optuna study required — the trial is a local stub.
"""

from __future__ import annotations

from dataclasses import make_dataclass

import pytest
from scripts.hyperopt_rate_model import (
    DD_WEIGHT,
    MIN_TRADES,
    NUMERIC_SPACE,
    PENALTY_SCORE,
    build_param_space,
    config_diff,
    score_result,
    suggest_params,
)

from src.strategies.rate_diff_mean_reversion import RateDiffMRConfig

BASE_PARAMS = [
    "lookback_days",
    "entry_z_threshold",
    "exit_z_threshold",
    "stop_loss_z",
    "max_holding_days",
]


# =============================================================================
# score_result — objective math
# =============================================================================


class TestScoreResult:
    def test_penalty_below_min_trades(self):
        metrics = {"sharpe": 2.5, "max_drawdown": -0.01}
        assert score_result(metrics, MIN_TRADES - 1) == PENALTY_SCORE

    def test_zero_trades_penalized(self):
        # The exact degenerate case the penalty exists for: flat returns
        # score sharpe=0, dd=0 → 0.0 without the guard, beating honest
        # losing configs.
        assert score_result({"sharpe": 0.0, "max_drawdown": 0.0}, 0) == PENALTY_SCORE

    def test_at_min_trades_uses_formula(self):
        metrics = {"sharpe": 1.2, "max_drawdown": -0.10}
        expected = 1.2 - DD_WEIGHT * 0.10
        assert score_result(metrics, MIN_TRADES) == pytest.approx(expected)

    def test_drawdown_sign_is_absolute(self):
        # max_drawdown is negative by convention; a (weird) positive
        # value must penalize identically.
        neg = score_result({"sharpe": 1.0, "max_drawdown": -0.2}, 20)
        pos = score_result({"sharpe": 1.0, "max_drawdown": 0.2}, 20)
        assert neg == pos == pytest.approx(1.0 - DD_WEIGHT * 0.2)

    def test_missing_keys_default_to_zero(self):
        assert score_result({}, 100) == 0.0

    def test_penalty_is_below_any_plausible_real_score(self):
        # Real scores live in roughly [-4.5, 3]; the penalty must sit
        # strictly below so no-trade regions can never win.
        worst_real = -3.0 - DD_WEIGHT * 1.0
        assert worst_real > PENALTY_SCORE


# =============================================================================
# build_param_space — config introspection
# =============================================================================


def _base_config_cls():
    return make_dataclass(
        "BaseCfg",
        [
            ("pair", str, "EURUSD"),
            ("lookback_days", int, 252),
            ("entry_z_threshold", float, 1.5),
            ("exit_z_threshold", float, 0.3),
            ("stop_loss_z", float, 3.5),
            ("max_holding_days", int, 30),
            ("volatility_target", float, 0.10),
        ],
    )


def _filtered_config_cls():
    return make_dataclass(
        "FilteredCfg",
        [
            ("lookback_days", int, 252),
            ("entry_z_threshold", float, 1.5),
            ("exit_z_threshold", float, 0.3),
            ("stop_loss_z", float, 3.5),
            ("max_holding_days", int, 30),
            ("momentum_lookback_days", int, 5),
            ("regime_max_vol_z", float, 2.0),
            ("use_momentum_filter", bool, False),
            ("regime_filter_enabled", bool, True),
            ("verbose", bool, False),  # bool but not a filter flag
        ],
    )


class TestBuildParamSpace:
    def test_base_config_only_base_params(self):
        space = build_param_space(_base_config_cls())
        assert sorted(space) == sorted(BASE_PARAMS)

    def test_non_searched_numeric_fields_excluded(self):
        space = build_param_space(_base_config_cls())
        assert "volatility_target" not in space
        assert "pair" not in space

    def test_filter_params_included_when_present(self):
        space = build_param_space(_filtered_config_cls())
        assert space["momentum_lookback_days"] == ("int", 3, 10)
        assert space["regime_max_vol_z"] == ("float", 1.0, 3.0)
        assert space["use_momentum_filter"] == ("bool",)
        assert space["regime_filter_enabled"] == ("bool",)

    def test_unrelated_bool_not_searched(self):
        space = build_param_space(_filtered_config_cls())
        assert "verbose" not in space

    def test_bounds_match_spec_table(self):
        space = build_param_space(_base_config_cls())
        assert space["lookback_days"] == ("int", 60, 504)
        assert space["entry_z_threshold"] == ("float", 1.0, 2.8)
        assert space["exit_z_threshold"] == ("float", 0.1, 0.8)
        assert space["stop_loss_z"] == ("float", 2.5, 4.5)
        assert space["max_holding_days"] == ("int", 10, 45)

    def test_real_config_includes_base_params(self):
        # Works both before and after the concurrent filter-param change:
        # base params must always be present; anything extra must come
        # from the NUMERIC_SPACE table or be a filter bool.
        space = build_param_space(RateDiffMRConfig)
        for name in BASE_PARAMS:
            assert name in space, name
        for name, spec in space.items():
            assert spec == NUMERIC_SPACE.get(name, ("bool",)), name

    def test_string_annotations_supported(self):
        # Postponed annotations (from __future__ import annotations)
        # leave field.type as a string — introspection must still work.
        cls = make_dataclass(
            "StrAnnCfg",
            [
                ("lookback_days", "int", 252),
                ("momentum_filter_on", "bool", True),
            ],
        )
        space = build_param_space(cls)
        assert space["lookback_days"] == ("int", 60, 504)
        assert space["momentum_filter_on"] == ("bool",)


# =============================================================================
# suggest_params — trial mapping
# =============================================================================


class _StubTrial:
    """Records suggest_* calls; returns the low bound / first choice."""

    def __init__(self):
        self.calls: list[tuple] = []

    def suggest_int(self, name, low, high):
        self.calls.append(("int", name, low, high))
        return low

    def suggest_float(self, name, low, high):
        self.calls.append(("float", name, low, high))
        return low

    def suggest_categorical(self, name, choices):
        self.calls.append(("cat", name, tuple(choices)))
        return choices[0]


class TestSuggestParams:
    def test_maps_kinds_to_trial_calls(self):
        space = {
            "lookback_days": ("int", 60, 504),
            "entry_z_threshold": ("float", 1.0, 2.8),
            "use_momentum_filter": ("bool",),
        }
        trial = _StubTrial()
        params = suggest_params(trial, space)
        assert params == {
            "lookback_days": 60,
            "entry_z_threshold": 1.0,
            "use_momentum_filter": False,
        }
        assert ("int", "lookback_days", 60, 504) in trial.calls
        assert ("float", "entry_z_threshold", 1.0, 2.8) in trial.calls
        assert ("cat", "use_momentum_filter", (False, True)) in trial.calls

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown spec kind"):
            suggest_params(_StubTrial(), {"x": ("weird", 0, 1)})


# =============================================================================
# config_diff — operator-facing diff (never auto-applied)
# =============================================================================


class TestConfigDiff:
    def test_changed_and_unchanged_marked(self):
        defaults = RateDiffMRConfig()
        lines = config_diff({
            "entry_z_threshold": defaults.entry_z_threshold,  # unchanged
            "max_holding_days": defaults.max_holding_days + 5,
        })
        joined = "\n".join(lines)
        assert f"entry_z_threshold: {defaults.entry_z_threshold} -> " in joined
        assert "(unchanged)" in joined
        assert (
            f"max_holding_days: {defaults.max_holding_days} -> "
            f"{defaults.max_holding_days + 5}" in joined
        )

    def test_param_absent_from_config_reported(self):
        lines = config_diff({"not_a_real_field": 1})
        assert lines == ["  not_a_real_field: <absent> -> 1"]
