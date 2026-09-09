"""Tests for the risk-profile loader (CL-risk-profile)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.risk.risk_profile import (
    BiasConfig,
    HoldingConfig,
    KillSwitchConfig,
    RiskProfile,
    SizingConfig,
    StrategyGatesConfig,
    load_active_profile,
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "risk_profile.yaml"
    p.write_text(body)
    return p


class TestActiveResolution:
    def test_missing_config_defaults_only_in_explicit_development(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="config missing"):
            load_active_profile(tmp_path / "missing.yaml")
        out = load_active_profile(tmp_path / "missing.yaml", allow_development_defaults=True)
        assert out.name == "conservative"
        assert out.sizing.kelly_fraction == 0.25

    def test_active_field_is_default(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """
active: aggressive
profiles:
  conservative: {}
  aggressive:
    sizing:
      kelly_fraction: 0.5
""",
        )
        out = load_active_profile(path)
        assert out.name == "aggressive"
        assert out.sizing.kelly_fraction == 0.5

    def test_env_var_overrides_active(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """
active: conservative
profiles:
  conservative: {}
  aggressive:
    sizing:
      kelly_fraction: 0.5
""",
        )
        with patch.dict(os.environ, {"CURLIT_RISK_PROFILE": "aggressive"}):
            out = load_active_profile(path)
        assert out.name == "aggressive"

    def test_unknown_profile_refuses_startup(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """
active: brain_freeze
profiles:
  conservative: {}
""",
        )
        with pytest.raises(ValueError, match="unknown profile"):
            load_active_profile(path)


class TestInheritance:
    def test_one_deep_inheritance_merges(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """
active: aggressive_short
profiles:
  aggressive:
    sizing:
      kelly_fraction: 0.5
      max_position_pct: 0.4
    kill_switches:
      daily_loss_limit_pct: -0.1
  aggressive_short:
    inherits: aggressive
    bias:
      long_signal_multiplier: 0.0
      short_signal_multiplier: 1.0
""",
        )
        out = load_active_profile(path)
        assert out.name == "aggressive_short"
        # Inherited values:
        assert out.sizing.kelly_fraction == 0.5
        assert out.sizing.max_position_pct == 0.4
        assert out.kill_switches.daily_loss_limit_pct == -0.1
        # Child-specific:
        assert out.bias.long_signal_multiplier == 0.0
        assert out.bias.short_signal_multiplier == 1.0

    def test_child_overrides_parent_within_section(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """
active: child
profiles:
  parent:
    sizing:
      kelly_fraction: 0.5
      max_position_pct: 0.4
  child:
    inherits: parent
    sizing:
      kelly_fraction: 0.75   # override only this — max_position_pct unchanged
""",
        )
        out = load_active_profile(path)
        assert out.sizing.kelly_fraction == 0.75
        # Parent's max_position_pct survives the deep-merge.
        assert out.sizing.max_position_pct == 0.4

    def test_inherits_from_unknown_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """
active: orphan
profiles:
  orphan:
    inherits: ghost
""",
        )
        with pytest.raises(ValueError, match="unknown profile"):
            load_active_profile(path)


class TestRealConfig:
    def test_shipped_config_loads_clean(self) -> None:
        out = load_active_profile()
        assert isinstance(out, RiskProfile)
        # Whichever profile is active, the dataclasses must be populated.
        assert isinstance(out.sizing, SizingConfig)
        assert isinstance(out.kill_switches, KillSwitchConfig)
        assert isinstance(out.strategy_gates, StrategyGatesConfig)
        assert isinstance(out.holding, HoldingConfig)
        assert isinstance(out.bias, BiasConfig)

    def test_aggressive_is_actually_more_aggressive(self) -> None:
        with patch.dict(os.environ, {"CURLIT_RISK_PROFILE": "conservative"}):
            cons = load_active_profile()
        with patch.dict(os.environ, {"CURLIT_RISK_PROFILE": "aggressive"}):
            agg = load_active_profile()

        # Sanity gates: aggressive must be looser in every direction.
        assert agg.sizing.kelly_fraction > cons.sizing.kelly_fraction
        assert agg.sizing.max_position_pct > cons.sizing.max_position_pct
        assert agg.sizing.polymarket_min_edge < cons.sizing.polymarket_min_edge
        assert agg.kill_switches.daily_loss_limit_pct < cons.kill_switches.daily_loss_limit_pct
        assert agg.kill_switches.drawdown_limit_pct < cons.kill_switches.drawdown_limit_pct
        assert agg.strategy_gates.min_r_squared < cons.strategy_gates.min_r_squared
        assert agg.strategy_gates.entry_z_threshold < cons.strategy_gates.entry_z_threshold
        assert agg.holding.max_holding_days < cons.holding.max_holding_days
        # CL-ep0c: trailing stop + open-position correlation looser too.
        assert agg.kill_switches.trailing_stop_pct > cons.kill_switches.trailing_stop_pct
        assert (
            agg.kill_switches.trailing_stop_cooldown_days
            < cons.kill_switches.trailing_stop_cooldown_days
        )
        assert (
            agg.kill_switches.open_position_corr_threshold
            > cons.kill_switches.open_position_corr_threshold
        )

    def test_cl_ep0c_keys_present_in_both_profiles(self) -> None:
        """CL-ep0c config keys ship with explicit values in the yaml."""
        with patch.dict(os.environ, {"CURLIT_RISK_PROFILE": "conservative"}):
            cons = load_active_profile().kill_switches
        with patch.dict(os.environ, {"CURLIT_RISK_PROFILE": "aggressive"}):
            agg = load_active_profile().kill_switches
        assert cons.trailing_stop_pct == 0.10
        assert cons.trailing_stop_cooldown_days == 7
        assert cons.open_position_corr_threshold == 0.85
        assert cons.open_position_corr_lookback_days == 60
        assert agg.trailing_stop_pct == 0.20
        assert agg.trailing_stop_cooldown_days == 3
        assert agg.open_position_corr_threshold == 0.92
        assert agg.open_position_corr_lookback_days == 60

    def test_aggressive_short_inherits_and_biases(self) -> None:
        with patch.dict(os.environ, {"CURLIT_RISK_PROFILE": "aggressive_short"}):
            out = load_active_profile()
        assert out.name == "aggressive_short"
        # Sizing inherits from aggressive:
        assert out.sizing.kelly_fraction == 0.5
        # Bias-specific knobs:
        assert out.bias.long_signal_multiplier == 0.0
        assert out.bias.short_signal_multiplier == 1.0
        assert out.bias.prefer_polymarket_no is True


class TestTypoRejection:
    def test_unknown_field_in_section_rejected(self, tmp_path: Path) -> None:
        # CL-0deu.9 intentionally reverses the old unsafe typo-tolerance policy.
        path = _write_yaml(
            tmp_path,
            """
active: oops
profiles:
  oops:
    sizing:
      kelly_fraction: 0.3
      misspelled_field: 99   # invalid trading configuration
""",
        )
        with pytest.raises(ValueError, match="sizing: unknown field"):
            load_active_profile(path)
