"""Risk-profile loader (CL-risk-profile).

A single config dial that scales the system's risk posture. Strategies
and risk managers call ``load_active_profile()`` at boot and read the
returned dataclass.

Why a separate module: each strategy needs to read the same numbers
(kelly_fraction, max_position_pct, daily_loss_limit_pct, etc) at
construction time. Centralizing the load + parse keeps the strategy
classes free of YAML-parsing logic and gives operators ONE place to
audit the active risk settings.

Override resolution:
  1. ``CURLIT_RISK_PROFILE`` env var (highest priority)
  2. ``active:`` field in configs/risk_profile.yaml
  3. Hardcoded fallback to "conservative"

Aggressive-only knobs (``bias`` block) are present on the
aggressive_short profile only — strategies that don't bias their
signal distributions ignore the field gracefully.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH: Path = Path("configs/risk_profile.yaml")
_FALLBACK_PROFILE: str = "conservative"
_ENV_OVERRIDE: str = "CURLIT_RISK_PROFILE"


@dataclass(frozen=True)
class SizingConfig:
    kelly_fraction: float = 0.25
    volatility_target_pct: float = 0.10
    max_position_pct: float = 0.20
    max_polymarket_market_fraction: float = 0.05
    polymarket_min_edge: float = 0.03


@dataclass(frozen=True)
class KillSwitchConfig:
    daily_loss_limit_pct: float = -0.03
    drawdown_limit_pct: float = -0.20
    single_strategy_dd_pct: float = -0.25
    strategy_correlation_spike: float = 0.90
    # Polymarket absolute loss caps in USD (CL-983f) — consumed by
    # src/risk/polymarket_loss_caps.py. Sized against the $100 initial
    # mainnet cap: one market can burn at most $25, one UTC day $50.
    polymarket_per_market_loss_cap_usd: float = 25.0
    polymarket_per_day_loss_cap_usd: float = 50.0


@dataclass(frozen=True)
class StrategyGatesConfig:
    min_r_squared: float = 0.10
    entry_z_threshold: float = 1.5
    paper_relevance_floor: float = 5.0
    min_paper_sharpe: float = 0.5
    min_paper_days: int = 90


@dataclass(frozen=True)
class HoldingConfig:
    max_holding_days: int = 30


@dataclass(frozen=True)
class BiasConfig:
    """Optional per-direction signal multipliers. Used by
    aggressive_short to bias the portfolio toward short positions."""

    long_signal_multiplier: float = 1.0
    short_signal_multiplier: float = 1.0
    prefer_polymarket_no: bool = False


@dataclass(frozen=True)
class RiskProfile:
    name: str
    sizing: SizingConfig
    kill_switches: KillSwitchConfig
    strategy_gates: StrategyGatesConfig
    holding: HoldingConfig
    bias: BiasConfig = field(default_factory=BiasConfig)


def load_active_profile(
    config_path: Path | str = _DEFAULT_CONFIG_PATH,
) -> RiskProfile:
    """Resolve which profile is active + return its parsed config.

    Resolution order:
      1. ``CURLIT_RISK_PROFILE`` env var (operator can flip at boot)
      2. ``active:`` field in the YAML
      3. Hardcoded fallback "conservative"

    Profile inheritance (``inherits: parent`` on the child) is one-deep;
    the child's keys override the parent's. Useful for ``aggressive_short``
    inheriting from ``aggressive``.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "risk profile config not found at %s — using conservative defaults",
            path,
        )
        return _build_profile(_FALLBACK_PROFILE, {})

    raw = yaml.safe_load(path.read_text()) or {}
    profiles = raw.get("profiles") or {}

    env_override = os.environ.get(_ENV_OVERRIDE, "").strip()
    name = env_override or raw.get("active") or _FALLBACK_PROFILE

    if name not in profiles:
        logger.warning(
            "risk profile %r not in config — falling back to %s",
            name, _FALLBACK_PROFILE,
        )
        name = _FALLBACK_PROFILE
        if name not in profiles:
            return _build_profile(name, {})

    body = dict(profiles[name])
    # Resolve one-level inheritance (e.g. aggressive_short inherits aggressive).
    parent = body.pop("inherits", None)
    if parent:
        if parent not in profiles:
            msg = (
                f"risk profile {name!r} inherits from unknown profile "
                f"{parent!r}"
            )
            raise ValueError(msg)
        merged: dict[str, Any] = dict(profiles[parent])
        # Per-section deep-merge: child keys override parent keys
        # within a section, but missing sections fall through to parent.
        for section, child_val in body.items():
            if (
                section in merged
                and isinstance(merged[section], dict)
                and isinstance(child_val, dict)
            ):
                section_merged = dict(merged[section])
                section_merged.update(child_val)
                merged[section] = section_merged
            else:
                merged[section] = child_val
        body = merged

    profile = _build_profile(name, body)
    logger.info(
        "risk profile active: %s (kelly_fraction=%s max_position_pct=%s "
        "daily_loss=%s dd_limit=%s)",
        profile.name,
        profile.sizing.kelly_fraction,
        profile.sizing.max_position_pct,
        profile.kill_switches.daily_loss_limit_pct,
        profile.kill_switches.drawdown_limit_pct,
    )
    return profile


def _build_profile(name: str, body: dict[str, Any]) -> RiskProfile:
    """Construct a frozen RiskProfile from a parsed YAML body. Missing
    sections fall back to the dataclass defaults (which equal the
    conservative values)."""

    def _sub(key: str, cls: type) -> Any:
        block = body.get(key) or {}
        if not isinstance(block, dict):
            block = {}
        # Only keep fields the dataclass actually accepts — defends
        # against typo'd keys in the YAML.
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in block.items() if k in valid}
        return cls(**clean)

    return RiskProfile(
        name=name,
        sizing=_sub("sizing", SizingConfig),
        kill_switches=_sub("kill_switches", KillSwitchConfig),
        strategy_gates=_sub("strategy_gates", StrategyGatesConfig),
        holding=_sub("holding", HoldingConfig),
        bias=_sub("bias", BiasConfig),
    )
