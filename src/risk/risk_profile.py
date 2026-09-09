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
  3. Named "conservative" profile when no explicit selector was supplied.

Unknown names, malformed values and missing trading configuration fail startup.
An absent file may use built-in defaults only with allow_development_defaults=True
and no environment override. This is an explicit local-development mode.

Aggressive-only knobs (``bias`` block) are present on the
aggressive_short profile only — strategies that don't bias their
signal distributions ignore the field gracefully.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH: Path = Path("configs/risk_profile.yaml")
_FALLBACK_PROFILE: str = "conservative"
_ENV_OVERRIDE: str = "CURLIT_RISK_PROFILE"


def _load_risk_yaml(data: str) -> Any:
    """Safe parser boundary; reject duplicates before constructing mappings."""
    loader = yaml.SafeLoader(data)
    visited: set[int] = set()

    def validate(node: Any) -> None:
        if id(node) in visited:
            return  # YAML anchors may share nodes; do not recurse forever.
        visited.add(id(node))
        if isinstance(node, yaml.MappingNode):
            keys = set()
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=True)
                if not isinstance(key, str) or key in keys:
                    raise ValueError("risk config: duplicate YAML key or non-string key")
                keys.add(key)
                validate(value_node)
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                validate(child)

    try:
        node = loader.get_single_node()
        if node is None:
            return None
        validate(node)
        return loader.construct_document(node)
    finally:
        loader.dispose()


@dataclass(frozen=True)
class SizingConfig:
    kelly_fraction: float = 0.25
    volatility_target_pct: float = 0.10
    max_position_pct: float = 0.20
    max_polymarket_market_fraction: float = 0.05
    polymarket_min_edge: float = 0.03


@dataclass(frozen=True)
class KillSwitchConfig:
    # daily_loss_limit / drawdown_limit thresholds — wired into the
    # KillSwitchManager conditions since CL-i4tx (previously the switches
    # hardcoded -0.03/-0.20 and ignored the profile).
    daily_loss_limit_pct: float = -0.03
    drawdown_limit_pct: float = -0.20
    # NOTE (CL-i4tx): single_strategy_dd_pct and strategy_correlation_spike
    # were removed — the switches they configured were deleted because their
    # inputs (per-strategy equity/returns series) are never populated in
    # production. Only knobs with a real consumer stay.
    # Polymarket absolute loss caps in USD (CL-983f) — consumed by
    # src/risk/polymarket_loss_caps.py. Sized against the $100 initial
    # mainnet cap: one market can burn at most $25, one UTC day $50.
    polymarket_per_market_loss_cap_usd: float = 25.0
    polymarket_per_day_loss_cap_usd: float = 50.0
    # CL-ep0c equity-curve trailing stop — consumed by
    # src/risk/kill_switches.py. Halt new trades once equity falls
    # trailing_stop_pct below its persisted peak; the halt persists for
    # trailing_stop_cooldown_days across restarts (state under data/).
    trailing_stop_pct: float = 0.10
    trailing_stop_cooldown_days: float = 7
    # CL-ep0c open-position correlation kill — with >=2 open positions,
    # fire reduce_50pct when the mean direction-adjusted pairwise return
    # correlation (daily closes over the lookback) exceeds the threshold.
    open_position_corr_threshold: float = 0.85
    open_position_corr_lookback_days: int = 60


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
    *,
    allow_development_defaults: bool = False,
) -> RiskProfile:
    """Resolve which profile is active + return its parsed config.

    Resolution order:
      1. ``CURLIT_RISK_PROFILE`` env var (operator can flip at boot)
      2. ``active:`` field in the YAML
      3. Named conservative profile (must exist in the configuration)

    Profile inheritance is acyclic; child section keys override the parent's.
    Every declared profile and inheritance chain is validated before selection.
    """
    path = Path(config_path)
    if not path.exists():
        if not allow_development_defaults or _ENV_OVERRIDE in os.environ:
            raise ValueError("risk profile config missing; trading startup refused")
        logger.warning(
            "risk profile config not found at %s — using conservative defaults",
            path,
        )
        return _build_profile(_FALLBACK_PROFILE, {})

    config_text = path.read_text()
    raw = _load_risk_yaml(config_text)
    if not isinstance(raw, dict) or set(raw) - {"active", "profiles"}:
        raise ValueError("risk config: expected active/profiles mapping; unknown top-level key")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("risk config: profiles must be a nonempty mapping")
    name = os.environ.get(_ENV_OVERRIDE, raw.get("active", _FALLBACK_PROFILE))
    if not isinstance(name, str) or not name.strip() or name.strip() not in profiles:
        raise ValueError("risk config: explicitly selected unknown profile")
    name = name.strip()
    # Validate every declared profile, including inactive sections. A child must
    # not mask a malformed parent setting with an apparently valid override.
    for profile_name, spec in profiles.items():
        if not isinstance(profile_name, str) or not profile_name or not isinstance(spec, dict):
            raise ValueError("risk config: invalid profile name/body")
        _build_profile(profile_name, {k: v for k, v in spec.items() if k != "inherits"})

    def resolve(selected: str, visiting: frozenset[str]) -> dict[str, Any]:
        if selected in visiting:
            raise ValueError("risk config: cyclic profile inheritance")
        spec = dict(profiles[selected])
        parent = spec.pop("inherits", None)
        if parent is None and "inherits" not in profiles[selected]:
            return spec
        if not isinstance(parent, str) or parent not in profiles:
            raise ValueError("risk config: inherits from unknown profile")
        merged = resolve(parent, visiting | {selected})
        for key, value in spec.items():
            merged[key] = {**merged.get(key, {}), **value}
        return merged

    # Validate inheritance even on inactive profiles rather than delaying failure.
    for profile_name in profiles:
        resolve(profile_name, frozenset())
    body = resolve(name, frozenset())

    profile = _build_profile(name, body)
    logger.info(
        "risk profile active: %s (kelly_fraction=%s max_position_pct=%s daily_loss=%s dd_limit=%s)",
        profile.name,
        profile.sizing.kelly_fraction,
        profile.sizing.max_position_pct,
        profile.kill_switches.daily_loss_limit_pct,
        profile.kill_switches.drawdown_limit_pct,
        extra={
            "extra_data": {
                "effective_risk_profile": asdict(profile),
                "config_sha256": sha256(config_text.encode()).hexdigest(),
                "selected_by": "environment" if _ENV_OVERRIDE in os.environ else "configuration",
            }
        },
    )
    return profile


def _build_profile(name: str, body: dict[str, Any]) -> RiskProfile:
    """Construct a frozen RiskProfile from a parsed YAML body. Missing
    sections fall back to the dataclass defaults (which equal the
    conservative values)."""

    sections = {"sizing", "kill_switches", "strategy_gates", "holding", "bias"}
    if not isinstance(body, dict) or set(body) - sections:
        raise ValueError("risk profile: unknown section")

    def _sub(key: str, cls: type[Any]) -> Any:
        block = body.get(key, {})
        if not isinstance(block, dict):
            raise ValueError(f"risk profile.{key}: expected mapping")
        defaults = vars(cls())
        if set(block) - set(defaults):
            raise ValueError(f"risk profile.{key}: unknown field")
        for field_name, value in block.items():
            path = f"risk profile.{key}.{field_name}"
            default = defaults[field_name]
            if type(default) is bool:
                if type(value) is not bool:
                    raise ValueError(path + ": expected boolean")
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(path + ": expected finite number, not string/bool")
            if type(default) is int and type(value) is not int:
                raise ValueError(path + ": expected integer")
            # Unitless fractions/correlations are mathematically bounded, not
            # strategy tuning. Negative loss thresholds use the documented sign.
            if field_name in {"daily_loss_limit_pct", "drawdown_limit_pct"}:
                valid = -1 <= value < 0
            elif key == "sizing" or field_name in {
                "trailing_stop_pct",
                "open_position_corr_threshold",
                "min_r_squared",
            }:
                valid = 0 <= value <= 1
            elif field_name == "paper_relevance_floor":
                valid = 0 <= value <= 10  # Existing research relevance score scale.
            elif field_name == "open_position_corr_lookback_days":
                valid = value >= 2  # Correlation needs at least two observations.
            elif key == "bias":
                valid = True  # Signed multipliers, including zero/-1, are documented.
            elif field_name == "trailing_stop_cooldown_days":
                valid = value >= 0
            else:
                valid = value > 0
            if not valid:
                raise ValueError(path + ": outside permitted range")
        return cls(**block)

    return RiskProfile(
        name=name,
        sizing=_sub("sizing", SizingConfig),
        kill_switches=_sub("kill_switches", KillSwitchConfig),
        strategy_gates=_sub("strategy_gates", StrategyGatesConfig),
        holding=_sub("holding", HoldingConfig),
        bias=_sub("bias", BiasConfig),
    )
