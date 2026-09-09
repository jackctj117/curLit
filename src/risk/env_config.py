"""Strict trading environment parsing (CL-0deu.9); invalid is never a default."""

from __future__ import annotations

import math
import os


def env_float(name: str, default: float) -> float:
    try:
        result = float(os.environ.get(name, default))
        if not math.isfinite(result):
            raise ValueError
        return result
    except ValueError:
        raise ValueError(f"{name}: expected finite number; startup refused") from None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        raise ValueError(f"{name}: expected integer; startup refused") from None


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}: expected explicit boolean; startup refused")


def validate_execution_config(config: object) -> None:
    """Validate known execution fields without changing documented disable sentinels."""
    for name, value in vars(config).items():
        if name == "selection":
            continue  # ContractSelectionConfig has a separate instrument contract.
        if name.startswith("require_") or name == "allow_short":
            if type(value) is not bool:
                raise ValueError(f"{name}: expected boolean")
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name}: expected finite number, not bool/string")
        if (
            name.endswith(("_days", "_min"))
            and name != "min_idea_life_days"
            and type(value) is not int
        ):
            raise ValueError(f"{name}: expected integer duration")
        if name == "min_alignment":
            valid = -1 <= value <= 1 or value == -1.01  # Documented disable sentinel.
        elif name in {"entry_delay_override_conf", "max_entry_spread_pct", "max_stop_spread_pct"}:
            valid = 0 <= value <= 1 or value == 1.01  # Documented disable sentinels.
        elif name in {"min_confidence", "stop_loss_pct", "entry_day_extreme_stop_pct"}:
            valid = 0 <= value <= 1
        elif name in {"default_moneyness", "strike_window"}:
            valid = 0 <= value < 1  # Keep put strikes and strike-window lower bounds positive.
        elif name == "min_urgency":
            valid = type(value) is int and 0 <= value <= 10  # Existing urgency score scale.
        elif name in {
            "max_premium_usd",
            "notional_usd",
            "qty",
            "default_time_stop_days",
            "default_dte_days",
        }:
            valid = value > 0
        else:
            valid = value >= 0
        if not valid:
            raise ValueError(f"{name}: outside permitted range")
