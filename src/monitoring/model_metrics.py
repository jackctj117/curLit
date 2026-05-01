"""Model metrics instrumentation — wire Prometheus into model refits and correlation."""

import logging

from src.monitoring.metrics import (
    correlation_regime,
    model_last_refit_timestamp,
    model_r_squared,
    model_refit_duration,
    model_residual_std,
    strategy_pair_correlation,
)

logger = logging.getLogger(__name__)


def instrument_model_fit(strategy_id: str, r_squared: float, residual_std: float, duration: float) -> None:
    import time
    model_refit_duration.labels(strategy_id=strategy_id, model_type="ols").observe(duration)
    model_r_squared.labels(strategy_id=strategy_id).set(r_squared)
    model_residual_std.labels(strategy_id=strategy_id).set(residual_std)
    model_last_refit_timestamp.labels(strategy_id=strategy_id).set(time.time())


def instrument_correlation_regime(regime: str) -> None:
    val = {"normal": 0, "stressed": 1, "crisis": 2}.get(regime, 0)
    correlation_regime.set(val)


def instrument_pair_correlation(strategy_a: str, strategy_b: str, value: float) -> None:
    strategy_pair_correlation.labels(strategy_a=strategy_a, strategy_b=strategy_b).set(value)
