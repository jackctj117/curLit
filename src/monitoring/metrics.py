"""
Prometheus metrics definitions for curLit.
Centralised gauge, counter, and histogram registry with utility decorators.
"""

from collections.abc import Callable
from functools import wraps
from time import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ======================================================================
# Infrastructure — ingestion
# ======================================================================

ingestion_runs = Counter(
    "fx_ingestion_runs_total",
    "Total ingestion runs",
    ["source", "status"],
)

ingestion_duration = Histogram(
    "fx_ingestion_duration_seconds",
    "Ingestion duration",
    ["source"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

ingestion_records = Counter(
    "fx_ingestion_records_total",
    "Records ingested",
    ["source"],
)

data_freshness_seconds = Gauge(
    "fx_data_freshness_seconds",
    "Seconds since last successful data update",
    ["symbol", "source"],
)

# ======================================================================
# Models
# ======================================================================

model_refit_duration = Histogram(
    "fx_model_refit_duration_seconds",
    "Model refit duration",
    ["strategy_id", "model_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0, 60.0),
)

model_r_squared = Gauge(
    "fx_model_r_squared",
    "Current model R squared",
    ["strategy_id"],
)

model_residual_std = Gauge(
    "fx_model_residual_std",
    "Current model residual standard deviation",
    ["strategy_id"],
)

model_last_refit_timestamp = Gauge(
    "fx_model_last_refit_timestamp",
    "Unix timestamp of last successful refit",
    ["strategy_id"],
)

# ======================================================================
# Trading
# ======================================================================

signals_generated = Counter(
    "fx_signals_generated_total",
    "Signals generated",
    ["strategy_id", "action"],
)

signal_value = Gauge(
    "fx_signal_value",
    "Current signal value",
    ["strategy_id", "pair"],
)

signal_z_score = Gauge(
    "fx_signal_z_score",
    "Current signal Z-score",
    ["strategy_id", "pair"],
)

edge_severity = Gauge(
    "fx_edge_severity",
    "Live-vs-backtest edge severity per strategy: 0=on_track, 1=underperforming, "
    "2=significantly_underperforming, 3=severely_underperforming",
    ["strategy_id"],
)

model_drift_severity = Gauge(
    "fx_model_drift_severity",
    "Model drift severity vs baseline: 0=normal, 1=degraded, 2=alarm",
    ["model_id"],
)

tca_component_bps = Histogram(
    "fx_tca_component_bps",
    "Per-fill TCA component cost in bps, broken out by component "
    "(queue / impact / broker). Positive = cost to us.",
    ["pair", "component"],
    buckets=(-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)

tca_implementation_shortfall_bps = Histogram(
    "fx_tca_implementation_shortfall_bps",
    "Per-fill total implementation shortfall vs arrival mid in bps. "
    "Positive = adverse fill price.",
    ["pair", "side"],
    buckets=(-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)

edge_verdict = Gauge(
    "fx_edge_verdict",
    "Unified per-strategy edge verdict: 0=insufficient_data, "
    "1=no_edge_detected, 2=edge_decayed, 3=weak_edge, 4=strong_edge",
    ["strategy_id"],
)

orders_placed = Counter(
    "fx_orders_placed_total",
    "Orders placed",
    ["strategy_id", "pair", "side", "order_type"],
)

orders_filled = Counter(
    "fx_orders_filled_total",
    "Orders filled",
    ["strategy_id", "pair", "side"],
)

orders_rejected = Counter(
    "fx_orders_rejected_total",
    "Orders rejected",
    ["pair", "reason"],
)

pretrade_rejections = Counter(
    "fx_pretrade_rejections_total",
    "Intents rejected by PreTradeValidator before reaching OMS",
    ["pair", "reason"],
)

blackout_size_down = Counter(
    "fx_blackout_size_down_total",
    "Intents whose size was reduced by the coordinator due to an "
    "economic-calendar blackout window (CL-k74b)",
    ["pair"],
)

order_fill_duration = Histogram(
    "fx_order_fill_duration_seconds",
    "Time from placement to fill",
    ["pair", "order_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0, 60.0, 300.0),
)

slippage_bps = Histogram(
    "fx_slippage_bps",
    "Realised slippage in basis points",
    ["pair", "side"],
    buckets=(-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)

# ======================================================================
# Positions & risk
# ======================================================================

positions_open = Gauge(
    "fx_positions_open",
    "Open positions (1=open, 0=closed)",
    ["strategy_id", "pair"],
)

position_notional_usd = Gauge(
    "fx_position_notional_usd",
    "Position notional in USD",
    ["strategy_id", "pair"],
)

position_unrealized_pnl_usd = Gauge(
    "fx_position_unrealized_pnl_usd",
    "Unrealised P&L in USD",
    ["strategy_id", "pair"],
)

account_equity_usd = Gauge(
    "fx_account_equity_usd",
    "Account equity in USD",
)

account_margin_used_usd = Gauge(
    "fx_account_margin_used_usd",
    "Margin in use in USD",
)

portfolio_drawdown_pct = Gauge(
    "fx_portfolio_drawdown_pct",
    "Current drawdown from peak equity",
)

daily_pnl_usd = Gauge(
    "fx_daily_pnl_usd",
    "Today's realised + unrealised P&L",
    ["strategy_id"],
)

# ======================================================================
# Risk system
# ======================================================================

kill_switch_triggered = Counter(
    "fx_kill_switch_triggered_total",
    "Kill switches triggered",
    ["switch_name", "action"],
)

kill_switch_state = Gauge(
    "fx_kill_switch_state",
    "Kill switch armed (1=armed, 0=triggered/disarmed)",
    ["switch_name"],
)

reconciliation_mismatches = Gauge(
    "fx_reconciliation_mismatches",
    "Count of position mismatches between internal and broker state",
)

correlation_regime = Gauge(
    "fx_correlation_regime",
    "Portfolio correlation regime (0=normal, 1=stressed, 2=crisis)",
)

strategy_pair_correlation = Gauge(
    "fx_strategy_pair_correlation",
    "Rolling correlation between strategy pairs",
    ["strategy_a", "strategy_b"],
)

# ======================================================================
# External / market
# ======================================================================

price_stream_connected = Gauge(
    "fx_price_stream_connected",
    "Price stream connection (1=connected, 0=disconnected)",
    ["broker"],
)

price_stream_last_tick_timestamp = Gauge(
    "fx_price_stream_last_tick_timestamp",
    "Unix timestamp of last received price tick",
    ["symbol"],
)

price_bid_ask_spread_bps = Gauge(
    "fx_price_bid_ask_spread_bps",
    "Current bid-ask spread in basis points",
    ["symbol"],
)

# ======================================================================
# Health
# ======================================================================

service_up = Gauge(
    "fx_service_up",
    "Service liveness (1=up, 0=down)",
    ["service"],
)

service_last_heartbeat = Gauge(
    "fx_service_last_heartbeat_timestamp",
    "Unix timestamp of last heartbeat",
    ["service"],
)

errors_total = Counter(
    "fx_errors_total",
    "Errors logged",
    ["service", "severity", "category"],
)

# ======================================================================
# Decorators & helpers
# ======================================================================


def track_duration(
    histogram: Histogram, **labels: str,
) -> Callable[[Callable], Callable]:
    """Decorator that observes function duration in *histogram*."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start = time()
            try:
                return func(*args, **kwargs)
            finally:
                histogram.labels(**labels).observe(time() - start)
        return wrapper
    return decorator


def track_errors(
    service_name: str, category: str = "unknown",
) -> Callable[[Callable], Callable]:
    """Decorator that increments fx_errors_total on exception."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                return func(*args, **kwargs)
            except Exception:
                errors_total.labels(
                    service=service_name,
                    severity="error",
                    category=category,
                ).inc()
                raise
        return wrapper
    return decorator


class HeartbeatTracker:
    """Periodic heartbeat emitter for a service."""

    def __init__(self, service_name: str, interval_sec: int = 30) -> None:
        self.service_name = service_name
        self.interval = interval_sec
        self._running = False

    def start(self) -> None:
        import time as _time

        self._running = True
        service_up.labels(service=self.service_name).set(1)

        def _loop() -> None:
            while self._running:
                service_last_heartbeat.labels(service=self.service_name).set(_time.time())
                _time.sleep(self.interval)

        t = threading.Thread(target=_loop, daemon=True)  # type: ignore[unused-ignore]
        t.start()

    def stop(self) -> None:
        self._running = False
        service_up.labels(service=self.service_name).set(0)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus /metrics HTTP endpoint."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        start_http_server(port)
    except OSError:
        logger.warning("Port %d unavailable — metrics server not started", port)

import threading  # noqa: E402
