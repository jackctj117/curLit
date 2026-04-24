# 10 — Monitoring and Observability

Prometheus metrics, structured logging, Loki/Grafana configs, alert rules.

## Prometheus Metrics

**Location:** `src/monitoring/metrics.py`
**Purpose:** All Prometheus metrics across infrastructure, application, trading, and risk layers.

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server


# === INFRASTRUCTURE LAYER ===

service_up = Gauge(
    'fx_service_up', 
    '1 if service is running and healthy', 
    ['service']
)

heartbeat_seconds = Gauge(
    'fx_heartbeat_seconds',
    'Unix timestamp of last heartbeat from main loop'
)

errors_total = Counter(
    'fx_errors_total',
    'Total errors encountered',
    ['service', 'severity', 'category']
)

queue_depth = Gauge(
    'fx_queue_depth',
    'Items waiting in async queues',
    ['queue_name']
)


# === APPLICATION LAYER ===

# Data ingestion
ingestion_runs = Counter(
    'fx_ingestion_runs_total',
    'Number of ingestion runs',
    ['source', 'status']
)

ingestion_lag_seconds = Gauge(
    'fx_ingestion_lag_seconds',
    'Seconds since last successful ingestion',
    ['source']
)

ingestion_rows = Counter(
    'fx_ingestion_rows_total',
    'Rows ingested',
    ['source']
)

# Strategy signals
signals_generated = Counter(
    'fx_signals_generated_total',
    'Strategy signals generated',
    ['strategy_id']
)

signal_latency = Histogram(
    'fx_signal_latency_seconds',
    'Time from data update to signal generation',
    ['strategy_id'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60]
)

# Price stream
prices_received = Counter(
    'fx_prices_received_total',
    'Price ticks received',
    ['symbol']
)

price_staleness = Gauge(
    'fx_price_staleness_seconds',
    'Seconds since last price update per symbol',
    ['symbol']
)


# === TRADING LAYER ===

orders_submitted = Counter(
    'fx_orders_submitted_total',
    'Orders sent to broker',
    ['strategy_id', 'symbol', 'side']
)

orders_filled = Counter(
    'fx_orders_filled_total',
    'Orders successfully filled',
    ['strategy_id', 'symbol']
)

orders_rejected = Counter(
    'fx_orders_rejected_total',
    'Orders rejected',
    ['strategy_id', 'reason']
)

order_latency = Histogram(
    'fx_order_latency_seconds',
    'Time from submission to fill',
    ['strategy_id'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5]
)

slippage_bps = Histogram(
    'fx_slippage_bps',
    'Realized slippage in basis points',
    ['strategy_id', 'symbol'],
    buckets=[-50, -10, -5, -1, 0, 1, 5, 10, 50]
)

positions_open = Gauge(
    'fx_positions_open',
    'Number of currently open positions'
)

# Account state
account_equity = Gauge(
    'fx_account_equity_usd',
    'Current account equity in USD'
)

account_balance = Gauge(
    'fx_account_balance_usd',
    'Current cash balance in USD'
)

margin_used = Gauge(
    'fx_margin_used_usd',
    'Margin currently used in USD'
)

unrealized_pnl = Gauge(
    'fx_unrealized_pnl_usd',
    'Unrealized P&L in USD',
    ['strategy_id']
)

realized_pnl_today = Gauge(
    'fx_realized_pnl_today_usd',
    'Realized P&L since UTC midnight',
    ['strategy_id']
)


# === RISK LAYER ===

position_size = Gauge(
    'fx_position_size',
    'Current position size in base currency',
    ['symbol']
)

position_notional_usd = Gauge(
    'fx_position_notional_usd',
    'Position notional value in USD',
    ['symbol']
)

leverage_ratio = Gauge(
    'fx_leverage_ratio',
    'Current leverage (gross notional / equity)'
)

drawdown_from_peak = Gauge(
    'fx_drawdown_from_peak_pct',
    'Current drawdown from equity peak as percent'
)

correlation_to_dxy = Gauge(
    'fx_correlation_to_dxy_30d',
    'Rolling 30-day correlation to DXY',
    ['symbol']
)


# === PORTFOLIO LAYER ===

portfolio_gross_leverage = Gauge(
    'fx_portfolio_gross_leverage',
    'Current gross leverage'
)

portfolio_net_leverage = Gauge(
    'fx_portfolio_net_leverage', 
    'Current net leverage'
)

strategy_allocation = Gauge(
    'fx_strategy_allocation',
    'Current allocation weight per strategy',
    ['strategy_id']
)

strategy_exposure_mult = Gauge(
    'fx_strategy_exposure_mult',
    'Current exposure multiplier per strategy',
    ['strategy_id']
)

correlation_regime = Gauge(
    'fx_correlation_regime',
    'Correlation regime (0=normal, 1=stressed, 2=crisis)'
)

strategy_attributed_pnl = Gauge(
    'fx_strategy_attributed_pnl_usd',
    'Cumulative attributed P&L per strategy',
    ['strategy_id']
)

strategy_attributed_sharpe = Gauge(
    'fx_strategy_attributed_sharpe',
    'Rolling 60-day Sharpe per strategy',
    ['strategy_id']
)

portfolio_conflicts_rate = Gauge(
    'fx_portfolio_conflicts_rate',
    'Conflicts per hour (strategies disagreeing on same symbol)'
)


# === SERVICE INFO ===

service_info = Gauge(
    'fx_service_info',
    'Build info',
    ['git_sha', 'deploy_ts', 'python_version']
)


def setup_metrics_server(port: int = 8000):
    """Start Prometheus exposition endpoint."""
    start_http_server(port)
```

## Heartbeat Tracker

**Location:** `src/monitoring/heartbeat.py`
**Purpose:** Track per-component heartbeats. Watchdog uses this to detect frozen components.

```python
from datetime import datetime, timedelta
from prometheus_client import Gauge


component_heartbeat_seconds = Gauge(
    'fx_component_heartbeat_seconds',
    'Last heartbeat per component',
    ['component']
)


class HeartbeatTracker:
    def __init__(self):
        self.last_heartbeat = {}
    
    def beat(self, component: str):
        now = datetime.utcnow()
        self.last_heartbeat[component] = now
        component_heartbeat_seconds.labels(component=component).set(now.timestamp())
    
    def is_healthy(self, component: str, max_age: timedelta) -> bool:
        last = self.last_heartbeat.get(component)
        if last is None:
            return False
        return (datetime.utcnow() - last) < max_age
    
    def stale_components(self, max_age: timedelta) -> list[str]:
        now = datetime.utcnow()
        return [
            c for c, t in self.last_heartbeat.items()
            if (now - t) > max_age
        ]
```

## Logging Config

**Location:** `src/monitoring/logging_config.py`
**Purpose:** JSON-structured logging with context filter for trace IDs.

```python
import logging
import json
import sys
from datetime import datetime
import contextvars

# Context for adding trace IDs to all log lines
log_context = contextvars.ContextVar('log_context', default={})


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            'ts': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add context (trace_id, strategy_id, etc.)
        ctx = log_context.get()
        data.update(ctx)
        
        # Add exception info if present
        if record.exc_info:
            data['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key in ('name', 'msg', 'args', 'created', 'filename', 'funcName',
                       'levelname', 'levelno', 'lineno', 'module', 'msecs',
                       'message', 'pathname', 'process', 'processName',
                       'relativeCreated', 'thread', 'threadName', 'exc_info',
                       'exc_text', 'stack_info', 'taskName'):
                continue
            try:
                json.dumps(value)
                data[key] = value
            except TypeError:
                data[key] = str(value)
        
        return json.dumps(data)


class ContextFilter(logging.Filter):
    """Filter that adds context fields to log records."""
    def filter(self, record):
        ctx = log_context.get()
        for key, value in ctx.items():
            setattr(record, key, value)
        return True


def setup_logging(level: str = 'INFO'):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    handler.addFilter(ContextFilter())
    
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def with_context(**kwargs):
    """Context manager to add fields to all log lines in scope."""
    class _CtxMgr:
        def __enter__(self):
            current = log_context.get().copy()
            current.update(kwargs)
            self.token = log_context.set(current)
            return self
        def __exit__(self, *args):
            log_context.reset(self.token)
    return _CtxMgr()
```

## Promtail Config

**Location:** `docker/promtail/promtail-config.yml`
**Purpose:** Ship logs from journald to Loki for centralized search.

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: fx_services
    journal:
      max_age: 12h
      labels:
        job: systemd-journal
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
      - source_labels: ['__journal__hostname']
        target_label: 'host'
    pipeline_stages:
      - match:
          selector: '{unit=~"fx-.*"}'
          stages:
            - json:
                expressions:
                  level: level
                  trace_id: trace_id
                  strategy_id: strategy_id
                  message: message
            - labels:
                level:
                strategy_id:
```

## Loki Config

**Location:** `docker/loki/loki-config.yml`
**Purpose:** Loki log storage backend.

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2024-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v12
      index:
        prefix: index_
        period: 24h

limits_config:
  retention_period: 30d
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

## Prometheus Config

**Location:** `docker/prometheus/prometheus.yml`
**Purpose:** Scrape config for Prometheus.

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/rules/*.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'fx_live_engine'
    static_configs:
      - targets: ['host.docker.internal:8000']
        labels:
          service: 'live_engine'
  
  - job_name: 'fx_ingestion'
    static_configs:
      - targets: ['host.docker.internal:8001']
        labels:
          service: 'ingestion'
  
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['node_exporter:9100']
  
  - job_name: 'postgres_exporter'
    static_configs:
      - targets: ['postgres_exporter:9187']
```

## Alert Rules

**Location:** `docker/prometheus/rules/fx_alerts.yml`
**Purpose:** Alerting rules covering critical, high, medium, and low severity conditions.

```yaml
groups:
  - name: fx_critical
    interval: 30s
    rules:
      - alert: LiveEngineDown
        expr: time() - fx_heartbeat_seconds > 90
        for: 1m
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "Live engine heartbeat stale"
          description: "No heartbeat received in {{ $value }}s"
          runbook: "Check systemctl status fx-live-engine; journalctl -u fx-live-engine -n 100"
      
      - alert: LargeUnrealizedLoss
        expr: sum(fx_unrealized_pnl_usd) < -1000
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Large unrealized loss: ${{ $value }}"
      
      - alert: AccountEquityDrop
        expr: |
          (fx_account_equity_usd - fx_account_equity_usd offset 1h) 
          / fx_account_equity_usd offset 1h < -0.05
        labels:
          severity: critical
        annotations:
          summary: "Account equity dropped >5% in 1 hour"
      
      - alert: PostgresDown
        expr: up{job="postgres_exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
      
      - alert: VaultAgentDown
        expr: up{service="vault_agent"} == 0
        for: 30s
        labels:
          severity: critical
  
  - name: fx_high
    interval: 1m
    rules:
      - alert: HighOrderRejectionRate
        expr: |
          rate(fx_orders_rejected_total[5m]) / 
          rate(fx_orders_submitted_total[5m]) > 0.1
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "Order rejection rate >10%"
      
      - alert: DataIngestionStale
        expr: fx_ingestion_lag_seconds > 7200
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "Ingestion lag for {{ $labels.source }} is {{ $value }}s"
      
      - alert: PriceStaleSeveral
        expr: |
          count(fx_price_staleness_seconds > 60) > 3
        for: 2m
        labels:
          severity: high
        annotations:
          summary: "Multiple price feeds stale"
      
      - alert: HighSlippage
        expr: |
          histogram_quantile(0.95, 
            rate(fx_slippage_bps_bucket[15m])) > 5
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "P95 slippage exceeds 5 bps"
      
      - alert: ApproachingDailyLossLimit
        expr: sum(fx_realized_pnl_today_usd) < -800
        labels:
          severity: high
        annotations:
          summary: "Approaching daily loss limit"
  
  - name: fx_medium
    interval: 5m
    rules:
      - alert: HighLeverage
        expr: fx_leverage_ratio > 5
        for: 10m
        labels:
          severity: medium
      
      - alert: SignalGenerationLatency
        expr: |
          histogram_quantile(0.95,
            rate(fx_signal_latency_seconds_bucket[15m])) > 10
        for: 10m
        labels:
          severity: medium
      
      - alert: DiskSpaceLow
        expr: node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes < 0.15
        for: 30m
        labels:
          severity: medium
  
  - name: fx_low
    interval: 30m
    rules:
      - alert: NoSignalsGenerated
        expr: |
          increase(fx_signals_generated_total[6h]) == 0
        labels:
          severity: low
      
      - alert: WeeklyDrawdownReview
        expr: fx_drawdown_from_peak_pct > 0.05
        for: 6h
        labels:
          severity: low
        annotations:
          summary: "Drawdown >5% — review at next checkpoint"
```

## Alertmanager Config

**Location:** `docker/alertmanager/alertmanager.yml`
**Purpose:** Route alerts to Pushover/Telegram with severity-based priorities.

```yaml
global:
  resolve_timeout: 5m

route:
  receiver: 'pushover'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: 'pushover-critical'
      continue: true
    - match:
        severity: critical
      receiver: 'telegram'
    - match:
        severity: high
      receiver: 'pushover'

receivers:
  - name: 'pushover'
    pushover_configs:
      - user_key: ${PUSHOVER_USER_KEY}
        token: ${PUSHOVER_API_TOKEN}
        priority: '1'
        title: 'FX: {{ .GroupLabels.alertname }}'
        message: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}
  
  - name: 'pushover-critical'
    pushover_configs:
      - user_key: ${PUSHOVER_USER_KEY}
        token: ${PUSHOVER_API_TOKEN}
        priority: '2'  # Emergency — bypasses quiet hours
        retry: '60s'
        expire: '1h'
        title: 'FX CRITICAL: {{ .GroupLabels.alertname }}'
  
  - name: 'telegram'
    webhook_configs:
      - url: 'http://localhost:8002/telegram-alert'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: critical
    target_match:
      severity: high
    equal: ['alertname']
```

## Service Info Tracking

**Location:** `src/monitoring/service_info.py`
**Purpose:** Emit deployment metadata as a metric for "what's running right now" visibility.

```python
import subprocess
import sys
from datetime import datetime
from src.monitoring.metrics import service_info


def emit_service_info():
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()
    except subprocess.CalledProcessError:
        sha = 'unknown'
    
    service_info.labels(
        git_sha=sha[:8],
        deploy_ts=datetime.utcnow().isoformat(),
        python_version=sys.version.split()[0],
    ).set(1)
```
