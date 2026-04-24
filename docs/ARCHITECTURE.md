# curLit — Technical Architecture

## Table of Contents

- [Tech Stack Summary](#tech-stack-summary)
- [Directory Structure](#directory-structure)
- [Development Roadmap](#development-roadmap)
- [1. Data Layer](#1-data-layer)
- [2. Feature Engineering & Modeling](#2-feature-engineering--modeling)
- [3. NLP Pipeline](#3-nlp-pipeline)
- [4. Backtesting Infrastructure](#4-backtesting-infrastructure)
- [5. Risk Management](#5-risk-management)
- [6. Execution Layer](#6-execution-layer)
- [7. Strategies](#7-strategies)
- [8. Observability & Monitoring](#8-observability--monitoring)
- [9. Deployment](#9-deployment)
- [10. Data Vendors & Broker Comparison](#10-data-vendors--broker-comparison)
- [11. FinBERT Fine-Tuning](#11-finbert-fine-tuning)
- [12. FX Overnight Rolls & Funding](#12-fx-overnight-rolls--funding)
- [Data Sources Reference](#data-sources-reference)
- [What Will Go Wrong](#what-will-go-wrong)

---

## Tech Stack Summary

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Language** | Python 3.11+ | Numerical ecosystem, ML libraries |
| **Database** | PostgreSQL + TimescaleDB | Time-series native, full SQL |
| **Columnar storage** | Parquet | Fast compressed analytics |
| **Data processing** | Pandas / Polars | Flexible; Polars for large ops |
| **Scheduling** | Apache Airflow or Prefect | DAG orchestration |
| **Reproducibility** | Docker | Consistent environments |
| **Testing** | pytest + hypothesis | Property-based + standard |
| **Experiment tracking** | MLflow or Weights & Biases | Model versioning |
| **NLP** | FinBERT + spaCy + HuggingFace Transformers | Fine-tuned on CB text |
| **Modeling** | statsmodels, scikit-learn | OLS regression, ML |
| **Observability** | Prometheus + Grafana + Loki + Alertmanager | Metrics, dashboards, logs, alerts |
| **Broker API** | OANDA v20 (start) → Interactive Brokers (scale) | Clean API → lower costs |
| **Process management** | systemd (trading) + Docker Compose (observability) | Reliability vs. isolation |
| **CI/CD** | GitHub Actions or Jenkins | Scheduled data pipeline runs |
| **Monitoring dashboards** | Grafana | P&L, signals, health, risk |

---

## Directory Structure

```
fx-system/
├── data/                    # Raw + processed data storage
│   ├── raw/                 # API dumps, unchanged
│   ├── processed/           # Cleaned, aligned time series
│   └── vintage/             # Point-in-time data for backtesting
├── src/
│   ├── data/                # Data providers (FRED, Stooq, CME, OANDA)
│   ├── features/            # Feature engineering, signal computation
│   ├── models/              # Rate diff, reaction function, factor models
│   ├── rates/               # OIS curve, day-count, calendars
│   ├── nlp/                 # Scrapers, preprocessing, lexicon, transformers, diff, inference
│   ├── signals/             # Signal generation logic
│   ├── backtest/            # Walk-forward framework, analytics, swap models
│   ├── risk/                # Correlation monitor, sizing, kill switches, stress tests
│   ├── execution/           # Broker interface, OMS, broker implementations
│   ├── strategies/          # Strategy classes, state store
│   ├── runtime/             # Live engine
│   ├── monitoring/          # Metrics, logging, heartbeat
│   └── ops/                 # Watchdog, backup, runbook scripts
├── notebooks/               # Research and ad-hoc analysis
├── tests/                   # Unit + integration tests
├── configs/                 # YAML configs per strategy/model
├── airflow/                 # DAGs for scheduled runs
├── docker/                  # Containerization
├── labeling/                # NLP labeling tool and corpus
├── training/                # FinBERT fine-tuning scripts
├── models/                  # Saved model artifacts
└── docs/
    └── ARCHITECTURE.md
```

---

## Development Roadmap

**Months 0–3**: Data pipeline end-to-end. FRED, Stooq for prices, CFTC for COT. PostgreSQL + TimescaleDB. Basic Airflow/cron. Tests.

**Months 3–6**: Feature computation, simple rate differential model. Walk-forward framework. First end-to-end backtest on EUR/USD.

**Months 6–9**: OIS curve construction, improved model, multi-pair support. NLP pipeline (scrapers, preprocessing, lexicon). CB sentiment scoring.

**Months 9–12**: Options data if available, reaction function model, CB NLP fine-tuning. More rigorous backtesting (bootstrap CIs, regime analysis, parameter sensitivity). Paper broker.

**Months 12–18**: Portfolio construction, risk management, paper trade daily. Dashboard monitoring. Compare paper results to backtest expectations.

**Months 18–24**: Live with very small size (1–5% of capital). Scale only as consistent performance validates the system.

**Year 2+**: Iterative improvement, new strategies, factor models, ensemble methods.

---

## 1. Data Layer

### 1.1 Database Schema (TimescaleDB)

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Price data (FX, futures, equities, yields)
CREATE TABLE prices (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC,
    PRIMARY KEY (ts, symbol, source)
);
SELECT create_hypertable('prices', 'ts');
CREATE INDEX idx_prices_symbol ON prices (symbol, ts DESC);

-- Macro economic data with vintage (release dates) — prevents look-ahead bias
CREATE TABLE macro_data (
    observation_date DATE NOT NULL,
    release_date TIMESTAMPTZ NOT NULL,
    series_id TEXT NOT NULL,
    value NUMERIC,
    revision INT DEFAULT 0,
    source TEXT NOT NULL,
    PRIMARY KEY (observation_date, release_date, series_id)
);
CREATE INDEX idx_macro_series ON macro_data (series_id, observation_date DESC);

-- Rate curves (OIS, treasuries) — snapshot by date
CREATE TABLE rate_curves (
    ts TIMESTAMPTZ NOT NULL,
    curve_id TEXT NOT NULL,
    tenor_days INT NOT NULL,
    rate NUMERIC,
    PRIMARY KEY (ts, curve_id, tenor_days)
);
SELECT create_hypertable('rate_curves', 'ts');

-- Positioning data (COT)
CREATE TABLE cot_positioning (
    report_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    category TEXT NOT NULL,
    longs INT,
    shorts INT,
    spreads INT,
    open_interest INT,
    PRIMARY KEY (report_date, symbol, category)
);

-- Options data (IV, risk reversals, butterflies)
CREATE TABLE fx_options (
    ts TIMESTAMPTZ NOT NULL,
    pair TEXT NOT NULL,
    tenor TEXT NOT NULL,
    atm_iv NUMERIC,
    rr_25d NUMERIC,
    bf_25d NUMERIC,
    rr_10d NUMERIC,
    bf_10d NUMERIC,
    PRIMARY KEY (ts, pair, tenor)
);
SELECT create_hypertable('fx_options', 'ts');

-- Computed features (cached)
CREATE TABLE features (
    ts TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value NUMERIC,
    PRIMARY KEY (ts, symbol, feature_name)
);
SELECT create_hypertable('features', 'ts');

-- CB sentiment
CREATE TABLE cb_sentiment (
    ts TIMESTAMPTZ NOT NULL,
    doc_id TEXT NOT NULL,
    cb TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    sentence_idx INT NOT NULL,
    sentence TEXT NOT NULL,
    lex_hawkish INT,
    lex_dovish INT,
    lex_net NUMERIC,
    tfm_dovish NUMERIC,
    tfm_neutral NUMERIC,
    tfm_hawkish NUMERIC,
    tfm_score NUMERIC,
    PRIMARY KEY (doc_id, sentence_idx)
);
CREATE INDEX ON cb_sentiment (cb, ts DESC);

CREATE TABLE cb_diff_events (
    ts TIMESTAMPTZ NOT NULL,
    cb TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    prev_doc_id TEXT NOT NULL,
    net_shift NUMERIC,
    added_hawkish NUMERIC,
    removed_hawkish NUMERIC,
    change_ratio NUMERIC,
    PRIMARY KEY (doc_id)
);
```

### 1.2 Vintage Data (Point-in-Time)

The `release_date` column is critical for avoiding look-ahead bias. Every backtest query filters `WHERE release_date <= @as_of_date` to get only data known at that point.

```python
def get_macro_as_of(engine, series_id: str, as_of: datetime) -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (observation_date)
            observation_date, value, release_date
        FROM macro_data
        WHERE series_id = %(series_id)s
          AND release_date <= %(as_of)s
        ORDER BY observation_date, release_date DESC
    """
    return pd.read_sql(query, engine, params={
        'series_id': series_id, 'as_of': as_of
    })
```

### 1.3 Data Quality Monitoring

```python
import great_expectations as ge

def validate_price_data(df: pd.DataFrame) -> bool:
    dfe = ge.from_pandas(df)
    dfe.expect_column_values_to_not_be_null('close')
    dfe.expect_column_values_to_be_between('close', 0, 1e6)
    dfe.expect_column_values_to_be_unique(['ts', 'symbol'])
    ts_gaps = df.set_index('ts').resample('D').size()
    assert (ts_gaps == 0).sum() < len(ts_gaps) * 0.05  # <5% missing
    return dfe.validate().success
```

### 1.4 Data Ingestion Pattern

```python
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from tenacity import retry, stop_after_attempt, wait_exponential


class BaseIngester(ABC):
    def __init__(self, db_url: str, source_name: str):
        self.engine = create_engine(db_url)
        self.source = source_name

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch raw data from source."""

    @abstractmethod
    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize to schema."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def run(self, start: datetime, end: datetime) -> int:
        raw = self.fetch(start, end)
        if raw.empty:
            return 0
        df = self.transform(raw)
        df = self.validate(df)
        rows = self.upsert(df)
        return rows

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['ts', 'symbol'])
        df = df.drop_duplicates(subset=['ts', 'symbol'])
        return df
```

---

## 2. Feature Engineering & Modeling

### 2.1 Feature Computation

```python
def rate_differential_zscore(us_2y: pd.Series, de_2y: pd.Series,
                              window: int = 252) -> pd.Series:
    diff = us_2y - de_2y
    rolling_mean = diff.rolling(window).mean()
    rolling_std = diff.rolling(window).std()
    return (diff - rolling_mean) / rolling_std


def realized_vol(price: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(price / price.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def cot_zscore(net_position: pd.Series, window: int = 156) -> pd.Series:
    rolling_mean = net_position.rolling(window).mean()
    rolling_std = net_position.rolling(window).std()
    return (net_position - rolling_mean) / rolling_std


def carry(target_rate: pd.Series, base_rate: pd.Series) -> pd.Series:
    return target_rate - base_rate


def momentum_signal(price: pd.Series, lookback: int = 63) -> pd.Series:
    return price / price.shift(lookback) - 1
```

### 2.2 Rate Differential Model

```python
import statsmodels.api as sm


class RateDiffModel:
    def __init__(self, pair: str, factors: list[str], window_days: int = 756):
        self.pair = pair
        self.factors = factors
        self.window = window_days

    def fit(self, df: pd.DataFrame) -> dict:
        y = df['target']
        X = sm.add_constant(df[self.factors])
        model = sm.OLS(y, X).fit()
        self.result = {
            'coefficients': model.params,
            'residual_std': model.resid.std(),
            'r_squared': model.rsquared,
        }
        return self.result

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = sm.add_constant(df[self.factors])
        return X @ self.result['coefficients']

    def deviation_zscore(self, df: pd.DataFrame) -> pd.Series:
        fair_value = self.predict(df)
        deviation = df['target'] - fair_value
        return deviation / self.result['residual_std']
```

### 2.3 OIS Curve Construction

**Day-count conventions:**

| Currency | OIS Rate | Day Count |
|----------|----------|-----------|
| USD | SOFR OIS | ACT/360 |
| EUR | €STR OIS | ACT/360 |
| GBP | SONIA OIS | ACT/365 |
| JPY | TONA OIS | ACT/365 |
| CAD | CORRA OIS | ACT/365 |

Business day convention: Modified Following.

**Bootstrap method:** For single-payment OIS (≤ 1Y), `DF(T) = 1 / (1 + S × τ)`. For multi-payment (> 1Y), iterate using previously-solved discount factors for coupon dates.

**Forward rate:** `F(t₁, t₂) = (DF(t₁)/DF(t₂) - 1) / τ(t₁, t₂)`

**Implied policy path:** Compute forward rates at meeting dates. Use `meeting_probability()` to back out hike/hold/cut probabilities: `probability = (implied - current) / 0.0025` for 25bp moves.

Key classes: `OISCurve`, `DayCountConvention`, `Calendar`, `OISQuote`, `BusinessDayConvention`. Full implementation in `src/rates/ois_curve.py`.

### 2.4 Reaction Function Model (Fed)

```python
class FedReactionFunction:
    def __init__(self):
        self.pce_target = 2.0
        self.u_star = 4.2  # Estimated natural rate
        self.r_star = 0.5  # Real neutral rate

    def implied_rate(self, core_pce, unemployment, fci=0.0):
        inflation_gap = core_pce - self.pce_target
        unemployment_gap = self.u_star - unemployment
        rate = (self.r_star + core_pce
                + 0.6 * 1.5 * inflation_gap  # Taylor principle (>1)
                + 0.3 * unemployment_gap
                + 0.1 * (0.0 - fci))
        return max(0, rate)
```

### 2.5 Cross-Asset Correlation Monitoring

Build rolling correlation matrices (20-day, 60-day, 250-day). Watch for:
- **Correlation breakdowns** — EUR/USD and US-DE 2Y correlation drops from 0.7 to 0.2
- **Correlation flips** — Gold switches from negatively to positively correlated with USD
- **Correlation extremes** — At multi-year highs between two assets, tend to mean-revert

### 2.6 Options-Based Signals

For FX vol regime detection when full options data is unavailable:
- **CVIX (Deutsche Bank Currency Vol Index)** — free daily close, general FX vol proxy
- **EVZ (CBOE EuroCurrency Volatility Index)** — EUR/USD options IV, free daily
- **VIX and VIX term structure** — cross-asset risk sentiment, free from CBOE
- **CFTC COT** — positioning, free weekly

---

## 3. NLP Pipeline

### 3.1 Scraping Central Bank Documents

Sources:
- **Fed**: `federalreserve.gov/monetarypolicy/fomccalendars.htm` (statements, minutes)
- **ECB**: `ecb.europa.eu/press/pr/date/`
- **BoE**: `bankofengland.co.uk/monetary-policy-summary-and-minutes`
- **BoJ**: `boj.or.jp/en/mopo/mpmsche_minu/`
- **BoC, RBA, RBNZ, SNB**: Similar patterns

Base class `CBScraper` with `list_documents()`, `fetch_url()`, `parse_document()`, `save_document()`. Per-CB subclasses handle site-specific HTML parsing. Documents stored as individual JSON files keyed by ID. Re-running daily is idempotent — only pulls what's new.

### 3.2 Text Preprocessing

```python
class TextPreprocessor:
    BOILERPLATE_PATTERNS = [
        r'For release at.*?ET',
        r'Implementation Note issued.*?$',
        r'^\s*\d+\s*$',  # Page numbers
        r'[A-Z][A-Z\s]{20,}',  # Long uppercase headers
    ]

    def clean(self, text: str) -> str:
        for pat in self.BOILERPLATE_PATTERNS:
            text = re.sub(pat, '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
```

### 3.3 Three-Layer Sentiment Scoring

**Layer 1: Lexicon-based** — Domain-specific hawkish/dovish terms (e.g., "additional firming", "persistent", "patient", "gradual"). Computes `net_score = (hawkish - dovish) / (hawkish + dovish + 1)`. Fast and interpretable.

**Layer 2: Transformer-based (FinBERT)** — Fine-tuned on 2,000+ labeled CB sentences. 3-class: dovish (0), neutral (1), hawkish (2). Expected F1 macro 0.78–0.83.

**Layer 3: Diff analysis** — `difflib.SequenceMatcher` between consecutive statements. Identifies added/removed/modified sentences, scores the net hawkish shift. This is the highest-signal output.

### 3.4 Labeling Workflow

- Target: 2,000–3,000 labeled sentences
- Stratified by CB, doc type, regime, time period
- Active learning: label 200 seed, train, select highest-uncertainty sentences, repeat
- Temporal-document split: no sentence from the same document appears in both train and test; test is always later than train
- Self-consistency: re-label random 5% after 2 weeks, target Cohen's κ > 0.85
- Class weights: ~50-60% neutral, 20-25% hawkish, 15-20% dovish → use weighted loss

### 3.5 Persistence

```sql
CREATE MATERIALIZED VIEW cb_daily_sentiment AS
SELECT
    DATE(ts) as date, cb, doc_type,
    AVG(tfm_score) as avg_tfm_score,
    AVG(lex_net) as avg_lex_score
FROM cb_sentiment
GROUP BY DATE(ts), cb, doc_type;
```

### 3.6 Data Sources for Options & Positioning

| Source | Data | Cost | Frequency |
|--------|------|------|-----------|
| CME FedWatch | Implied rate probabilities | Free | Real-time |
| CFTC COT | Spec/hedger positioning | Free | Weekly (Fri) |
| CBOE VIX/EVZ | Equity & FX vol indices | Free | Daily |
| Saxo Research | RR & butterfly summaries | Free | Periodic |
| CME FX Options | Exchange-traded FX options | Free (delayed) | Daily |
| Bloomberg OVDV | Full IV surfaces, RR, butterflies | Paid ($25k+/yr) | Real-time |

---

## 4. Backtesting Infrastructure

### 4.1 Walk-Forward Framework

Classical non-overlapping walk-forward loop:

```
1. Define IS window (3 years / ~756 days) and OOS window (3 months / ~63 days)
2. Train model / select parameters on IS window
3. Test on OOS window — record results, no modifications
4. Advance both windows forward by OOS length
5. Repeat
6. Concatenate all OOS results → real performance
```

**Only measure performance on OOS periods.** IS performance is a curve-fitting artifact.

Key class: `WalkForwardRunner` with `WalkForwardConfig`, `CostModel`. Strategy must implement `fit(train_data)` and `generate_signals(test_data)`.

### 4.2 Performance Metrics

```python
class PerformanceAnalytics:
    @staticmethod
    def metrics(returns: pd.Series) -> dict:
        return {
            'total_return': (1 + returns).prod() - 1,
            'sharpe': returns.mean() / returns.std() * np.sqrt(252),
            'sortino': ...,
            'max_drawdown': ...,
            'hit_rate': hits / len(trades),
            'profit_factor': gross_wins / abs(gross_losses),
        }
```

Track: Sharpe, Sortino, max drawdown, drawdown duration, hit rate, avg win/loss, profit factor, rolling Sharpe over 6-month windows, performance by year, coefficient stability across refits.

### 4.3 Bootstrap Confidence Intervals

Sharpe ratios have wide confidence intervals. Use **stationary bootstrap** (Politis-Romano) to generate honest CIs that preserve serial correlation. A Sharpe of 1.2 with bootstrapped 95% CI of [0.3, 2.1] is humbling — the true Sharpe could plausibly be anywhere in that range.

### 4.4 Transaction Costs

Model conservatively:
- **Spread**: 0.5 bps on majors (EUR/USD), 1-3 bps on crosses
- **Slippage**: 0.3 bps market orders in normal conditions, 2-5 bps during news
- **Financing (swap)**: Target rate minus base rate, × 1/360 daily, triple on Wednesdays
- **Commission**: $3-7 per standard lot round-trip (ECN broker)
- **Baseline**: 1 bp round-trip per trade for majors

### 4.5 Avoiding Pitfalls

- **Data snooping**: Keep a locked holdout (final 20% of data) never touched during development
- **Look-ahead through data revisions**: Use ALFRED vintage data (point-in-time) for backtesting
- **Parameter sensitivity**: Run strategy with ±20% on every parameter; if Sharpe collapses, the model is overfit
- **Multiple testing**: Bonferroni-correct significance thresholds or use White's Reality Check

### 4.6 Regime-Segmented Analysis

Segment OOS results by regime:
- Risk-on vs. risk-off (VIX above/below 20)
- Trending vs. ranging (ADX > 25 vs < 20)
- Rate-hiking vs. rate-cutting environments
- High vs. low rates volatility (MOVE index quartiles)

A strategy with 1.2 Sharpe overall that's 2.0 in hiking and -0.3 in cutting cycles isn't a strategy — it's a one-trick pony.

### 4.7 Event Backtesting (for CB Sentiment)

For event-driven strategies (not continuous signals): `EventBacktester` class processes events with expanding-window thresholds. Enters at close of event day, exits after N business days with trailing/hard/time stops. Trades per year: 15-40 (8-12 CB events per CB per year).

---

## 5. Risk Management

### 5.1 Position Sizing

```python
class PositionSizer:
    @staticmethod
    def volatility_target(capital, target_vol, realized_vol, price):
        """Size so position contributes target_vol annualized."""
        notional = capital * target_vol / realized_vol
        return notional / price

    @staticmethod
    def kelly(edge, odds, kelly_fraction=0.25):
        """Fractional Kelly for survival."""
        full_kelly = edge - (1 - edge) / odds
        return max(0, full_kelly * kelly_fraction)

    @staticmethod
    def risk_parity_weights(cov_matrix):
        """Assign weights so each asset contributes equal risk."""
```

### 5.2 Portfolio-Level Risk Checks

```python
class RiskManager:
    def check_trade(self, proposed, current, covariance):
        # Leverage: gross ≤ max_leverage
        # Concentration: per-symbol ≤ max_position_pct
        # VaR: 1.65 × portfolio_vol ≤ max_var_pct
        return ok, msg
```

### 5.3 Correlation Breakdown Detection

The most dangerous scenario in multi-strategy portfolios: correlation regime shift during a stress event. Strategies that looked decorrelated in calm markets all drawdown together during a crisis.

```python
class CorrelationMonitor:
    def check_correlations(self, strategy_returns, vix_series):
        # Compute 20-day rolling correlation vs 250-day baseline correlation
        # Z-score of how far recent correlation is from long-run mean
        # Regime: stable / diverging / crisis
        # Recommendation: Continue normal / Monitor / Reduce 50% / Halt
```

### 5.4 Regime-Aware Sizing

```python
class RegimeAwareSizer:
    def compute_adjustment(self, vix, portfolio_dd, correlation_regime, max_pair_corr):
        # Vol multiplier: VIX ≥ 25 → 0.6, ≥ 30 → 0.4, ≥ 40 → 0.2
        # Drawdown throttle: DD ≤ -10% → 0.75, ≤ -20% → 0.25
        # Correlation multiplier: crisis → 0.4, stressed → 0.7
        final = vol_mult * dd_mult * corr_mult
```

### 5.5 Kill Switches

Six switches that trigger automatically:
1. **Daily loss limit** → halt new trades
2. **Drawdown limit** → flatten all positions
3. **VIX spike** (>35, up 50% intraday) → reduce 50%
4. **FX vol spike** (CVIX Z-score > 3) → halt new
5. **Reconciliation failure** → halt new
6. **Stale prices** (no tick in 10 min) → halt new

### 5.6 Scenario Stress Testing

Run each strategy against historical crises:
- GFC 2008, Eurozone Crisis 2011, Taper Tantrum 2013, CHF Shock 2015, Brexit 2016, COVID 2020, Inflation Shock 2022, SVB 2023, Yen Intervention 2024

If the portfolio would lose 40% during COVID, fix the strategy or sizing before going live.

---

## 6. Execution Layer

### 6.1 Broker Abstraction

```python
class Broker(ABC):
    def place_order(self, order: Order) -> Order: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> list[Position]: ...
    def get_account(self) -> Account: ...
    def get_price(self, symbol: str) -> tuple[float, float]: ...
    def stream_prices(self, symbols: list[str]): ...
```

Implementations: `OandaBroker`, `IBKRBroker`, `PaperBroker`. Swap implementations in ~1 week.

### 6.2 OANDA Implementation

- **REST API**: `api-fxtrade.oanda.com` (live) / `api-fxpractice.oanda.com` (demo)
- **Streaming API**: `stream-fxtrade.oanda.com`
- **Symbol mapping**: `EURUSD` → `EUR_USD`
- **Order types**: MARKET (FOK), LIMIT (GTC)
- **Position tracking**: `GET /positions/{instrument}`
- **Account summary**: `GET /accounts/{id}/summary`

### 6.3 Order Management System (OMS)

The OMS sits between strategy and broker. It handles:
- Retries, partial fills, order tracking
- Failure recovery
- Position reconciliation (compare internal vs. broker state every 5 min)
- Risk pre-checks before placing orders

`OrderIntent` captures what the strategy wants. OMS translates to broker orders, respecting urgency (normal/urgent/passive).

### 6.4 Live Engine

Event-driven, asyncio-based. Runs:
- `_price_stream_task()` — tick ingestion
- `_signal_generation_task()` — strategy evaluation
- `_reconciliation_task()` — every 5 minutes
- `_health_check_task()` — every 60 seconds
- `_metrics_publisher_task()` — every 15 seconds

Trading window enforcement: Sunday 22:00 UTC to Friday 22:00 UTC. No trading during first/last 15 min of major sessions, 5 min before/after major data releases, or year-end thin markets (Dec 23 – Jan 2).

### 6.5 Graceful Shutdown

SIGTERM → halt new trades → wait for pending orders (30s timeout) → persist state → close broker connections. On startup, check persisted state and decide: resume / verify / require manual ack.

### 6.6 Pre-Live Checklist

1. Paper trade for ≥ 3 months
2. Independent P&L calculation matching broker to the penny
3. Kill switch tested regularly
4. Circuit breakers: max daily loss, max drawdown, max orders/min, price sanity checks
5. Logging everything (every tick, signal, order, fill)
6. Alerting (critical → Pushover priority 2, warning → email, info → log)
7. Time-of-day awareness
8. Start absurdly small (1/100th of backtest sizing)

---

## 7. Strategies

### 7.1 Strategy 1: Rate Differential Mean Reversion on EUR/USD

**Thesis**: EUR/USD has a strong relationship with the US 2Y minus German 2Y spread. Deviations from fair value mean-revert within weeks.

**Model**: Rolling OLS regression: `EURUSD = α + β × (US2Y − DE2Y) + ε`

**Entry**: Z(deviation) > ±1.5
**Exit**: Z reverts to ±0.3, or stop at Z = ±3.5, or time stop at 30 days
**Sizing**: Volatility-targeted, conviction-scaled, capped at 20% equity
**Quality gate**: R² < 0.25 → sit out (relationship too weak)
**Refit**: Weekly, or if model > 7 days stale

**Expected performance** (out-of-sample on EUR/USD, 5+ years):
- Sharpe: 0.5–1.0 (after costs)
- Max drawdown: 10–18%
- Hit rate: 55–65%
- Avg trade duration: 8–15 days
- Trades per year: 15–30

### 7.2 Strategy 2: CB Sentiment Shift

**Thesis**: Significant hawkish/dovish shifts in CB statements produce FX trends persisting 2–4 weeks post-event.

**Entry**: CB statement released AND diff score in top/bottom 15% of historical diffs
**Direction**: Hawkish → long CB's currency vs USD; Dovish → short
**CB-to-pair mapping**:
- Fed hawkish → short EUR/USD; ECB hawkish → long EUR/USD
- BoE → GBP/USD; BoJ → USD/JPY; BoC → USDCAD
**Hold**: 10 trading days with trailing stop
**Exit**: Time (10 days), trailing stop (1.5% from 2% profit), hard stop (−1.5%)
**Sizing**: 1% of equity at risk per trade, max 3 concurrent positions

**Expected performance**: Trades per year ~30-50 (across all CBs). Expected Sharpe 0.5-0.8. Event-driven and decorrelated from Strategy 1.

### 7.3 Portfolio Integration

Both strategies run concurrently. Risk manager enforces per-pair notional cap (30% of equity) regardless of strategy source.

**Portfolio-level pattern**:
- Normal markets: Strategy 1 does most of the work
- CB event weeks: Strategy 2 fires; Strategy 1 may be temporarily offside but rides through
- Combined Sharpe target: 1.2–1.5 after costs
- Max drawdown target: 12–15%

### 7.4 Future Strategies

1. **USD/JPY rate differential mean reversion** (US 10Y - JGB 10Y)
2. **Carry trade with risk filter** (VIX < 20 only)
3. **COT positioning reversal** (fade extreme spec positions, 4-8 week horizon)
4. **Commodity currency momentum** (AUD/JPY on copper/gold ratio)

---

## 8. Observability & Monitoring

### 8.1 Metrics Taxonomy (4 Layers)

**Layer 1 — Infrastructure**: CPU, memory, disk, network, process uptime, DB connections
**Layer 2 — Application**: Ingestion success rate, records/run, feature staleness, model refit duration & R²
**Layer 3 — Trading**: Signal rate & latency, orders placed/filled/rejected, fill quality slippage, position reconciliation, kill switch state
**Layer 4 — P&L & Risk**: Equity, drawdown curves, per-strategy P&L, per-pair exposure, correlation matrix, VaR utilization

### 8.2 Prometheus Metrics (Python client)

Key metrics registered:
- Gauge: `fx_signal_value`, `fx_signal_z_score`, `fx_positions_open`, `fx_position_notional_usd`, `fx_position_unrealized_pnl_usd`, `fx_account_equity_usd`, `fx_portfolio_drawdown_pct`, `fx_kill_switch_state`, `fx_model_r_squared`, `fx_correlation_regime`, `fx_price_stream_connected`, `fx_data_freshness_seconds`
- Counter: `fx_signals_generated_total`, `fx_orders_placed_total`, `fx_orders_filled_total`, `fx_orders_rejected_total`, `fx_kill_switch_triggered_total`, `fx_ingestion_runs_total`, `fx_ingestion_records_total`, `fx_errors_total`
- Histogram: `fx_ingestion_duration_seconds`, `fx_model_refit_duration_seconds`, `fx_order_fill_duration_seconds`, `fx_slippage_bps`

Decorators for clean instrumentation: `@track_duration(histogram)`, `@track_errors(service, category)`, `HeartbeatTracker` class.

### 8.3 Logging

**Rules**: Structured JSON logging only. Context via thread-local filter. Every log line carries `strategy_id`, `symbol`, `trade_id` when relevant.

```python
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'ts': ...,
            'level': record.levelname,
            'logger': record.name,
            'msg': record.getMessage(),
            # + context from thread-local filter
        })

with LogContext(strategy_id='eurusd_rate_diff_mr'):
    logger.info("Signal computed", extra={'extra_data': {...}})
```

**Loki queries**: `{job="fx-services", level="ERROR"}`, `{strategy_id="cb_sentiment_shift"}`, `{symbol="EURUSD"} |= "kill switch"`

### 8.4 Alerting

**Severity levels**:
- **Critical (page via Pushover priority 2)**: Kill switch triggered, drawdown > 15%, reconciliation mismatch, engine down, price stream disconnected, order reject rate > 20%
- **Warning (Pushover normal + email)**: Drawdown > 10%, daily loss > 2%, model R² degraded (< 0.20), data stale (> 5 min), P90 slippage > 3 bps, spread widening (> 5 bps), correlation regime stressed
- **Info (log only)**: All other

**Alert rules**: Prometheus YAML with `for:` clauses for debounce. Inhibit: engine down → suppress all downstream alerts.

### 8.5 Dashboards (Grafana)

Three viewing contexts:
1. **Glance** (always on): Today's P&L, equity curve, exposure, active signals, system health, market regime
2. **Daily review**: P&L attribution by strategy/pair, execution quality, model health, correlation heatmap
3. **Incident**: Services up/down, kill switches, open positions, recent errors, live log stream

### 8.6 Runbooks

Every critical alert has a Markdown runbook with: what it means, immediate actions, common causes, resolution steps, escalation notes. Runbooks are for 3 AM half-asleep you.

### 8.7 Watchdog Service

Separate systemd service polling `/metrics` endpoint every 30 seconds. If engine heartbeat > 120s old for 3 consecutive checks → critical alert + automatic restart.

### 8.8 Backup & Recovery

- **Daily**: DB dump, configs, model artifacts (encrypted via `age`). Retention: 30 daily, 12 weekly, 12 monthly.
- **Quarterly**: Full disaster recovery drill to separate machine. Target: full recovery < 2 hours.
- **Test monthly**: Random file restore from backup, verify integrity.

---

## 9. Deployment

### 9.1 Server Requirements

- 8 cores, 32GB RAM, 1TB NVMe SSD
- Ubuntu 24.04 LTS
- UPS (power blip during market hours with open positions is bad)
- Wired ethernet (never Wi-Fi for trading)

### 9.2 Service Architecture

```
Host: fx-server
├── Systemd services (native)
│   ├── fx-ingestion.service
│   ├── fx-live-engine.service
│   ├── fx-risk-monitor.service
│   └── fx-health-check.service (watchdog)
│
├── Docker Compose stack (observability)
│   ├── prometheus + grafana + loki + promtail + alertmanager
│   ├── node_exporter (host metrics)
│   └── postgres_exporter (DB metrics)
│
└── Caddy/nginx reverse proxy
    └── grafana.local → :3000
```

**Why systemd for trading, Docker for observability?** Trading services benefit from direct host access, simple restart, easy debugging. Observability services benefit from Docker Compose isolation and easy upgrades.

### 9.3 Systemd Unit (live engine)

```ini
[Service]
User=fx
WorkingDirectory=/opt/fx-system
Environment="PYTHONPATH=/opt/fx-system"
Environment="FX_ENV=production"
ExecStart=/opt/fx-system/venv/bin/python -m src.runtime.run_engine
Restart=on-failure
RestartSec=15s
MemoryMax=8G
ProtectSystem=strict
ReadWritePaths=/opt/fx-system/logs /opt/fx-system/data
```

### 9.4 Weekly Operating Rhythm

- **Monday AM**: Review weekend price action, verify scheduled ingestions ran
- **Daily**: 5-min morning glance, 15-min evening review
- **Saturday**: Performance review, model health, rolling Sharpe vs expectations, alert review
- **Monthly**: Full correlation matrix, strategy attribution, parameter sensitivity, backup verify
- **Quarterly**: Disaster recovery drill, reaction function recalibration, dependency updates

### 9.5 Anti-Patterns

- Dashboarding wrong metrics (if it doesn't drive a decision, remove it)
- Alerting on symptoms, not causes
- Logging everything at INFO (reserve for meaningful state changes)
- Ignoring alerts because "that one is usually fine"
- Not testing alert channels monthly
- Backing up without testing restore
- Hand-editing configs without version control
- Running engine from user shell (use systemd)

---

## 10. Data Vendors & Broker Comparison

### 10.1 Recommended Data Stack (Free / Near-Free)

| Source | Data | Update | Usage |
|--------|------|--------|-------|
| **Stooq** | Daily FX, yields, commodities, indices | Daily | Historical + live daily |
| **FRED (ALFRED)** | US macro (CPI, NFP, GDP, rates) | Varies | Fundamental data, vintage backtesting |
| **ECB SDW** | European macro, ESTR, HICP | Varies | EUR fundamentals |
| **BoJ / BoE / BoC** | Respective country macro | Varies | Cross-country |
| **BIS** | REER, cross-country stats | Monthly | Valuation overlay |
| **CME** | SOFR futures settlements | Daily | OIS curve construction |
| **CME FedWatch** | Implied rate probabilities | Real-time | Meeting-specific expectations |
| **CFTC** | COT positioning reports | Weekly (Fri) | Positioning |
| **Broker API** | Real-time FX spot prices | Streaming | Live trading |
| **CBOE** | VIX, EVZ vol indices | Daily | Risk regime detection |

**Total cost: $0**. Proven sufficient for daily signal strategies.

### 10.2 Constructing OIS Curves from CME Futures

SOFR 1-month (SR1) and 3-month (SR3) futures settlement prices are freely downloadable from CME. Build the curve:
1. Fetch settlement prices for active contracts
2. Implied rate = 100 − settlement price (as decimal)
3. Use SR1 for first ~12 months (higher resolution); SR3 beyond (deeper liquidity at longer tenors)
4. Build `OISCurve` from synthetic par rates

Accuracy: error at 1Y < 2 bps, at 2Y 3–5 bps. Usable for retail.

### 10.3 Broker Comparison

| Criterion | OANDA | IBKR |
|-----------|-------|------|
| **API quality** | Clean REST + streaming, modern v20 API | TWS API, stateful, old design. Use `ib_insync` wrapper |
| **EUR/USD spread** | 0.6–1.2 pips | 0.1–0.2 pips |
| **Commission** | Spread-embedded | $2 per $1M notional minimum |
| **Min deposit** | $0 (practical: $1k) | $0 (practical: $10k) |
| **Leverage** | 50:1 majors (US) | 50:1 majors (US) |
| **Instrument coverage** | FX spot + CFDs | FX, futures, equities, bonds, options |
| **Regulation** | CFTC/NFA, FCA, MAS | SEC, FINRA, CFTC, FCA |
| **Practice environment** | Excellent (fxpractice) | Good (paper account) |
| **Data subscriptions** | Free | $10-50/month |
| **Recommendation** | **Start here** | **Scale here** |

**Cost comparison** (EUR/USD, 50 trades/year, $50k avg notional):
- OANDA: ~$2,000/year (spread only)
- IBKR: ~$620-760/year (spread + commission + data)

At $50k account doing 15% annual returns ($7,500), OANDA costs eat 27% of gross profit. IBKR eats 10%. This gap widens with more notional.

**Migration path**: OANDA (build + prove) → IBKR (scale). Broker abstraction layer makes the swap ~1 week.

### 10.4 Connection Architecture (IBKR)

```
fx-server → TWS Gateway (auto-restart via ibc) → IBKR servers
```

TWS Gateway needs: 1 vCPU, 1-2GB RAM, nightly restart during weekend (via `ibc` wrapper to handle auth/2FA).

---

## 11. FinBERT Fine-Tuning

### 11.1 Training Configuration

- **Base model**: `ProsusAI/finbert`
- **Classes**: 0 = dovish, 1 = neutral, 2 = hawkish
- **Labels**: 2,000–3,000 sentences across CBs, doc types, regimes, time periods
- **Split**: Temporal-document split (70/15/15)
- **Batch size**: 16 (GPU) / 8 (CPU)
- **Epochs**: 6, early stopping at patience 2
- **Learning rate**: 2e-5, linear schedule, warmup 10%
- **Loss**: Cross-entropy with class weights (balanced)
- **Max length**: 256 tokens
- **Hardware**: RTX 3090/4090 → 15-30 min; CPU 16+ cores → 8-12 hours; Cloud (Lambda A10) → $1-2 total

### 11.2 Expected Metrics

| Label Count | Accuracy | F1 Macro | Per-Class F1 |
|-------------|----------|----------|--------------|
| Out-of-box (0) | 55-62% | 0.45-0.55 | — |
| 500 labels | 72-78% | 0.65-0.72 | — |
| 2,000 labels | 82-86% | 0.78-0.83 | D: 0.74-0.80, N: 0.85-0.89, H: 0.76-0.82 |
| 5,000+ labels | 85-88% | 0.82-0.86 | Plateau region |

**Human ceiling** (re-label same 100 sentences 2 weeks later): ~90-92% Cohen's κ. Model at 85% is competitive.

### 11.3 Calibration

Softmax outputs are usually overconfident. Apply **temperature scaling**: fit a single temperature parameter on validation set via LBFGS. Typically temperature ≈ 1.5–2.0, shrinking max confidences by 10–15%. ECE drops from ~0.08 to ~0.02.

### 11.4 Inference Deployment

Serve via local FastAPI on `127.0.0.1:8100`. Endpoints: `/score` (sentence list), `/score_document` (aggregate), `/health`. Model loaded once, serves many requests. Strategies hit this endpoint rather than loading model directly.

---

## 12. FX Overnight Rolls & Funding

### 12.1 Mechanics

FX spot: **T+2 settlement** (T+1 for USD/CAD). Holding past settlement → broker automatically rolls (swaps) the position forward one day. The roll cost = overnight interest rate differential between the two currencies.

```
Daily swap = Notional × (currency_held_rate − currency_shorted_rate) × (1/360) × multiplier
```

**Triple swap Wednesday**: Positions held over Wednesday night roll through three calendar days (Fri+Sat+Sun). Wednesday swap includes 3× the normal daily rate.

### 12.2 Broker Markup

Retail swap rates are worse than interbank:
- **IBKR**: 0.25-0.50% markup (best retail)
- **OANDA**: 0.50-1.50% markup
- **Most other retail**: 1-3%

On a $100k carry position with 3% rate differential: broker might credit you 2-2.5% instead of 3%. Over a year on large notional, thousands in hidden cost.

### 12.3 Swap Modeling in Backtests

```python
class SwapModel:
    def compute_daily_swap(self, pair, position, target_rate, base_rate, day_of_week):
        rate_diff = target_rate - base_rate
        daily_rate_diff = rate_diff / 360
        markup_penalty = broker_markup_pct / 100 / 360
        multiplier = 3 if day_of_week == 2 else 1  # Wednesday
        if day_of_week in (5, 6): return 0  # Weekend
        effective = daily_rate_diff - markup_penalty * np.sign(position * rate_diff)
        return abs(position) * effective * multiplier
```

**Impact on strategy P&L**: A strategy showing 8% annual return ignoring swaps might only show 5% with realistic modeling. Swap costs eat 12-20% of gross profit for mean-reversion strategies holding ~10 days. For carry strategies, swap is the majority of expected return — getting it wrong completely invalidates the backtest.

### 12.4 Cross-Currency Basis

The FX swap market prices currency-specific funding pressure. When USD is scarce globally (year-end, crisis), EUR/USD basis goes deeply negative — European banks pay extra to borrow dollars synthetically. Monitoring basis is a leading signal for dollar funding stress. Sources: BIS (monthly, free), Bloomberg (daily, paid).

---

## Data Sources Reference

| Source | Coverage | API/Access | Cost | Key Series |
|--------|----------|------------|------|------------|
| **FRED** | US macro, yields, FX | REST API (free key) | Free | DFF, DGS2, DGS10, CPIAUCSL, PCEPILFE, UNRATE, PAYEMS, GDPC1, DTWEXBGS |
| **ALFRED** | US vintage data | FRED API subset | Free | Same as FRED, point-in-time |
| **ECB SDW** | European macro, ESTR | REST API | Free | Policy rates, HICP, GDP, yield curves |
| **BoJ** | Japanese macro, TONA | Web/API | Free | Policy rate, CPI, GDP |
| **BoE** | UK macro, SONIA | Web/database | Free | Bank rate, CPI, GDP |
| **BIS** | Cross-country, REER | Web/database | Free | Effective exchange rates, banking stats |
| **OECD** | Cross-country macro | API | Free | Standardized GDP, CPI, employment |
| **CME** | SOFR futures, FedWatch | REST/web | Free | SR1, SR3 settlements |
| **CFTC** | COT positioning | Text/Excel download | Free | Weekly TFF reports |
| **CBOE** | VIX, VIX term, EVZ | Web | Free | Vol indices |
| **Stooq** | Daily FX, yields, commodities | CSV download | Free | EUR/USD, 10Y yields, gold, oil |
| **Dukascopy** | Tick FX historical | Desktop tool | Free | Tick-by-tick back to 2003 |
| **OANDA API** | Live FX pricing | REST + streaming | Free (with account) | All major/minor pairs |

---

## What Will Go Wrong

A catalogue of problems every retail quant hits:

- **Data inconsistencies** eat days. Different vendors handle corporate actions, weekends, holidays differently. Build defensive checks everywhere.
- **Timezones will burn you.** FX trades 24/5 across timezones. Pin everything to UTC internally; only convert at the edges.
- **Your first backtest will look amazing** — it's probably buggy. When Sharpe > 3, look for look-ahead bias.
- **Your second backtest will look worse** — that's closer to truth.
- **Real performance will be 30-50% worse than backtest.** Slippage, partial fills, emotional overrides, model drift.
- **You'll want to over-engineer.** Ship a working v1 of everything before optimizing any single component.
- **Most quit in months 3-9** because the work is hard and early results are usually disappointing.
- **Risk management matters more than prediction.** A 55% accurate system with 2:1 reward/risk and proper sizing crushes a 70% system with 1:1 and overleverage.
- **Drawdowns of 20-30% happen to good systems.** Without emotional discipline, you'll abandon good strategies at drawdown bottoms and chase dead ones.
- **Start small, scale slowly.** Goal for year 1: don't blow up while learning. Profitability in year 2+ is realistic if the foundation is right.
- **The edge is shrinking everywhere.** Every signal described is known to institutions. Retail edge comes from being small enough to fit in illiquid niches, longer holding periods, ruthless execution discipline, or combining many weak signals.
