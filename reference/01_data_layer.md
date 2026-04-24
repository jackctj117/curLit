# 01 — Data Layer

Database schema, ingestion base classes, and concrete data source clients.

## Database Schema

**Location:** `scripts/init_db_schema.py` or SQL migration file
**Purpose:** PostgreSQL + TimescaleDB schema for all time-series data

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Price data (FX, futures, equities, etc.)
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

-- Simpler FX prices table used by backtest scripts
CREATE TABLE fx_prices (
    date DATE NOT NULL,
    pair TEXT NOT NULL,
    close NUMERIC NOT NULL,
    high NUMERIC,
    low NUMERIC,
    volume NUMERIC,
    PRIMARY KEY (date, pair)
);
CREATE INDEX idx_fx_pair_date ON fx_prices(pair, date DESC);

-- Macro economic data with vintage (release dates) — critical for backtesting
CREATE TABLE macro_data (
    observation_date DATE NOT NULL,
    release_date TIMESTAMPTZ NOT NULL,  -- When this value was published
    series_id TEXT NOT NULL,
    value NUMERIC,
    revision INT DEFAULT 0,
    source TEXT NOT NULL,
    PRIMARY KEY (observation_date, release_date, series_id)
);
CREATE INDEX idx_macro_series ON macro_data (series_id, observation_date DESC);

-- Interest rates (simpler table for monthly series)
CREATE TABLE interest_rates (
    date DATE NOT NULL,
    currency TEXT NOT NULL,
    rate_pct NUMERIC NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (date, currency, source)
);
CREATE INDEX idx_rates_ccy_date ON interest_rates(currency, date DESC);

-- Rate curves (OIS, treasuries, etc.) — snapshot by date
CREATE TABLE rate_curves (
    ts TIMESTAMPTZ NOT NULL,
    curve_id TEXT NOT NULL,
    tenor_days INT NOT NULL,
    rate NUMERIC,
    PRIMARY KEY (ts, curve_id, tenor_days)
);
SELECT create_hypertable('rate_curves', 'ts');

-- Volatility indices
CREATE TABLE fx_volatility (
    date DATE NOT NULL,
    index_name TEXT NOT NULL,
    value NUMERIC NOT NULL,
    PRIMARY KEY (date, index_name)
);

-- COT positioning data (weekly)
CREATE TABLE cot_positioning (
    report_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    category TEXT NOT NULL,  -- 'lev_funds', 'asset_mgr', 'dealer'
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

-- Strategy tracking tables
CREATE TABLE strategy_signals (
    ts TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    signal_data JSONB,
    PRIMARY KEY (ts, strategy_id)
);

CREATE TABLE strategy_trades (
    ts TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    action TEXT NOT NULL,  -- 'entry', 'exit'
    details JSONB,
    PRIMARY KEY (ts, strategy_id, action)
);

-- Central bank NLP tables
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

-- Research papers tracking
CREATE TABLE research_papers (
    paper_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    authors JSONB,
    abstract TEXT,
    url TEXT,
    pdf_url TEXT,
    published_date TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL,
    keywords JSONB,
    categories JSONB,
    relevance_score NUMERIC DEFAULT 0,
    read_status TEXT DEFAULT 'unread',
    my_notes TEXT,
    implementation_priority INT DEFAULT 0,
    evaluation_data JSONB
);
CREATE INDEX idx_papers_status ON research_papers(read_status, relevance_score DESC);
```

## Base Ingester

**Location:** `src/data/base.py`
**Purpose:** Abstract base class for all data ingestion. Every concrete ingester inherits from this.

```python
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseIngester(ABC):
    def __init__(self, db_url: str, source_name: str):
        self.engine = create_engine(db_url)
        self.source = source_name
    
    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch raw data from source."""
        pass
    
    @abstractmethod
    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize to schema."""
        pass
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def run(self, start: datetime, end: datetime) -> int:
        logger.info(f"Ingesting {self.source} from {start} to {end}")
        raw = self.fetch(start, end)
        if raw.empty:
            logger.warning(f"No data for {self.source}")
            return 0
        df = self.transform(raw)
        df = self.validate(df)
        rows = self.upsert(df)
        logger.info(f"{self.source}: wrote {rows} rows")
        return rows
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['ts', 'symbol'])
        df = df.drop_duplicates(subset=['ts', 'symbol'])
        return df
    
    def upsert(self, df: pd.DataFrame) -> int:
        # Implementation depends on table — use COPY + ON CONFLICT for performance
        raise NotImplementedError
```

## FRED Provider

**Location:** `src/data/fred.py`
**Purpose:** Fetch US macroeconomic data from FRED API. Supports vintage (ALFRED) data for proper backtesting.

```python
import os
import time
import httpx
import pandas as pd
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential


class FREDProvider:
    BASE_URL = 'https://api.stlouisfed.org/fred'
    RATE_LIMIT_PER_SEC = 0.5
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ['FRED_API_KEY']
        self._last_request = 0
    
    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 1 / self.RATE_LIMIT_PER_SEC:
            time.sleep(1 / self.RATE_LIMIT_PER_SEC - elapsed)
        self._last_request = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def _get(self, endpoint: str, params: dict) -> dict:
        self._wait_for_rate_limit()
        params = {**params, 'api_key': self.api_key, 'file_type': 'json'}
        resp = httpx.get(f'{self.BASE_URL}/{endpoint}', params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    
    def get_series(self, series_id: str, 
                    start: datetime = None, end: datetime = None) -> pd.DataFrame:
        params = {'series_id': series_id}
        if start:
            params['observation_start'] = start.strftime('%Y-%m-%d')
        if end:
            params['observation_end'] = end.strftime('%Y-%m-%d')
        
        data = self._get('series/observations', params)
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.set_index('date')[['value']]
        df.columns = [series_id]
        return df
    
    def get_vintage(self, series_id: str, as_of_date: datetime) -> pd.DataFrame:
        """ALFRED: series values as they existed on a specific date."""
        params = {
            'series_id': series_id,
            'realtime_start': as_of_date.strftime('%Y-%m-%d'),
            'realtime_end': as_of_date.strftime('%Y-%m-%d'),
        }
        data = self._get('series/observations', params)
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.set_index('date')[['value']]
    
    def get_all_releases(self, series_id: str) -> pd.DataFrame:
        """Full release history with realtime_start dates."""
        data = self._get('series/observations', {
            'series_id': series_id,
            'realtime_start': '1776-07-04',
            'realtime_end': '9999-12-31',
        })
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['realtime_start'] = pd.to_datetime(df['realtime_start'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df


# Common FRED series IDs for reference
FRED_SERIES = {
    'DFF': 'Effective Federal Funds Rate',
    'DGS2': 'US Treasury 2Y',
    'DGS10': 'US Treasury 10Y',
    'SOFR': 'Secured Overnight Financing Rate',
    'DFEDTARL': 'Fed Funds Target Lower',
    'DFEDTARU': 'Fed Funds Target Upper',
    'CPIAUCSL': 'CPI All Urban',
    'CPILFESL': 'Core CPI',
    'PCEPI': 'PCE Price Index',
    'PCEPILFE': 'Core PCE',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payrolls',
    'CES0500000003': 'Avg Hourly Earnings',
    'JTSJOL': 'Job Openings',
    'GDPC1': 'Real GDP',
    'INDPRO': 'Industrial Production',
    'RSAFS': 'Retail Sales',
    'UMCSENT': 'Consumer Sentiment',
    'DTWEXBGS': 'Trade Weighted USD Index (broad)',
    'IRLTLT01DEM156N': 'Germany 10Y',
    'IRLTLT01JPM156N': 'Japan 10Y',
    'IRLTLT01GBM156N': 'UK 10Y',
}

# 3-month interbank/OIS rate series for carry strategy
RATE_SERIES = {
    'USD': 'IR3TIB01USM156N',
    'EUR': 'IR3TIB01EZM156N',
    'JPY': 'IR3TIB01JPM156N',
    'GBP': 'IR3TIB01GBM156N',
    'CHF': 'IR3TIB01CHM156N',
    'CAD': 'IR3TIB01CAM156N',
    'AUD': 'IR3TIB01AUM156N',
    'NZD': 'IR3TIB01NZM156N',
    'NOK': 'IR3TIB01NOM156N',
    'SEK': 'IR3TIB01SEM156N',
}
```

## FRED Concrete Ingester

**Location:** `src/data/fred_ingester.py`
**Purpose:** BaseIngester subclass for FRED data

```python
from src.data.base import BaseIngester
from fredapi import Fred


class FREDIngester(BaseIngester):
    def __init__(self, db_url: str, api_key: str, series_ids: list[str]):
        super().__init__(db_url, 'fred')
        self.api_key = api_key
        self.series_ids = series_ids
    
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        fred = Fred(api_key=self.api_key)
        frames = []
        for series_id in self.series_ids:
            # ALFRED gives vintage data — critical for backtesting
            data = fred.get_series_all_releases(series_id)
            data['series_id'] = series_id
            frames.append(data)
        return pd.concat(frames, ignore_index=True)
    
    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={
            'date': 'observation_date',
            'realtime_start': 'release_date',
        })
        df = df[['observation_date', 'release_date', 'series_id', 'value']]
        df['source'] = self.source
        return df
```

## Stooq FX Provider

**Location:** `src/data/stooq.py`
**Purpose:** Free daily FX, indices, commodities data. No API key required.

```python
import httpx
import pandas as pd
from io import StringIO
from datetime import datetime


class StooqDataProvider:
    BASE_URL = 'https://stooq.com/q/d/l/'
    
    SYMBOL_MAP = {
        'EURUSD': 'eurusd',
        'GBPUSD': 'gbpusd',
        'USDJPY': 'usdjpy',
        'USDCAD': 'usdcad',
        'AUDUSD': 'audusd',
        'NZDUSD': 'nzdusd',
        'USDCHF': 'usdchf',
        'DXY': '^dxy',
        'GOLD': 'xauusd',
        'OIL_WTI': 'cl.f',
        'COPPER': 'hg.f',
    }
    
    def fetch_daily(self, symbol: str, start: datetime, end: datetime,
                     interval: str = 'd') -> pd.DataFrame:
        """interval: d (daily), w (weekly), m (monthly)"""
        code = self.SYMBOL_MAP.get(symbol, symbol.lower())
        params = {
            's': code,
            'i': interval,
            'd1': start.strftime('%Y%m%d'),
            'd2': end.strftime('%Y%m%d'),
        }
        
        resp = httpx.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        
        df = pd.read_csv(StringIO(resp.text))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df.columns = [c.lower() for c in df.columns]
        df['symbol'] = symbol
        return df
```

## CME SOFR Futures Provider

**Location:** `src/data/cme_sofr.py`
**Purpose:** Fetch SOFR futures settlement prices from CME for OIS curve construction.

```python
import httpx
import pandas as pd
from datetime import datetime, date


class CMESOFRProvider:
    """Fetch SOFR futures settlement data from CME."""
    
    SETTLEMENTS_URL = 'https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/Settlements/{product_id}/FUT'
    
    PRODUCT_IDS = {
        'SR1': '8463',  # 1-month SOFR
        'SR3': '8462',  # 3-month SOFR
    }
    
    def fetch_settlements(self, product: str, trade_date: date = None) -> pd.DataFrame:
        """Fetch settlement prices for all active contracts."""
        params = {}
        if trade_date:
            params['tradeDate'] = trade_date.strftime('%m/%d/%Y')
        
        url = self.SETTLEMENTS_URL.format(product_id=self.PRODUCT_IDS[product])
        resp = httpx.get(url, params=params, 
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        rows = []
        for settlement in data.get('settlements', []):
            month_code = settlement['month']
            price_str = settlement['settle']
            if price_str in ('-', 'Cab'):
                continue
            price = float(price_str)
            # SOFR futures: implied rate = 100 - price
            implied_rate = (100 - price) / 100
            
            expiry = self._parse_month_code(month_code)
            rows.append({
                'product': product,
                'month_code': month_code,
                'expiry': expiry,
                'settle_price': price,
                'implied_rate': implied_rate,
                'trade_date': data.get('tradeDate'),
            })
        
        return pd.DataFrame(rows)
    
    def _parse_month_code(self, code: str) -> date:
        """Parse 'MAR 26' to date(2026, 3, 15) (approximate expiry)."""
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
        }
        parts = code.strip().split()
        month = month_map[parts[0].upper()]
        year = 2000 + int(parts[1])
        # Third Wednesday approximation for SOFR settlement
        d = date(year, month, 15)
        while d.weekday() != 2:
            d = date(year, month, d.day + 1)
        return d
```

## Complete Carry Data Ingestion Script

**Location:** `scripts/ingest_carry_data.py`
**Purpose:** End-to-end ingestion for carry+vol strategy backtest. Runs all needed sources.

```python
import os
import time
from datetime import datetime, timedelta, date
import logging

import pandas as pd
import numpy as np
import httpx
from fredapi import Fred
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


RATE_SERIES = {
    'USD': 'IR3TIB01USM156N',
    'EUR': 'IR3TIB01EZM156N',
    'JPY': 'IR3TIB01JPM156N',
    'GBP': 'IR3TIB01GBM156N',
    'CHF': 'IR3TIB01CHM156N',
    'CAD': 'IR3TIB01CAM156N',
    'AUD': 'IR3TIB01AUM156N',
    'NZD': 'IR3TIB01NZM156N',
    'NOK': 'IR3TIB01NOM156N',
    'SEK': 'IR3TIB01SEM156N',
}

FX_SYMBOLS = {
    'EURUSD': 'eurusd',
    'USDJPY': 'usdjpy',
    'GBPUSD': 'gbpusd',
    'USDCHF': 'usdchf',
    'USDCAD': 'usdcad',
    'AUDUSD': 'audusd',
    'NZDUSD': 'nzdusd',
    'USDNOK': 'usdnok',
    'USDSEK': 'usdsek',
}


class DataIngester:
    def __init__(self, db_url: str, fred_api_key: str):
        self.engine = create_engine(db_url)
        self.fred = Fred(api_key=fred_api_key)
        self._ensure_schema()
    
    def _ensure_schema(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fx_prices (
                    date DATE NOT NULL,
                    pair TEXT NOT NULL,
                    close NUMERIC NOT NULL,
                    high NUMERIC,
                    low NUMERIC,
                    volume NUMERIC,
                    PRIMARY KEY (date, pair)
                );
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS interest_rates (
                    date DATE NOT NULL,
                    currency TEXT NOT NULL,
                    rate_pct NUMERIC NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (date, currency, source)
                );
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fx_volatility (
                    date DATE NOT NULL,
                    index_name TEXT NOT NULL,
                    value NUMERIC NOT NULL,
                    PRIMARY KEY (date, index_name)
                );
            """))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def ingest_fred_rates(self, start: date = date(2000, 1, 1)) -> int:
        total_rows = 0
        for ccy, series_id in RATE_SERIES.items():
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start.strftime('%Y-%m-%d')
                )
                df = series.to_frame('rate_pct').dropna()
                df.index.name = 'date'
                df = df.reset_index()
                df['currency'] = ccy
                df['source'] = 'fred'
                
                with self.engine.begin() as conn:
                    for _, row in df.iterrows():
                        conn.execute(text("""
                            INSERT INTO interest_rates (date, currency, rate_pct, source)
                            VALUES (:date, :ccy, :rate, :src)
                            ON CONFLICT (date, currency, source) 
                            DO UPDATE SET rate_pct = EXCLUDED.rate_pct
                        """), {
                            'date': row['date'].date(),
                            'ccy': ccy,
                            'rate': float(row['rate_pct']),
                            'src': 'fred',
                        })
                
                total_rows += len(df)
                logger.info(f"Ingested {len(df)} rate observations for {ccy}")
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to ingest {ccy} rates: {e}")
        
        return total_rows
    
    def ingest_stooq_fx(self, start: date = date(2000, 1, 1), 
                        end: date = None) -> int:
        end = end or date.today()
        total_rows = 0
        
        for pair, stooq_code in FX_SYMBOLS.items():
            try:
                url = 'https://stooq.com/q/d/l/'
                params = {
                    's': stooq_code,
                    'i': 'd',
                    'd1': start.strftime('%Y%m%d'),
                    'd2': end.strftime('%Y%m%d'),
                }
                resp = httpx.get(url, params=params, timeout=30)
                resp.raise_for_status()
                
                if 'No data' in resp.text or len(resp.text) < 100:
                    logger.warning(f"No Stooq data for {pair}")
                    continue
                
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df.columns = [c.lower() for c in df.columns]
                
                with self.engine.begin() as conn:
                    for _, row in df.iterrows():
                        conn.execute(text("""
                            INSERT INTO fx_prices (date, pair, close, high, low, volume)
                            VALUES (:date, :pair, :close, :high, :low, :vol)
                            ON CONFLICT (date, pair) 
                            DO UPDATE SET close = EXCLUDED.close
                        """), {
                            'date': row['date'],
                            'pair': pair,
                            'close': float(row['close']),
                            'high': float(row.get('high', row['close'])),
                            'low': float(row.get('low', row['close'])),
                            'vol': float(row.get('volume', 0)) if 'volume' in row else None,
                        })
                
                total_rows += len(df)
                logger.info(f"Ingested {len(df)} FX observations for {pair}")
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Failed to ingest {pair}: {e}")
        
        return total_rows
    
    def compute_realized_vol_proxy(self, start: date = date(2000, 1, 1)) -> int:
        """Compute equal-weighted realized vol across G10 pairs as CVIX proxy."""
        pairs = list(FX_SYMBOLS.keys())
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT date, pair, close FROM fx_prices 
                WHERE pair = ANY(:pairs) AND date >= :start
                ORDER BY date
            """), conn, params={'pairs': pairs, 'start': start})
        
        df['date'] = pd.to_datetime(df['date'])
        pivot = df.pivot(index='date', columns='pair', values='close')
        log_returns = np.log(pivot / pivot.shift(1))
        realized_vol = log_returns.rolling(20).std() * np.sqrt(252) * 100
        avg_vol = realized_vol.mean(axis=1).dropna()
        
        with self.engine.begin() as conn:
            for date_val, vol_val in avg_vol.items():
                conn.execute(text("""
                    INSERT INTO fx_volatility (date, index_name, value)
                    VALUES (:date, 'G10_RV20', :val)
                    ON CONFLICT (date, index_name) 
                    DO UPDATE SET value = EXCLUDED.value
                """), {'date': date_val.date(), 'val': float(vol_val)})
        
        logger.info(f"Computed {len(avg_vol)} realized vol observations")
        return len(avg_vol)


def main():
    db_url = os.environ.get('DATABASE_URL', 'postgresql://localhost/fx')
    fred_key = os.environ['FRED_API_KEY']
    ingester = DataIngester(db_url, fred_key)
    
    n_rates = ingester.ingest_fred_rates()
    logger.info(f"Total rates ingested: {n_rates}")
    n_fx = ingester.ingest_stooq_fx()
    logger.info(f"Total FX observations ingested: {n_fx}")
    n_vol = ingester.compute_realized_vol_proxy()
    logger.info(f"Volatility observations computed: {n_vol}")


if __name__ == '__main__':
    main()
```

## Data Quality Validation

**Location:** `src/data/validation.py`
**Purpose:** Post-ingestion quality checks using great_expectations or similar.

```python
import great_expectations as ge
import pandas as pd


def validate_price_data(df: pd.DataFrame) -> bool:
    dfe = ge.from_pandas(df)
    dfe.expect_column_values_to_not_be_null('close')
    dfe.expect_column_values_to_be_between('close', 0, 1e6)
    dfe.expect_column_values_to_be_unique(['ts', 'symbol'])
    # Check for gaps
    ts_gaps = df.set_index('ts').resample('D').size()
    assert (ts_gaps == 0).sum() < len(ts_gaps) * 0.05  # <5% missing
    return dfe.validate().success
```

## Vintage Data Query Helper

**Location:** `src/data/vintage.py`
**Purpose:** Retrieve point-in-time data for backtesting without look-ahead bias.

```python
from datetime import datetime
import pandas as pd


def get_macro_as_of(engine, series_id: str, as_of: datetime) -> pd.DataFrame:
    """Return the vintage of a series as it existed on `as_of`."""
    query = """
        SELECT DISTINCT ON (observation_date) 
            observation_date, value, release_date
        FROM macro_data
        WHERE series_id = %(series_id)s
          AND release_date <= %(as_of)s
        ORDER BY observation_date, release_date DESC
    """
    return pd.read_sql(query, engine, params={
        'series_id': series_id, 
        'as_of': as_of
    })
```

## Data Provider (Live Use)

**Location:** `src/data/provider.py`
**Purpose:** High-level interface for live strategies to query data.

```python
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import text


class DataProvider:
    def __init__(self, engine):
        self.engine = engine
    
    def get_aligned_series(self, symbols: list[str], start: datetime, 
                            end: datetime) -> pd.DataFrame:
        """Return aligned daily closes for given symbols."""
        query = text("""
            SELECT ts::date as date, symbol, close
            FROM prices
            WHERE symbol = ANY(:symbols)
              AND ts >= :start AND ts <= :end
            ORDER BY ts
        """)
        df = pd.read_sql(query, self.engine, 
                         params={'symbols': symbols, 'start': start, 'end': end})
        pivot = df.pivot(index='date', columns='symbol', values='close')
        pivot.index = pd.to_datetime(pivot.index)
        return pivot
    
    async def get_latest_rate_spread(self) -> float | None:
        query = text("""
            SELECT (us.close - de.close) as spread
            FROM prices us, prices de
            WHERE us.symbol = 'US_2Y' AND de.symbol = 'DE_2Y'
              AND us.ts::date = de.ts::date
            ORDER BY us.ts DESC
            LIMIT 1
        """)
        result = self.engine.execute(query).fetchone()
        return float(result[0]) if result else None
    
    async def get_recent_vol(self, symbol: str, window: int = 20) -> float | None:
        end = datetime.utcnow()
        start = end - timedelta(days=window * 2)
        df = self.get_aligned_series([symbol], start, end)
        if len(df) < window:
            return None
        returns = np.log(df[symbol] / df[symbol].shift(1)).dropna().tail(window)
        return float(returns.std() * np.sqrt(252))
```
