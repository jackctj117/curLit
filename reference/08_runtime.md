# 08 — Runtime / Live Engine

The async engine that runs everything live: price streaming, signal generation, OMS reconciliation, monitoring.

## Live Engine

**Location:** `src/runtime/live_engine.py`
**Purpose:** Top-level coordinator running all async tasks. Single instance per deployment.

```python
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Optional

from src.execution.broker import Broker
from src.execution.oms import OMS
from src.portfolio.coordinator import PortfolioCoordinator
from src.monitoring import metrics as m

logger = logging.getLogger(__name__)


class LiveEngine:
    def __init__(self, strategies: list, coordinator: PortfolioCoordinator,
                 oms: OMS, broker: Broker, config):
        self.strategies = strategies
        self.coordinator = coordinator
        self.oms = oms
        self.broker = broker
        self.config = config
        
        self.running = False
        self._tasks: list[asyncio.Task] = []
        self._last_signal_times: dict[str, datetime] = {}
        self._last_prices: dict[str, dict] = {}
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        logger.info("Starting LiveEngine")
        self.running = True
        
        # Restore state from previous run
        for strategy in self.strategies:
            state = strategy.state.load_current_position(strategy.id)
            if state:
                logger.info(f"Restored {strategy.id} state: {state}")
        
        # Verify broker connectivity
        try:
            account = self.broker.get_account()
            logger.info(f"Connected to broker. Equity: {account.equity}")
            m.account_equity.set(account.equity)
        except Exception as e:
            logger.error(f"Cannot connect to broker: {e}")
            raise
        
        # Spawn async tasks
        symbols = list(set(s for strat in self.strategies for s in strat.symbols))
        
        self._tasks = [
            asyncio.create_task(self._price_stream_task(symbols), 
                               name='price_stream'),
            asyncio.create_task(self._signal_generation_task(), 
                               name='signal_generation'),
            asyncio.create_task(self._reconciliation_task(), 
                               name='reconciliation'),
            asyncio.create_task(self._health_check_task(), 
                               name='health_check'),
            asyncio.create_task(self._metrics_publisher_task(), 
                               name='metrics_publisher'),
            asyncio.create_task(self._rebalance_task(), 
                               name='rebalance'),
        ]
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._initiate_shutdown)
        
        # Wait for shutdown signal or task failure
        done, pending = await asyncio.wait(
            self._tasks + [asyncio.create_task(self._shutdown_event.wait(),
                                                name='shutdown_waiter')],
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        # Check if a task failed
        for task in done:
            if task.get_name() == 'shutdown_waiter':
                continue
            if task.exception():
                logger.exception(f"Task {task.get_name()} failed")
                m.errors_total.labels(
                    service='live_engine',
                    severity='critical',
                    category=f'task_{task.get_name()}_died',
                ).inc()
        
        # Initiate shutdown
        await self.graceful_shutdown()
    
    async def _price_stream_task(self, symbols: list[str]):
        retry_delay = 1
        while self.running:
            try:
                logger.info(f"Streaming prices for {symbols}")
                async for price in self.broker.stream_prices(symbols):
                    self._last_prices[price['symbol']] = price
                    m.prices_received.labels(symbol=price['symbol']).inc()
                    retry_delay = 1
            except Exception as e:
                logger.exception(f"Price stream error: {e}")
                m.errors_total.labels(
                    service='live_engine',
                    severity='warning',
                    category='price_stream_disconnect',
                ).inc()
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
    
    async def _signal_generation_task(self):
        while self.running:
            now = datetime.utcnow()
            if not self._in_trading_window(now):
                await asyncio.sleep(60)
                continue
            
            # Collect intents from all strategies
            raw_intents: dict[str, list] = {}
            
            for strategy in self.strategies:
                last = self._last_signal_times.get(strategy.id, datetime.min)
                if (now - last).total_seconds() < strategy.signal_interval_seconds:
                    continue
                
                try:
                    intents = await strategy.generate_intents(
                        self._last_prices, self.broker
                    )
                    if intents:
                        raw_intents[strategy.id] = intents
                    self._last_signal_times[strategy.id] = now
                    m.signals_generated.labels(strategy_id=strategy.id).inc()
                except Exception as e:
                    logger.exception(f"Strategy {strategy.id} error: {e}")
                    m.errors_total.labels(
                        service='live_engine',
                        severity='error',
                        category=f'strategy_{strategy.id}',
                    ).inc()
            
            # Send to coordinator (will scale, resolve conflicts, apply constraints, submit)
            if raw_intents:
                await self.coordinator.process_intents(raw_intents)
            
            await asyncio.sleep(10)
    
    async def _reconciliation_task(self):
        while self.running:
            try:
                await self.oms.reconcile()
            except Exception as e:
                logger.exception(f"Reconciliation error: {e}")
                m.errors_total.labels(
                    service='live_engine',
                    severity='error',
                    category='reconciliation_failed',
                ).inc()
            await asyncio.sleep(2)
    
    async def _health_check_task(self):
        while self.running:
            try:
                # Check broker connection
                account = self.broker.get_account()
                m.account_equity.set(account.equity)
                m.account_balance.set(account.balance)
                m.margin_used.set(account.margin_used)
                
                # Check positions
                positions = self.broker.get_positions()
                m.positions_open.set(len(positions))
                
                # Check price freshness
                now = datetime.utcnow()
                for symbol, price_data in self._last_prices.items():
                    age = (now - price_data['ts']).total_seconds()
                    m.price_staleness.labels(symbol=symbol).set(age)
                    if age > 60:
                        logger.warning(f"Stale price for {symbol}: {age:.0f}s")
                
            except Exception as e:
                logger.exception(f"Health check error: {e}")
                m.errors_total.labels(
                    service='live_engine',
                    severity='error',
                    category='health_check',
                ).inc()
            
            await asyncio.sleep(30)
    
    async def _metrics_publisher_task(self):
        while self.running:
            # Heartbeat
            m.heartbeat_seconds.set(datetime.utcnow().timestamp())
            await asyncio.sleep(5)
    
    async def _rebalance_task(self):
        """Monthly portfolio reallocation."""
        while self.running:
            try:
                await self.coordinator.rebalance_allocations()
            except Exception as e:
                logger.exception(f"Rebalance error: {e}")
            await asyncio.sleep(86400)  # Daily check
    
    def _in_trading_window(self, now: datetime) -> bool:
        # FX trades 24/5 — closed Friday 21:00 UTC to Sunday 22:00 UTC
        weekday = now.weekday()
        hour = now.hour
        if weekday == 4 and hour >= 21:  # Friday after 21:00 UTC
            return False
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and hour < 22:  # Sunday before 22:00 UTC
            return False
        return True
    
    def _initiate_shutdown(self):
        logger.info("Shutdown signal received")
        self._shutdown_event.set()
    
    async def graceful_shutdown(self):
        logger.info("Initiating graceful shutdown")
        self.running = False
        
        # Final reconciliation
        try:
            await asyncio.wait_for(self.oms.reconcile(), timeout=10)
        except asyncio.TimeoutError:
            logger.error("Final reconciliation timeout")
        
        # Persist state
        for strategy in self.strategies:
            try:
                state = strategy.get_current_state()
                strategy.state.persist_state(strategy.id, state)
            except Exception as e:
                logger.error(f"Failed to persist {strategy.id} state: {e}")
        
        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Shutdown complete")
```

## Engine Entry Point

**Location:** `src/runtime/run_engine.py`
**Purpose:** Wires up all dependencies and starts the engine.

```python
import asyncio
import os
import logging
from sqlalchemy import create_engine

from src.runtime.live_engine import LiveEngine
from src.execution.oanda import OandaBroker
from src.execution.oms import OMS
from src.portfolio.coordinator import PortfolioCoordinator, PortfolioConstraints
from src.security.vault_client import VaultClient
from src.data.provider import DataProvider
from src.nlp.provider import NLPDataProvider
from src.strategies.state import StrategyStateStore
from src.strategies.rate_diff_mean_reversion import (
    RateDiffMeanReversionStrategy, StrategyConfig
)
from src.strategies.cb_sentiment_shift import (
    CBSentimentShiftStrategy, CBSentimentConfig
)
from src.strategies.carry_vol_filter import (
    CarryVolFilterStrategy, CarryVolFilterConfig
)
from src.monitoring.metrics import setup_metrics_server
from src.monitoring.logging_config import setup_logging


async def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    setup_metrics_server(port=8000)
    
    vault = VaultClient()
    
    db_url = (f"postgresql://fx:{vault.get('POSTGRES_FX_PASSWORD')}"
              f"@localhost:5432/fx")
    db_engine = create_engine(db_url)
    
    broker = OandaBroker(
        api_key=vault.get('OANDA_API_KEY'),
        account_id=vault.get('OANDA_ACCOUNT_ID'),
        practice=os.environ.get('FX_ENV') != 'production',
    )
    
    data_provider = DataProvider(db_engine)
    nlp_provider = NLPDataProvider(db_engine)
    state_store = StrategyStateStore(db_engine)
    
    strategies = [
        RateDiffMeanReversionStrategy(
            StrategyConfig(pair='EURUSD'),
            data_provider, state_store
        ),
        CBSentimentShiftStrategy(
            CBSentimentConfig(),
            data_provider, nlp_provider, state_store
        ),
        CarryVolFilterStrategy(
            CarryVolFilterConfig(),
            data_provider, state_store
        ),
    ]
    
    oms = OMS(broker)
    
    coordinator = PortfolioCoordinator(
        strategies=strategies,
        oms=oms,
        broker=broker,
        state_store=state_store,
        constraints=PortfolioConstraints(),
    )
    coordinator.initialize_allocations({
        s.id: 1.0/len(strategies) for s in strategies
    })
    
    engine = LiveEngine(
        strategies=strategies,
        coordinator=coordinator,
        oms=oms,
        broker=broker,
        config={},
    )
    
    await engine.start()


if __name__ == '__main__':
    asyncio.run(main())
```

## Ingestion Scheduler

**Location:** `src/ingestion/run_scheduler.py`
**Purpose:** Periodically run all data ingestion jobs. Runs as separate systemd service.

```python
import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine

from src.security.vault_client import VaultClient
from src.data.fred import FREDProvider
from src.data.stooq import StooqDataProvider
from src.data.cme_sofr import CMESOFRProvider
from src.nlp.scrapers.fed import FedStatementScraper
from src.monitoring.logging_config import setup_logging


logger = logging.getLogger(__name__)


class IngestionScheduler:
    def __init__(self, db_engine, vault):
        self.engine = db_engine
        self.vault = vault
    
    async def run(self):
        while True:
            now = datetime.utcnow()
            
            try:
                await self._run_daily_jobs()
            except Exception as e:
                logger.exception(f"Daily job error: {e}")
            
            try:
                if now.hour == 22 and now.minute < 10:
                    await self._run_cb_scrape()
            except Exception as e:
                logger.exception(f"CB scrape error: {e}")
            
            try:
                if now.weekday() == 4 and now.hour == 23:  # Friday 23:00 UTC
                    await self._run_cot_ingest()
            except Exception as e:
                logger.exception(f"COT ingest error: {e}")
            
            await asyncio.sleep(600)  # 10-min loop
    
    async def _run_daily_jobs(self):
        logger.info("Running daily ingestion")
        # FRED, Stooq, CME SOFR
        pass
    
    async def _run_cb_scrape(self):
        logger.info("Running CB scrape")
        scraper = FedStatementScraper(raw_dir='/opt/fx-system/data/raw/cb')
        scraper.run(since=datetime.utcnow() - timedelta(days=30))
    
    async def _run_cot_ingest(self):
        logger.info("Running COT ingestion")
        # CFTC TFF parsing


async def main():
    setup_logging()
    vault = VaultClient()
    db_url = (f"postgresql://fx:{vault.get('POSTGRES_FX_PASSWORD')}"
              f"@localhost:5432/fx")
    engine = create_engine(db_url)
    
    scheduler = IngestionScheduler(engine, vault)
    await scheduler.run()


if __name__ == '__main__':
    asyncio.run(main())
```

## Watchdog

**Location:** `src/ops/watchdog.py`
**Purpose:** Independent process that monitors the live engine. Restarts on failure, sends alerts.

```python
import asyncio
import logging
import time
from datetime import datetime, timedelta
import httpx
import subprocess

from src.security.vault_client import VaultClient


logger = logging.getLogger(__name__)


class Watchdog:
    HEARTBEAT_TIMEOUT_SEC = 60
    METRICS_URL = 'http://localhost:8000/metrics'
    
    def __init__(self, vault):
        self.vault = vault
        self.last_alert_time = {}
    
    async def run(self):
        while True:
            try:
                await self._check_heartbeat()
                await self._check_systemd_services()
                await self._check_disk_space()
            except Exception as e:
                logger.exception(f"Watchdog error: {e}")
            await asyncio.sleep(30)
    
    async def _check_heartbeat(self):
        try:
            resp = httpx.get(self.METRICS_URL, timeout=5)
            metrics_text = resp.text
            
            for line in metrics_text.split('\n'):
                if line.startswith('fx_heartbeat_seconds '):
                    last_hb = float(line.split()[1])
                    age = time.time() - last_hb
                    if age > self.HEARTBEAT_TIMEOUT_SEC:
                        await self._alert('heartbeat_stale', 
                                          f'Engine heartbeat is {age:.0f}s old')
                        self._restart_service('fx-live-engine')
                    return
            
            await self._alert('heartbeat_missing', 
                              'No heartbeat metric in /metrics')
        except Exception as e:
            await self._alert('metrics_unreachable', f'Cannot reach metrics: {e}')
    
    async def _check_systemd_services(self):
        services = ['fx-live-engine', 'fx-ingestion', 'fx-vault-agent', 'postgresql']
        for svc in services:
            result = subprocess.run(
                ['systemctl', 'is-active', svc],
                capture_output=True, text=True
            )
            if result.stdout.strip() != 'active':
                await self._alert(f'service_down_{svc}', 
                                  f'Service {svc} is not active')
    
    async def _check_disk_space(self):
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if line.endswith(' /'):
                use_pct = int(line.split()[4].rstrip('%'))
                if use_pct > 90:
                    await self._alert('disk_space_low', 
                                      f'Disk usage at {use_pct}%')
    
    def _restart_service(self, service: str):
        logger.warning(f"Restarting {service}")
        subprocess.run(['systemctl', 'restart', service])
    
    async def _alert(self, key: str, message: str):
        now = datetime.utcnow()
        last = self.last_alert_time.get(key)
        if last and (now - last) < timedelta(minutes=15):
            return  # Rate limit
        
        self.last_alert_time[key] = now
        logger.error(f"ALERT [{key}]: {message}")
        
        # Send via Pushover or Telegram
        try:
            httpx.post(
                'https://api.pushover.net/1/messages.json',
                data={
                    'token': self.vault.get('PUSHOVER_API_TOKEN'),
                    'user': self.vault.get('PUSHOVER_USER_KEY'),
                    'title': f'FX System: {key}',
                    'message': message,
                    'priority': 1,
                },
                timeout=10,
            )
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")


async def main():
    vault = VaultClient()
    watchdog = Watchdog(vault)
    await watchdog.run()


if __name__ == '__main__':
    asyncio.run(main())
```
