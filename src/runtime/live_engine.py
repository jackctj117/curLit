"""Live trading engine — asyncio event loop for price stream, signal gen, reconciliation."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from src.monitoring.logging_setup import LogContext
from src.monitoring.metrics import start_metrics_server

logger = logging.getLogger(__name__)


class LiveEngine:
    def __init__(self, strategies: list, oms, broker) -> None:
        self.strategies = strategies
        self.oms = oms
        self.broker = broker
        self.running = False
        self._last_prices: dict[str, dict] = {}
        self._last_signal_times: dict[str, datetime] = {}

    async def run(self) -> None:
        self.running = True
        start_metrics_server(port=8090)
        logger.info("Live engine starting")
        await asyncio.gather(
            self._price_stream_task(),
            self._signal_generation_task(),
            self._reconciliation_task(),
            self._health_check_task(),
        )

    async def _price_stream_task(self) -> None:
        symbols = list(set(s for s in self.strategies for s in s.symbols))
        while self.running:
            try:
                async for tick in self.broker.stream_prices(symbols):
                    self._last_prices[tick["symbol"]] = tick
            except Exception:
                logger.exception("Price stream error")
                await asyncio.sleep(5)

    async def _signal_generation_task(self) -> None:
        while self.running:
            now = datetime.now(timezone.utc)
            if not self._in_trading_window(now):
                await asyncio.sleep(60)
                continue
            for strategy in self.strategies:
                with LogContext(strategy_id=strategy.id):
                    try:
                        last = self._last_signal_times.get(strategy.id, datetime.min.replace(tzinfo=timezone.utc))
                        interval = getattr(strategy, "signal_interval_seconds", 60)
                        if (now - last).total_seconds() < interval:
                            continue
                        intents = await strategy.generate_intents(self._last_prices, self.broker)
                        for intent in intents:
                            self.oms.submit_intent(intent)
                        self._last_signal_times[strategy.id] = now
                    except Exception:
                        logger.exception("Strategy %s error", strategy.id)
            await asyncio.sleep(10)

    async def _reconciliation_task(self) -> None:
        while self.running:
            await asyncio.sleep(300)
            try:
                self.oms.reconcile()
            except Exception:
                logger.exception("Reconciliation error")

    async def _health_check_task(self) -> None:
        while self.running:
            await asyncio.sleep(60)
            try:
                account = self.broker.get_account()
                logger.debug("Health: equity=%.2f", account.equity)
            except Exception:
                logger.exception("Health check error")

    @staticmethod
    def _in_trading_window(ts: datetime) -> bool:
        wd = ts.weekday()
        h = ts.hour
        if wd == 5: return False
        if wd == 6 and h < 22: return False
        if wd == 4 and h >= 22: return False
        return True

    async def graceful_shutdown(self, timeout: int = 30) -> None:
        logger.info("Graceful shutdown")
        self.running = False
        self.oms.halt_new_trades()
        start = time.time()
        while self.oms.has_pending() and (time.time() - start) < timeout:
            await asyncio.sleep(0.5)
        logger.info("Shutdown complete")
