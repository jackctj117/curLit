"""Live trading engine — asyncio event loop for price stream, signal gen, reconciliation."""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from src.execution.oms import OrderIntent
from src.monitoring.logging_setup import LogContext
from src.monitoring.metrics import HeartbeatTracker, start_metrics_server

logger = logging.getLogger(__name__)


# Daily rebalance check interval (seconds). The coordinator's own
# _REBALANCE_MIN_INTERVAL_DAYS gating prevents over-frequent risk-parity refits;
# this loop just gives the coordinator a daily opportunity to decide.
_REBALANCE_TASK_INTERVAL_SEC: int = 86_400


class LiveEngine:
    def __init__(
        self,
        strategies: list[Any],
        oms: Any,
        broker: Any,
        coordinator: Any | None = None,
        cold_start_reconciler: Any | None = None,
    ) -> None:
        self.strategies = strategies
        self.oms = oms
        self.broker = broker
        self.coordinator = coordinator
        self.cold_start_reconciler = cold_start_reconciler
        self.running = False
        self._last_prices: dict[str, dict[str, Any]] = {}
        self._last_signal_times: dict[str, datetime] = {}
        self._last_reconciliation_report: Any | None = None
        # Tracked async tasks — populated in run(), used by graceful_shutdown()
        # to cancel each so the run() gather can return and the process exit.
        self._tasks: list[asyncio.Task[Any]] = []

        if coordinator is None:
            logger.warning(
                "LiveEngine started without PortfolioCoordinator — strategy intents "
                "will route directly to OMS (legacy mode). Wire a coordinator for "
                "portfolio-level scaling, aggregation, and constraints.",
            )
        else:
            logger.info(
                "LiveEngine using PortfolioCoordinator with %d strategies",
                len(strategies),
            )

    async def run(self) -> None:
        self.running = True
        start_metrics_server(port=8099)
        self._heartbeat = HeartbeatTracker("live_engine", interval_sec=30)
        self._heartbeat.start()
        logger.info("Live engine starting")

        # Cold-start reconciliation runs BEFORE any signal generation so the
        # engine starts with a clean broker-vs-internal alignment.
        if self.cold_start_reconciler is not None:
            try:
                self._last_reconciliation_report = self.cold_start_reconciler.reconcile()
                logger.info(
                    "Cold-start reconciliation: %d entries, mismatches=%s",
                    len(self._last_reconciliation_report.entries),
                    self._last_reconciliation_report.has_mismatches,
                )
            except Exception:
                logger.exception(
                    "Cold-start reconciliation failed — continuing with engine startup"
                )

        # Track tasks as asyncio.Task so graceful_shutdown() can cancel them.
        # Without this, any task that doesn't poll self.running between
        # awaits (uvicorn server, the broker price-stream async-for) keeps
        # the gather alive forever and the process can never exit on SIGTERM.
        coros = [
            ("price_stream", self._price_stream_task()),
            ("signal_gen", self._signal_generation_task()),
            ("reconciliation", self._reconciliation_task()),
            ("health_check", self._health_check_task()),
            ("web_server", self._web_server_task()),
        ]
        if self.coordinator is not None:
            coros.append(("rebalance", self._rebalance_task()))
        self._tasks = [
            asyncio.create_task(coro, name=name) for name, coro in coros
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Live engine tasks cancelled — exiting")

    async def _web_server_task(self) -> None:
        import uvicorn
        from src.web.api import app, set_runtime
        set_runtime(self.broker, self.oms, self.strategies)
        config = uvicorn.Config(app, host="127.0.0.1", port=8200, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

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
            now = datetime.now(UTC)
            if not self._in_trading_window(now):
                await asyncio.sleep(60)
                continue

            intents_by_strategy: dict[str, list[OrderIntent]] = {}
            for strategy in self.strategies:
                with LogContext(strategy_id=strategy.id):
                    try:
                        last = self._last_signal_times.get(
                            strategy.id, datetime.min.replace(tzinfo=UTC),
                        )
                        interval = getattr(strategy, "signal_interval_seconds", 60)
                        if (now - last).total_seconds() < interval:
                            continue
                        intents = await strategy.generate_intents(
                            self._last_prices, self.broker,
                        )
                        if intents:
                            intents_by_strategy[strategy.id] = list(intents)
                        self._last_signal_times[strategy.id] = now
                    except Exception:
                        logger.exception("Strategy %s error", strategy.id)

            if intents_by_strategy:
                await self._dispatch_intents(intents_by_strategy)

            await asyncio.sleep(10)

    async def _dispatch_intents(
        self,
        intents_by_strategy: dict[str, list[OrderIntent]],
    ) -> None:
        """Forward gathered intents either through the coordinator or directly to OMS.

        Coordinator path is preferred — it scales by allocation, applies portfolio
        constraints, and aggregates by symbol. Legacy direct-OMS path is used only
        when no coordinator was wired (warning emitted at startup).
        """
        if self.coordinator is not None:
            try:
                await self.coordinator.process_intents(intents_by_strategy)
            except Exception:
                logger.exception(
                    "Coordinator process_intents failed; intents dropped this tick"
                )
            return

        # Legacy fallback: submit each intent directly. Preserved so the engine
        # can run while operators wire the coordinator (D7 migration safety net).
        for sid, intents in intents_by_strategy.items():
            with LogContext(strategy_id=sid):
                for intent in intents:
                    self.oms.submit_intent(intent)

    async def _rebalance_task(self) -> None:
        """Daily rebalance trigger.

        The coordinator's own minimum-interval gating handles whether a real refit
        runs; this loop just ticks the opportunity once per day so we don't miss
        a rebalance window.
        """
        if self.coordinator is None:
            return
        while self.running:
            await asyncio.sleep(_REBALANCE_TASK_INTERVAL_SEC)
            try:
                await self.coordinator.rebalance_allocations()
            except Exception:
                logger.exception("Rebalance task error")

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
        """Halt new trades, drain pending OMS work, then cancel all tasks.

        Idempotent — repeat calls are no-ops once shutdown has run. Cancels
        the tracked tasks so run()'s asyncio.gather raises CancelledError and
        the awaiting caller (run_engine.run_engine) can return cleanly.
        """
        if not self.running:
            return  # already shut down
        logger.info("Graceful shutdown")
        self.running = False
        self.oms.halt_new_trades()
        # Wait briefly for OMS to drain in-flight orders.
        start = time.time()
        while self.oms.has_pending() and (time.time() - start) < timeout:
            await asyncio.sleep(0.5)
        # Cancel tasks so run()'s gather can return.
        for task in self._tasks:
            if not task.done():
                task.cancel()
        logger.info("Shutdown complete")
