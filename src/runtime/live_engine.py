"""Live trading engine — asyncio event loop for price stream, signal gen, reconciliation."""

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from src.execution.oms import OrderIntent
from src.monitoring.logging_setup import LogContext
from src.monitoring.metrics import HeartbeatTracker, start_metrics_server
from src.risk.risk_context import RiskContextBuilder

logger = logging.getLogger(__name__)


# Daily rebalance check interval (seconds). The coordinator's own
# _REBALANCE_MIN_INTERVAL_DAYS gating prevents over-frequent risk-parity refits;
# this loop just gives the coordinator a daily opportunity to decide.
_REBALANCE_TASK_INTERVAL_SEC: int = 86_400

# Broker-vs-internal alignment check interval (CL-i4tx). Same 300s cadence
# the old OrderManager.reconcile() stub ran at, now doing a real comparison.
_ALIGNMENT_CHECK_INTERVAL_SEC: int = 300

# A mismatch must persist across this many consecutive alignment checks
# (i.e. >= ~5 minutes) before the reconciliation_failure kill switch sees
# position_mismatch=True — fill latency between an OMS submit and the
# strategy book update must not halt the engine.
_ALIGNMENT_MISMATCH_STREAK_TO_FLAG: int = 2


class LiveEngine:
    def __init__(
        self,
        strategies: list[Any],
        oms: Any,
        broker: Any,
        coordinator: Any | None = None,
        cold_start_reconciler: Any | None = None,
        kill_switch_manager: Any | None = None,
        risk_context_builder: RiskContextBuilder | None = None,
    ) -> None:
        self.strategies = strategies
        self.oms = oms
        self.broker = broker
        self.coordinator = coordinator
        self.cold_start_reconciler = cold_start_reconciler
        # CL-ep0c: KillSwitchManager, evaluated on the health tick. None
        # preserves pre-CL-ep0c behavior (no automated kill switches).
        self.kill_switch_manager = kill_switch_manager
        self.running = False
        self._last_prices: dict[str, dict[str, Any]] = {}
        self._last_signal_times: dict[str, datetime] = {}
        self._last_reconciliation_report: Any | None = None
        # CL-i4tx periodic alignment state: None = unknown (no reconciler /
        # no report yet); True only after a mismatch persists across
        # _ALIGNMENT_MISMATCH_STREAK_TO_FLAG consecutive checks.
        self._position_mismatch: bool | None = None
        self._alignment_mismatch_streak = 0
        # CL-i4tx: RiskContextBuilder assembles the REAL kill-switch context
        # each health tick (daily PnL, portfolio DD, VIX/CVIX, price age,
        # position mismatch) — before this only {"equity"} was passed and
        # most switches could never fire. Injectable for tests; constructed
        # here by default so it can share the engine's live _last_prices
        # reference and trading-window predicate. Construction failure (e.g.
        # corrupt state file) raises — fail loud, same posture as the
        # trailing stop's state load.
        self.risk_context_builder: RiskContextBuilder | None = risk_context_builder
        if self.risk_context_builder is None and kill_switch_manager is not None:
            self.risk_context_builder = RiskContextBuilder(
                data_provider=getattr(kill_switch_manager, "data_provider", None),
                last_prices=self._last_prices,
                position_mismatch=(
                    self._current_position_mismatch
                    if cold_start_reconciler is not None else None
                ),
                in_trading_window=self._in_trading_window,
            )
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
        # CL-oluv: port env-overridable for containerized deploys; default
        # matches the historical hard-coded value so native runs are unchanged.
        start_metrics_server(port=int(os.environ.get("CURLIT_METRICS_PORT", "8099")))
        self._heartbeat = HeartbeatTracker("live_engine", interval_sec=30)
        self._heartbeat.start()
        logger.info("Live engine starting")

        # Cold-start reconciliation runs BEFORE any signal generation so the
        # engine starts with a clean broker-vs-internal alignment. It does
        # sync broker HTTP on the loop thread, which is fine ONLY here:
        # no engine tasks exist yet (they are created below), so there is
        # nothing to starve (CL-8lv6 sync-I/O audit).
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

        # CL-i4tx boot-time honesty: one ARMED/UNARMED line per kill switch
        # so operators know which brakes are actually connected.
        if self.kill_switch_manager is not None:
            provided = (
                self.risk_context_builder.provided_keys()
                if self.risk_context_builder is not None else {"equity"}
            )
            self.kill_switch_manager.log_arming(provided)

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
        # CL-vj74: consume the OANDA transaction stream for real, low-latency,
        # per-order fill confirmation — only when the broker exposes it (OANDA;
        # the paper broker fills synchronously and has no stream).
        if hasattr(self.broker, "stream_transactions") and hasattr(
            self.oms, "on_fill",
        ):
            coros.append(("transaction_stream", self._transaction_stream_task()))
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
        # CL-8lv6: pass the kill-switch manager so this re-call can't
        # clobber the /api/system/resume re-arm wiring; TypeError fallback
        # covers a set_runtime that predates the parameter.
        try:
            set_runtime(
                self.broker, self.oms, self.strategies,
                kill_switch_manager=self.kill_switch_manager,
            )
        except TypeError:
            set_runtime(self.broker, self.oms, self.strategies)
        # CL-oluv: host/port env-overridable. Defaults preserve native
        # behavior (loopback-only on 8200). Containers set
        # CURLIT_API_HOST=0.0.0.0 so the published port is reachable.
        host = os.environ.get("CURLIT_API_HOST", "127.0.0.1")
        port = int(os.environ.get("CURLIT_API_PORT", "8200"))
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
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

    async def _transaction_stream_task(self) -> None:
        """Drive OMS fill events from the OANDA transaction stream (CL-vj74):
        each ORDER_FILL becomes a real ORDER_FILLED journal row and clears the
        matching PENDING intent in ms, instead of waiting on the ~300s position
        poll. The stream self-heals internally (reconnect + backoff); a
        permanent 4xx propagates out and ends the task (bad creds can't self-
        fix). The poll-based confirmation stays as the backstop."""
        while self.running:
            try:
                async for fill in self.broker.stream_transactions():
                    try:
                        self.oms.on_fill(fill)
                    except Exception:
                        logger.exception("on_fill failed for %r", fill)
            except Exception:
                logger.exception("Transaction stream error")
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
                        # Strategy generate_intents implementations are
                        # async-signature but their bodies do synchronous
                        # DB/broker I/O and never await, so awaiting them
                        # directly blocks the event loop for the whole tick
                        # (CL-8lv6; extends the CL-xdnh offload to the
                        # strategy tick). Run each on a worker thread via a
                        # private event loop. Strategies are invoked
                        # sequentially here and are never called
                        # concurrently with themselves, so strategy state
                        # needs no locking. NOTE: offloaded code must not
                        # touch the engine's running loop; thread-local
                        # LogContext (strategy_id) does not propagate into
                        # the worker thread — same tradeoff as CL-xdnh.
                        # The prices snapshot is taken on the loop thread so
                        # the strategy sees a stable dict even while the
                        # price-stream task keeps mutating _last_prices.
                        intents = await asyncio.to_thread(
                            asyncio.run,
                            strategy.generate_intents(
                                dict(self._last_prices), self.broker,
                            ),
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
                    # submit_intent does sync broker HTTP + retry sleeps —
                    # keep it off the event loop (CL-xdnh).
                    await self.oms.submit_intent_async(intent)

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
        """Periodic broker-vs-internal alignment check (CL-i4tx).

        Replaces the deleted ``OrderManager.reconcile()`` stub (which
        CRITICALed on EVERY open position every 300s without comparing any
        internal book). Delegates to the real reconciler's classification-
        only ``check_alignment()`` and feeds the result to the
        ``reconciliation_failure`` kill switch via the context builder.
        """
        if self.cold_start_reconciler is None:
            logger.warning(
                "No reconciler wired — periodic alignment checks disabled; "
                "the reconciliation_failure kill switch has NO FEED and "
                "will never fire (CL-r8gv)",
            )
            return
        while self.running:
            await asyncio.sleep(_ALIGNMENT_CHECK_INTERVAL_SEC)
            try:
                # check_alignment does sync broker HTTP (CL-xdnh).
                report = await asyncio.to_thread(
                    self.cold_start_reconciler.check_alignment,
                )
            except Exception:
                logger.exception("Alignment check error")
                continue
            self._record_alignment_report(report)

    def _record_alignment_report(self, report: Any | None) -> None:
        """Fold one alignment report into the mismatch streak/flag.

        None (broker unreachable) leaves the current state untouched —
        alignment is UNKNOWN, not mismatched.
        """
        if report is None:
            return
        if report.has_mismatches:
            self._alignment_mismatch_streak += 1
            mismatched = [
                e.to_dict() for e in report.entries
                if e.status.value != "matched"
            ]
            logger.warning(
                "Alignment check: %d/%d entries mismatched (streak=%d): %s",
                len(mismatched), len(report.entries),
                self._alignment_mismatch_streak, mismatched,
            )
        else:
            self._alignment_mismatch_streak = 0
        self._position_mismatch = (
            self._alignment_mismatch_streak >= _ALIGNMENT_MISMATCH_STREAK_TO_FLAG
        )

    def _current_position_mismatch(self) -> bool | None:
        """Supplier for RiskContextBuilder — see _record_alignment_report."""
        return self._position_mismatch

    async def _health_check_task(self) -> None:
        while self.running:
            await asyncio.sleep(60)
            try:
                # _health_tick does sync broker HTTP and kill-switch
                # evaluation (which can submit OMS intents) — keep the
                # event loop free (CL-xdnh).
                await asyncio.to_thread(self._health_tick)
            except Exception:
                logger.exception("Health check error")

    def _health_tick(self) -> None:
        """One health-check evaluation (sync; extracted for testability).

        CL-i4tx: the kill-switch context is now assembled by the
        RiskContextBuilder (daily PnL, portfolio DD, VIX/CVIX, price-stream
        age, position mismatch) instead of the old equity-only dict, and
        ``reset_daily`` re-arms the once-per-day trigger dedup at UTC-day
        rollover so a fired switch can fire again tomorrow after
        ``/api/system/resume``.
        """
        account = self.broker.get_account()
        logger.debug("Health: equity=%.2f", account.equity)
        if self.kill_switch_manager is None:
            return
        if self.risk_context_builder is not None:
            context = self.risk_context_builder.build(float(account.equity))
            if self.risk_context_builder.consume_day_rollover():
                # CL-ssoh (P1): automatic rollover re-arms the daily trigger
                # dedup but must NOT clear active halt causes while the OMS is
                # still halted — that would strand it (auto-resume would read
                # the empty cause set as a manual halt and never lift it). The
                # manual /api/system/resume path keeps the default (clears).
                self.kill_switch_manager.reset_daily(clear_causes=False)
        else:
            context = {"equity": account.equity}
        self.kill_switch_manager.check(context)
        # CL-nxjx: after evaluating, auto-lift a halt whose ONLY cause was a
        # data-availability gate (stale_prices) that has since cleared — so
        # a network blip / laptop-wake recovers on its own instead of
        # staying halted until a manual restart. Risk halts stay sticky.
        if hasattr(self.kill_switch_manager, "attempt_auto_resume"):
            self.kill_switch_manager.attempt_auto_resume(context)

    @staticmethod
    def _in_trading_window(ts: datetime) -> bool:
        wd = ts.weekday()
        h = ts.hour
        if wd == 5:
            return False
        if wd == 6 and h < 22:
            return False
        return not (wd == 4 and h >= 22)

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
        # halt_new_trades takes the OMS lock, which an offloaded
        # submit_intent may hold across broker HTTP + retry sleeps —
        # keep the wait off the event loop (CL-8lv6, extends CL-xdnh).
        await asyncio.to_thread(self.oms.halt_new_trades)
        # Wait briefly for OMS to drain in-flight orders.
        start = time.time()
        while self.oms.has_pending() and (time.time() - start) < timeout:
            await asyncio.sleep(0.5)
        # Cancel tasks so run()'s gather can return.
        for task in self._tasks:
            if not task.done():
                task.cancel()
        logger.info("Shutdown complete")
