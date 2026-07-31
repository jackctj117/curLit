"""Unit tests — live_engine routing through PortfolioCoordinator (D7 / CL-amf)."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

from src.execution.oms import OrderIntent
from src.runtime.live_engine import LiveEngine


class _RecordingOMS:
    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    def submit_intent(self, intent: OrderIntent) -> str:
        self.submitted.append(intent)
        return intent.intent_id

    async def submit_intent_async(self, intent: OrderIntent) -> str:
        # Mirrors OrderManager's event-loop-safe wrapper (CL-xdnh).
        return self.submit_intent(intent)

    def halt_new_trades(self) -> None:
        pass


class _RecordingCoordinator:
    """Minimal PortfolioCoordinator double — captures process_intents calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, list[OrderIntent]]] = []
        self.rebalance_calls: int = 0

    async def process_intents(
        self,
        raw_intents: dict[str, list[OrderIntent]],
    ) -> dict[str, dict[str, Any]]:
        self.calls.append({k: list(v) for k, v in raw_intents.items()})
        return {}

    async def rebalance_allocations(self, force: bool = False) -> None:
        self.rebalance_calls += 1


class _RaisingCoordinator:
    """Coordinator that raises on process_intents — used to verify isolation."""

    async def process_intents(
        self,
        raw_intents: dict[str, list[OrderIntent]],
    ) -> dict[str, dict[str, Any]]:
        raise RuntimeError("simulated coordinator failure")

    async def rebalance_allocations(self, force: bool = False) -> None:
        pass


# =============================================================================
# Construction
# =============================================================================


class TestLiveEngineConstruction:
    def test_accepts_coordinator(self) -> None:
        coord = _RecordingCoordinator()
        engine = LiveEngine(
            strategies=[],
            oms=_RecordingOMS(),
            broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        assert engine.coordinator is coord

    def test_coordinator_optional(self) -> None:
        engine = LiveEngine(
            strategies=[],
            oms=_RecordingOMS(),
            broker=None,  # type: ignore[arg-type]
        )
        assert engine.coordinator is None


# =============================================================================
# _dispatch_intents — coordinator path
# =============================================================================


class TestDispatchIntents:
    def test_coordinator_path_calls_process_intents(self) -> None:
        coord = _RecordingCoordinator()
        oms = _RecordingOMS()
        engine = LiveEngine(
            strategies=[],
            oms=oms,
            broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        intents = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100)],
            "s2": [OrderIntent(strategy_id="s2", symbol="USDJPY", target_position=200)],
        }
        asyncio.run(engine._dispatch_intents(intents))
        assert len(coord.calls) == 1
        # OMS bypassed when coordinator is present.
        assert oms.submitted == []

    def test_coordinator_path_does_not_double_submit_to_oms(self) -> None:
        coord = _RecordingCoordinator()
        oms = _RecordingOMS()
        engine = LiveEngine(
            strategies=[],
            oms=oms,
            broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        intents = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100)],
        }
        asyncio.run(engine._dispatch_intents(intents))
        # Coordinator received once; OMS received 0 (coordinator owns OMS interaction).
        assert len(coord.calls) == 1
        assert oms.submitted == []

    def test_legacy_path_routes_to_oms_directly(self) -> None:
        oms = _RecordingOMS()
        engine = LiveEngine(
            strategies=[],
            oms=oms,
            broker=None,  # type: ignore[arg-type]
            coordinator=None,
        )
        intents = {
            "s1": [
                OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100),
                OrderIntent(strategy_id="s1", symbol="USDJPY", target_position=200),
            ],
            "s2": [
                OrderIntent(strategy_id="s2", symbol="GBPUSD", target_position=300),
            ],
        }
        asyncio.run(engine._dispatch_intents(intents))
        # All 3 intents flow to OMS.
        assert len(oms.submitted) == 3
        symbols = {i.symbol for i in oms.submitted}
        assert symbols == {"EURUSD", "USDJPY", "GBPUSD"}

    def test_coordinator_failure_is_isolated(self) -> None:
        """A coordinator exception must not crash the engine; intents are dropped this tick."""
        coord = _RaisingCoordinator()
        oms = _RecordingOMS()
        engine = LiveEngine(
            strategies=[],
            oms=oms,
            broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        intents = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100)],
        }
        # No exception expected to propagate.
        asyncio.run(engine._dispatch_intents(intents))
        # When coordinator fails, we do NOT fall back to direct OMS — the
        # tick is silently dropped to avoid bypassing portfolio constraints.
        assert oms.submitted == []

    def test_empty_intents_noop(self) -> None:
        coord = _RecordingCoordinator()
        engine = LiveEngine(
            strategies=[],
            oms=_RecordingOMS(),
            broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        asyncio.run(engine._dispatch_intents({}))
        # Coordinator receives empty dict (it handles gracefully).
        assert coord.calls == [{}]


# =============================================================================
# _signal_generation_task — strategy I/O off the event loop (CL-8lv6)
# =============================================================================


class _ThreadRecordingStrategy:
    """Strategy double whose generate_intents records its executing thread
    (same idiom as test_portfolio_coordinator.TestBrokerIoOffEventLoop)."""

    def __init__(self) -> None:
        self.id = "s1"
        self.symbols = ["EURUSD"]
        self.signal_interval_seconds = 0
        self.call_threads: list[int] = []
        self.saw_live_prices_dict: list[bool] = []
        self._engine: LiveEngine | None = None  # set by the test

    async def generate_intents(
        self,
        prices: dict[str, Any],
        broker: Any,
    ) -> list[OrderIntent]:
        self.call_threads.append(threading.get_ident())
        # The engine must hand strategies a loop-thread snapshot, not the
        # live _last_prices dict the price-stream task keeps mutating.
        assert self._engine is not None
        self.saw_live_prices_dict.append(prices is self._engine._last_prices)
        return [
            OrderIntent(
                strategy_id=self.id,
                symbol="EURUSD",
                target_position=100.0,
            )
        ]


def _always_in_window(ts: Any) -> bool:
    return True


class TestStrategyIoOffEventLoop:
    def test_generate_intents_never_runs_on_loop_thread(self) -> None:
        """Strategy generate_intents bodies do synchronous DB/broker I/O
        (they are async-signature but never await) — the engine must run
        each on a worker thread so the tick can't freeze the price stream,
        health ticks, and kill switches (CL-8lv6, extends CL-xdnh)."""
        strategy = _ThreadRecordingStrategy()
        coord = _RecordingCoordinator()
        engine = LiveEngine(
            strategies=[strategy],
            oms=_RecordingOMS(),
            broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        strategy._engine = engine
        engine.running = True
        # Deterministic regardless of when the test runs (weekend gate).
        engine._in_trading_window = _always_in_window  # type: ignore[method-assign]

        loop_thread: list[int] = []

        async def run() -> None:
            loop_thread.append(threading.get_ident())
            task = asyncio.create_task(engine._signal_generation_task())
            try:
                for _ in range(500):  # up to ~5s — normally a few ms
                    if coord.calls:
                        break
                    await asyncio.sleep(0.01)
            finally:
                engine.running = False
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        asyncio.run(run())

        assert strategy.call_threads, "generate_intents was never called"
        assert all(t != loop_thread[0] for t in strategy.call_threads), (
            "strategy generate_intents ran on the event-loop thread"
        )
        # The intents must still flow through to the coordinator.
        assert coord.calls and "s1" in coord.calls[0]
        assert coord.calls[0]["s1"][0].symbol == "EURUSD"
        # And the strategy saw a snapshot, not the live shared dict.
        assert strategy.saw_live_prices_dict == [False]


# =============================================================================
# _rebalance_task
# =============================================================================


class TestRebalanceTask:
    def test_returns_immediately_when_no_coordinator(self) -> None:
        engine = LiveEngine(
            strategies=[],
            oms=_RecordingOMS(),
            broker=None,  # type: ignore[arg-type]
            coordinator=None,
        )
        # Should return without blocking even though running is False.
        engine.running = False
        asyncio.run(engine._rebalance_task())
        # Nothing to assert beyond not hanging.


# =============================================================================
# CL-i4tx — health tick kill-switch wiring + periodic alignment streak
# =============================================================================


from datetime import UTC, datetime, timedelta  # noqa: E402

from src.portfolio.reconciler import (  # noqa: E402
    ReconciliationEntry,
    ReconciliationReport,
    ReconciliationStatus,
)
from src.risk.risk_context import RiskContextBuilder  # noqa: E402

_T0 = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _FakeClock:
    def __init__(self, now: datetime = _T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _HealthBroker:
    def __init__(self, equity: float) -> None:
        self.equity = equity

    def get_account(self) -> Any:
        class _Account:
            pass

        account = _Account()
        account.equity = self.equity
        return account


class _RecordingKSM:
    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []
        self.resets = 0
        self.armed_with: set[str] | None = None
        self.data_provider = None

    def check(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        self.contexts.append(context)
        return []

    def reset_daily(self, *, clear_causes: bool = True) -> None:
        # CL-ssoh: the automatic rollover path now passes clear_causes=False.
        self.resets += 1

    def log_arming(self, provided: Any) -> None:
        self.armed_with = set(provided)


def _entry(symbol: str, status: ReconciliationStatus) -> ReconciliationEntry:
    return ReconciliationEntry(
        symbol=symbol,
        broker_quantity=1.0,
        internal_quantity=0.0,
        contributing_strategies=[],
        status=status,
    )


class TestHealthTickKillSwitchWiring:
    def _engine(self, clock: _FakeClock) -> tuple[LiveEngine, _RecordingKSM]:
        ksm = _RecordingKSM()
        builder = RiskContextBuilder(state_path=None, clock=clock)
        engine = LiveEngine(
            strategies=[],
            oms=_RecordingOMS(),
            broker=_HealthBroker(100_000.0),
            kill_switch_manager=ksm,
            risk_context_builder=builder,
        )
        return engine, ksm

    def test_context_is_richer_than_equity_only(self) -> None:
        engine, ksm = self._engine(_FakeClock())
        engine._health_tick()
        ctx = ksm.contexts[-1]
        # The facade passed only {"equity"}; the real builder feeds the
        # PnL/DD switches too.
        assert ctx["equity"] == 100_000.0
        assert "daily_pnl_pct" in ctx
        assert "portfolio_dd" in ctx

    def test_reset_daily_called_on_utc_rollover(self) -> None:
        clock = _FakeClock()
        engine, ksm = self._engine(clock)
        engine._health_tick()
        assert ksm.resets == 0
        clock.now = _T0 + timedelta(days=1)
        engine._health_tick()
        assert ksm.resets == 1
        engine._health_tick()
        assert ksm.resets == 1  # once per rollover, not per tick

    def test_no_kill_switch_manager_is_noop(self) -> None:
        engine = LiveEngine(
            strategies=[],
            oms=_RecordingOMS(),
            broker=_HealthBroker(1.0),
        )
        assert engine.risk_context_builder is None
        engine._health_tick()  # must not raise


class TestAlignmentStreak:
    def _engine(self) -> LiveEngine:
        return LiveEngine(
            strategies=[],
            oms=_RecordingOMS(),
            broker=None,  # type: ignore[arg-type]
        )

    @staticmethod
    def _mismatch_report() -> ReconciliationReport:
        return ReconciliationReport(
            entries=[
                _entry("EURUSD", ReconciliationStatus.ORPHANED_BROKER),
            ]
        )

    @staticmethod
    def _clean_report() -> ReconciliationReport:
        return ReconciliationReport(
            entries=[
                _entry("EURUSD", ReconciliationStatus.MATCHED),
            ]
        )

    def test_single_mismatch_does_not_flag(self) -> None:
        engine = self._engine()
        engine._record_alignment_report(self._mismatch_report())
        # Transient (fill latency) mismatches must not halt the engine.
        assert engine._current_position_mismatch() is False

    def test_persistent_mismatch_flags_after_two_checks(self) -> None:
        engine = self._engine()
        engine._record_alignment_report(self._mismatch_report())
        engine._record_alignment_report(self._mismatch_report())
        assert engine._current_position_mismatch() is True

    def test_clean_report_resets_streak(self) -> None:
        engine = self._engine()
        engine._record_alignment_report(self._mismatch_report())
        engine._record_alignment_report(self._clean_report())
        engine._record_alignment_report(self._mismatch_report())
        assert engine._current_position_mismatch() is False

    def test_none_report_leaves_state_unknown(self) -> None:
        engine = self._engine()
        engine._record_alignment_report(None)
        # Broker unreachable -> alignment UNKNOWN, not a mismatch.
        assert engine._current_position_mismatch() is None


class TestTradingWindowDST:
    """CL-azgh: the FX week is Sun 17:00 → Fri 17:00 NEW YORK time.

    The old implementation compared UTC hours against 22:00 — right only
    under EST. Under EDT (summer) the close is 21:00 UTC, so the engine
    kept the window "open" for the hour after the real Friday close (no
    ticks can arrive → stale_prices fired and paged, 2026-07-31) and kept
    it "closed" for the first hour of the real Sunday reopen."""

    @staticmethod
    def _at(iso: str) -> bool:
        from datetime import datetime

        return LiveEngine._in_trading_window(datetime.fromisoformat(iso))

    # --- summer (EDT, UTC-4): the cases the old code got WRONG ---
    def test_summer_friday_after_close_is_shut(self) -> None:
        # 21:10 UTC = 17:10 ET — the exact 2026-07-31 false halt.
        assert self._at("2026-07-31T21:10:00+00:00") is False

    def test_summer_sunday_first_hour_is_open(self) -> None:
        # 21:10 UTC = 17:10 ET Sunday — real trading the old code skipped.
        assert self._at("2026-08-02T21:10:00+00:00") is True

    # --- summer boundary sanity ---
    def test_summer_friday_before_close_is_open(self) -> None:
        assert self._at("2026-07-31T20:59:00+00:00") is True

    def test_summer_sunday_before_open_is_shut(self) -> None:
        assert self._at("2026-08-02T20:59:00+00:00") is False

    # --- winter (EST, UTC-5): the old boundary was correct — keep it ---
    def test_winter_friday_before_close_is_open(self) -> None:
        assert self._at("2026-01-16T21:30:00+00:00") is True  # 16:30 ET Fri

    def test_winter_friday_after_close_is_shut(self) -> None:
        assert self._at("2026-01-16T22:01:00+00:00") is False  # 17:01 ET Fri

    def test_winter_sunday_reopen(self) -> None:
        assert self._at("2026-01-18T22:01:00+00:00") is True  # 17:01 ET Sun
        assert self._at("2026-01-18T21:30:00+00:00") is False  # 16:30 ET Sun

    # --- UTC/local weekday crossover + midweek ---
    def test_saturday_always_shut(self) -> None:
        assert self._at("2026-08-01T12:00:00+00:00") is False

    def test_friday_late_evening_utc_saturday(self) -> None:
        # 04:30 UTC Sat = 23:30 ET Fri — closed (post-close), and the local
        # weekday (Friday) must be the one consulted, not the UTC Saturday.
        assert self._at("2026-01-17T04:30:00+00:00") is False

    def test_midweek_open(self) -> None:
        assert self._at("2026-07-29T12:00:00+00:00") is True
