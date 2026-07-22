"""Unit tests — live_engine routing through PortfolioCoordinator (D7 / CL-amf)."""

from __future__ import annotations

import asyncio
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
            strategies=[], oms=_RecordingOMS(), broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        assert engine.coordinator is coord

    def test_coordinator_optional(self) -> None:
        engine = LiveEngine(
            strategies=[], oms=_RecordingOMS(), broker=None,  # type: ignore[arg-type]
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
            strategies=[], oms=oms, broker=None,  # type: ignore[arg-type]
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
            strategies=[], oms=oms, broker=None,  # type: ignore[arg-type]
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
            strategies=[], oms=oms, broker=None,  # type: ignore[arg-type]
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
            strategies=[], oms=oms, broker=None,  # type: ignore[arg-type]
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
            strategies=[], oms=_RecordingOMS(), broker=None,  # type: ignore[arg-type]
            coordinator=coord,
        )
        asyncio.run(engine._dispatch_intents({}))
        # Coordinator receives empty dict (it handles gracefully).
        assert coord.calls == [{}]


# =============================================================================
# _rebalance_task
# =============================================================================


class TestRebalanceTask:
    def test_returns_immediately_when_no_coordinator(self) -> None:
        engine = LiveEngine(
            strategies=[], oms=_RecordingOMS(), broker=None,  # type: ignore[arg-type]
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

    def reset_daily(self) -> None:
        self.resets += 1

    def log_arming(self, provided: Any) -> None:
        self.armed_with = set(provided)


def _entry(symbol: str, status: ReconciliationStatus) -> ReconciliationEntry:
    return ReconciliationEntry(
        symbol=symbol, broker_quantity=1.0, internal_quantity=0.0,
        contributing_strategies=[], status=status,
    )


class TestHealthTickKillSwitchWiring:
    def _engine(self, clock: _FakeClock) -> tuple[LiveEngine, _RecordingKSM]:
        ksm = _RecordingKSM()
        builder = RiskContextBuilder(state_path=None, clock=clock)
        engine = LiveEngine(
            strategies=[], oms=_RecordingOMS(), broker=_HealthBroker(100_000.0),
            kill_switch_manager=ksm, risk_context_builder=builder,
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
            strategies=[], oms=_RecordingOMS(), broker=_HealthBroker(1.0),
        )
        assert engine.risk_context_builder is None
        engine._health_tick()  # must not raise


class TestAlignmentStreak:
    def _engine(self) -> LiveEngine:
        return LiveEngine(
            strategies=[], oms=_RecordingOMS(), broker=None,  # type: ignore[arg-type]
        )

    @staticmethod
    def _mismatch_report() -> ReconciliationReport:
        return ReconciliationReport(entries=[
            _entry("EURUSD", ReconciliationStatus.ORPHANED_BROKER),
        ])

    @staticmethod
    def _clean_report() -> ReconciliationReport:
        return ReconciliationReport(entries=[
            _entry("EURUSD", ReconciliationStatus.MATCHED),
        ])

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
