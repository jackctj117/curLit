"""Unit tests — portfolio.pretrade: validator rejection cases + coordinator wiring."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from src.data.economic_calendar import (
    BlackoutEvaluator,
    EconomicCalendar,
    EconomicEvent,
    SeverityTier,
)
from src.execution.broker import Position
from src.execution.oms import OrderIntent
from src.execution.paper_broker import PaperBroker
from src.portfolio.coordinator import (
    PortfolioConstraints,
    PortfolioCoordinator,
    StrategyAllocation,
)
from src.portfolio.pretrade import (
    PreTradeValidator,
    RejectionReason,
)

# =============================================================================
# Test fakes
# =============================================================================


class _RecordingOMS:
    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    def submit_intent(self, intent: OrderIntent, **kwargs: object) -> str:
        self.submitted.append(intent)
        return intent.intent_id

    def halt_new_trades(self) -> None:
        pass


class _FakeState:
    def record_reallocation(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_portfolio_order(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_positions_by_strategy(self, strategy_id: str) -> list[Any]:
        return []

    def load_strategy_returns_history(self, *args: Any, **kwargs: Any) -> Any:
        import pandas as pd
        return pd.DataFrame()


class _StrategyDouble:
    def __init__(self, sid: str) -> None:
        self.id = sid


class _AlwaysHaltedTradability:
    def is_tradable(self, symbol: str) -> bool:  # noqa: ARG002
        return False


class _RaisingTradability:
    def is_tradable(self, symbol: str) -> bool:  # noqa: ARG002
        raise RuntimeError("tradability check exploded")


# =============================================================================
# Builders
# =============================================================================


def _make_broker(
    capital: float = 100_000.0,
    positions: list[Position] | None = None,
    eurusd_mid: float = 1.10,
    usdjpy_mid: float = 150.0,
    gbpusd_mid: float = 1.25,
) -> PaperBroker:
    broker = PaperBroker(initial_capital=capital)
    broker.set_price("EURUSD", eurusd_mid - 0.0001, eurusd_mid + 0.0001)
    broker.set_price("USDJPY", usdjpy_mid - 0.01, usdjpy_mid + 0.01)
    broker.set_price("GBPUSD", gbpusd_mid - 0.0001, gbpusd_mid + 0.0001)
    if positions:
        for pos in positions:
            broker._positions[pos.symbol] = pos  # type: ignore[attr-defined]
    return broker


def _make_validator(
    broker: PaperBroker | None = None,
    constraints: PortfolioConstraints | None = None,
    **kwargs: Any,
) -> tuple[PreTradeValidator, PaperBroker, PortfolioConstraints]:
    broker = broker or _make_broker()
    constraints = constraints or PortfolioConstraints()
    validator = PreTradeValidator(broker, constraints, **kwargs)
    return validator, broker, constraints


# =============================================================================
# Per-rejection-class tests
# =============================================================================


class TestInstrumentSupported:
    def test_unsupported_symbol_rejected(self) -> None:
        validator, *_ = _make_validator(supported_instruments={"EURUSD", "USDJPY"})
        intent = OrderIntent(strategy_id="s1", symbol="XYZUSD", target_position=100)
        rejection = validator.validate(intent, current_positions=[])
        assert rejection is not None
        assert rejection.reason == RejectionReason.INSTRUMENT_UNSUPPORTED

    def test_supported_symbol_passes_check(self) -> None:
        validator, *_ = _make_validator(supported_instruments={"EURUSD", "USDJPY"})
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100)
        # Other checks may pass with small position; assert no INSTRUMENT_UNSUPPORTED reject.
        rejection = validator.validate(intent, current_positions=[])
        if rejection is not None:
            assert rejection.reason != RejectionReason.INSTRUMENT_UNSUPPORTED

    def test_no_whitelist_passes_all_symbols(self) -> None:
        validator, *_ = _make_validator(supported_instruments=None)
        intent = OrderIntent(strategy_id="s1", symbol="EXOTIC1", target_position=100)
        rejection = validator.validate(intent, current_positions=[])
        if rejection is not None:
            assert rejection.reason != RejectionReason.INSTRUMENT_UNSUPPORTED


class TestInstrumentTradable:
    def test_halted_symbol_rejected(self) -> None:
        validator, *_ = _make_validator(tradability=_AlwaysHaltedTradability())
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100)
        rejection = validator.validate(intent, current_positions=[])
        assert rejection is not None
        assert rejection.reason == RejectionReason.INSTRUMENT_HALTED

    def test_tradability_failure_does_not_block_other_checks(self) -> None:
        """If tradability raises, validator continues other checks (broker is final safety net)."""
        validator, *_ = _make_validator(tradability=_RaisingTradability())
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100)
        rejection = validator.validate(intent, current_positions=[])
        # No INSTRUMENT_HALTED specifically.
        if rejection is not None:
            assert rejection.reason != RejectionReason.INSTRUMENT_HALTED


class TestPerPairConcentration:
    def test_pair_over_cap_rejected(self) -> None:
        # 100k equity, 35% pair cap = 35k notional max.
        # 50k notional EURUSD = 45,454 units at 1.10 → exceeds 35k cap.
        broker = _make_broker(capital=100_000)
        validator = PreTradeValidator(broker, PortfolioConstraints())
        intent = OrderIntent(
            strategy_id="s1", symbol="EURUSD", target_position=50_000 / 1.10,
        )
        rejection = validator.validate(intent, current_positions=[])
        assert rejection is not None
        assert rejection.reason == RejectionReason.PER_PAIR_CONCENTRATION

    def test_pair_under_cap_passes_concentration(self) -> None:
        broker = _make_broker(capital=100_000)
        validator = PreTradeValidator(broker, PortfolioConstraints())
        intent = OrderIntent(
            strategy_id="s1", symbol="EURUSD", target_position=10_000 / 1.10,
        )
        rejection = validator.validate(intent, current_positions=[])
        # Could fail other checks but not concentration.
        if rejection is not None:
            assert rejection.reason != RejectionReason.PER_PAIR_CONCENTRATION


class TestMargin:
    def test_insufficient_margin_rejected(self) -> None:
        # Tiny equity → cannot cover even small trade.
        broker = _make_broker(capital=100.0)
        # Use very loose bounds so margin is the only binding constraint.
        # max_notional_per_pair_pct multiplies equity → 1000.0 means 100*1000 = 100k cap.
        constraints = PortfolioConstraints(
            max_notional_per_pair_pct=1000.0,
            max_gross_leverage=10_000.0,
            max_net_leverage=10_000.0,
            max_directional_exposure_per_currency=1000.0,
        )
        validator = PreTradeValidator(broker, constraints, margin_requirement_pct=0.05)
        # 10k notional → 500 required margin > 100 equity.
        intent = OrderIntent(
            strategy_id="s1", symbol="EURUSD", target_position=10_000 / 1.10,
        )
        rejection = validator.validate(intent, current_positions=[])
        assert rejection is not None
        assert rejection.reason == RejectionReason.INSUFFICIENT_MARGIN


class TestPostTradeLeverage:
    def test_gross_leverage_breach_rejected(self) -> None:
        # 100k equity, 3x cap = 300k gross max.
        # Existing 200k EURUSD long + new 200k USDJPY long → 400k gross > 300k cap.
        broker = _make_broker(capital=100_000)
        # Existing position: 200k notional EURUSD long = 181,818 units at 1.10.
        existing = Position(symbol="EURUSD", quantity=200_000 / 1.10, avg_price=1.10)
        broker._positions[existing.symbol] = existing  # type: ignore[attr-defined]
        # Relax per-pair so concentration doesn't bite first.
        constraints = PortfolioConstraints(max_notional_per_pair_pct=10.0)
        validator = PreTradeValidator(broker, constraints)
        # New 200k USDJPY long.
        intent = OrderIntent(
            strategy_id="s1", symbol="USDJPY", target_position=200_000 / 150.0,
        )
        rejection = validator.validate(intent)
        assert rejection is not None
        assert rejection.reason == RejectionReason.GROSS_LEVERAGE_BREACH

    def test_net_leverage_breach_rejected(self) -> None:
        # 100k equity, 2x net cap = 200k net max, 3x gross cap = 300k gross max.
        # Existing 100k EURUSD long + new 110k USDJPY long → net 210k > 200k cap;
        # gross 210k < 300k cap. So only net breaches.
        broker = _make_broker(capital=100_000)
        existing = Position(
            symbol="EURUSD", quantity=100_000 / 1.10, avg_price=1.10,
        )
        broker._positions[existing.symbol] = existing  # type: ignore[attr-defined]
        constraints = PortfolioConstraints(
            max_notional_per_pair_pct=10.0,
            max_directional_exposure_per_currency=10.0,
        )
        validator = PreTradeValidator(broker, constraints)
        # 110k USDJPY long → push past net 200k.
        intent = OrderIntent(
            strategy_id="s1", symbol="USDJPY", target_position=110_000 / 150.0,
        )
        rejection = validator.validate(intent)
        assert rejection is not None
        assert rejection.reason == RejectionReason.NET_LEVERAGE_BREACH


class TestCurrencyExposure:
    def test_currency_breach_rejected(self) -> None:
        # 100k equity, 40% per-currency cap = 40k per currency max.
        # Existing 30k EURUSD long → +30k EUR. New 30k EURJPY long → +30k EUR more
        # → 60k total EUR exposure > 40k cap. (Use very loose other constraints.)
        broker = _make_broker(capital=100_000, eurusd_mid=1.10)
        broker.set_price("EURJPY", 162.95, 162.99)
        existing = Position(
            symbol="EURUSD", quantity=30_000 / 1.10, avg_price=1.10,
        )
        broker._positions[existing.symbol] = existing  # type: ignore[attr-defined]
        constraints = PortfolioConstraints(
            max_notional_per_pair_pct=10.0,
            max_gross_leverage=10.0,
            max_net_leverage=10.0,
        )
        validator = PreTradeValidator(broker, constraints)
        intent = OrderIntent(
            strategy_id="s1", symbol="EURJPY", target_position=30_000 / 162.97,
        )
        rejection = validator.validate(intent)
        assert rejection is not None
        assert rejection.reason == RejectionReason.PER_CURRENCY_EXPOSURE


# =============================================================================
# Bookkeeping
# =============================================================================


class TestRejectionLog:
    def test_rejections_recorded(self) -> None:
        validator, *_ = _make_validator(supported_instruments={"EURUSD"})
        intent = OrderIntent(strategy_id="s1", symbol="XYZUSD", target_position=100)
        validator.validate(intent, current_positions=[])
        log = validator.rejections_log
        assert len(log) == 1
        d = log[0].to_dict()
        assert d["symbol"] == "XYZUSD"
        assert d["reason"] == RejectionReason.INSTRUMENT_UNSUPPORTED.value


# =============================================================================
# Coordinator integration
# =============================================================================


class TestCoordinatorWithValidator:
    def test_rejected_intent_does_not_reach_oms(self) -> None:
        broker = _make_broker(capital=100_000)
        oms = _RecordingOMS()
        constraints = PortfolioConstraints()
        validator = PreTradeValidator(
            broker,
            constraints,
            supported_instruments={"EURUSD"},  # exclude USDJPY
        )
        coord = PortfolioCoordinator(
            strategies=[_StrategyDouble("s1")],
            oms=oms,  # type: ignore[arg-type]
            broker=broker,
            state=_FakeState(),  # type: ignore[arg-type]
            constraints=constraints,
            pre_trade_validator=validator,
        )
        coord.allocations["s1"] = StrategyAllocation(
            strategy_id="s1", target_weight=1.0,
        )
        # USDJPY is not in whitelist → rejection.
        raw = {
            "s1": [OrderIntent(strategy_id="s1", symbol="USDJPY", target_position=100)],
        }
        accepted = asyncio.run(coord.process_intents(raw))
        assert oms.submitted == []
        assert "USDJPY" not in accepted
        assert len(validator.rejections_log) == 1

    def test_accepted_intent_reaches_oms(self) -> None:
        broker = _make_broker(capital=100_000)
        oms = _RecordingOMS()
        constraints = PortfolioConstraints()
        validator = PreTradeValidator(
            broker,
            constraints,
            supported_instruments={"EURUSD"},
        )
        coord = PortfolioCoordinator(
            strategies=[_StrategyDouble("s1")],
            oms=oms,  # type: ignore[arg-type]
            broker=broker,
            state=_FakeState(),  # type: ignore[arg-type]
            constraints=constraints,
            pre_trade_validator=validator,
        )
        coord.allocations["s1"] = StrategyAllocation(
            strategy_id="s1", target_weight=1.0,
        )
        raw = {
            "s1": [
                OrderIntent(
                    strategy_id="s1", symbol="EURUSD",
                    target_position=5_000 / 1.10,  # small, well under all caps
                ),
            ],
        }
        accepted = asyncio.run(coord.process_intents(raw))
        assert len(oms.submitted) == 1
        assert oms.submitted[0].symbol == "EURUSD"
        assert "EURUSD" in accepted
        assert validator.rejections_log == []


# =============================================================================
# Blackout-window tests (CL-c8th)
# =============================================================================


def _evaluator_with_event(
    minutes_until: float,
    tier: SeverityTier = SeverityTier.TIER_1,
    currency: str | None = None,
) -> BlackoutEvaluator:
    """Build a BlackoutEvaluator whose calendar has one upcoming event."""
    when = datetime.now(UTC) + timedelta(minutes=minutes_until)
    cal = EconomicCalendar([
        EconomicEvent(
            ts=when,
            event_type="NFP",
            severity_tier=tier,
            description="test",
            currency=currency,
        ),
    ])
    return BlackoutEvaluator(cal)


class TestBlackoutWindow:
    def test_new_entry_blocked_within_exit_flat_window(self) -> None:
        # Default tier-1 EXIT_FLAT window is 30 minutes — event in 5 min hits it.
        evaluator = _evaluator_with_event(minutes_until=5)
        validator, *_ = _make_validator(blackout_evaluator=evaluator)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1_000)
        rejection = validator.validate(intent, current_positions=[])
        assert rejection is not None
        assert rejection.reason == RejectionReason.BLACKOUT_EXIT_FLAT

    def test_new_entry_blocked_within_pause_window(self) -> None:
        # Default tier-1 PAUSE window is 1h, EXIT_FLAT is 30min — event at 45min
        # falls strictly into PAUSE.
        evaluator = _evaluator_with_event(minutes_until=45)
        validator, *_ = _make_validator(blackout_evaluator=evaluator)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1_000)
        rejection = validator.validate(intent, current_positions=[])
        assert rejection is not None
        assert rejection.reason == RejectionReason.BLACKOUT_PAUSE

    def test_size_down_window_does_not_block(self) -> None:
        # Tier-1 SIZE_DOWN_50PCT window is 24h, PAUSE is 1h — event at 6h falls
        # into SIZE_DOWN. Validator does NOT enforce size-down (coordinator's
        # job); validate must accept.
        evaluator = _evaluator_with_event(minutes_until=360)
        validator, *_ = _make_validator(blackout_evaluator=evaluator)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1_000)
        rejection = validator.validate(intent, current_positions=[])
        if rejection is not None:
            assert rejection.reason not in (
                RejectionReason.BLACKOUT_PAUSE,
                RejectionReason.BLACKOUT_EXIT_FLAT,
            )

    def test_flatten_during_exit_flat_allowed(self) -> None:
        # Existing 1_000 EURUSD long. Intent target=0 means "flatten" — must be
        # allowed even during EXIT_FLAT.
        evaluator = _evaluator_with_event(minutes_until=5)
        broker = _make_broker(
            positions=[Position(symbol="EURUSD", quantity=1_000, avg_price=1.10)],
        )
        validator, *_ = _make_validator(broker=broker, blackout_evaluator=evaluator)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=0)
        rejection = validator.validate(intent)
        if rejection is not None:
            assert rejection.reason not in (
                RejectionReason.BLACKOUT_PAUSE,
                RejectionReason.BLACKOUT_EXIT_FLAT,
            )

    def test_reduction_during_pause_allowed(self) -> None:
        # Existing 2_000 long, intent reduces to 500 — magnitude shrinks. Allow.
        evaluator = _evaluator_with_event(minutes_until=45)
        broker = _make_broker(
            positions=[Position(symbol="EURUSD", quantity=2_000, avg_price=1.10)],
        )
        validator, *_ = _make_validator(broker=broker, blackout_evaluator=evaluator)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=500)
        rejection = validator.validate(intent)
        if rejection is not None:
            assert rejection.reason not in (
                RejectionReason.BLACKOUT_PAUSE,
                RejectionReason.BLACKOUT_EXIT_FLAT,
            )

    def test_scale_up_during_pause_blocked(self) -> None:
        # Existing 500 long, intent grows to 2_000 — magnitude increases. Block.
        evaluator = _evaluator_with_event(minutes_until=45)
        broker = _make_broker(
            positions=[Position(symbol="EURUSD", quantity=500, avg_price=1.10)],
        )
        validator, *_ = _make_validator(broker=broker, blackout_evaluator=evaluator)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=2_000)
        rejection = validator.validate(intent)
        assert rejection is not None
        assert rejection.reason == RejectionReason.BLACKOUT_PAUSE

    def test_no_evaluator_does_not_block(self) -> None:
        # Default validator has no blackout evaluator — never rejects on blackout.
        validator, *_ = _make_validator()
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1_000)
        rejection = validator.validate(intent, current_positions=[])
        if rejection is not None:
            assert rejection.reason not in (
                RejectionReason.BLACKOUT_PAUSE,
                RejectionReason.BLACKOUT_EXIT_FLAT,
            )

    def test_currency_filter_passes_base_currency(self) -> None:
        # Event tagged USD; intent on EUR pair. Evaluator's currency filter is
        # base-of-pair (EUR), so a USD-only event should not block. EconomicCalendar
        # treats event currency as a hard match if both filter and event are set;
        # this is the documented behavior.
        evaluator = _evaluator_with_event(
            minutes_until=5, currency="USD",
        )
        validator, *_ = _make_validator(blackout_evaluator=evaluator)
        intent = OrderIntent(strategy_id="s1", symbol="EURJPY", target_position=1_000)
        rejection = validator.validate(intent, current_positions=[])
        if rejection is not None:
            assert rejection.reason not in (
                RejectionReason.BLACKOUT_PAUSE,
                RejectionReason.BLACKOUT_EXIT_FLAT,
            )
