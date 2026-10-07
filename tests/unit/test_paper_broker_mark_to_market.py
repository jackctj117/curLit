"""PaperBroker marks open positions to market (CL-9ird).

Before CL-9ird ``get_account().equity`` moved only on REALIZED deltas, so
an open losing position was invisible to the drawdown / daily-loss kill
switches on the paper venue. Requirements under test:

* equity = cash + unrealized, unrealized marked at the exit side of the
  latest quote, in ACCOUNT currency; realized reported separately;
* a flat book's equity equals capital + realized - costs (no mark);
* an open position that cannot be marked (no price, invalid price, or a
  non-account P&L currency — conversion is CL-vfw7) makes equity UNKNOWN
  (raise), never "cash only" / a zero mark;
* through the REAL LiveEngine health tick + RiskContextBuilder +
  KillSwitchManager, an open losing position trips ``drawdown_limit``, and
  an unmarkable book is counted as an account-read failure, never a value.

Oracle: hand-computed P&L from the quotes set in each test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.execution.broker import Order, OrderType
from src.execution.oms import OrderManager
from src.execution.paper_broker import AccountMarkUnavailableError, PaperBroker
from src.risk.kill_switches import KillSwitchManager
from src.risk.risk_context import RiskContextBuilder
from src.runtime.live_engine import LiveEngine

# 1 bp of fill notional (PaperBroker's simulated cost model).
_COST_RATE = 0.0001


def _order(side: str, qty: float, symbol: str = "EURUSD") -> Order:
    return Order(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET)


class TestMarkToMarket:
    def test_open_long_marked_at_bid(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 10_000))  # fill @ 1.1000 ask
        cost = 10_000 * 1.1000 * _COST_RATE
        b.set_price("EURUSD", 1.0898, 1.0900)  # 100 pips lower
        acct = b.get_account()
        expected_unrealized = 10_000 * (1.0898 - 1.1000)  # -102.0
        assert acct.unrealized_pnl == pytest.approx(expected_unrealized)
        assert acct.realized_pnl == pytest.approx(0.0)
        assert acct.equity == pytest.approx(100_000.0 - cost + expected_unrealized)
        assert b.equity == pytest.approx(acct.equity)
        assert b.cash == pytest.approx(100_000.0 - cost)

    def test_open_short_marked_at_ask(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("EUR_USD", 1.1000, 1.1002)
        b.place_order(_order("sell", 5_000, "EUR_USD"))  # fill @ 1.1000 bid
        cost = 5_000 * 1.1000 * _COST_RATE
        b.set_price("EUR_USD", 1.1100, 1.1102)
        acct = b.get_account()
        expected_unrealized = -5_000 * (1.1102 - 1.1000)  # short loses as price rises
        assert acct.unrealized_pnl == pytest.approx(expected_unrealized)
        assert acct.equity == pytest.approx(100_000.0 - cost + expected_unrealized)

    def test_flat_book_equity_equals_realized(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 10_000))  # @ 1.1000
        b.set_price("EURUSD", 1.1050, 1.1052)
        b.place_order(_order("sell", 10_000))  # @ 1.1050
        costs = 10_000 * 1.1000 * _COST_RATE + 10_000 * 1.1050 * _COST_RATE
        realized = 10_000 * (1.1050 - 1.1000)
        acct = b.get_account()
        assert acct.realized_pnl == pytest.approx(realized)
        assert acct.unrealized_pnl == pytest.approx(0.0)
        assert acct.equity == pytest.approx(100_000.0 + realized - costs)
        # A flat book needs no price: the mark must not consult the quote.
        del b._prices["EURUSD"]
        assert b.get_account().equity == pytest.approx(100_000.0 + realized - costs)

    def test_partial_close_splits_realized_and_unrealized(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("GBPUSD", 1.2498, 1.2500)
        b.place_order(_order("buy", 10_000, "GBPUSD"))  # @ 1.2500
        b.set_price("GBPUSD", 1.2400, 1.2402)
        b.place_order(_order("sell", 4_000, "GBPUSD"))  # @ 1.2400
        acct = b.get_account()
        assert acct.realized_pnl == pytest.approx(4_000 * (1.2400 - 1.2500))
        assert acct.unrealized_pnl == pytest.approx(6_000 * (1.2400 - 1.2500))


class TestUnknownMarkNeverZero:
    def test_missing_price_raises(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 10_000))
        del b._prices["EURUSD"]  # quote lost while the position is open
        with pytest.raises(AccountMarkUnavailableError, match="EURUSD"):
            b.get_account()
        with pytest.raises(AccountMarkUnavailableError):
            _ = b.equity

    def test_nonfinite_price_raises(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 10_000))
        b.set_price("EURUSD", float("nan"), float("nan"))
        with pytest.raises(AccountMarkUnavailableError):
            b.get_account()

    @pytest.mark.parametrize(
        ("symbol", "bid", "ask"),
        [
            ("USDJPY", 149.99, 150.01),  # JPY-quoted: P&L in JPY
            ("EURGBP", 0.8499, 0.8501),  # cross: P&L in GBP
            ("SPX500_USD", 5000.0, 5000.5),  # not a currency pair
        ],
    )
    def test_non_account_currency_pnl_raises(self, symbol: str, bid: float, ask: float) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price(symbol, bid, ask)
        b.place_order(_order("buy", 1_000, symbol))
        with pytest.raises(AccountMarkUnavailableError, match="CL-vfw7"):
            b.get_account()

    def test_non_account_currency_round_trip_stays_unknown_when_flat(self) -> None:
        # Codex r1: closing USDJPY realizes -1,000 JPY. That must not be
        # reported as -1,000 USD realized, and going flat must not make the
        # unconverted result "known" again.
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("USDJPY", 149.99, 150.00)
        b.place_order(_order("buy", 1_000, "USDJPY"))  # @ 150.00
        b.set_price("USDJPY", 149.00, 149.01)
        b.place_order(_order("sell", 1_000, "USDJPY"))  # @ 149.00 → -1,000 JPY
        assert next(p for p in b.get_positions() if p.symbol == "USDJPY").quantity == 0.0
        with pytest.raises(AccountMarkUnavailableError, match="JPY"):
            b.get_account()
        # Account-currency cash was never touched by the JPY amounts.
        assert b.cash == pytest.approx(100_000.0)

    def test_account_currency_is_configurable(self) -> None:
        b = PaperBroker(initial_capital=10_000_000.0, account_currency="JPY")
        b.set_price("USDJPY", 149.99, 150.01)
        b.place_order(_order("buy", 1_000, "USDJPY"))  # @ 150.01
        b.set_price("USDJPY", 148.99, 149.01)
        assert b.get_account().unrealized_pnl == pytest.approx(1_000 * (148.99 - 150.01))


# =============================================================================
# Kill-switch path: real LiveEngine health tick + builder + manager
# =============================================================================


class _Clock:
    def __init__(self) -> None:
        # A Wednesday mid-session: inside the FX trading window.
        self.now = datetime(2026, 10, 7, 14, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _engine(broker: PaperBroker, tmp_path: Any) -> tuple[LiveEngine, KillSwitchManager]:
    clock = _Clock()
    oms = OrderManager(broker)
    ksm = KillSwitchManager(
        broker=broker,
        oms=oms,
        config={},  # profile defaults: drawdown -20%, daily loss -3%
        clock=clock,
        trailing_state_path=tmp_path / "trailing.json",
    )
    builder = RiskContextBuilder(state_path=None, clock=clock)
    engine = LiveEngine(
        strategies=[],
        oms=oms,
        broker=broker,
        kill_switch_manager=ksm,
        risk_context_builder=builder,
    )
    return engine, ksm


class TestKillSwitchSeesOpenLosses:
    def test_open_losing_position_trips_drawdown(self, tmp_path: Any) -> None:
        broker = PaperBroker(initial_capital=100_000.0)
        broker.set_price("EURUSD", 1.0998, 1.1000)
        broker.place_order(_order("buy", 1_000_000))  # @ 1.1000, 1.1M notional
        engine, ksm = _engine(broker, tmp_path)

        engine._health_tick()  # establishes peak / day-start equity
        assert "drawdown_limit" not in ksm._active_halt_causes

        # 300 pips lower: unrealized -30,000 on ~100k equity = ~-30% < -20%.
        broker.set_price("EURUSD", 1.0700, 1.0702)
        assert broker.get_account().equity < 0.8 * 100_000.0
        engine._health_tick()

        assert "drawdown_limit" in ksm._active_halt_causes
        # flatten_all reached the paper venue: the loss is now realized.
        pos = next(p for p in broker.get_positions() if p.symbol == "EURUSD")
        assert pos.quantity == pytest.approx(0.0)
        assert broker.get_account().realized_pnl == pytest.approx(1_000_000 * (1.0700 - 1.1000))

    def test_unmarkable_book_is_an_account_read_failure_not_a_value(
        self,
        tmp_path: Any,
    ) -> None:
        broker = PaperBroker(initial_capital=100_000.0)
        broker.set_price("EURUSD", 1.0998, 1.1000)
        broker.place_order(_order("buy", 1_000_000))
        engine, ksm = _engine(broker, tmp_path)
        engine._health_tick()
        assert engine._account_read_failures == 0

        del broker._prices["EURUSD"]
        engine._health_tick()
        assert engine._account_read_failures == 1
        # No equity value was fed to the risk context: the builder's state
        # still reflects only the first (marked) tick.
        assert engine.risk_context_builder is not None
        assert engine.risk_context_builder._peak_equity == pytest.approx(
            100_000.0 - 1_000_000 * 1.1000 * _COST_RATE + 1_000_000 * (1.0998 - 1.1000)
        )
