"""Tests for Polymarket per-market + per-day loss caps (CL-983f).

Tracker math (realized + mark-to-market on PER-TOKEN books), the order
gate (breach -> LossCapExceededError, under-cap -> allowed, per-market
isolation, UTC-day rollover, position-bounded reduce-only exemption),
condition_id cap aggregation, persistence (restart continuity + fill
dedup), and the broker wiring — paper broker records fills/marks and
delegates its position book to the tracker; live broker gates before
anything is signed, records immediate CLOB matches, ingests reconciled
on-chain fills idempotently, and warns when trading with an unfed
tracker (mocked ClobClient, no network).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.execution.broker import Order, OrderStatus, OrderType
from src.execution.polymarket_reconciler import OnchainFill
from src.risk.polymarket_loss_caps import (
    LossCapConfig,
    LossCapExceededError,
    PolymarketLossCapTracker,
)

_DAY_1 = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
_DAY_2 = datetime(2026, 7, 15, 0, 5, tzinfo=UTC)


class _FakeClock:
    def __init__(self, now: datetime = _DAY_1) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _tracker(
    market_cap: str = "25",
    day_cap: str = "50",
    clock: _FakeClock | None = None,
    state_path: Path | None = None,
    market_key_fn=None,
) -> PolymarketLossCapTracker:
    kwargs = {}
    if market_key_fn is not None:
        kwargs["market_key_fn"] = market_key_fn
    return PolymarketLossCapTracker(
        config=LossCapConfig(
            per_market_loss_cap_usd=Decimal(market_cap),
            per_day_loss_cap_usd=Decimal(day_cap),
            state_path=state_path,
        ),
        clock=clock or _FakeClock(),
        **kwargs,
    )


class TestPnlMath:
    def test_realized_loss_on_round_trip(self) -> None:
        t = _tracker()
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_fill("mkt-a", "sell", 100, "0.40")
        assert t.market_pnl("mkt-a") == Decimal("-10.0")
        assert t.day_pnl() == Decimal("-10.0")

    def test_mark_to_market_loss_counts(self) -> None:
        t = _tracker()
        t.record_fill("mkt-a", "buy", 100, "0.50")
        assert t.market_pnl("mkt-a") == Decimal("0")  # mark = fill price
        t.record_mark("mkt-a", "0.30")
        assert t.market_pnl("mkt-a") == Decimal("-20.0")
        assert t.day_pnl() == Decimal("-20.0")

    def test_fees_count_against_realized(self) -> None:
        t = _tracker()
        t.record_fill("mkt-a", "buy", 100, "0.50", fee="1.25")
        assert t.market_pnl("mkt-a") == Decimal("-1.25")

    def test_short_position_mark_to_market(self) -> None:
        # Sold 100 @ 0.50; price rises to 0.60 -> -10 unrealized.
        t = _tracker()
        t.record_fill("mkt-a", "sell", 100, "0.50")
        t.record_mark("mkt-a", "0.60")
        assert t.market_pnl("mkt-a") == Decimal("-10.0")

    def test_unknown_market_pnl_is_zero(self) -> None:
        assert _tracker().market_pnl("never-traded") == Decimal("0")


class TestGate:
    def test_under_cap_allowed(self) -> None:
        t = _tracker(market_cap="25", day_cap="50")
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_fill("mkt-a", "sell", 100, "0.40")  # -10, under both caps
        t.check_order_allowed("mkt-a", side="buy")  # no raise

    def test_market_breach_rejects(self) -> None:
        t = _tracker(market_cap="25", day_cap="500")
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_fill("mkt-a", "sell", 100, "0.20")  # -30 realized
        with pytest.raises(LossCapExceededError, match="scope=market") as exc_info:
            t.check_order_allowed("mkt-a", side="buy")
        assert exc_info.value.scope == "market"
        assert exc_info.value.market_key == "mkt-a"

    def test_mark_to_market_breach_rejects(self) -> None:
        t = _tracker(market_cap="25", day_cap="500")
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_mark("mkt-a", "0.20")  # -30 unrealized
        with pytest.raises(LossCapExceededError, match="scope=market"):
            t.check_order_allowed("mkt-a", side="buy")

    def test_per_market_isolation(self) -> None:
        # Market A burned its cap; market B must remain tradable.
        t = _tracker(market_cap="25", day_cap="500")
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_fill("mkt-a", "sell", 100, "0.20")  # A: -30
        with pytest.raises(LossCapExceededError):
            t.check_order_allowed("mkt-a", side="buy")
        t.check_order_allowed("mkt-b", side="buy")  # no raise

    def test_day_breach_blocks_all_markets(self) -> None:
        # Three markets each lose 20 — none breaches its own 25 cap,
        # but the day is -60 <= -50: everything is blocked.
        t = _tracker(market_cap="25", day_cap="50")
        for mkt in ("mkt-a", "mkt-b", "mkt-c"):
            t.record_fill(mkt, "buy", 100, "0.50")
            t.record_fill(mkt, "sell", 100, "0.30")
        with pytest.raises(LossCapExceededError, match="scope=day") as exc_info:
            t.check_order_allowed("mkt-fresh", side="buy")
        assert exc_info.value.scope == "day"

    def test_day_rollover_resets_daily_accumulator(self) -> None:
        clock = _FakeClock(_DAY_1)
        t = _tracker(market_cap="25", day_cap="50", clock=clock)
        for mkt in ("mkt-a", "mkt-b", "mkt-c"):
            t.record_fill(mkt, "buy", 100, "0.50")
            t.record_fill(mkt, "sell", 100, "0.30")
        with pytest.raises(LossCapExceededError, match="scope=day"):
            t.check_order_allowed("mkt-fresh", side="buy")

        clock.now = _DAY_2  # UTC midnight passed
        assert t.day_pnl() == Decimal("0")
        t.check_order_allowed("mkt-fresh", side="buy")  # no raise

    def test_market_cap_survives_day_rollover(self) -> None:
        # Per-market losses are lifetime — a burned market stays
        # blocked after midnight; only the daily bucket resets.
        clock = _FakeClock(_DAY_1)
        t = _tracker(market_cap="25", day_cap="500", clock=clock)
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_fill("mkt-a", "sell", 100, "0.20")  # -30

        clock.now = _DAY_2
        with pytest.raises(LossCapExceededError, match="scope=market"):
            t.check_order_allowed("mkt-a", side="buy")

    def test_reducing_order_allowed_despite_breach(self) -> None:
        # Long a burned market: sell (exit) allowed, buy (add) refused.
        t = _tracker(market_cap="25", day_cap="500")
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_mark("mkt-a", "0.20")  # -30 unrealized
        t.check_order_allowed("mkt-a", side="sell")  # no raise
        with pytest.raises(LossCapExceededError):
            t.check_order_allowed("mkt-a", side="buy")


class TestReduceOnlyBound:
    """The reduce exemption is bounded by |position| — an opposing order
    LARGER than the position would flip through zero into new risk in a
    capped market, so the whole (atomic) order is rejected."""

    def _burned_long(self) -> PolymarketLossCapTracker:
        t = _tracker(market_cap="25", day_cap="500")
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_mark("mkt-a", "0.20")  # -30: breached
        return t

    def test_oversized_reducing_order_rejected(self) -> None:
        t = self._burned_long()
        with pytest.raises(LossCapExceededError, match="scope=market"):
            t.check_order_allowed("mkt-a", side="sell", quantity=150)

    def test_exact_flat_reduce_allowed(self) -> None:
        t = self._burned_long()
        t.check_order_allowed("mkt-a", side="sell", quantity=100)  # no raise

    def test_partial_reduce_allowed(self) -> None:
        t = self._burned_long()
        t.check_order_allowed("mkt-a", side="sell", quantity=40)  # no raise

    def test_short_position_buy_reduce_bound(self) -> None:
        t = _tracker(market_cap="25", day_cap="500")
        t.record_fill("mkt-a", "sell", 100, "0.50")
        t.record_mark("mkt-a", "0.80")  # -30: breached
        t.check_order_allowed("mkt-a", side="buy", quantity=100)  # flat: ok
        with pytest.raises(LossCapExceededError, match="scope=market"):
            t.check_order_allowed("mkt-a", side="buy", quantity=101)

    def test_day_cap_oversized_reduce_also_rejected(self) -> None:
        t = _tracker(market_cap="500", day_cap="25")
        t.record_fill("mkt-a", "buy", 100, "0.50")
        t.record_mark("mkt-a", "0.20")  # day -30: breached
        t.check_order_allowed("mkt-a", side="sell", quantity=100)
        with pytest.raises(LossCapExceededError, match="scope=day"):
            t.check_order_allowed("mkt-a", side="sell", quantity=200)


class TestConditionIdAggregation:
    """market_key_fn pools tokens for CAP AGGREGATION only — position
    books stay per-token, so a YES/NO hedge nets to ~0 combined P&L
    instead of a phantom loss from a pooled qty/avg book."""

    @staticmethod
    def _cond_fn(token_id: str) -> str:
        return "cond-1" if token_id in ("tok-yes", "tok-no") else token_id

    def _hedged(self) -> PolymarketLossCapTracker:
        t = _tracker(market_cap="25", day_cap="500", market_key_fn=self._cond_fn)
        t.record_fill("tok-yes", "buy", 100, "0.70")
        t.record_fill("tok-no", "buy", 100, "0.30")
        return t

    def test_per_token_books_kept(self) -> None:
        t = self._hedged()
        books = t.open_positions()
        assert books["tok-yes"].qty == Decimal("100")
        assert books["tok-yes"].avg_price == Decimal("0.70")
        assert books["tok-no"].qty == Decimal("100")
        assert books["tok-no"].avg_price == Decimal("0.30")

    def test_hedged_pair_has_zero_combined_pnl(self) -> None:
        t = self._hedged()
        assert t.market_pnl("cond-1") == Decimal("0")
        # Complementary marks move against each other — still ~0, no
        # phantom -40 from a pooled 200-share 0.50-avg book.
        t.record_mark("tok-yes", "0.60")
        t.record_mark("tok-no", "0.40")
        assert t.market_pnl("cond-1") == Decimal("0")
        assert t.day_pnl() == Decimal("0")
        t.check_order_allowed("tok-yes", side="buy")  # no phantom breach

    def test_genuine_market_loss_blocks_both_tokens(self) -> None:
        t = self._hedged()
        # YES collapses more than NO gains: -30 + 5 = -25 <= -25 cap.
        t.record_mark("tok-yes", "0.40")
        t.record_mark("tok-no", "0.35")
        assert t.market_pnl("cond-1") == Decimal("-25.0")
        with pytest.raises(LossCapExceededError, match="market=cond-1"):
            t.check_order_allowed("tok-yes", side="buy")
        with pytest.raises(LossCapExceededError, match="market=cond-1"):
            t.check_order_allowed("tok-no", side="buy")

    def test_other_condition_unaffected(self) -> None:
        t = self._hedged()
        t.record_mark("tok-yes", "0.40")
        t.record_mark("tok-no", "0.35")
        t.check_order_allowed("tok-other", side="buy")  # no raise


class TestPersistence:
    """Cap state survives restarts: burned markets stay blocked, the
    daily bucket keeps its running total, and fill idempotency keys
    persist so re-reconciliation cannot double-count."""

    def test_round_trip_blocks_burned_market(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        clock = _FakeClock(_DAY_1)
        t1 = _tracker(market_cap="25", day_cap="500", clock=clock, state_path=p)
        t1.record_fill("mkt-a", "buy", 100, "0.50")
        t1.record_fill("mkt-a", "sell", 100, "0.20")  # -30 realized

        t2 = _tracker(market_cap="25", day_cap="500", clock=clock, state_path=p)
        assert t2.market_pnl("mkt-a") == Decimal("-30.0")
        with pytest.raises(LossCapExceededError, match="scope=market"):
            t2.check_order_allowed("mkt-a", side="buy")

    def test_restart_mid_day_daily_cap_continuity(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        clock = _FakeClock(_DAY_1)
        t1 = _tracker(market_cap="25", day_cap="50", clock=clock, state_path=p)
        for mkt in ("mkt-a", "mkt-b"):
            t1.record_fill(mkt, "buy", 100, "0.50")
            t1.record_fill(mkt, "sell", 100, "0.20")  # -30 each; day: -60

        # Restart mid-day: the day does NOT get a fresh $50 budget.
        t2 = _tracker(market_cap="25", day_cap="50", clock=clock, state_path=p)
        assert t2.day_pnl() == Decimal("-60.0")
        with pytest.raises(LossCapExceededError, match="scope=day"):
            t2.check_order_allowed("mkt-fresh", side="buy")

        # Midnight rolls the daily bucket, but burned markets stay
        # blocked (lifetime realized persisted).
        clock.now = _DAY_2
        t2.check_order_allowed("mkt-fresh", side="buy")  # no raise
        with pytest.raises(LossCapExceededError, match="scope=market"):
            t2.check_order_allowed("mkt-a", side="buy")

    def test_marks_and_positions_persist(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        t1 = _tracker(market_cap="25", day_cap="500", state_path=p)
        t1.record_fill("mkt-a", "buy", 100, "0.50")
        t1.record_mark("mkt-a", "0.20")  # -30 unrealized

        t2 = _tracker(market_cap="25", day_cap="500", state_path=p)
        assert t2.market_pnl("mkt-a") == Decimal("-30.0")
        st = t2.open_positions()["mkt-a"]
        assert st.qty == Decimal("100")
        assert st.avg_price == Decimal("0.50")

    def test_fill_dedup_survives_restart(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        t1 = _tracker(state_path=p)
        assert t1.record_fill("mkt-a", "buy", 10, "0.50", fill_id="tx1:oh1")
        assert not t1.record_fill("mkt-a", "buy", 10, "0.50", fill_id="tx1:oh1")

        t2 = _tracker(state_path=p)
        assert t2.has_recorded_fill("tx1:oh1")
        assert not t2.record_fill("mkt-a", "buy", 10, "0.50", fill_id="tx1:oh1")
        assert t2.open_positions()["mkt-a"].qty == Decimal("10")

    def test_missing_file_starts_fresh(self, tmp_path: Path) -> None:
        t = _tracker(state_path=tmp_path / "nope.json")
        assert not t.has_activity
        t.check_order_allowed("mkt-a", side="buy")  # no raise

    def test_corrupt_state_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        p.write_text("{not json")
        with pytest.raises(ValueError, match="corrupt"):
            _tracker(state_path=p)

    def test_no_state_path_never_writes(self, tmp_path: Path) -> None:
        t = _tracker(state_path=None)
        t.record_fill("mkt-a", "buy", 10, "0.50")
        t.record_mark("mkt-a", "0.40")
        assert list(tmp_path.iterdir()) == []


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LossCapConfig()
        assert cfg.per_market_loss_cap_usd == Decimal("25")
        assert cfg.per_day_loss_cap_usd == Decimal("50")
        assert cfg.state_path == Path("data/polymarket_loss_caps_state.json")

    def test_nonpositive_caps_rejected(self) -> None:
        with pytest.raises(ValueError, match="per_market"):
            LossCapConfig(per_market_loss_cap_usd=Decimal("0"))
        with pytest.raises(ValueError, match="per_day"):
            LossCapConfig(per_day_loss_cap_usd=Decimal("-5"))

    def test_from_active_profile_reads_kill_switches(self) -> None:
        profile = MagicMock()
        profile.kill_switches.polymarket_per_market_loss_cap_usd = 12.5
        profile.kill_switches.polymarket_per_day_loss_cap_usd = 40
        with patch(
            "src.risk.risk_profile.load_active_profile",
            return_value=profile,
        ):
            cfg = LossCapConfig.from_active_profile()
        assert cfg.per_market_loss_cap_usd == Decimal("12.5")
        assert cfg.per_day_loss_cap_usd == Decimal("40")


# ---------------------------------------------------------------------- #
# Broker wiring — the choke point is place_order
# ---------------------------------------------------------------------- #


def _paper_broker(tracker: PolymarketLossCapTracker):
    from src.execution.polymarket_data_source import (
        BookLevel,
        OrderBookSnapshot,
        PolymarketDataSource,
    )
    from src.execution.polymarket_paper_broker import PolymarketPaperBroker

    book = OrderBookSnapshot(
        token_id="tok-1",
        bids=[BookLevel(Decimal("0.49"), Decimal("500"))],
        asks=[BookLevel(Decimal("0.50"), Decimal("500"))],
    )
    ds = MagicMock(spec=PolymarketDataSource)
    ds.resolve_symbol.side_effect = lambda _s: "tok-1"
    ds.get_book.return_value = book
    return PolymarketPaperBroker(data_source=ds, loss_cap_tracker=tracker)


def _crashed_book():
    """A tok-1 book whose mid (0.20) marks a 0.50 entry deep underwater."""
    from src.execution.polymarket_data_source import (
        BookLevel,
        OrderBookSnapshot,
    )

    return OrderBookSnapshot(
        token_id="tok-1",
        bids=[BookLevel(Decimal("0.19"), Decimal("500"))],
        asks=[BookLevel(Decimal("0.21"), Decimal("500"))],
    )


class TestPaperBrokerWiring:
    def test_breach_refuses_new_order(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        tracker.record_fill("tok-1", "buy", 100, "0.50")
        tracker.record_mark("tok-1", "0.20")  # -30: breached
        broker = _paper_broker(tracker)
        order = Order(
            symbol="POLY:tok-1",
            side="buy",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=0.50,
        )
        with pytest.raises(LossCapExceededError, match="loss cap breached"):
            broker.place_order(order)
        # The rejected order changed nothing — the (delegated) position
        # book still shows exactly the pre-seeded 100 shares.
        positions = broker.get_positions()
        assert [(p.symbol, p.quantity) for p in positions] == [("tok-1", 100.0)]

    def test_under_cap_order_fills(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker = _paper_broker(tracker)
        order = Order(
            symbol="POLY:tok-1",
            side="buy",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=0.50,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.FILLED

    def test_paper_fills_feed_the_tracker(self) -> None:
        # Paper runs must exercise the same accounting the live path
        # uses — the broker records its own fills.
        tracker = _tracker(market_cap="25", day_cap="50")
        broker = _paper_broker(tracker)
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="buy",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        assert broker.loss_caps is tracker
        assert tracker.open_positions()["tok-1"].qty == Decimal("10")

    def test_reducing_exit_allowed_when_breached(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        broker = _paper_broker(tracker)
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="buy",
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        tracker.record_mark("tok-1", "0.20")  # -30: breached
        out = broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="sell",
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=0.49,
            )
        )
        assert out.status == OrderStatus.FILLED

    def test_oversized_flip_through_breached_cap_refused(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        broker = _paper_broker(tracker)
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="buy",
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        tracker.record_mark("tok-1", "0.20")  # -30: breached
        # Sell 300 against a 100-long would flip into a 200 short —
        # new risk in a capped market. Whole order refused.
        with pytest.raises(LossCapExceededError, match="loss cap breached"):
            broker.place_order(
                Order(
                    symbol="POLY:tok-1",
                    side="sell",
                    quantity=300,
                    order_type=OrderType.LIMIT,
                    limit_price=0.49,
                )
            )


class TestPaperMarkObservation:
    """The paper broker feeds book mids to the tracker wherever quotes
    enter (place_order / get_price / stream_prices), so unrealized
    losses between trades arm the cap gate."""

    def test_get_price_records_mark_and_blocks_next_order(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        broker = _paper_broker(tracker)
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="buy",
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        # Market collapses; the broker observes it via get_price.
        broker._data.get_book.return_value = _crashed_book()  # noqa: SLF001
        broker.get_price("POLY:tok-1")
        assert tracker.open_positions()["tok-1"].last_mark == Decimal("0.20")
        # -30 unrealized (plus the fill's fee against realized).
        assert tracker.market_pnl("tok-1") <= Decimal("-30.0")
        with pytest.raises(LossCapExceededError, match="scope=market"):
            broker.place_order(
                Order(
                    symbol="POLY:tok-1",
                    side="buy",
                    quantity=10,
                    order_type=OrderType.LIMIT,
                    limit_price=0.21,
                )
            )

    def test_place_order_book_fetch_records_mark(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        broker = _paper_broker(tracker)
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="buy",
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        broker._data.get_book.return_value = _crashed_book()  # noqa: SLF001
        # A (reducing, resting) sell still observes the crashed book.
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="sell",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.99,
            )
        )
        assert tracker.open_positions()["tok-1"].last_mark == Decimal("0.20")
        assert tracker.market_pnl("tok-1") <= Decimal("-30.0")

    def test_stream_prices_records_marks(self) -> None:
        import asyncio

        tracker = _tracker(market_cap="25", day_cap="500")
        broker = _paper_broker(tracker)
        broker._data.get_book.return_value = _crashed_book()  # noqa: SLF001

        async def _one_tick():
            agen = broker.stream_prices(["POLY:tok-1"])
            try:
                return await agen.__anext__()
            finally:
                await agen.aclose()

        asyncio.run(_one_tick())
        assert tracker.open_positions() == {}  # no position, just a mark
        tracker.record_fill("tok-1", "buy", 100, "0.50")
        tracker.record_mark("tok-1", "0.20")
        assert tracker.market_pnl("tok-1") == Decimal("-30.0")


class TestPaperBookDelegation:
    """Finding: the paper broker kept a second, divergent position book.
    Now get_positions is derived from the tracker's per-token book."""

    def test_flip_through_zero_rebases_avg_at_fill_price(self) -> None:
        from src.execution.polymarket_data_source import (
            BookLevel,
            OrderBookSnapshot,
            PolymarketDataSource,
        )
        from src.execution.polymarket_paper_broker import PolymarketPaperBroker

        book = OrderBookSnapshot(
            token_id="tok-1",
            bids=[BookLevel(Decimal("0.40"), Decimal("500"))],
            asks=[BookLevel(Decimal("0.50"), Decimal("500"))],
        )
        ds = MagicMock(spec=PolymarketDataSource)
        ds.resolve_symbol.side_effect = lambda _s: "tok-1"
        ds.get_book.return_value = book
        tracker = _tracker(market_cap="500", day_cap="500")
        broker = PolymarketPaperBroker(data_source=ds, loss_cap_tracker=tracker)

        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="buy",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="sell",
                quantity=30,
                order_type=OrderType.LIMIT,
                limit_price=0.40,
            )
        )
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == -20.0
        # Old duplicate book kept the stale 0.50 avg on a flip; the
        # tracker (single book) re-bases at the flip's fill price.
        assert positions[0].avg_price == 0.40

    def test_positions_view_is_the_tracker_view(self) -> None:
        tracker = _tracker(market_cap="500", day_cap="500")
        broker = _paper_broker(tracker)
        broker.place_order(
            Order(
                symbol="POLY:tok-1",
                side="buy",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        # A fill recorded straight into the tracker (e.g. by the engine)
        # is visible through the broker — one book, no divergence.
        tracker.record_fill("tok-1", "buy", 5, "0.60")
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 15.0
        assert positions[0].realized_pnl == pytest.approx(
            float(tracker.open_positions()["tok-1"].realized),
        )


# ---------------------------------------------------------------------- #
# Live broker
# ---------------------------------------------------------------------- #


def _live_broker(tracker: PolymarketLossCapTracker):
    from src.execution.polymarket_secrets import PolymarketCreds

    creds = PolymarketCreds(
        signer_pk="0x" + "11" * 32,
        api_key="k",
        api_secret="s",
        api_passphrase="p",
        funder_address="0xFunder",
        chain_id=80002,
        rpc_url="https://example/rpc",
    )
    with (
        patch(
            "src.execution.polymarket_broker.load_polymarket_creds",
            return_value=creds,
        ),
        patch("py_clob_client.client.ClobClient") as client_cls,
    ):
        from src.execution.polymarket_broker import PolymarketBroker

        broker = PolymarketBroker(env="amoy", loss_cap_tracker=tracker)
    return broker, client_cls.return_value


class TestLiveBrokerWiring:
    def test_breach_refuses_before_signing(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        tracker.record_fill("tok-live", "buy", 100, "0.50")
        tracker.record_mark("tok-live", "0.20")  # -30: breached
        broker, client = _live_broker(tracker)

        order = Order(
            symbol="tok-live",
            side="buy",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=0.50,
        )
        with pytest.raises(LossCapExceededError, match="loss cap breached"):
            broker.place_order(order)
        client.create_order.assert_not_called()
        client.post_order.assert_not_called()

    def test_under_cap_order_submits(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = _live_broker(tracker)
        client.post_order.return_value = {"success": True, "orderID": "ord-9"}

        order = Order(
            symbol="tok-live",
            side="buy",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=0.50,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.PENDING
        assert out.order_id == "ord-9"
        client.create_order.assert_called_once()

    def test_day_breach_blocks_other_market_on_live(self) -> None:
        tracker = _tracker(market_cap="500", day_cap="50")
        tracker.record_fill("tok-a", "buy", 200, "0.50")
        tracker.record_fill("tok-a", "sell", 200, "0.20")  # -60 day
        broker, client = _live_broker(tracker)

        order = Order(
            symbol="tok-other",
            side="buy",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=0.50,
        )
        with pytest.raises(LossCapExceededError, match="scope=day"):
            broker.place_order(order)
        client.create_order.assert_not_called()

    def test_oversized_flip_refused_on_live(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        tracker.record_fill("tok-live", "buy", 100, "0.50")
        tracker.record_mark("tok-live", "0.20")  # -30: breached
        broker, client = _live_broker(tracker)

        with pytest.raises(LossCapExceededError, match="loss cap breached"):
            broker.place_order(
                Order(
                    symbol="tok-live",
                    side="sell",
                    quantity=150,
                    order_type=OrderType.LIMIT,
                    limit_price=0.20,
                )
            )
        client.create_order.assert_not_called()

    def test_tracker_exposed_for_fill_loop(self) -> None:
        tracker = _tracker()
        broker, _client = _live_broker(tracker)
        assert broker.loss_caps is tracker


class TestLiveImmediateFills:
    def test_matched_response_records_fill(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = _live_broker(tracker)
        client.post_order.return_value = {
            "success": True,
            "orderID": "0x" + "ab" * 32,
            "status": "matched",
            "makingAmount": "5",  # USDC paid
            "takingAmount": "10",  # tokens received
            "transactionsHashes": ["0x" + "cd" * 32],
        }
        out = broker.place_order(
            Order(
                symbol="tok-live",
                side="buy",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        assert out.status == OrderStatus.FILLED
        st = tracker.open_positions()["tok-live"]
        assert st.qty == Decimal("10")
        assert st.avg_price == Decimal("0.5")

    def test_matched_without_amounts_books_full_order(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = _live_broker(tracker)
        client.post_order.return_value = {
            "success": True,
            "orderID": "0xabc",
            "status": "matched",
        }
        out = broker.place_order(
            Order(
                symbol="tok-live",
                side="buy",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        assert out.status == OrderStatus.FILLED
        assert tracker.open_positions()["tok-live"].qty == Decimal("10")

    def test_unmatched_response_records_nothing(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = _live_broker(tracker)
        client.post_order.return_value = {"success": True, "orderID": "ord-9"}
        broker.place_order(
            Order(
                symbol="tok-live",
                side="buy",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        assert tracker.open_positions() == {}


def _onchain_fill(**overrides) -> OnchainFill:
    """Funder-as-maker BUY: pays 5 USDC for 10 tokens of asset 123."""
    base = dict(
        order_hash="aa" * 32,
        maker="0xFunder",
        taker="0xCounterparty",
        maker_asset_id=0,  # USDC out
        taker_asset_id=123,  # tokens in
        maker_amount_filled=5_000_000,  # 5 USDC (6 decimals)
        taker_amount_filled=10_000_000,  # 10 tokens
        fee=0,
        block_number=100,
        tx_hash="0x" + "ee" * 32,
    )
    base.update(overrides)
    return OnchainFill(**base)


class TestReconciledFillIngestion:
    def test_buy_fill_ingested(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, _client = _live_broker(tracker)
        assert broker.ingest_reconciled_fills([_onchain_fill()]) == 1
        st = tracker.open_positions()["123"]
        assert st.qty == Decimal("10")
        assert st.avg_price == Decimal("0.5")

    def test_reingestion_is_idempotent(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, _client = _live_broker(tracker)
        fills = [_onchain_fill()]
        assert broker.ingest_reconciled_fills(fills) == 1
        # Overlapping block range re-reconciled — nothing double-counted.
        assert broker.ingest_reconciled_fills(fills) == 0
        assert tracker.open_positions()["123"].qty == Decimal("10")
        assert tracker.day_pnl() == Decimal("0")

    def test_sell_fill_and_fee(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, _client = _live_broker(tracker)
        broker.ingest_reconciled_fills([_onchain_fill()])  # long 10 @ 0.50
        sell = _onchain_fill(
            order_hash="bb" * 32,
            tx_hash="0x" + "ff" * 32,
            maker_asset_id=123,  # tokens out
            taker_asset_id=0,  # USDC in
            maker_amount_filled=10_000_000,  # 10 tokens
            taker_amount_filled=4_000_000,  # 4 USDC -> sell @ 0.40
            fee=100_000,  # 0.10 USDC fee
        )
        assert broker.ingest_reconciled_fills([sell]) == 1
        # Realized: (0.40 - 0.50) * 10 - 0.10 fee = -1.10.
        assert tracker.market_pnl("123") == Decimal("-1.10")

    def test_funder_as_taker_perspective(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, _client = _live_broker(tracker)
        # Funder is the (lowercased) taker: gives takerAsset tokens,
        # receives maker's USDC — a sell of 10 @ 0.50.
        fill = _onchain_fill(
            maker="0xSomeoneElse",
            taker="0xfunder",
            maker_asset_id=0,
            taker_asset_id=456,
            maker_amount_filled=5_000_000,
            taker_amount_filled=10_000_000,
        )
        assert broker.ingest_reconciled_fills([fill]) == 1
        assert tracker.open_positions()["456"].qty == Decimal("-10")

    def test_unrelated_fill_skipped(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, _client = _live_broker(tracker)
        fill = _onchain_fill(maker="0xNotUs", taker="0xAlsoNotUs")
        assert broker.ingest_reconciled_fills([fill]) == 0
        assert tracker.open_positions() == {}

    def test_immediate_fill_not_double_counted_by_reconcile(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = _live_broker(tracker)
        tx = "0x" + "cd" * 32
        order_hash = "0x" + "ab" * 32
        client.post_order.return_value = {
            "success": True,
            "orderID": order_hash,
            "status": "matched",
            "makingAmount": "5",
            "takingAmount": "10",
            "transactionsHashes": [tx],
        }
        broker.place_order(
            Order(
                symbol="123",
                side="buy",
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=0.50,
            )
        )
        # The same match later surfaces on-chain: same tx + order hash
        # (reconciler hex() strings carry no 0x prefix).
        onchain = _onchain_fill(
            order_hash="ab" * 32,
            tx_hash="cd" * 32,
        )
        assert broker.ingest_reconciled_fills([onchain]) == 0
        assert tracker.open_positions()["123"].qty == Decimal("10")


class TestUnfedTrackerWarning:
    """Fail-loud on the wiring gap: live orders while the tracker has
    never seen a fill/mark AND exposure exists -> WARNING w/ counter."""

    def _order(self) -> Order:
        return Order(
            symbol="tok-live",
            side="buy",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=0.50,
        )

    def test_warns_once_not_per_order(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = _live_broker(tracker)
        client.get_orders.return_value = [{"id": "resting-1"}]
        client.get_positions.return_value = []
        client.post_order.return_value = {"success": True, "orderID": "x"}
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                broker.place_order(self._order())
        unfed = [r for r in caplog.records if "no fills or marks" in r.getMessage()]
        assert len(unfed) == 1  # first offense only, not per-order spam

    def test_no_warning_without_exposure(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = _live_broker(tracker)
        client.get_orders.return_value = []
        client.get_positions.return_value = []
        client.post_order.return_value = {"success": True, "orderID": "x"}
        with caplog.at_level(logging.WARNING):
            broker.place_order(self._order())
        assert not [r for r in caplog.records if "no fills or marks" in r.getMessage()]

    def test_no_warning_once_fed(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        tracker.record_mark("tok-live", "0.50")  # any activity counts
        broker, client = _live_broker(tracker)
        client.get_orders.return_value = [{"id": "resting-1"}]
        client.post_order.return_value = {"success": True, "orderID": "x"}
        with caplog.at_level(logging.WARNING):
            broker.place_order(self._order())
        assert not [r for r in caplog.records if "no fills or marks" in r.getMessage()]
