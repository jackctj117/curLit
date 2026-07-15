"""Tests for Polymarket per-market + per-day loss caps (CL-983f).

Tracker math (realized + mark-to-market), the order gate (breach ->
LossCapExceededError, under-cap -> allowed, per-market isolation, UTC-day
rollover, reduce-only exemption), and the broker wiring — paper broker
records/gates automatically; live broker gates before anything is
signed (mocked ClobClient, no network).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.execution.broker import Order, OrderStatus, OrderType
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
) -> PolymarketLossCapTracker:
    return PolymarketLossCapTracker(
        config=LossCapConfig(
            per_market_loss_cap_usd=Decimal(market_cap),
            per_day_loss_cap_usd=Decimal(day_cap),
        ),
        clock=clock or _FakeClock(),
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


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LossCapConfig()
        assert cfg.per_market_loss_cap_usd == Decimal("25")
        assert cfg.per_day_loss_cap_usd == Decimal("50")

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


class TestPaperBrokerWiring:
    def test_breach_refuses_new_order(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        tracker.record_fill("tok-1", "buy", 100, "0.50")
        tracker.record_mark("tok-1", "0.20")  # -30: breached
        broker = _paper_broker(tracker)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        )
        with pytest.raises(LossCapExceededError, match="loss cap breached"):
            broker.place_order(order)
        assert broker.get_positions() == []

    def test_under_cap_order_fills(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker = _paper_broker(tracker)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.FILLED

    def test_paper_fills_feed_the_tracker(self) -> None:
        # Paper runs must exercise the same accounting the live path
        # uses — the broker records its own fills.
        tracker = _tracker(market_cap="25", day_cap="50")
        broker = _paper_broker(tracker)
        broker.place_order(Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        ))
        assert broker.loss_caps is tracker
        assert tracker._markets["tok-1"].qty == Decimal("10")  # noqa: SLF001

    def test_reducing_exit_allowed_when_breached(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        broker = _paper_broker(tracker)
        broker.place_order(Order(
            symbol="POLY:tok-1", side="buy", quantity=100,
            order_type=OrderType.LIMIT, limit_price=0.50,
        ))
        tracker.record_mark("tok-1", "0.20")  # -30: breached
        out = broker.place_order(Order(
            symbol="POLY:tok-1", side="sell", quantity=100,
            order_type=OrderType.LIMIT, limit_price=0.49,
        ))
        assert out.status == OrderStatus.FILLED


class TestLiveBrokerWiring:
    def _live_broker(self, tracker: PolymarketLossCapTracker):
        from src.execution.polymarket_secrets import PolymarketCreds

        creds = PolymarketCreds(
            signer_pk="0x" + "11" * 32,
            api_key="k", api_secret="s", api_passphrase="p",
            funder_address="0xFunder", chain_id=80002,
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

    def test_breach_refuses_before_signing(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="500")
        tracker.record_fill("tok-live", "buy", 100, "0.50")
        tracker.record_mark("tok-live", "0.20")  # -30: breached
        broker, client = self._live_broker(tracker)

        order = Order(
            symbol="tok-live", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        )
        with pytest.raises(LossCapExceededError, match="loss cap breached"):
            broker.place_order(order)
        client.create_order.assert_not_called()
        client.post_order.assert_not_called()

    def test_under_cap_order_submits(self) -> None:
        tracker = _tracker(market_cap="25", day_cap="50")
        broker, client = self._live_broker(tracker)
        client.post_order.return_value = {"success": True, "orderID": "ord-9"}

        order = Order(
            symbol="tok-live", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.PENDING
        assert out.order_id == "ord-9"
        client.create_order.assert_called_once()

    def test_day_breach_blocks_other_market_on_live(self) -> None:
        tracker = _tracker(market_cap="500", day_cap="50")
        tracker.record_fill("tok-a", "buy", 200, "0.50")
        tracker.record_fill("tok-a", "sell", 200, "0.20")  # -60 day
        broker, client = self._live_broker(tracker)

        order = Order(
            symbol="tok-other", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        )
        with pytest.raises(LossCapExceededError, match="scope=day"):
            broker.place_order(order)
        client.create_order.assert_not_called()

    def test_tracker_exposed_for_fill_loop(self) -> None:
        tracker = _tracker()
        broker, _client = self._live_broker(tracker)
        assert broker.loss_caps is tracker
