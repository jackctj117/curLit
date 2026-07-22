"""Slippage-limit enforcement at execution (CL-qyav P2).

OrderIntent.max_slippage_bps was journaled but never enforced. Now:

  - OMS copies the intent's limit onto the outgoing Order.
  - OandaBroker computes a direction-aware FOK ``priceBound`` from the live
    reference quote (buy → above ask, sell → below bid) at the venue's own
    quote precision, so fills beyond tolerance are rejected by OANDA itself.
  - PaperBroker simulates the same check against the mid and REJECTS with a
    reject_reason, which the OMS raises through the BrokerRejectedOrderError
    / RejectionHandler path.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.execution.broker import Order, OrderStatus, OrderType
from src.execution.oanda_broker import OandaBroker, _price_bound_str
from src.execution.oms import OrderIntent, OrderManager
from src.execution.paper_broker import PaperBroker

# ---------------------------------------------------------------------------
# OANDA: pure bound math
# ---------------------------------------------------------------------------


class TestPriceBoundStr:
    def test_buy_bound_is_above_reference(self) -> None:
        bound = Decimal(_price_bound_str("buy", "1.10020", 2.0))
        assert bound > Decimal("1.10020")
        # 1.10020 * 1.0002 = 1.10042004 → floor to 5 dp
        assert bound == Decimal("1.10042")

    def test_sell_bound_is_below_reference(self) -> None:
        bound = Decimal(_price_bound_str("sell", "1.10000", 2.0))
        assert bound < Decimal("1.10000")
        # 1.10000 * 0.9998 = 1.09978 → ceiling to 5 dp
        assert bound == Decimal("1.09978")

    def test_precision_follows_reference_quote(self) -> None:
        # JPY-style 3-decimal quote must not gain extra decimals
        # (MARKET_ORDER_PRICE_BOUND_PRECISION_EXCEEDED at the venue).
        bound = _price_bound_str("buy", "155.123", 2.0)
        frac = bound.split(".")[1]
        assert len(frac) == 3

    def test_rounding_never_widens_tolerance(self) -> None:
        # Buy: raw bound floors DOWN toward the reference; sell: ceils UP.
        ref = Decimal("155.123")
        buy = Decimal(_price_bound_str("buy", "155.123", 2.0))
        sell = Decimal(_price_bound_str("sell", "155.123", 2.0))
        assert buy <= ref * (1 + Decimal("2.0") / 10_000)
        assert sell >= ref * (1 - Decimal("2.0") / 10_000)


# ---------------------------------------------------------------------------
# OANDA: payload wiring (no network — transport monkeypatched)
# ---------------------------------------------------------------------------


def _broker() -> OandaBroker:
    b = OandaBroker.__new__(OandaBroker)  # no network in __init__ path
    b.account_id = "ACC"
    return b


def _pricing_client(bid: str, ask: str) -> SimpleNamespace:
    payload = {"prices": [{"bids": [{"price": bid}], "asks": [{"price": ask}]}]}
    return SimpleNamespace(
        get=lambda url, params=None: SimpleNamespace(
            json=lambda: payload, raise_for_status=lambda: None,
        ),
    )


def _capture_post(b: OandaBroker, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    bodies: list[dict] = []

    def fake_post(url: str, body: dict) -> SimpleNamespace:
        bodies.append(body)
        return SimpleNamespace(json=lambda: {"orderFillTransaction": {"id": "1"}})

    monkeypatch.setattr(b, "_post_following_307", fake_post)
    return bodies


class TestOandaPriceBoundPayload:
    def test_buy_includes_bound_above_ask(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        b = _broker()
        b.client = _pricing_client("1.10000", "1.10020")
        bodies = _capture_post(b, monkeypatch)
        order = Order(
            symbol="EUR_USD", side="buy", quantity=1000,
            order_type=OrderType.MARKET, max_slippage_bps=2.0,
        )
        out = b.place_order(order)
        assert out.status == OrderStatus.FILLED
        bound = Decimal(bodies[0]["order"]["priceBound"])
        assert bound > Decimal("1.10020")  # above the ask reference

    def test_sell_includes_bound_below_bid(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        b = _broker()
        b.client = _pricing_client("1.10000", "1.10020")
        bodies = _capture_post(b, monkeypatch)
        order = Order(
            symbol="EUR_USD", side="sell", quantity=1000,
            order_type=OrderType.MARKET, max_slippage_bps=2.0,
        )
        b.place_order(order)
        bound = Decimal(bodies[0]["order"]["priceBound"])
        assert bound < Decimal("1.10000")  # below the bid reference

    def test_no_slippage_cap_means_no_bound(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        b = _broker()
        bodies = _capture_post(b, monkeypatch)
        order = Order(
            symbol="EUR_USD", side="buy", quantity=1000,
            order_type=OrderType.MARKET,
        )
        b.place_order(order)
        assert "priceBound" not in bodies[0]["order"]

    def test_pricing_failure_places_without_bound(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Fail-open by design: this path also carries kill-switch de-risking
        # orders; a /pricing hiccup must not block a flatten.
        b = _broker()

        def boom(url: str, params: dict | None = None) -> SimpleNamespace:
            raise OSError("pricing endpoint down")

        b.client = SimpleNamespace(get=boom)
        bodies = _capture_post(b, monkeypatch)
        order = Order(
            symbol="EUR_USD", side="buy", quantity=1000,
            order_type=OrderType.MARKET, max_slippage_bps=2.0,
        )
        out = b.place_order(order)
        assert out.status == OrderStatus.FILLED
        assert "priceBound" not in bodies[0]["order"]


# ---------------------------------------------------------------------------
# PaperBroker: simulated enforcement
# ---------------------------------------------------------------------------


class TestPaperBrokerSlippage:
    def test_buy_beyond_bound_rejected(self) -> None:
        broker = PaperBroker()
        # Wide book: mid 1.1000, ask 9.09 bps above mid — beyond 2 bps.
        broker.set_price("EURUSD", 1.0990, 1.1010)
        order = Order(
            symbol="EURUSD", side="buy", quantity=1000,
            order_type=OrderType.MARKET, max_slippage_bps=2.0,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.REJECTED
        assert "SLIPPAGE_EXCEEDED" in (out.reject_reason or "")
        assert broker.get_positions() == []  # nothing filled

    def test_sell_beyond_bound_rejected(self) -> None:
        broker = PaperBroker()
        broker.set_price("EURUSD", 1.0990, 1.1010)
        order = Order(
            symbol="EURUSD", side="sell", quantity=1000,
            order_type=OrderType.MARKET, max_slippage_bps=2.0,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.REJECTED
        assert "SLIPPAGE_EXCEEDED" in (out.reject_reason or "")

    def test_within_tolerance_fills(self) -> None:
        broker = PaperBroker()
        # Half-spread ≈ 0.9 bps < 2 bps tolerance.
        broker.set_price("EURUSD", 1.1000, 1.1002)
        order = Order(
            symbol="EURUSD", side="buy", quantity=1000,
            order_type=OrderType.MARKET, max_slippage_bps=2.0,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.FILLED

    def test_no_cap_never_rejects(self) -> None:
        broker = PaperBroker()
        broker.set_price("EURUSD", 1.0900, 1.1100)  # absurd spread
        order = Order(
            symbol="EURUSD", side="buy", quantity=1000,
            order_type=OrderType.MARKET,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# OMS wiring: intent limit reaches the order; reject flows to the handler
# ---------------------------------------------------------------------------


class _RecordingHandler:
    """Minimal RejectionHandler double — records the exception, no retry."""

    def __init__(self) -> None:
        self.exceptions: list[BaseException] = []

    def handle(self, intent, order, exc, attempt):  # noqa: ANN001, ANN201
        self.exceptions.append(exc)
        return SimpleNamespace(
            should_retry=False,
            halt_strategy=False,
            sleep_sec=0.0,
            next_size_fraction=1.0,
            final_resolution=SimpleNamespace(value="abort"),
        )

    def sleep(self, sec: float) -> None:  # pragma: no cover - not retried
        pass


class TestOmsSlippageWiring:
    def test_intent_limit_copied_onto_order(self) -> None:
        broker = PaperBroker()
        broker.set_price("EURUSD", 1.1000, 1.1002)
        seen: list[Order] = []
        original = broker.place_order

        def spy(order: Order) -> Order:
            seen.append(order)
            return original(order)

        broker.place_order = spy  # type: ignore[method-assign]
        oms = OrderManager(broker)
        oms.submit_intent(OrderIntent(
            strategy_id="s", symbol="EURUSD", target_position=500,
            max_slippage_bps=7.5,
        ))
        assert seen and seen[0].max_slippage_bps == 7.5

    def test_slippage_reject_reaches_rejection_handler(self) -> None:
        from src.execution.broker import BrokerRejectedOrderError

        broker = PaperBroker()
        broker.set_price("EURUSD", 1.0990, 1.1010)  # beyond 2 bps default
        handler = _RecordingHandler()
        oms = OrderManager(broker, rejection_handler=handler)
        oms.submit_intent(OrderIntent(
            strategy_id="s", symbol="EURUSD", target_position=500,
        ))
        assert len(handler.exceptions) == 1
        exc = handler.exceptions[0]
        assert isinstance(exc, BrokerRejectedOrderError)
        assert "SLIPPAGE_EXCEEDED" in str(exc)
        assert broker.get_positions() == []
