"""Order Management System — intent to orders, retry, reconciliation.

Rejection handling: when broker.place_order raises, an optional RejectionHandler
classifies the failure and decides retry/halve/abort/halt-strategy per
docs/runbooks/OrderRejected.md. Without a handler attached, OMS falls back to
the legacy log-and-drop behavior to preserve backward compatibility for tests
and ad-hoc usage.
"""

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .broker import Broker, Order, OrderType

logger = logging.getLogger(__name__)


@dataclass
class OrderIntent:
    strategy_id: str
    symbol: str
    target_position: float
    urgency: str = "normal"
    max_slippage_bps: float = 2.0
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        rejection_handler: Any | None = None,
        on_strategy_halt: Callable[[str, str], None] | None = None,
    ) -> None:
        """Construct an OMS.

        rejection_handler: optional RejectionHandler driving classified retry
        behavior. When None, place_order failures are logged and the intent is
        dropped (legacy behavior).
        on_strategy_halt: optional callback invoked with (strategy_id, reason)
        when a rejection class demands the strategy be halted (e.g. instrument
        halted indefinitely). Caller is responsible for actually pausing the
        strategy.
        """
        self.broker = broker
        self.rejection_handler = rejection_handler
        self.on_strategy_halt = on_strategy_halt
        self._lock = threading.Lock()
        self._halted = False
        self._pending: dict[str, list[Order]] = {}

    def submit_intent(self, intent: OrderIntent) -> str:
        with self._lock:
            if self._halted:
                logger.warning("OMS halted — rejecting intent %s", intent.intent_id)
                return intent.intent_id

            current_positions = {p.symbol: p.quantity for p in self.broker.get_positions()}
            current_qty = current_positions.get(intent.symbol, 0.0)
            delta = intent.target_position - current_qty

            if abs(delta) < self._min_trade_size(intent.symbol):
                return intent.intent_id

            side = "buy" if delta > 0 else "sell"
            self._submit_with_retry(intent, side, abs(delta))
            return intent.intent_id

    def _submit_with_retry(
        self,
        intent: OrderIntent,
        side: str,
        original_qty: float,
    ) -> None:
        """Place the order, applying RejectionHandler policy on broker failures."""
        attempt = 1
        size_fraction = 1.0
        while True:
            qty = original_qty * size_fraction
            order = Order(
                symbol=intent.symbol,
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
            )
            try:
                placed = self.broker.place_order(order)
                self._pending[intent.intent_id] = [placed]
                logger.info(
                    "Placed %s %s %.4f (attempt=%d, fraction=%.2f)",
                    intent.symbol, side, qty, attempt, size_fraction,
                )
                return
            except Exception as exc:
                if self.rejection_handler is None:
                    # Legacy behavior: log and drop.
                    logger.exception("Order failed for %s", intent.intent_id)
                    return

                outcome = self.rejection_handler.handle(
                    intent=intent,
                    order=order,
                    exc=exc,
                    attempt=attempt,
                )

                if outcome.halt_strategy and self.on_strategy_halt is not None:
                    try:
                        self.on_strategy_halt(intent.strategy_id, intent.symbol)
                    except Exception:
                        logger.exception(
                            "on_strategy_halt callback failed for %s",
                            intent.strategy_id,
                        )

                if not outcome.should_retry:
                    logger.warning(
                        "Giving up on %s after %d attempts (resolution=%s)",
                        intent.intent_id, attempt, outcome.final_resolution.value,
                    )
                    return

                if outcome.sleep_sec > 0:
                    self.rejection_handler.sleep(outcome.sleep_sec)
                size_fraction = outcome.next_size_fraction
                attempt += 1

    def reconcile(self) -> dict[str, dict]:
        broker_pos = {p.symbol: p.quantity for p in self.broker.get_positions()}
        mismatches = {}
        for sym in broker_pos:
            if abs(broker_pos[sym]) > 1e-6:
                mismatches.setdefault(sym, {})
        if mismatches:
            logger.critical("POSITION MISMATCH: %s", mismatches)
        return mismatches

    def halt_new_trades(self) -> None:
        self._halted = True
        logger.warning("OMS: new trades halted")

    def has_pending(self) -> bool:
        return len(self._pending) > 0

    def resume_trades(self) -> None:
        with self._lock:
            self._halted = False
            logger.info("OMS: trades resumed")

    @staticmethod
    def _min_trade_size(symbol: str) -> float:
        return 1.0
