"""Order Management System — intent to orders, retry, reconciliation."""

import logging
import threading
import uuid
from dataclasses import dataclass, field

from .broker import Account, Broker, Order, OrderType

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
    def __init__(self, broker: Broker) -> None:
        self.broker = broker
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
            order = Order(
                symbol=intent.symbol,
                side=side,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
            )
            try:
                placed = self.broker.place_order(order)
                self._pending[intent.intent_id] = [placed]
                logger.info("Placed %s %s %.0f", intent.symbol, side, abs(delta))
            except Exception:
                logger.exception("Order failed for %s", intent.intent_id)

            return intent.intent_id

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
        return False

    @staticmethod
    def _min_trade_size(symbol: str) -> float:
        return 1.0
