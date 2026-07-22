"""Order Management System — intent to orders, retry, reconciliation.

Rejection handling: when broker.place_order raises, an optional RejectionHandler
classifies the failure and decides retry/halve/abort/halt-strategy per
docs/runbooks/OrderRejected.md. Without a handler attached, OMS falls back to
the legacy log-and-drop behavior to preserve backward compatibility for tests
and ad-hoc usage.

Audit trail: when a TradeJournal is supplied, every intent emits
INTENT_SUBMITTED on entry, ORDER_PLACED after broker.place_order returns, and
ORDER_FILLED if the broker reports FILLED status synchronously (paper broker;
OANDA fills arrive async via stream and would need a separate fill-stream
callback to emit ORDER_FILLED). Without a journal, OMS is silent — the
journal is optional so unit tests and ad-hoc usage don't require a DB.
"""

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .broker import Broker, Order, OrderStatus, OrderType, canonical_symbol
from .trade_journal import EventType, TradeJournal

logger = logging.getLogger(__name__)


@dataclass
class OrderIntent:
    strategy_id: str
    symbol: str
    target_position: float
    urgency: str = "normal"
    # 2 bps = typical OANDA spread on majors at normal liquidity. Above
    # this, refuse the fill rather than chase a runaway book. Strategies
    # with looser tolerances (event-driven, breakout) bump this in their
    # OrderIntent construction.
    max_slippage_bps: float = 2.0
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Free-form per-intent metadata. Strategies attach feature-snapshot ids
    # (CL-xpw9) here so the trade journal can record reproducibility info
    # without coupling OrderIntent's schema to feature-versioning internals.
    metadata: dict[str, Any] = field(default_factory=dict)


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        rejection_handler: Any | None = None,
        on_strategy_halt: Callable[[str, str], None] | None = None,
        journal: TradeJournal | None = None,
    ) -> None:
        """Construct an OMS.

        rejection_handler: optional RejectionHandler driving classified retry
        behavior. When None, place_order failures are logged and the intent is
        dropped (legacy behavior).
        on_strategy_halt: optional callback invoked with (strategy_id, reason)
        when a rejection class demands the strategy be halted (e.g. instrument
        halted indefinitely). Caller is responsible for actually pausing the
        strategy.
        journal: optional TradeJournal — every intent / placement / fill event
        is appended for audit. None disables journaling (used in unit tests).
        """
        self.broker = broker
        self.rejection_handler = rejection_handler
        self.on_strategy_halt = on_strategy_halt
        self.journal = journal
        self._lock = threading.Lock()
        self._halted = False
        self._pending: dict[str, list[Order]] = {}

    def submit_intent(self, intent: OrderIntent) -> str:
        with self._lock:
            if self._halted:
                logger.warning("OMS halted — rejecting intent %s", intent.intent_id)
                return intent.intent_id

            # Position matching MUST use the canonical key (CL-qqra): broker
            # positions come back compact ("USDCAD") while event intents are
            # OANDA-underscore ("USD_CAD"). A raw .get() always missed →
            # exit deltas of 0 (positions never closed at the broker) and
            # entries that stacked on an existing position. Routing below
            # still uses intent.symbol (broker _to_oanda is idempotent).
            current_positions = {
                canonical_symbol(p.symbol): p.quantity
                for p in self.broker.get_positions()
            }
            current_qty = current_positions.get(canonical_symbol(intent.symbol), 0.0)
            delta = intent.target_position - current_qty

            intent_payload: dict[str, Any] = {
                "target_position": intent.target_position,
                "current_position": current_qty,
                "delta": delta,
                "urgency": intent.urgency,
                "max_slippage_bps": intent.max_slippage_bps,
            }
            # Strategy-attached metadata (e.g. feature snapshot id from
            # CL-xpw9) flows through to the journal so reconstruct_features
            # can find the snapshot from the intent's audit row.
            if intent.metadata:
                intent_payload.update(intent.metadata)
            self._journal_event(
                EventType.INTENT_SUBMITTED,
                intent=intent,
                payload=intent_payload,
            )

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
                self._journal_event(
                    EventType.ORDER_PLACED,
                    intent=intent,
                    payload={
                        "side": side,
                        "quantity": qty,
                        "attempt": attempt,
                        "size_fraction": size_fraction,
                        "order_type": order.order_type.value,
                    },
                )
                # Synchronous fill (PaperBroker) — emit ORDER_FILLED now. For
                # OANDA, fills arrive via stream and would need a separate
                # fill-stream wiring; ORDER_PLACED is all OMS sees here.
                if placed.status == OrderStatus.FILLED:
                    self._journal_event(
                        EventType.ORDER_FILLED,
                        intent=intent,
                        payload={
                            "side": side,
                            "quantity": qty,
                            "attempt": attempt,
                        },
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

    def reconcile(self) -> dict[str, dict[str, Any]]:
        broker_pos = {p.symbol: p.quantity for p in self.broker.get_positions()}
        mismatches: dict[str, dict[str, Any]] = {}
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

    def _journal_event(
        self,
        event_type: EventType,
        intent: OrderIntent,
        payload: dict[str, Any],
    ) -> None:
        """Append one event to the trade journal — silent no-op without one.

        Journal failures must NOT propagate: trading is more important than
        bookkeeping, and a DB hiccup must not drop or duplicate orders.
        """
        if self.journal is None:
            return
        try:
            self.journal.record(
                event_type=event_type,
                payload=payload,
                intent_id=intent.intent_id,
                strategy_id=intent.strategy_id,
                symbol=intent.symbol,
            )
        except Exception:
            logger.exception(
                "Trade journal append failed (event=%s intent=%s) — continuing",
                event_type.value, intent.intent_id,
            )
