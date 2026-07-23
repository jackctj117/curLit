"""Order Management System — intent to orders, retry, rejection policy.

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

import asyncio
import enum
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .broker import (
    Broker,
    BrokerRejectedOrderError,
    Order,
    OrderStatus,
    OrderType,
    canonical_symbol,
)
from .trade_journal import EventType, TradeJournal

logger = logging.getLogger(__name__)


class Urgency(enum.StrEnum):
    """Canonical intent-urgency vocabulary (CL-ikz2; review §6.1.2/§9.1).

    Declared in ESCALATION ORDER (least → most urgent): the portfolio
    coordinator ranks same-symbol escalation by declaration order, so any
    string outside this set can never escalate a netted intent. That is
    exactly how the legacy event-exit ``"high"`` silently lost escalation
    — emit ``Urgency.<LEVEL>.value``, never ad-hoc strings.
    """

    PASSIVE = "passive"
    NORMAL = "normal"
    URGENT = "urgent"


@dataclass
class OrderIntent:
    strategy_id: str
    symbol: str
    target_position: float
    # Kept a plain str for wire/journal compatibility; canonical values
    # are the Urgency enum members above.
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
        # RLock: halt_new_trades now takes the lock (CL-8lv6 TOCTOU) and
        # may be reached from callbacks fired while submit_intent holds
        # it — re-entrancy beats a deadlock footgun.
        self._lock = threading.RLock()
        # CL-sbrp: per-canonical-symbol reservation. Two concurrent submits for
        # the SAME symbol on different threads (a strategy entry and a
        # kill-switch flatten) must not both size from the same pre-fill book
        # and double the position. The reservation is held ACROSS place_order;
        # different symbols never block each other (so the CL-8s2a win — a
        # cross-symbol emergency de-risk not queuing behind a slow order —
        # stands). Bounded by the instrument universe, so no cleanup needed.
        self._inflight_guard = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}
        self._halted = False
        self._pending: dict[str, list[Order]] = {}

    def submit_intent(
        self,
        intent: OrderIntent,
        *,
        bypass_halt: bool = False,
        positions: list[Any] | None = None,
    ) -> str:
        """Convert one intent into a broker order (delta vs current position).

        bypass_halt (CL-i4tx): reserved for the risk layer's emergency
        exposure-REDUCING intents (kill-switch flatten_all / reduce_50pct).
        "Halt new trades" must never block de-risking — e.g. the trailing
        stop halts at -20%, then drawdown_limit needs to flatten at -40%.
        Strategy paths must never set it.

        positions (CL-a0sv): optional pre-fetched broker snapshot. The
        coordinator submits a post-aggregation batch (max ONE intent per
        symbol), so one snapshot is delta-accurate for the whole batch —
        reusing it drops N per-intent get_positions round-trips. When
        None, fetch live (correct default for standalone callers).

        Lock width (CL-8s2a): the DECISION (delta compute + halt gate +
        INTENT_SUBMITTED journal + min-size skip) runs under the lock so a
        health-tick halt racing a strategy place still serializes and the
        halt-TOCTOU fix (CL-8lv6) holds. But the BLOCKING part — place_order
        HTTP plus the RejectionHandler retry-sleep loop — is released from the
        lock: previously an emergency de-risk (bypass_halt) could queue for
        seconds behind a stuck/retrying strategy order that held the RLock the
        whole time. _pending is re-acquired under the lock for its own writes
        inside _submit_with_retry, so shutdown-drain accounting stays correct.
        """
        # CL-sbrp: reserve this symbol for the whole submit (delta + place).
        # If another submit for the same symbol is in flight, block until it
        # reaches a terminal state, THEN recompute our delta against the now
        # post-fill book — the passed snapshot predates that order and is stale.
        csym = canonical_symbol(intent.symbol)
        with self._inflight_guard:
            sym_lock = self._inflight.setdefault(csym, threading.Lock())
        contended = not sym_lock.acquire(blocking=False)
        if contended:
            sym_lock.acquire()
        try:
            return self._submit_reserved(
                intent, bypass_halt=bypass_halt,
                positions=None if contended else positions,
            )
        finally:
            sym_lock.release()

    def _submit_reserved(
        self,
        intent: OrderIntent,
        *,
        bypass_halt: bool = False,
        positions: list[Any] | None = None,
    ) -> str:
        """submit_intent body, run while holding the per-symbol reservation
        (CL-sbrp). ``positions`` is forced to None when the reservation was
        contended, so the delta is recomputed from a fresh, post-fill book."""
        with self._lock:
            # Position matching MUST use the canonical key (CL-qqra): broker
            # positions come back compact ("USDCAD") while event intents are
            # OANDA-underscore ("USD_CAD"). A raw .get() always missed →
            # exit deltas of 0 (positions never closed at the broker) and
            # entries that stacked on an existing position. Routing below
            # still uses intent.symbol (broker _to_oanda is idempotent).
            snapshot = (
                positions if positions is not None
                else self.broker.get_positions()
            )
            current_positions = {
                canonical_symbol(p.symbol): p.quantity for p in snapshot
            }
            current_qty = current_positions.get(canonical_symbol(intent.symbol), 0.0)
            delta = intent.target_position - current_qty

            # "Halt new trades" means exactly that (CL-8lv6): risk-REDUCING
            # intents still pass — a halted OMS must never trap a strategy's
            # exit while its book already closed (broker keeps the risk, book
            # says flat). Reducing = smaller absolute size, same side (or
            # flat); flips and adds are blocked. This gate MUST stay inside
            # the lock (halt-TOCTOU, CL-8lv6): halt_new_trades takes the same
            # lock, so the flag read and the place decision are atomic.
            if self._halted and not bypass_halt:
                reducing = (
                    abs(intent.target_position) < abs(current_qty)
                    and (intent.target_position == 0.0
                         or intent.target_position * current_qty > 0)
                )
                if not reducing:
                    logger.warning(
                        "OMS halted — rejecting non-reducing intent %s "
                        "(%s target=%.4f current=%.4f)",
                        intent.intent_id, intent.symbol,
                        intent.target_position, current_qty,
                    )
                    return intent.intent_id
                logger.warning(
                    "OMS halted — allowing risk-REDUCING intent %s "
                    "(%s target=%.4f current=%.4f)",
                    intent.intent_id, intent.symbol,
                    intent.target_position, current_qty,
                )

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

        # Lock RELEASED before the blocking submit (CL-8s2a): place_order HTTP
        # + RejectionHandler retry sleeps no longer hold the RLock, so a
        # concurrent bypass_halt emergency de-risk isn't queued behind a slow
        # strategy order. _submit_with_retry re-acquires the lock only for its
        # short _pending mutations.
        self._submit_with_retry(intent, side, abs(delta),
                                emergency=bypass_halt)
        return intent.intent_id

    async def submit_intent_async(
        self,
        intent: OrderIntent,
        *,
        bypass_halt: bool = False,
        positions: list[Any] | None = None,
    ) -> str:
        """Async wrapper around submit_intent for event-loop callers (CL-xdnh).

        submit_intent does synchronous broker HTTP (get_positions +
        place_order, with retry sleeps) — calling it directly from a
        coroutine blocks the event loop for the full round-trip. This
        offloads the whole submission (including the OMS lock) to a worker
        thread; submit_intent stays as-is for sync callers (web API, risk
        kill switches, reconciler).
        """
        return await asyncio.to_thread(
            self.submit_intent, intent,
            bypass_halt=bypass_halt, positions=positions,
        )

    def _submit_with_retry(
        self,
        intent: OrderIntent,
        side: str,
        original_qty: float,
        *,
        emergency: bool = False,
    ) -> None:
        """Place the order, applying RejectionHandler policy on broker failures."""
        attempt = 1
        size_fraction = 1.0
        while True:
            if attempt > 1:
                # CL-a13q (P1): a TRANSIENT retry may follow a fill-then-timeout
                # — the order actually filled at OANDA but the HTTP read timed
                # out and was classified retryable. OANDA has no client
                # idempotency key (unlike Alpaca's client_order_id), so before
                # re-submitting, re-read the book: if the position already
                # reached the target, the prior attempt filled — abort rather
                # than double the position. (Held under the per-symbol
                # reservation from CL-sbrp, so no concurrent submit races this
                # read.) If the fill isn't visible yet, fall through and retry
                # as before — strictly no worse than the old behavior.
                try:
                    fresh = {
                        canonical_symbol(p.symbol): p.quantity
                        for p in self.broker.get_positions()
                    }
                    residual = intent.target_position - fresh.get(
                        canonical_symbol(intent.symbol), 0.0,
                    )
                    if abs(residual) < self._min_trade_size(intent.symbol):
                        logger.warning(
                            "Retry ABORTED for %s — position already at target "
                            "%.4f (a prior attempt filled despite the error); "
                            "not double-submitting",
                            intent.symbol, intent.target_position,
                        )
                        return
                except Exception:
                    logger.debug(
                        "retry re-read failed for %s; proceeding with retry",
                        intent.symbol, exc_info=True,
                    )
            qty = original_qty * size_fraction
            order = Order(
                symbol=intent.symbol,
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
                # Enforced at the venue (CL-qyav): OANDA turns this into a
                # FOK priceBound; PaperBroker simulates the same check. The
                # intent's limit was previously journaled but never enforced.
                max_slippage_bps=intent.max_slippage_bps,
                # Kill-switch de-risk orders (bypass_halt) may place UNBOUND
                # when the slippage reference is unavailable — getting flat
                # beats slippage protection. Normal orders fail closed
                # (CL-8lv6).
                emergency=emergency,
            )
            try:
                placed = self.broker.place_order(order)
                if placed.status == OrderStatus.REJECTED:
                    # Ultrareview #2: a REJECTED status must flow through the
                    # SAME rejection policy as a transport exception — before
                    # this it entered _pending forever (poisoning
                    # has_pending), was journaled ORDER_PLACED, and never
                    # reached the RejectionHandler's halved-retry/halt logic.
                    raise BrokerRejectedOrderError(
                        placed.reject_reason or "broker rejected order",
                    )
                # _pending means "submitted, not yet terminal" (CL-8lv6):
                # a synchronously-FILLED order must NOT linger — it poisoned
                # has_pending() forever and graceful_shutdown always burned
                # its full drain timeout. Guarded by a SHORT re-acquire of the
                # lock (CL-8s2a): submit_intent released it before this
                # blocking loop, so the _pending write must re-lock to stay
                # consistent with has_pending()/shutdown drain and concurrent
                # submitters. RLock → safe even if a callback path re-enters.
                with self._lock:
                    if placed.status == OrderStatus.FILLED:
                        self._pending.pop(intent.intent_id, None)
                    else:
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

    # NOTE (CL-i4tx): the old OrderManager.reconcile() was deleted. It
    # flagged EVERY non-zero broker position as "POSITION MISMATCH" at
    # CRITICAL every 300s without comparing any internal book — a false
    # positive machine that trained operators to ignore real desync. The
    # live engine's periodic alignment check now delegates to
    # src/portfolio/reconciler.PositionReconciler.check_alignment().

    def halt_new_trades(self) -> None:
        # Under the same lock that guards submit_intent's halt check —
        # a health-tick halt racing a strategy place must serialize
        # (CL-8lv6 TOCTOU).
        with self._lock:
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
