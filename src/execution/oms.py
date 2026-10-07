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

#: CL-vj74: cap on the fill-dedup set. Redelivered fills are always recent
#: (a stream reconnect resumes near now), so evicting the OLDEST ids past this
#: bound is safe. Sized far above any realistic per-session fill count, and
#: at least the catch-up's per-reconnect replay bound (fill_stream: 20 pages x
#: 500 transactions), so a replay's ids are still remembered when the same
#: fills arrive on the live stream.
_SEEN_FILLS_CAP = 10_000


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


class SubmissionStatus(enum.StrEnum):
    """CL-o9sq: acknowledgement is not execution, and a timeout is not rejection."""

    FILLED = "filled"
    AT_TARGET = "at_target"
    WORKING = "working"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    UNKNOWN = "submission_unknown"


@dataclass(frozen=True)
class SubmissionResult:
    intent_id: str
    status: SubmissionStatus
    target_reached: bool = False
    order_id: str | None = None
    #: CL-pksi: absolute units of the order actually sent to the broker (None
    #: when nothing was sent). Lets the risk layer compare cumulative fills
    #: against what was requested instead of trusting a status label.
    requested_qty: float | None = None
    #: CL-pksi: absolute units the broker REPORTED as executed by a synchronous
    #: fill (None = not reported). Never inferred from the requested size.
    filled_qty: float | None = None


@dataclass(frozen=True)
class FillOutcome:
    """Result of :meth:`OrderManager.process_fill` (CL-pksi catch-up)."""

    #: First delivery of this venue fill (post-dedup).
    newly_processed: bool
    #: The fill's ORDER_FILLED row is known to be journaled (or there is no
    #: journal): a durable transaction checkpoint may advance past it.
    durable: bool
    #: The fill's journal state is UNKNOWN and its ORDER_FILLED append was
    #: deferred to a journal-checked replay (integration review r3): the
    #: durable checkpoint must be rewound below it.
    deferred: bool = False


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        rejection_handler: Any | None = None,
        on_strategy_halt: Callable[[str, str], None] | None = None,
        journal: TradeJournal | None = None,
        halt_store: Any | None = None,
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
        # CL-0deu.2: the DURABLE account-wide halt (src.risk.trading_halt).
        # ``_halted`` above stays the engine-local brake (kill switches, cold
        # start); the store is shared with the Alpaca books, survives restarts
        # and fails closed when unreadable. None = not wired (unit tests and
        # legacy callers keep the previous behavior).
        self.halt_store = halt_store
        # Non-reducing placements that passed the gate but whose place_order
        # has not returned yet. The durable halt may only be ACKNOWLEDGED when
        # this is zero, so "applied" proves no pre-halt entry is in flight.
        self._inflight_entries = 0
        self._pending: dict[str, list[Order]] = {}
        # CL-vj74: intent kept alongside a PENDING order so an async
        # ORDER_FILL (OANDA transaction stream) can journal ORDER_FILLED and
        # clear the pending entry by client id; _seen_fills dedups by the
        # venue transaction id (the stream can redeliver on reconnect).
        self._pending_intents: dict[str, OrderIntent] = {}
        self._seen_fills: set[str] = set()
        # Insertion order of _seen_fills, so the cap evicts the OLDEST ids
        # (CL-pksi: a replay overlaps the live stream by recent ids only).
        self._seen_order: dict[str, None] = {}
        # CL-pksi catch-up: fills whose ORDER_FILLED journal append FAILED.
        # Their ids are NOT kept in ``_seen_fills`` (so a replay from the
        # durable stream checkpoint can journal them again); the resolved
        # intent is kept here so the retry keeps its attribution, and the
        # listeners — which already ran — are not fired twice.
        self._unjournaled_fills: dict[str, OrderIntent | None] = {}
        # Fills whose journal state was UNKNOWN when they arrived live (lookup
        # failed): claimed and fed to listeners, never appended blind; the
        # next journal-checked replay appends them only if no row exists.
        # Value: (intent, the fill itself) — the fill is kept so a replay can
        # journal it even when the venue page no longer contains it.
        self._deferred_fills: dict[str, tuple[OrderIntent | None, dict[str, Any]]] = {}
        # Fill ids claimed by a delivery whose journal append is in progress.
        self._fills_journaling: set[str] = set()
        # CL-80tv: per-canonical-symbol change counter, bumped (under _lock)
        # whenever a fill for that symbol is processed. Broker position reads
        # run OUTSIDE _lock; a submit captures the counter before its read
        # and re-checks it under _lock before deciding, so a fill that landed
        # during the read is detected instead of sizing off a stale book.
        self._symbol_versions: dict[str, int] = {}
        # CL-pksi: consumers of NEWLY processed streamed fills (the kill-switch
        # manager resolves emergency-order fences from them). Called after the
        # transaction-id dedup above, so a redelivered fill never re-fires.
        self._fill_listeners: list[Callable[[dict[str, Any]], None]] = []
        # CL-pksi: canonical symbols with an unresolved emergency order at the
        # broker. EVERY submit for a fenced symbol is refused (strategy,
        # reconciler, manual and emergency alike): sizing a new order off a
        # snapshot that may not yet include that order's fill can duplicate
        # or reverse the close. Value = (reason, owner intent id): the owner is
        # the emergency intent being submitted under the fence (None once its
        # outcome is known to be unresolved — then nobody is exempt).
        self._symbol_fences: dict[str, tuple[str, str | None]] = {}
        # CL-pksi: account-wide submission block (reason) — set when the
        # emergency-attempt record could not be recovered, so unknown
        # outstanding emergency orders cannot be duplicated by ANY writer.
        self._submission_block: str | None = None

    #: CL-80tv: position reads per submit when a fill for the same symbol is
    #: processed during the (lock-free) broker read. One re-read covers the
    #: usual case — a fill already in flight when the read started; a second
    #: change means the book is moving under us, so the intent is refused
    #: (BLOCKED, nothing sent) instead of sized off a stale snapshot.
    _POSITION_READ_ATTEMPTS = 2

    def submit_intent(
        self,
        intent: OrderIntent,
        *,
        bypass_halt: bool = False,
        positions: list[Any] | None = None,
    ) -> str:
        """Legacy ID-only interface; the returned ID is NOT proof of execution."""
        return self.submit_intent_result(
            intent, bypass_halt=bypass_halt, positions=positions
        ).intent_id

    def submit_intent_result(
        self,
        intent: OrderIntent,
        *,
        bypass_halt: bool = False,
        positions: list[Any] | None = None,
    ) -> SubmissionResult:
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
                intent,
                bypass_halt=bypass_halt,
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
    ) -> SubmissionResult:
        """submit_intent body, run while holding the per-symbol reservation
        (CL-sbrp). ``positions`` is forced to None when the reservation was
        contended, so the delta is recomputed from a fresh, post-fill book."""
        csym = canonical_symbol(intent.symbol)
        # Cheap pre-check so a blocked/fenced submit never pays for a broker
        # read. Re-checked under the lock at decision time (below), because a
        # fence can be set while the read is in flight.
        with self._lock:
            refused = self._refusal_locked(intent)
        if refused is not None:
            return refused
        decision: SubmissionResult | tuple[str, float, bool] | None = None
        for read_attempt in range(1, self._POSITION_READ_ATTEMPTS + 1):
            version: int | None = None
            if positions is not None:
                snapshot = positions
            else:
                # CL-80tv: the broker read (HTTP, possibly slow) runs OUTSIDE
                # _lock — holding it here stalled every halt_new_trades(),
                # fill and other-symbol submit behind one slow GET. The
                # per-symbol reservation (CL-sbrp) still excludes every other
                # OMS submit for this symbol; the only same-symbol state change
                # that can land meanwhile is a processed fill, which bumps the
                # symbol version re-checked below.
                with self._lock:
                    version = self._symbol_versions.get(csym, 0)
                logger.debug(
                    "OMS: reading broker positions for %s outside the state lock "
                    "(intent %s, attempt %d/%d, version %d)",
                    csym,
                    intent.intent_id,
                    read_attempt,
                    self._POSITION_READ_ATTEMPTS,
                    version,
                )
                snapshot = self.broker.get_positions()
            with self._lock:
                if version is not None and self._symbol_versions.get(csym, 0) != version:
                    # A fill for this symbol was processed while the read was
                    # in flight: the snapshot may or may not include it, so a
                    # delta from it could double or reverse the position.
                    logger.warning(
                        "OMS: %s changed during the position read for intent %s "
                        "(version %d -> %d, attempt %d/%d) — discarding the snapshot",
                        csym,
                        intent.intent_id,
                        version,
                        self._symbol_versions.get(csym, 0),
                        read_attempt,
                        self._POSITION_READ_ATTEMPTS,
                    )
                    continue
                # The DECISION (gates + delta + journal + in-flight count)
                # stays atomic with the halt flag (CL-8lv6).
                decision = self._decide_locked(intent, snapshot, bypass_halt=bypass_halt)
                break
        if decision is None:
            logger.critical(
                "OMS: %s kept changing during %d position reads — refusing intent %s "
                "(%s target=%.4f) rather than sizing off a stale book",
                csym,
                self._POSITION_READ_ATTEMPTS,
                intent.intent_id,
                intent.strategy_id,
                intent.target_position,
            )
            return SubmissionResult(intent.intent_id, SubmissionStatus.BLOCKED)
        if isinstance(decision, SubmissionResult):
            return decision
        side, qty, counts_as_entry = decision

        # Lock RELEASED before the blocking submit (CL-8s2a): place_order HTTP
        # + RejectionHandler retry sleeps no longer hold the RLock, so a
        # concurrent bypass_halt emergency de-risk isn't queued behind a slow
        # strategy order. _submit_with_retry re-acquires the lock only for its
        # short _pending mutations.
        if counts_as_entry:
            try:
                return self._submit_with_retry(intent, side, qty, emergency=bypass_halt)
            finally:
                with self._lock:
                    self._inflight_entries -= 1
        return self._submit_with_retry(intent, side, qty, emergency=bypass_halt)

    def _refusal_locked(self, intent: OrderIntent) -> SubmissionResult | None:
        """BLOCKED result when an account block or a foreign symbol fence
        refuses ``intent`` (CL-pksi), else None. Caller holds ``_lock``."""
        # CL-pksi: checked HERE, under the per-symbol reservation, so a
        # submit that queued behind an emergency order sees the fence that
        # order set before it was placed (a pre-reservation check would be
        # stale by the time the reservation is granted).
        if self._submission_block is not None:
            logger.critical(
                "OMS: ALL submissions blocked (%s) — refusing intent %s (%s %s)",
                self._submission_block,
                intent.intent_id,
                intent.strategy_id,
                intent.symbol,
            )
            return SubmissionResult(intent.intent_id, SubmissionStatus.BLOCKED)
        fence = self._symbol_fences.get(canonical_symbol(intent.symbol))
        if fence is not None and fence[1] != intent.intent_id:
            logger.critical(
                "OMS: %s is FENCED (%s) — refusing intent %s from %s (target=%.4f); "
                "an emergency order there is unresolved",
                canonical_symbol(intent.symbol),
                fence[0],
                intent.intent_id,
                intent.strategy_id,
                intent.target_position,
            )
            return SubmissionResult(intent.intent_id, SubmissionStatus.BLOCKED)
        return None

    def _decide_locked(
        self,
        intent: OrderIntent,
        snapshot: list[Any],
        *,
        bypass_halt: bool,
    ) -> SubmissionResult | tuple[str, float, bool]:
        """Gate + size one intent against ``snapshot`` (caller holds ``_lock``).

        Returns a terminal :class:`SubmissionResult`, or ``(side, qty,
        counts_as_entry)`` for an order to place; an entry has already been
        added to ``_inflight_entries`` and the caller must decrement it.
        """
        # CL-pksi: re-checked HERE, under the per-symbol reservation AND the
        # state lock at decision time, so a submit that queued behind an
        # emergency order (or whose position read raced a new fence) sees it.
        refused = self._refusal_locked(intent)
        if refused is not None:
            return refused
        # Position matching MUST use the canonical key (CL-qqra): broker
        # positions come back compact ("USDCAD") while event intents are
        # OANDA-underscore ("USD_CAD"). A raw .get() always missed →
        # exit deltas of 0 (positions never closed at the broker) and
        # entries that stacked on an existing position. Routing below
        # still uses intent.symbol (broker _to_oanda is idempotent).
        current_positions = {canonical_symbol(p.symbol): p.quantity for p in snapshot}
        current_qty = current_positions.get(canonical_symbol(intent.symbol), 0.0)
        delta = intent.target_position - current_qty

        # "Halt new trades" means exactly that (CL-8lv6): risk-REDUCING
        # intents still pass — a halted OMS must never trap a strategy's
        # exit while its book already closed (broker keeps the risk, book
        # says flat). Reducing = smaller absolute size, same side (or
        # flat); flips and adds are blocked. This gate MUST stay inside
        # the lock (halt-TOCTOU, CL-8lv6): halt_new_trades takes the same
        # lock, so the flag read and the place decision are atomic.
        reducing = abs(intent.target_position) < abs(current_qty) and (
            intent.target_position == 0.0 or intent.target_position * current_qty > 0
        )
        if bypass_halt and not reducing and delta != 0:
            # A position can change between the risk snapshot and this
            # submit. An emergency label must never authorize buying back
            # an already-reduced leg or crossing through zero.
            logger.critical(
                "Emergency intent %s is no longer reducing (%s %.4f -> %.4f); blocked",
                intent.intent_id,
                intent.symbol,
                current_qty,
                intent.target_position,
            )
            return SubmissionResult(intent.intent_id, SubmissionStatus.BLOCKED)
        # CL-0deu.2: the durable account halt is consulted for every
        # non-reducing intent, inside the same lock as the local flag, so
        # the decision and the halt write serialize. Unknown state blocks.
        durable_block = None
        if not bypass_halt and not reducing and self.halt_store is not None:
            decision = self.halt_store.entry_decision()
            if not decision.allowed:
                durable_block = decision
        if durable_block is not None:
            logger.warning(
                "OMS: account halt (%s) — rejecting non-reducing intent %s "
                "(%s target=%.4f current=%.4f): %s",
                durable_block.reason_code,
                intent.intent_id,
                intent.symbol,
                intent.target_position,
                current_qty,
                durable_block.detail,
            )
            return SubmissionResult(intent.intent_id, SubmissionStatus.BLOCKED)
        if self._halted and not bypass_halt:
            if not reducing:
                logger.warning(
                    "OMS halted — rejecting non-reducing intent %s (%s target=%.4f current=%.4f)",
                    intent.intent_id,
                    intent.symbol,
                    intent.target_position,
                    current_qty,
                )
                return SubmissionResult(intent.intent_id, SubmissionStatus.BLOCKED)
            logger.warning(
                "OMS halted — allowing risk-REDUCING intent %s (%s target=%.4f current=%.4f)",
                intent.intent_id,
                intent.symbol,
                intent.target_position,
                current_qty,
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
            # Dust below venue minimum is not a completed flatten.
            return SubmissionResult(
                intent.intent_id,
                SubmissionStatus.AT_TARGET if delta == 0 else SubmissionStatus.SKIPPED,
                target_reached=delta == 0,
            )

        side = "buy" if delta > 0 else "sell"
        # Registered under the lock, before it is released for the HTTP
        # call: a concurrent acknowledge_portfolio_halt() then sees it.
        counts_as_entry = not reducing and not bypass_halt
        if counts_as_entry:
            self._inflight_entries += 1
        return side, abs(delta), counts_as_entry

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
            self.submit_intent,
            intent,
            bypass_halt=bypass_halt,
            positions=positions,
        )

    def _submit_with_retry(
        self,
        intent: OrderIntent,
        side: str,
        original_qty: float,
        *,
        emergency: bool = False,
    ) -> SubmissionResult:
        """Place the order, applying RejectionHandler policy on broker failures."""
        attempt = 1
        size_fraction = 1.0
        while True:
            sync_owns_fill = True
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
                #
                # CL-80tv: the read runs outside _lock (it always did — the
                # lock was released before this loop), and is now version-
                # checked like the submit-time read: a fill for this symbol
                # processed DURING the read (very likely the prior attempt's
                # own fill) triggers one re-read; if the book is still moving
                # the outcome is UNKNOWN rather than a blind re-send.
                csym = canonical_symbol(intent.symbol)
                try:
                    fresh: dict[str, float] | None = None
                    for read_attempt in range(1, self._POSITION_READ_ATTEMPTS + 1):
                        with self._lock:
                            version = self._symbol_versions.get(csym, 0)
                        snapshot = self.broker.get_positions()
                        with self._lock:
                            stable = self._symbol_versions.get(csym, 0) == version
                        if stable:
                            fresh = {canonical_symbol(p.symbol): p.quantity for p in snapshot}
                            break
                        logger.warning(
                            "Retry re-read for %s: a fill landed during the read "
                            "(attempt %d/%d) — re-reading",
                            intent.symbol,
                            read_attempt,
                            self._POSITION_READ_ATTEMPTS,
                        )
                    if fresh is None:
                        logger.critical(
                            "Retry ABORTED for %s (intent %s): position kept changing "
                            "during the re-read — outcome UNKNOWN, not re-submitting",
                            intent.symbol,
                            intent.intent_id,
                        )
                        return SubmissionResult(
                            intent.intent_id,
                            SubmissionStatus.UNKNOWN,
                            requested_qty=original_qty * size_fraction,
                        )
                    residual = intent.target_position - fresh.get(csym, 0.0)
                    if abs(residual) < self._min_trade_size(intent.symbol):
                        logger.warning(
                            "Retry ABORTED for %s — position already at target "
                            "%.4f (a prior attempt filled despite the error); "
                            "not double-submitting",
                            intent.symbol,
                            intent.target_position,
                        )
                        return SubmissionResult(
                            intent.intent_id,
                            SubmissionStatus.AT_TARGET
                            if residual == 0
                            else SubmissionStatus.SKIPPED,
                            target_reached=residual == 0,
                        )
                except Exception:
                    logger.debug(
                        "retry re-read failed for %s; proceeding with retry",
                        intent.symbol,
                        exc_info=True,
                    )
            qty = original_qty * size_fraction
            order = Order(
                symbol=intent.symbol,
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
                # CL-vj74: stable client id so an async ORDER_FILL from the
                # OANDA transaction stream maps back to this intent.
                client_order_id=intent.intent_id,
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
                if placed.status == OrderStatus.CANCELLED:
                    # A generic canceled order may already have partial fills;
                    # this adapter contract has no cumulative-fill field.
                    return SubmissionResult(
                        intent.intent_id,
                        SubmissionStatus.UNKNOWN,
                        order_id=placed.order_id or None,
                        requested_qty=qty,
                    )
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
                        self._pending_intents.pop(intent.intent_id, None)
                        # CL-vj74: a synchronous FOK fill journals ORDER_FILLED
                        # below; record its venue fill-txn id (placed.order_id
                        # is the orderFillTransaction id here) so the SAME fill
                        # redelivered on the transaction stream is deduped by
                        # on_fill and never double-journaled.
                        if placed.order_id:
                            if placed.order_id in self._seen_fills:
                                # The streamed / replayed copy of this fill was
                                # claimed (and is journaled or being journaled)
                                # before this response returned: it owns the
                                # ORDER_FILLED row.
                                sync_owns_fill = False
                            else:
                                # A failed earlier claim hands ownership to
                                # this path (its listeners already ran).
                                self._unjournaled_fills.pop(placed.order_id, None)
                                self._remember_fill_locked(placed.order_id)
                                # Claimed until ORDER_FILLED below is journaled,
                                # so a streamed duplicate is not reported
                                # durable early.
                                self._fills_journaling.add(placed.order_id)
                        self._bump_symbol_version_locked(intent.symbol)
                    else:
                        self._pending[intent.intent_id] = [placed]
                        # CL-vj74: keep the intent so a later async ORDER_FILL
                        # from the transaction stream can journal a real
                        # ORDER_FILLED and clear this pending entry.
                        self._pending_intents[intent.intent_id] = intent
                logger.info(
                    "Placed %s %s %.4f (attempt=%d, fraction=%.2f)",
                    intent.symbol,
                    side,
                    qty,
                    attempt,
                    size_fraction,
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
                if placed.status == OrderStatus.FILLED and not sync_owns_fill:
                    logger.info(
                        "sync fill %s for intent %s already journaled from the "
                        "transaction stream — not journaling it twice",
                        placed.order_id,
                        intent.intent_id,
                    )
                elif placed.status == OrderStatus.FILLED:
                    sync_journaled = self._journal_event(
                        EventType.ORDER_FILLED,
                        intent=intent,
                        payload={
                            "side": side,
                            "quantity": qty,
                            "attempt": attempt,
                            # Venue fill-transaction id (CL-pksi catch-up): a
                            # replay after restart finds this row and does not
                            # journal the same fill a second time.
                            "fill_id": placed.order_id or None,
                        },
                    )
                    if placed.order_id:
                        with self._lock:
                            self._fills_journaling.discard(placed.order_id)
                            if (
                                sync_journaled
                                and self.journal is not None
                                and self._deferred_fills.pop(placed.order_id, None) is not None
                            ):
                                # The streamed copy arrived first and was deferred
                                # (journal unreadable); this row makes it durable.
                                # Listeners already ran for the streamed copy.
                                logger.info(
                                    "deferred fill %s journaled by the placement "
                                    "response — deferral cleared",
                                    placed.order_id,
                                )
                            if not sync_journaled:
                                # Let the streamed / replayed copy journal it
                                # (its listeners never fired for a sync fill).
                                self._forget_fill_locked(placed.order_id)
                                self._unjournaled_fills[placed.order_id] = intent
                return SubmissionResult(
                    intent.intent_id,
                    SubmissionStatus.FILLED
                    if placed.status == OrderStatus.FILLED
                    else SubmissionStatus.WORKING,
                    target_reached=(
                        placed.status == OrderStatus.FILLED
                        and qty == original_qty
                        and placed.quantity == original_qty
                        # CL-pksi: a short (or unreported) executed quantity
                        # is not the target, whatever the status says.
                        and placed.filled_quantity is not None
                        and abs(placed.filled_quantity - qty) < self._min_trade_size(intent.symbol)
                    ),
                    order_id=placed.order_id or None,
                    requested_qty=qty,
                    filled_qty=(
                        placed.filled_quantity if placed.status == OrderStatus.FILLED else None
                    ),
                )
            except Exception as exc:
                # Emergency retries must not turn an ambiguous acceptance into
                # duplicate closes. Preserve uncertainty for reconciliation.
                # Explicit broker rejects are terminal and safe to reconsider
                # on the next risk tick against the original reduction target.
                if emergency:
                    status = (
                        SubmissionStatus.REJECTED
                        if isinstance(exc, BrokerRejectedOrderError)
                        else SubmissionStatus.UNKNOWN
                    )
                    logger.error(
                        "Emergency submission unresolved: intent=%s status=%s; no blind retry",
                        intent.intent_id,
                        status,
                        exc_info=True,
                    )
                    return SubmissionResult(intent.intent_id, status, requested_qty=qty)
                if self.rejection_handler is None:
                    # Legacy behavior: log and drop.
                    logger.exception("Order failed for %s", intent.intent_id)
                    return SubmissionResult(
                        intent.intent_id,
                        SubmissionStatus.REJECTED
                        if isinstance(exc, BrokerRejectedOrderError)
                        else SubmissionStatus.UNKNOWN,
                    )

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
                        intent.intent_id,
                        attempt,
                        outcome.final_resolution.value,
                    )
                    return SubmissionResult(
                        intent.intent_id,
                        SubmissionStatus.REJECTED
                        if isinstance(exc, BrokerRejectedOrderError)
                        else SubmissionStatus.UNKNOWN,
                    )

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

    def on_fill(self, fill: dict[str, Any]) -> bool:
        """Consume one ORDER_FILL from the broker's transaction stream (CL-vj74).

        Idempotent by the venue transaction id (the stream can redeliver on
        reconnect, and a synchronous FOK fill is recorded at placement so its
        streamed duplicate is skipped). Emits a real ORDER_FILLED journal row
        with per-order attribution and clears the matching PENDING intent by
        its client id — so a FOK order OANDA created-then-filled asynchronously
        no longer sits in ``_pending`` forever (poisoning has_pending / the
        shutdown drain). Returns True when the fill was newly processed.

        The position-poll confirmation (EventBook.confirm_entries/confirm_exits)
        stays as the backstop; this is the low-latency, attributed fast path.
        """
        return self.process_fill(fill).newly_processed

    def process_fill(
        self,
        fill: dict[str, Any],
        *,
        check_journal: bool = False,
        defer_on_lookup_failure: bool = False,
    ) -> FillOutcome:
        """:meth:`on_fill` plus the durability verdict the stream checkpoint
        needs (CL-pksi catch-up).

        ``FillOutcome.durable`` is True only when the fill's ORDER_FILLED row
        is known to be in the journal (or no journal is configured): a durable
        transaction-id checkpoint may then advance past it. A failed journal
        append is NOT durable — the id is dropped from the dedup set (so a
        replay journals it again) and the fill's intent is kept for that retry.

        ``check_journal`` (replay after a restart, when the in-memory dedup set
        is empty): look the fill up in the journal first, so a fill journaled
        before the restart is not journaled twice. That DB read runs outside
        ``_lock``; a lookup failure propagates (the caller must not advance).
        A fill found in the journal is still handed to the fill listeners
        (they dedup by transaction id): a crash between the journal commit and
        a listener's own persistence must not lose that evidence on replay.

        ``defer_on_lookup_failure`` (live stream, integration review): when
        the journal lookup fails the fill's journal state is UNKNOWN, so it is
        never appended. It is claimed (pending cleared, listeners fed) and
        parked as DEFERRED with ``durable=False``; the next catch-up replay
        looks it up again and appends it only if the row is really missing.

        Exactly-once: the claim (dedup check + pending pop + version bump) is
        one ``_lock`` section; journaling and listeners run outside it, and a
        concurrent duplicate reports ``durable=False`` until the claimant's
        journal append has finished.
        """
        fill_id = str(fill.get("transaction_id") or "")
        client_id = fill.get("client_order_id")
        if fill_id and not check_journal:
            with self._lock:
                if fill_id in self._deferred_fills:
                    # Journal state unknown: only a journal-checked replay may
                    # append it (never a blind path).
                    return FillOutcome(newly_processed=False, durable=False)
        if check_journal and fill_id:
            with self._lock:
                known = fill_id in self._seen_fills or fill_id in self._unjournaled_fills
                deferred = fill_id in self._deferred_fills
            if not known:
                try:
                    already = self._fill_already_journaled(fill_id, client_id)
                except Exception:
                    if not defer_on_lookup_failure:
                        raise
                    return self._defer_fill(fill, fill_id, client_id, deferred=deferred)
                was_deferred = False
                if already:
                    with self._lock:
                        if fill_id in self._unjournaled_fills:
                            already = False  # a failed append of OURS: retry below
                        else:
                            was_deferred = self._deferred_fills.pop(fill_id, False) is not False
                            self._remember_fill_locked(fill_id)
                    if already:
                        logger.info(
                            "fill %s already journaled — no second ORDER_FILLED row",
                            fill_id,
                        )
                        if not was_deferred:
                            # Listeners dedup by transaction id; the journal
                            # row alone does not prove they persisted it.
                            self._deliver_fill_listeners(fill, fill_id)
                        return FillOutcome(newly_processed=False, durable=True)
        with self._lock:
            if fill_id and fill_id in self._seen_fills:
                # Already handled (redelivery or sync fill); durable only once
                # the claimant's journal append has completed.
                durable = self.journal is not None and fill_id not in self._fills_journaling
                if durable and self._deferred_fills.pop(fill_id, None) is not None:
                    # Journaled by another path (e.g. the synchronous placement
                    # response) after it was deferred: no longer deferred, so
                    # the checkpoint may pass it (integration review r4).
                    logger.info("deferred fill %s is now journaled — deferral cleared", fill_id)
                return FillOutcome(newly_processed=False, durable=durable)
            retry = bool(fill_id) and (
                fill_id in self._unjournaled_fills or fill_id in self._deferred_fills
            )
            if retry:
                # A previous delivery claimed this fill but its journal append
                # failed (or was deferred and the journal now shows no row):
                # reuse its attribution, do not re-fire listeners.
                deferred_entry = self._deferred_fills.pop(fill_id, None)
                if fill_id in self._unjournaled_fills:
                    intent = self._unjournaled_fills.pop(fill_id)
                else:
                    assert deferred_entry is not None
                    intent = deferred_entry[0]
            else:
                intent = self._pending_intents.pop(str(client_id), None) if client_id else None
                if client_id:
                    self._pending.pop(str(client_id), None)
                # CL-80tv: a processed fill changes this symbol's book — a
                # submit whose (lock-free) position read overlapped it must
                # discard that snapshot.
                self._bump_symbol_version_locked(
                    str(fill.get("instrument") or (intent.symbol if intent else ""))
                )
            if fill_id:
                self._remember_fill_locked(fill_id)
                self._fills_journaling.add(fill_id)
        symbol = str(fill.get("instrument") or (intent.symbol if intent else ""))
        # Reuse the original intent so the ORDER_FILLED row links to the
        # INTENT_SUBMITTED row; fall back to a synthetic one (e.g. a fill for
        # an order already terminal via the sync path, or an unknown client id).
        journal_intent = intent or OrderIntent(
            strategy_id="oanda-fill-stream",
            symbol=symbol,
            target_position=0.0,
            intent_id=str(client_id) if client_id else (fill_id or "unknown"),
        )
        journaled = False
        try:
            journaled = self._journal_event(
                EventType.ORDER_FILLED,
                intent=journal_intent,
                payload={
                    "source": "oanda_transaction_stream",
                    "order_id": fill.get("order_id"),
                    "client_order_id": client_id,
                    "fill_id": fill_id,
                    "symbol": symbol,
                    "units": fill.get("units"),
                    "price": fill.get("price"),
                },
            )
        finally:
            if fill_id:
                with self._lock:
                    self._fills_journaling.discard(fill_id)
                    if not journaled:
                        # Not durable: forget the id so a replay can journal
                        # it, and keep the intent for that retry.
                        self._forget_fill_locked(fill_id)
                        if len(self._unjournaled_fills) < _SEEN_FILLS_CAP:
                            self._unjournaled_fills[fill_id] = intent
        if not journaled:
            logger.error(
                "ORDER_FILLED (stream) NOT journaled: fill=%s client=%s %s — "
                "kept replayable (the stream checkpoint will not advance past it)",
                fill_id or "?",
                client_id,
                symbol,
            )
        logger.info(
            "ORDER_FILLED (stream): client=%s order=%s %s units=%s @ %s%s",
            client_id,
            fill.get("order_id"),
            symbol,
            fill.get("units"),
            fill.get("price"),
            " (journal retry)" if retry else "",
        )
        if not retry:
            # CL-pksi: hand the NEWLY processed fill (post-dedup) to listeners —
            # the kill-switch manager resolves emergency-order fences from it.
            self._deliver_fill_listeners(fill, fill_id)
        # Durable only when a journal actually holds the row: with no journal
        # nothing survives a restart, so a checkpoint must never pass it.
        return FillOutcome(newly_processed=True, durable=journaled and self.journal is not None)

    def _deliver_fill_listeners(self, fill: dict[str, Any], fill_id: str) -> None:
        """Feed one fill to every listener (CL-pksi). Listeners must be
        idempotent per transaction id (the emergency-attempt ledger is); a
        listener failure must never break fill handling."""
        for listener in list(self._fill_listeners):
            try:
                listener(dict(fill))
            except Exception:
                logger.exception("fill listener failed for fill %s", fill_id or "?")

    def _defer_fill(
        self, fill: dict[str, Any], fill_id: str, client_id: Any, *, deferred: bool
    ) -> FillOutcome:
        """Journal state of a live fill is unknown (lookup failed): claim it
        WITHOUT appending, feed the listeners once, and park it for the next
        journal-checked replay (integration review defect 1)."""
        logger.error(
            "journal lookup for fill %s failed — ORDER_FILLED deferred to the next "
            "catch-up replay (no blind append)",
            fill_id,
            exc_info=True,
        )
        if deferred:
            return FillOutcome(newly_processed=False, durable=False, deferred=True)
        with self._lock:
            if fill_id in self._deferred_fills:
                return FillOutcome(newly_processed=False, durable=False, deferred=True)
            if fill_id in self._seen_fills:
                return FillOutcome(newly_processed=False, durable=False)
            intent = self._pending_intents.pop(str(client_id), None) if client_id else None
            if client_id:
                self._pending.pop(str(client_id), None)
            self._bump_symbol_version_locked(
                str(fill.get("instrument") or (intent.symbol if intent else ""))
            )
            if len(self._deferred_fills) < _SEEN_FILLS_CAP:
                self._deferred_fills[fill_id] = (intent, dict(fill))
            else:
                # Never silent: the caller's durable checkpoint rewind still
                # makes the next replay re-fetch (and journal) this fill.
                logger.critical(
                    "deferred-fill map full (%d) — fill %s relies on the durable "
                    "checkpoint rewind for its replay",
                    _SEEN_FILLS_CAP,
                    fill_id,
                )
        self._deliver_fill_listeners(fill, fill_id)
        return FillOutcome(newly_processed=True, durable=False, deferred=True)

    def deferred_fills(self) -> list[dict[str, Any]]:
        """Copies of the fills whose ORDER_FILLED append is deferred
        (integration review r3): a replay drains these independently of the
        venue page, through ``process_fill(check_journal=True)``."""
        with self._lock:
            return [dict(f) for _, f in self._deferred_fills.values()]

    def _fill_already_journaled(self, fill_id: str, client_id: Any) -> bool:
        """True when an ORDER_FILLED row for venue fill ``fill_id`` is already
        in the journal (CL-pksi catch-up). Rows are keyed by intent id — the
        client id, or the fill id for an unattributed fill — and carry the
        fill id in their payload. Raises when the journal cannot be read."""
        if self.journal is None:
            return False
        keys = [str(client_id)] if client_id else []
        if fill_id not in keys:
            keys.append(fill_id)
        for intent_key in keys:
            logger.debug("journal lookup for replayed fill %s (intent %s)", fill_id, intent_key)
            for event in self.journal.query_by_intent(intent_key):
                if (
                    event.event_type is EventType.ORDER_FILLED
                    and str(event.payload.get("fill_id") or "") == fill_id
                ):
                    return True
        return False

    def _remember_fill_locked(self, fill_id: str) -> None:
        """Add a venue fill id to the dedup set (caller holds ``_lock``).
        Bounded by evicting the OLDEST ids — redeliveries (live/replay
        overlap) are always recent, so the newest window is what matters."""
        if fill_id in self._seen_fills:
            return
        self._seen_fills.add(fill_id)
        self._seen_order.pop(fill_id, None)
        self._seen_order[fill_id] = None  # (re)insert as newest
        while len(self._seen_fills) > _SEEN_FILLS_CAP and self._seen_order:
            oldest = next(iter(self._seen_order))
            del self._seen_order[oldest]
            self._seen_fills.discard(oldest)

    def _forget_fill_locked(self, fill_id: str) -> None:
        """Drop a fill id from the dedup set AND its eviction order (caller
        holds ``_lock``), so a later retry is ordered as a fresh entry."""
        self._seen_fills.discard(fill_id)
        self._seen_order.pop(fill_id, None)

    def _bump_symbol_version_locked(self, symbol: str) -> None:
        """CL-80tv: record a state change for ``symbol`` (caller holds _lock)."""
        csym = canonical_symbol(symbol)
        if csym:
            self._symbol_versions[csym] = self._symbol_versions.get(csym, 0) + 1

    def add_fill_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Register a consumer of newly processed streamed fills (CL-pksi)."""
        assert callable(listener), "fill listener must be callable"
        with self._lock:
            if listener not in self._fill_listeners:
                self._fill_listeners.append(listener)

    def fence_symbol(self, symbol: str, reason: str, owner: str | None = None) -> None:
        """Refuse every submit for ``symbol`` until released (CL-pksi).

        Set while an emergency order on that instrument is unresolved at the
        broker, so no other writer (strategy, reconciler, manual trade, a
        second kill switch) sizes an order off a book that may not include it.
        ``owner`` exempts exactly one intent id: the emergency order about to
        be placed under the fence (set BEFORE its submission, so no other
        writer can slip in between its placement and its outcome).
        """
        csym = canonical_symbol(symbol)
        assert csym and reason, "fence needs a symbol and a reason"
        with self._lock:
            if self._symbol_fences.get(csym) != (reason, owner):
                logger.critical("OMS: fencing %s (owner=%s) — %s", csym, owner, reason)
            self._symbol_fences[csym] = (reason, owner)

    def release_symbol_fence(self, symbol: str) -> None:
        """Lift a fence set by :meth:`fence_symbol` (CL-pksi)."""
        csym = canonical_symbol(symbol)
        with self._lock:
            if self._symbol_fences.pop(csym, None) is not None:
                logger.warning("OMS: fence on %s released", csym)

    def block_all_submissions(self, reason: str) -> None:
        """Refuse EVERY submit, reducing or not (CL-pksi): outstanding
        emergency orders are unknown, so no symbol can be sized safely."""
        assert reason, "a block needs a reason"
        with self._lock:
            if self._submission_block != reason:
                logger.critical("OMS: blocking ALL submissions — %s", reason)
            self._submission_block = reason

    def unblock_all_submissions(self) -> None:
        """Lift :meth:`block_all_submissions` (CL-pksi)."""
        with self._lock:
            if self._submission_block is not None:
                logger.warning("OMS: account-wide submission block lifted")
            self._submission_block = None

    def submission_block(self) -> str | None:
        with self._lock:
            return self._submission_block

    def fenced_symbols(self) -> dict[str, str]:
        """Snapshot of fenced canonical symbols -> reason (CL-pksi)."""
        with self._lock:
            return {sym: fence[0] for sym, fence in self._symbol_fences.items()}

    def acknowledge_portfolio_halt(self) -> bool:
        """Acknowledge the durable halt for the FX path (CL-0deu.2).

        Only at a quiescent point: when a non-reducing placement that passed
        the gate is still inside place_order, acknowledging would let
        "applied" hide pre-halt exposure. Returns True when an ack was
        written. Every later non-reducing intent re-reads the store anyway.
        """
        if self.halt_store is None:
            return False
        with self._lock:
            if self._inflight_entries > 0:
                logger.info(
                    "OMS: %d entry placement(s) in flight — deferring account-halt ack",
                    self._inflight_entries,
                )
                return False
            from src.risk.trading_halt import PATH_FX_OMS  # noqa: PLC0415

            self.halt_store.observe(PATH_FX_OMS)
            return True

    def halt_new_trades(self) -> None:
        # Under the same lock that guards submit_intent's halt check —
        # a health-tick halt racing a strategy place must serialize
        # (CL-8lv6 TOCTOU).
        with self._lock:
            self._halted = True
            logger.warning("OMS: new trades halted")

    def is_halted(self) -> bool:
        """Engine-local halt flag (CL-d7ex): read by the kill-switch manager
        so a halt it did not record is never auto-resumed away."""
        with self._lock:
            return self._halted

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
    ) -> bool:
        """Append one event to the trade journal — silent no-op without one.

        Journal failures must NOT propagate: trading is more important than
        bookkeeping, and a DB hiccup must not drop or duplicate orders.
        Returns False only when an append was attempted and FAILED (CL-pksi:
        the stream checkpoint must not advance past an unjournaled fill).
        """
        if self.journal is None:
            return True
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
                event_type.value,
                intent.intent_id,
            )
            return False
        return True
