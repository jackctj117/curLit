"""Order rejection handling + retry/escalation policy (CL-yteo).

Codifies the response when the broker rejects an order. Without this, rejected
orders during stress (the moment we most need execution to work) become silent
failures or get retried indefinitely.

Rejection classes:
    LIQUIDITY    broker had no liquidity at requested size (FOK fail, partial)
    MARGIN       insufficient margin to cover the trade
    HALT         instrument halted, closed, or otherwise non-tradable now
    TRANSIENT    network error, 5xx HTTP, timeout — likely retryable
    MALFORMED    400/422 — bad request payload, programming error
    UNKNOWN      anything we can't classify

Per-class default resolutions (override via RejectionPolicy):
    LIQUIDITY  → RETRY_SMALLER  (halve size, up to 3 tries)
    MARGIN     → ABORT          (size won't help; alert operator)
    HALT       → ABORT_HALT_STRATEGY  (pause the originating strategy)
    TRANSIENT  → RETRY           (exponential backoff: 1s, 2s, 4s, ceiling 4 tries)
    MALFORMED  → ABORT           (alert operator — programming bug)
    UNKNOWN    → ABORT           (conservative: don't retry mystery errors)

Wired into OrderManager.submit_intent — when broker.place_order raises, the
handler classifies the error, emits structured logs + metric, and decides
whether to retry, halve, abort, or halt the strategy.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.execution.broker import Order
from src.execution.oms import OrderIntent
from src.execution.trade_journal import EventType, TradeJournal
from src.monitoring.metrics import orders_rejected

logger = logging.getLogger(__name__)


# Time multiplier between retry attempts under exponential backoff. 2x is the
# standard choice — fast enough to recover from transients, slow enough to
# avoid thundering-herd against a struggling broker.
_RETRY_BACKOFF_MULTIPLIER: float = 2.0

# Initial retry sleep in seconds. 1s lets a single transient blip clear without
# meaningfully impacting trade latency for normal flow.
_INITIAL_RETRY_SLEEP_SEC: float = 1.0

# Size reduction per RETRY_SMALLER attempt. Halving each attempt (1.0, 0.5,
# 0.25, 0.125) is the standard "halve until liquidity accepts" pattern.
_SIZE_REDUCTION_FACTOR: float = 0.5

# Smallest size we'll attempt before giving up (as a fraction of original
# request). Below this, the trade is too small to matter.
_MIN_SIZE_FRACTION: float = 0.10


class RejectionClass(Enum):
    LIQUIDITY = "liquidity"
    MARGIN = "margin"
    HALT = "halt"
    TRANSIENT = "transient"
    MALFORMED = "malformed"
    UNKNOWN = "unknown"


class RejectionResolution(Enum):
    RETRY = "retry"
    RETRY_SMALLER = "retry_smaller"
    ABORT = "abort"
    ABORT_HALT_STRATEGY = "abort_halt_strategy"


@dataclass
class ClassPolicy:
    """How to handle one rejection class."""

    resolution: RejectionResolution
    max_attempts: int
    initial_sleep_sec: float = _INITIAL_RETRY_SLEEP_SEC
    backoff_multiplier: float = _RETRY_BACKOFF_MULTIPLIER

    def __post_init__(self) -> None:
        assert self.max_attempts >= 1, f"max_attempts must be >= 1, got {self.max_attempts}"
        assert self.initial_sleep_sec >= 0, "initial_sleep_sec must be non-negative"
        assert self.backoff_multiplier >= 1.0, "backoff_multiplier must be >= 1.0"


@dataclass
class RejectionPolicy:
    """Per-class resolution map. Defaults match the runbook in
    docs/runbooks/OrderRejected.md.
    """

    by_class: dict[RejectionClass, ClassPolicy]

    @classmethod
    def default(cls) -> RejectionPolicy:
        return cls(
            by_class={
                RejectionClass.LIQUIDITY: ClassPolicy(
                    resolution=RejectionResolution.RETRY_SMALLER,
                    max_attempts=3,
                    initial_sleep_sec=0.1,
                ),
                RejectionClass.MARGIN: ClassPolicy(
                    resolution=RejectionResolution.ABORT,
                    max_attempts=1,
                ),
                RejectionClass.HALT: ClassPolicy(
                    resolution=RejectionResolution.ABORT_HALT_STRATEGY,
                    max_attempts=1,
                ),
                RejectionClass.TRANSIENT: ClassPolicy(
                    resolution=RejectionResolution.RETRY,
                    max_attempts=4,
                    initial_sleep_sec=_INITIAL_RETRY_SLEEP_SEC,
                ),
                RejectionClass.MALFORMED: ClassPolicy(
                    resolution=RejectionResolution.ABORT,
                    max_attempts=1,
                ),
                RejectionClass.UNKNOWN: ClassPolicy(
                    resolution=RejectionResolution.ABORT,
                    max_attempts=1,
                ),
            }
        )


@dataclass
class RejectionEvent:
    """Structured event recorded for every broker rejection."""

    intent_id: str
    symbol: str
    strategy_id: str
    rejection_class: RejectionClass
    resolution: RejectionResolution
    attempt: int
    detail: str
    ts: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "rejection_class": self.rejection_class.value,
            "resolution": self.resolution.value,
            "attempt": self.attempt,
            "detail": self.detail,
        }


# =============================================================================
# Classification
# =============================================================================


# Keyword patterns mapping exception text to rejection classes. Pattern wins
# in declaration order — more specific patterns first.
_CLASSIFICATION_PATTERNS: list[tuple[re.Pattern[str], RejectionClass]] = [
    (re.compile(r"insufficient[_\s]+margin", re.I), RejectionClass.MARGIN),
    (re.compile(r"margin[_\s]+(call|exhausted|insufficient)", re.I), RejectionClass.MARGIN),
    (re.compile(r"liquidity|fok\b|fill[_\s]+or[_\s]+kill", re.I), RejectionClass.LIQUIDITY),
    (re.compile(r"no[_\s]+liquidity|not[_\s]+enough[_\s]+units", re.I), RejectionClass.LIQUIDITY),
    (
        re.compile(r"halt|closed|trading[_\s]+suspended|market[_\s]+closed", re.I),
        RejectionClass.HALT,
    ),
    (re.compile(r"instrument[_\s]+(halted|unavailable|disabled)", re.I), RejectionClass.HALT),
    (
        re.compile(r"timeout|connection[_\s]+(reset|refused|aborted)", re.I),
        RejectionClass.TRANSIENT,
    ),
    (re.compile(r"5\d\d|service[_\s]+unavailable|gateway", re.I), RejectionClass.TRANSIENT),
    (
        re.compile(r"(400|422|bad[_\s]+request|invalid[_\s]+(json|payload|param))", re.I),
        RejectionClass.MALFORMED,
    ),
]


def classify_exception(
    exc: BaseException,
    response_text: str | None = None,
) -> RejectionClass:
    """Classify a broker exception into a RejectionClass.

    Inspects the exception message and optional response body text. Defaults to
    UNKNOWN when no pattern matches — the conservative default policy then
    aborts rather than retrying mystery errors.
    """
    text_parts = [str(exc), exc.__class__.__name__]
    if response_text:
        text_parts.append(response_text)
    text = " | ".join(text_parts)

    for pattern, cls in _CLASSIFICATION_PATTERNS:
        if pattern.search(text):
            return cls

    return RejectionClass.UNKNOWN


# =============================================================================
# Handler
# =============================================================================


@dataclass
class HandlerOutcome:
    """Result of one rejection-handling cycle.

    Conveys back to the caller whether to retry the order (with what size
    fraction), abort, or halt the strategy entirely.
    """

    should_retry: bool
    next_size_fraction: float = 1.0
    sleep_sec: float = 0.0
    halt_strategy: bool = False
    final_resolution: RejectionResolution = RejectionResolution.ABORT


class RejectionHandler:
    """Classifies broker exceptions and decides retry/abort according to policy.

    Stateless — caller supplies attempt count. Emits the
    fx_orders_rejected_total{pair, reason} counter and structured log per call.
    """

    def __init__(
        self,
        policy: RejectionPolicy | None = None,
        journal: TradeJournal | None = None,
    ) -> None:
        self.policy = policy or RejectionPolicy.default()
        self.journal = journal
        self._events: list[RejectionEvent] = []

    def handle(
        self,
        intent: OrderIntent,
        order: Order,
        exc: BaseException,
        attempt: int,
        response_text: str | None = None,
    ) -> HandlerOutcome:
        """Decide what to do about a rejection.

        attempt is 1-indexed. A returned should_retry=True means the caller
        should sleep `sleep_sec` and retry with size = original × next_size_fraction.
        """
        cls = classify_exception(exc, response_text=response_text)
        class_policy = self.policy.by_class[cls]

        ts = datetime.now(UTC)
        event = RejectionEvent(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            strategy_id=intent.strategy_id,
            rejection_class=cls,
            resolution=class_policy.resolution,
            attempt=attempt,
            detail=str(exc),
            ts=ts,
        )
        self._events.append(event)

        try:
            orders_rejected.labels(pair=intent.symbol, reason=cls.value).inc()
        except Exception:
            logger.exception("orders_rejected metric increment failed")

        logger.warning(
            "Order reject %s symbol=%s class=%s resolution=%s attempt=%d/%d detail=%s",
            intent.intent_id,
            intent.symbol,
            cls.value,
            class_policy.resolution.value,
            attempt,
            class_policy.max_attempts,
            exc,
        )

        # Audit the rejection. Journal failures must not propagate — trading
        # decisions are independent of audit-log availability.
        if self.journal is not None:
            try:
                self.journal.record(
                    event_type=EventType.ORDER_REJECTED,
                    payload={
                        "rejection_class": cls.value,
                        "resolution": class_policy.resolution.value,
                        "attempt": attempt,
                        "max_attempts": class_policy.max_attempts,
                        "detail": str(exc),
                        "order_quantity": order.quantity,
                        "order_side": order.side,
                    },
                    intent_id=intent.intent_id,
                    strategy_id=intent.strategy_id,
                    symbol=intent.symbol,
                )
            except Exception:
                logger.exception(
                    "Trade journal append failed for rejection of %s",
                    intent.intent_id,
                )

        # Decide based on class policy + attempt count.
        if attempt >= class_policy.max_attempts:
            # Out of retries — abort (or halt strategy if class demands it).
            return HandlerOutcome(
                should_retry=False,
                halt_strategy=(class_policy.resolution == RejectionResolution.ABORT_HALT_STRATEGY),
                final_resolution=class_policy.resolution,
            )

        if class_policy.resolution == RejectionResolution.RETRY:
            sleep = class_policy.initial_sleep_sec * (
                class_policy.backoff_multiplier ** (attempt - 1)
            )
            return HandlerOutcome(
                should_retry=True,
                next_size_fraction=1.0,
                sleep_sec=sleep,
                final_resolution=class_policy.resolution,
            )

        if class_policy.resolution == RejectionResolution.RETRY_SMALLER:
            # Each attempt halves size: attempt 1 → 1.0, attempt 2 → 0.5, etc.
            next_fraction = _SIZE_REDUCTION_FACTOR**attempt
            if next_fraction < _MIN_SIZE_FRACTION:
                # Don't bother going below the minimum useful fraction.
                return HandlerOutcome(
                    should_retry=False,
                    final_resolution=class_policy.resolution,
                )
            return HandlerOutcome(
                should_retry=True,
                next_size_fraction=next_fraction,
                sleep_sec=class_policy.initial_sleep_sec,
                final_resolution=class_policy.resolution,
            )

        # ABORT or ABORT_HALT_STRATEGY: no retry.
        return HandlerOutcome(
            should_retry=False,
            halt_strategy=(class_policy.resolution == RejectionResolution.ABORT_HALT_STRATEGY),
            final_resolution=class_policy.resolution,
        )

    @property
    def events(self) -> list[RejectionEvent]:
        return list(self._events)

    @staticmethod
    def sleep(seconds: float) -> None:
        """Wrapper around time.sleep — exposed so tests can patch."""
        if seconds > 0:
            time.sleep(seconds)
