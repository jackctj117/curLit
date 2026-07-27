"""Structural types for the LiveEngine's money-path collaborators (CL-sb5w).

``LiveEngine.__init__`` took ``broker: Any``, ``oms: Any``,
``kill_switch_manager: Any``... so mypy's discipline stopped exactly where
real money starts: a typo'd method name on the OMS or a broker missing
``get_account`` was a runtime AttributeError inside the trading loop, not a
type error at check time.

These are :class:`typing.Protocol` (STRUCTURAL), deliberately not the
concrete classes or the :class:`src.execution.broker.Broker` ABC:

* the engine is wired with several implementations (OANDA / paper /
  Polymarket brokers, a real vs recording OMS) and the unit suite passes
  duck-typed fakes that inherit nothing — a structural type accepts those
  unchanged, a nominal ABC would reject them and force test rewrites;
* each protocol declares only the surface :class:`LiveEngine` ACTUALLY
  calls, so it documents the real coupling instead of the full class.

Optional-by-design members live in their own protocols
(:class:`FillStreamingBroker`, :class:`FillHandlingOrderManager`) rather than
the base ones: the OANDA fill stream (CL-vj74) is feature-detected because the
paper broker has no ``stream_transactions``, so declaring it on
:class:`BrokerLike` would make that guard a lie. Being ``runtime_checkable``,
they replace the old ``hasattr`` pairs with an ``isinstance`` check that is
equivalent at runtime (Protocol isinstance tests method PRESENCE) but also
narrows the type for mypy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BrokerLike(Protocol):
    """The broker surface the engine's loop needs (see module note)."""

    def get_account(self) -> Any: ...

    def stream_prices(self, symbols: list[str]) -> AsyncIterator[dict[str, Any]]: ...


@runtime_checkable
class OrderManagerLike(Protocol):
    """OMS surface used by the engine (submission + halt/pending state)."""

    async def submit_intent_async(
        self,
        intent: Any,
        *,
        bypass_halt: bool = False,
        positions: list[Any] | None = None,
    ) -> str: ...

    def halt_new_trades(self) -> None: ...

    def has_pending(self) -> bool: ...


@runtime_checkable
class FillStreamingBroker(Protocol):
    """A broker that can push real-time fills (CL-vj74). OPTIONAL — the paper
    broker has no transaction stream, so this is feature-detected."""

    def stream_transactions(self) -> AsyncIterator[dict[str, Any]]: ...


@runtime_checkable
class FillHandlingOrderManager(Protocol):
    """An OMS that can consume a streamed fill (CL-vj74). OPTIONAL — paired
    with :class:`FillStreamingBroker`."""

    def on_fill(self, fill: dict[str, Any]) -> bool: ...


@runtime_checkable
class KillSwitchManagerLike(Protocol):
    """Safety layer surface evaluated on the engine's health tick."""

    def check(self, context: dict[str, Any]) -> list[dict[str, Any]]: ...

    def log_arming(self, provided_keys: Iterable[str]) -> None: ...

    def reset_daily(self, *, clear_causes: bool = True) -> None: ...

    def attempt_auto_resume(self, context: dict[str, Any]) -> bool: ...


@runtime_checkable
class ReconcilerLike(Protocol):
    """Cold-start + periodic position reconciliation surface."""

    def reconcile(self) -> Any: ...

    def check_alignment(self) -> Any: ...


@runtime_checkable
class CoordinatorLike(Protocol):
    """Portfolio coordinator surface driven by the engine loop."""

    async def process_intents(
        self,
        raw_intents: dict[str, list[Any]],
    ) -> dict[str, dict[str, Any]]: ...

    async def rebalance_allocations(self, force: bool = False) -> None: ...
