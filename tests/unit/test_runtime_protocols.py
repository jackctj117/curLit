"""Money-path structural types (CL-sb5w).

LiveEngine took ``broker: Any`` / ``oms: Any``, so mypy's discipline stopped
exactly where real money starts — a typo'd method name was a runtime
AttributeError inside the trading loop, not a type error.

These tests pin the two properties that make Protocols the right tool here:
duck-typed fakes (which inherit nothing) still satisfy them, and the OPTIONAL
fill-stream pair is detected by presence — the runtime behaviour the old
``hasattr`` guard had, now type-narrowing too.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from src.runtime.protocols import (
    BrokerLike,
    FillHandlingOrderManager,
    FillStreamingBroker,
    KillSwitchManagerLike,
    OrderManagerLike,
    ReconcilerLike,
)


class _PaperishBroker:
    """Inherits nothing — exactly like the suite's broker fakes."""

    def get_account(self) -> dict[str, float]:
        return {"equity": 100_000.0}

    def stream_prices(self, symbols: list[str]) -> AsyncIterator[dict[str, Any]]:  # noqa: ARG002
        raise NotImplementedError


class _StreamingBroker(_PaperishBroker):
    def stream_transactions(self) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError


class _Oms:
    async def submit_intent_async(
        self,
        intent: Any,
        *,
        bypass_halt: bool = False,
        positions: list[Any] | None = None,
    ) -> str:
        return "id"

    def halt_new_trades(self) -> None: ...

    def has_pending(self) -> bool:
        return False


class _FillOms(_Oms):
    def on_fill(self, fill: dict[str, Any]) -> bool:
        return True


class TestStructuralSatisfaction:
    def test_duck_typed_fakes_satisfy_the_protocols(self) -> None:
        # The whole reason these are Protocols and not the Broker ABC: the
        # unit suite's fakes inherit nothing and must keep working.
        assert isinstance(_PaperishBroker(), BrokerLike)
        assert isinstance(_Oms(), OrderManagerLike)

    def test_missing_method_does_not_satisfy(self) -> None:
        class _NoAccount:
            def stream_prices(self, symbols: list[str]) -> Any: ...

        assert not isinstance(_NoAccount(), BrokerLike)

    def test_real_implementations_satisfy(self) -> None:
        from src.execution.paper_broker import PaperBroker

        assert isinstance(PaperBroker(), BrokerLike)


class TestOptionalFillStream:
    """CL-vj74 pairing: the paper broker has no transaction stream, so the
    engine feature-detects. isinstance on a runtime_checkable Protocol tests
    method PRESENCE — same runtime semantics as the old hasattr pair."""

    def test_streaming_broker_detected(self) -> None:
        assert isinstance(_StreamingBroker(), FillStreamingBroker)
        assert isinstance(_FillOms(), FillHandlingOrderManager)

    def test_non_streaming_broker_not_detected(self) -> None:
        # A paper-style broker must NOT get the transaction-stream task.
        assert not isinstance(_PaperishBroker(), FillStreamingBroker)
        assert not isinstance(_Oms(), FillHandlingOrderManager)

    def test_optional_members_absent_from_base_protocol(self) -> None:
        # Declaring stream_transactions on BrokerLike would make the engine's
        # feature-detection a lie — the paper broker still satisfies BrokerLike.
        assert isinstance(_PaperishBroker(), BrokerLike)


class TestSafetyProtocols:
    def test_kill_switch_manager_and_reconciler(self) -> None:
        from src.risk.kill_switches import KillSwitchManager

        assert isinstance(KillSwitchManager([], oms=_Oms()), KillSwitchManagerLike)

        class _Rec:
            def reconcile(self) -> Any: ...
            def check_alignment(self) -> Any: ...

        assert isinstance(_Rec(), ReconcilerLike)
