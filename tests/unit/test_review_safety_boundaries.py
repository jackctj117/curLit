"""October review: unknown broker state and misleading rejection digits."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from src.execution.oms import OrderManager
from src.execution.rejection import RejectionClass, classify_exception
from src.portfolio.reconciler import PositionReconciler
from src.risk.kill_switches import KillSwitchManager
from src.runtime.live_engine import LiveEngine
from src.strategies.event_book import EventBook


@pytest.mark.parametrize(
    "message",
    [
        "SLIPPAGE_EXCEEDED at 1.105234",
        "https://broker/accounts/15001234/orders/422",
        "price=1.4002",
    ],
)
def test_numeric_prices_and_urls_do_not_classify_http_status(message: str) -> None:
    assert classify_exception(RuntimeError(message)) is RejectionClass.UNKNOWN


def test_structured_http_status_overrides_digits_in_url() -> None:
    request = httpx.Request("POST", "https://fixture.invalid/accounts/500123")
    response = httpx.Response(422, request=request)
    error = httpx.HTTPStatusError("order refused", request=request, response=response)
    assert classify_exception(error) is RejectionClass.MALFORMED


@pytest.mark.parametrize("failure", [RuntimeError("broker offline"), None])
def test_unavailable_reconcile_cannot_confirm_or_clear_book(failure: Exception | None) -> None:
    broker = Mock()
    broker.get_positions.side_effect = failure
    broker.get_positions.return_value = None  # malformed snapshot is not flat either
    book = Mock()
    state = Mock()
    oms = Mock()
    reconciler = PositionReconciler(broker, oms, state, [SimpleNamespace(id="event", book=book)])
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        reconciler.reconcile()
    book.confirm_entries.assert_not_called()
    oms.submit_intent.assert_not_called()
    assert not state.mock_calls


def test_account_read_failures_have_an_independent_halt_counter() -> None:
    broker = Mock()
    broker.get_positions.return_value = []
    oms = OrderManager(broker)
    manager = KillSwitchManager(broker, oms, {}, trailing_state_path=None)
    broker.get_account.side_effect = RuntimeError("account unavailable")
    engine = LiveEngine([], oms, broker)
    engine.kill_switch_manager = manager
    for _ in range(3):
        engine._health_tick()
    assert oms.is_halted
    assert "external:account_snapshot_unavailable" in manager.active_halt_causes()
    # A recovered account read does not silently approve resume.
    broker.get_account.side_effect = None
    broker.get_account.return_value = SimpleNamespace(equity=1000.0)
    engine._health_tick()
    assert oms.is_halted
    assert engine._account_read_failures == 0


@pytest.mark.parametrize("equity", [float("nan"), float("inf"), True, None])
def test_malformed_account_equity_cannot_bypass_failure_counter(equity: object) -> None:
    broker, oms = Mock(), Mock()
    broker.get_account.return_value = SimpleNamespace(equity=equity)
    engine = LiveEngine([], oms, broker)
    for _ in range(3):
        engine._health_tick()
    oms.halt_new_trades.assert_called_once()


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        "[]",
        '{"realized_pnl": "NaN"}',
        '{"open_positions": []}',
        '{"open_positions": {"EURUSD": {}}}',
        '{"pending_exits": {"EURUSD": {}}}',
        '{"pending_entries": {"EURUSD": {}}}',
    ],
)
def test_corrupt_book_cannot_drop_history_or_overwrite_evidence(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "book.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        EventBook(
            state_path=str(path),
            max_loss_pct=0.02,
            per_instrument_max_pct=0.55,
            haven_max_pct=0.60,
            max_holding_hours=4,
            reconcile_grace_sec=120,
        )
    assert path.read_text() == raw
    assert list(tmp_path.iterdir()) == [path]
