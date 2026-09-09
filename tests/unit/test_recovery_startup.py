"""Startup readiness is separate from process liveness (CL-0deu.14/18)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.runtime.live_engine import LiveEngine
from src.runtime.release_manifest import redact


@pytest.mark.parametrize("mode", ["unavailable", "raises", "mismatch"])
def test_failed_startup_reconciliation_blocks_entries(mode, monkeypatch):
    monkeypatch.delenv("CURLIT_START_ENTRY_PAUSED", raising=False)
    oms = Mock()
    reconciler = None if mode == "unavailable" else Mock()
    if reconciler is not None:
        if mode == "raises":
            reconciler.reconcile.side_effect = OSError("account unavailable")
        else:
            reconciler.reconcile.return_value = SimpleNamespace(
                entries=["mismatch"], has_mismatches=True
            )
    engine = LiveEngine([], oms, Mock(), cold_start_reconciler=reconciler)
    engine._reconcile_startup()
    oms.halt_new_trades.assert_called_once()
    oms.resume_trading.assert_not_called()
    oms.submit_intent.assert_not_called()


def test_requested_startup_pause_survives_successful_reconciliation(monkeypatch):
    monkeypatch.setenv("CURLIT_START_ENTRY_PAUSED", "1")
    oms = Mock()
    reconciler = Mock()
    reconciler.reconcile.return_value = SimpleNamespace(entries=[], has_mismatches=False)
    engine = LiveEngine([], oms, Mock(), cold_start_reconciler=reconciler)
    engine._reconcile_startup()
    oms.halt_new_trades.assert_called_once()
    oms.resume_trading.assert_not_called()


def test_nested_runtime_credentials_are_redacted_without_changing_limits():
    raw = {
        "risk": {"limit": 12},
        "api_key": "private",
        "children": [{"password": "hidden"}],
        "credentials": {"unusual_name": "also private"},
    }
    clean = redact(raw)
    assert clean == {
        "risk": {"limit": 12},
        "api_key": "[REDACTED]",
        "children": [{"password": "[REDACTED]"}],
        "credentials": "[REDACTED]",
    }
    assert raw["api_key"] == "private"
