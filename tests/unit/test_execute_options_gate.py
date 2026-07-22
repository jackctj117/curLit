"""Alpaca live-trading dual gate (CL-8lv6).

``ALPACA_PAPER=false`` alone must never arm real-money trading: live mode
requires BOTH env ``ALPACA_LIVE_UNLOCK=1`` AND the ``--confirm-live`` CLI
flag, mirroring OANDA's --confirm-live. Paper (the default) is unaffected.
"""

from __future__ import annotations

import pytest
from scripts.execute_options import _live_gate_error, main


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ALPACA_LIVE_UNLOCK", raising=False)
    monkeypatch.delenv("ALPACA_PAPER", raising=False)


def test_paper_mode_needs_no_gates():
    assert _live_gate_error(paper=True, confirm_live=False) is None


def test_live_refused_without_any_gate():
    err = _live_gate_error(paper=False, confirm_live=False)
    assert err is not None
    assert "ALPACA_LIVE_UNLOCK=1" in err
    assert "--confirm-live" in err


def test_live_refused_with_env_but_no_flag(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE_UNLOCK", "1")
    err = _live_gate_error(paper=False, confirm_live=False)
    assert err is not None
    assert "--confirm-live" in err
    # The satisfied half must not be named as missing.
    assert "ALPACA_LIVE_UNLOCK" not in err


def test_live_refused_with_flag_but_no_env():
    err = _live_gate_error(paper=False, confirm_live=True)
    assert err is not None
    assert "ALPACA_LIVE_UNLOCK=1" in err


def test_live_allowed_with_both_gates(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE_UNLOCK", "1")
    assert _live_gate_error(paper=False, confirm_live=True) is None


def test_main_refuses_live_without_gates(monkeypatch):
    """End-to-end: main() exits 4 at the gate — before any DB/broker work."""
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_OPTIONS_ENABLED", "1")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    assert main(["--once"]) == 4


def test_main_refuses_live_with_flag_but_no_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_OPTIONS_ENABLED", "1")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    assert main(["--once", "--confirm-live"]) == 4
