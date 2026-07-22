"""Tests for the broker-mode selection in run_engine (CL-920k).

The engine entrypoint now picks between three brokers via --broker:
  paper          — in-process PaperBroker (default)
  oanda-practice — OandaBroker against api-fxpractice.oanda.com
  oanda-live     — OandaBroker against api-fxtrade.oanda.com (requires
                   --confirm-live)

Tests verify build_broker honors each mode and FAILS FAST when OANDA
creds are missing (CL-qyav P1 — the old silent PaperBroker fallback made
the whole system lie about its broker; the downgrade now requires the
explicit ALLOW_PAPER_FALLBACK=1 opt-in).
"""

from __future__ import annotations

import pytest

from src.execution.broker import BrokerCredentialsError
from src.execution.paper_broker import PaperBroker
from src.runtime.run_engine import BROKER_MODES, build_broker


class TestBuildBroker:
    def test_paper_returns_paper_broker(self) -> None:
        broker = build_broker("paper")
        assert isinstance(broker, PaperBroker)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown broker mode"):
            build_broker("ghost")

    def test_oanda_practice_without_creds_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OANDA_API_KEY", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("ALLOW_PAPER_FALLBACK", raising=False)
        with pytest.raises(BrokerCredentialsError):
            build_broker("oanda-practice")

    def test_oanda_live_without_creds_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OANDA_API_KEY", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("ALLOW_PAPER_FALLBACK", raising=False)
        with pytest.raises(BrokerCredentialsError):
            build_broker("oanda-live")

    def test_explicit_fallback_optin_downgrades_to_paper(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OANDA_API_KEY", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
        monkeypatch.setenv("ALLOW_PAPER_FALLBACK", "1")
        broker = build_broker("oanda-practice")
        assert isinstance(broker, PaperBroker)

    def test_oanda_practice_with_creds_returns_oanda_broker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OANDA_API_KEY", "fake-key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "fake-id")
        from src.execution.oanda_broker import OandaBroker
        broker = build_broker("oanda-practice")
        assert isinstance(broker, OandaBroker)
        # Practice URL set
        assert "fxpractice" in str(broker.client.base_url)

    def test_oanda_live_with_creds_returns_oanda_broker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OANDA_API_KEY", "fake-key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "fake-id")
        from src.execution.oanda_broker import OandaBroker
        broker = build_broker("oanda-live")
        assert isinstance(broker, OandaBroker)
        assert "fxtrade" in str(broker.client.base_url)


class TestBrokerModes:
    def test_modes_constant(self) -> None:
        # Guard against drift — the CLI's --broker choices use this set
        assert "paper" in BROKER_MODES
        assert "oanda-practice" in BROKER_MODES
        assert "oanda-live" in BROKER_MODES
