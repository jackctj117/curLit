"""Tests for build_fx_broker — fail-fast on missing OANDA creds (CL-qyav P1).

The old engine factory silently fell back to an in-process PaperBroker when
OANDA_API_KEY/OANDA_ACCOUNT_ID were missing while an OANDA mode was
requested — the system then LIED about its broker mode. The policy under
test:

  - missing creds + OANDA mode        → BrokerCredentialsError (fail fast)
  - missing creds + ALLOW_PAPER_FALLBACK=1 → PaperBroker + CRITICAL log,
    and effective_broker_mode() reports "paper", never the requested mode
  - creds present                     → OandaBroker (practice/live URL)

All env access is via monkeypatch — tests never read the real .env.
"""

from __future__ import annotations

import logging

import pytest

from src.execution.broker import (
    ALLOW_PAPER_FALLBACK_ENV,
    BrokerCredentialsError,
    build_fx_broker,
    effective_broker_mode,
)
from src.execution.oanda_broker import OandaBroker
from src.execution.paper_broker import PaperBroker


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic env: no ambient OANDA creds or fallback opt-in leak in."""
    monkeypatch.delenv("OANDA_API_KEY", raising=False)
    monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
    monkeypatch.delenv(ALLOW_PAPER_FALLBACK_ENV, raising=False)


class TestBuildFxBroker:
    def test_paper_mode_returns_paper_broker(self) -> None:
        broker = build_fx_broker("paper")
        assert isinstance(broker, PaperBroker)

    def test_unknown_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown FX broker mode"):
            build_fx_broker("ghost")

    @pytest.mark.parametrize("mode", ["oanda-practice", "oanda-live"])
    def test_missing_creds_fails_fast(self, mode: str) -> None:
        with pytest.raises(BrokerCredentialsError, match="OANDA_API_KEY"):
            build_fx_broker(mode)

    def test_partial_creds_fail_fast_and_name_missing_var(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OANDA_API_KEY", "fake-key")
        with pytest.raises(BrokerCredentialsError, match="OANDA_ACCOUNT_ID"):
            build_fx_broker("oanda-practice")

    def test_fallback_opt_in_must_be_exactly_1(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for value in ("0", "true", "yes", ""):
            monkeypatch.setenv(ALLOW_PAPER_FALLBACK_ENV, value)
            with pytest.raises(BrokerCredentialsError):
                build_fx_broker("oanda-practice")

    def test_explicit_fallback_returns_paper_and_logs_critical(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv(ALLOW_PAPER_FALLBACK_ENV, "1")
        with caplog.at_level(logging.CRITICAL, logger="src.execution.broker"):
            broker = build_fx_broker("oanda-practice")
        assert isinstance(broker, PaperBroker)
        crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert crit, "fallback must log at CRITICAL"
        assert "NOT CONNECTED TO OANDA" in crit[0].getMessage()

    def test_with_creds_returns_practice_oanda_broker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OANDA_API_KEY", "fake-key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "fake-id")
        broker = build_fx_broker("oanda-practice")
        assert isinstance(broker, OandaBroker)
        assert "fxpractice" in str(broker.client.base_url)

    def test_with_creds_returns_live_oanda_broker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OANDA_API_KEY", "fake-key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "fake-id")
        broker = build_fx_broker("oanda-live")
        assert isinstance(broker, OandaBroker)
        assert "fxtrade" in str(broker.client.base_url)


class TestEffectiveBrokerMode:
    def test_fallback_reports_paper_not_oanda(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(ALLOW_PAPER_FALLBACK_ENV, "1")
        broker = build_fx_broker("oanda-practice")
        assert effective_broker_mode("oanda-practice", broker) == "paper"

    def test_real_oanda_reports_requested_mode(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OANDA_API_KEY", "fake-key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "fake-id")
        broker = build_fx_broker("oanda-practice")
        assert effective_broker_mode("oanda-practice", broker) == "oanda-practice"

    def test_paper_mode_reports_paper(self) -> None:
        broker = build_fx_broker("paper")
        assert effective_broker_mode("paper", broker) == "paper"
