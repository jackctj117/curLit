"""Tests for the BROKER_MODES + build_broker dispatch (CL-poly-2/3).

The mainnet hard-gate is the security-sensitive piece — verified by:
  * default behavior raises a clear error
  * the documented unlock env var changes the rejection
  * preflight failures still abort even when unlocked
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.runtime.run_engine import BROKER_MODES, build_broker


class TestRegistry:
    def test_polymarket_modes_in_registry(self) -> None:
        assert "polymarket-paper" in BROKER_MODES
        assert "polymarket-amoy" in BROKER_MODES
        assert "polymarket-mainnet" in BROKER_MODES

    def test_existing_modes_preserved(self) -> None:
        # Adding new modes must not regress the OANDA + paper paths.
        for legacy in ("paper", "oanda-practice", "oanda-live"):
            assert legacy in BROKER_MODES


class TestPaperBroker:
    def test_polymarket_paper_constructs(self) -> None:
        # No network — paper broker constructs without any external
        # service. Returns the right type.
        broker = build_broker("polymarket-paper")
        from src.execution.polymarket_paper_broker import PolymarketPaperBroker
        assert isinstance(broker, PolymarketPaperBroker)


class TestMainnetHardGate:
    def test_mainnet_default_rejected(self) -> None:
        # Without the unlock var, mainnet must raise.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POLYMARKET_MAINNET_UNLOCK", None)
            with pytest.raises(RuntimeError, match="HARD-GATED"):
                build_broker("polymarket-mainnet")

    def test_mainnet_with_unlock_proceeds_to_preflight(self) -> None:
        # Unlock flag flips the gate; we then expect preflight to be
        # called. Patch preflight to always fail so we can verify
        # we reach it (and don't accidentally instantiate a real
        # ClobClient with bad creds).
        with (
            patch.dict(os.environ, {"POLYMARKET_MAINNET_UNLOCK": "1"}),
            patch(
                "src.execution.polymarket_preflight.run",
                return_value=["fake failure: vault not provisioned"],
            ),
        ):
            with pytest.raises(RuntimeError, match="preflight failed"):
                build_broker("polymarket-mainnet")


class TestAmoyTestnet:
    def test_amoy_calls_preflight(self) -> None:
        # Amoy preflight is called with require_vault=False; we verify
        # the call shape by intercepting.
        captured: dict[str, object] = {}

        def fake_preflight(env: str, *, require_vault: bool, **_kw: object) -> list[str]:
            captured["env"] = env
            captured["require_vault"] = require_vault
            return ["amoy stub failure"]  # abort before py-clob-client

        with patch(
            "src.execution.polymarket_preflight.run",
            side_effect=fake_preflight,
        ):
            with pytest.raises(RuntimeError, match="amoy preflight failed"):
                build_broker("polymarket-amoy")

        assert captured["env"] == "amoy"
        assert captured["require_vault"] is False


class TestUnknown:
    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown broker mode"):
            build_broker("polymarket-evil")
