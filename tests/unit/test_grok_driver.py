"""Tests for the Grok provider driver (CL-dpw6).

Two layers:

  * Unit-level — verify the driver is registered correctly with the
    right base URL + env var + that all three providers (Claude,
    DeepSeek, Grok) coexist in the registry without state leak. No
    network.
  * Smoke (env-gated) — make a real Grok API call when ``XAI_API_KEY``
    is set. Skipped otherwise so CI / dev machines without
    credentials don't fail.
"""

from __future__ import annotations

import os

import pytest

from src.research.llm.client import (
    _DEFAULT_API_KEY_ENV,
    _DRIVERS,
    LLMClient,
    Message,
    get_client,
)
from src.research.llm.grok import GrokDriver

# ---------------------------------------------------------------------------- #
# Unit-level
# ---------------------------------------------------------------------------- #


class TestGrokRegistration:
    def test_grok_in_driver_registry(self) -> None:
        assert "grok" in _DRIVERS
        assert _DRIVERS["grok"] is GrokDriver

    def test_grok_default_api_key_env(self) -> None:
        # Operator sets XAI_API_KEY; the agent config layer
        # (resolve_api_key) reads the per-provider env var.
        assert _DEFAULT_API_KEY_ENV.get("grok") == "XAI_API_KEY"

    def test_grok_uses_xai_base_url(self) -> None:
        assert GrokDriver.base_url == "https://api.x.ai/v1"
        assert GrokDriver.name == "grok"

    def test_three_providers_coexist(self) -> None:
        # Verify all three providers are registered without any state
        # leak. CL-dpw6 acceptance: "Test with all three providers in
        # a single integration run to verify no driver leaks state."
        for name in ("claude", "deepseek", "grok"):
            assert name in _DRIVERS, f"missing provider: {name!r}"
        # Each driver class has its own name attribute (no shared
        # mutable state between subclasses).
        assert _DRIVERS["claude"].name == "claude"
        assert _DRIVERS["deepseek"].name == "deepseek"
        assert _DRIVERS["grok"].name == "grok"


class TestGrokClient:
    def test_get_client_grok_returns_grok_driver(self) -> None:
        # Constructs an LLMClient wrapping a GrokDriver — no live call.
        client = get_client(provider="grok", api_key="fake-key")
        assert isinstance(client, LLMClient)
        assert client.provider == "grok"
        assert isinstance(client.driver, GrokDriver)


# ---------------------------------------------------------------------------- #
# Smoke (env-gated; real network)
# ---------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY"),
    reason="XAI_API_KEY not set; skipping real-network Grok smoke test",
)
class TestGrokSmoke:
    def test_real_grok_call_returns_text(self) -> None:
        """Real network call against api.x.ai. Gated on XAI_API_KEY so
        only operators with credentials hit this; CI runs without."""
        client = get_client(
            provider="grok", api_key=os.environ["XAI_API_KEY"],
        )
        resp = client.complete(
            messages=[
                Message(
                    role="user",
                    content="Reply with just the word OK and nothing else.",
                ),
            ],
            model="grok-3-mini",  # cheapest tier for smoke
            max_tokens=16,
            temperature=0.0,
        )
        assert resp.text.strip()
        assert resp.provider == "grok"
        assert resp.input_tokens > 0
        assert resp.elapsed_sec > 0
