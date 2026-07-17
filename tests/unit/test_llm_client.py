"""Unit tests for the provider-agnostic LLM client (CL-zhix).

No live API calls — tests use a synthetic Driver subclass that returns
predetermined responses. Validates: provider routing, cost computation,
usage tracking, missing-key error handling, registration of new providers.
"""

from __future__ import annotations

import pytest

from src.research.llm.client import (
    Driver,
    LLMClient,
    LLMResponse,
    Message,
    _cost_usd,
    get_client,
    register_driver,
)


class _StubDriver(Driver):
    """In-memory driver returning canned responses, for testing."""

    name = "stub"

    def __init__(self, api_key: str = "stub-key") -> None:
        super().__init__(api_key)
        self.calls: list[dict] = []

    def complete(self, messages, model, max_tokens=4096, temperature=0.0, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": list(messages), "model": model})
        return LLMResponse(
            text="stub-response",
            model=model,
            provider=self.name,
            input_tokens=100,
            output_tokens=50,
            usd_cost=_cost_usd(model, 100, 50),
            elapsed_sec=0.01,
        )


# =============================================================================
# Cost helper
# =============================================================================


class TestCostHelper:
    def test_known_model_priced(self) -> None:
        # Fable 5: $10/Mtok in, $50/Mtok out
        cost = _cost_usd("claude-fable-5", 1_000_000, 1_000_000)
        assert cost == pytest.approx(10.0 + 50.0)

    def test_unknown_model_zero_cost(self) -> None:
        # Unknown model returns 0 — caller can decide to log/warn
        assert _cost_usd("not-a-real-model", 1000, 1000) == 0.0

    def test_grok_meaningfully_cheaper_than_fable(self) -> None:
        # Sanity check: grok-4.5 should be much cheaper for the same
        # token volume (this is the rationale for routing the bulk
        # roles — extractor / idea / resolver — to Grok)
        grok = _cost_usd("grok-4.5", 100_000, 100_000)
        fable = _cost_usd("claude-fable-5", 100_000, 100_000)
        assert grok * 5 < fable  # at least 5x cheaper


# =============================================================================
# Client behavior
# =============================================================================


class TestLLMClient:
    def test_complete_calls_driver_and_returns_response(self) -> None:
        client = LLMClient(driver=_StubDriver())
        resp = client.complete(
            messages=[Message(role="user", content="hi")],
            model="claude-opus-4-7",
        )
        assert resp.text == "stub-response"
        assert resp.provider == "stub"
        assert resp.model == "claude-opus-4-7"
        assert resp.input_tokens == 100

    def test_usage_accumulates_across_calls(self) -> None:
        client = LLMClient(driver=_StubDriver())
        for _ in range(3):
            client.complete(
                messages=[Message(role="user", content="x")],
                model="deepseek-chat",
            )
        summary = client.usage_summary()
        assert summary["calls"] == 3
        assert summary["input_tokens"] == 300
        assert summary["output_tokens"] == 150
        assert summary["total_tokens"] == 450
        # 3 * (100 * 0.27 + 50 * 1.10) / 1e6 = 0.000246
        assert summary["usd_cost"] == pytest.approx(0.000246)

    def test_provider_property_reflects_driver(self) -> None:
        client = LLMClient(driver=_StubDriver())
        assert client.provider == "stub"


# =============================================================================
# get_client factory + provider registry
# =============================================================================


class TestGetClient:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown LLM provider"):
            get_client(provider="not-a-provider", api_key="x")

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Strip env vars so the factory has nowhere to find a key
        for var in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError, match="no API key"):
            get_client(provider="claude")

    def test_explicit_api_key_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Register the stub so we don't need a real anthropic key for the test
        register_driver("test_stub", _StubDriver)
        # Even if env is empty, an explicit key constructs the driver
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = get_client(provider="test_stub", api_key="my-explicit-key")
        assert client.provider == "stub"
        assert client.driver.api_key == "my-explicit-key"

    def test_register_new_driver_extension_hook(self) -> None:
        """A new provider plugs in via register_driver — the extension hook
        for future Ollama / Bedrock / etc. drivers."""
        class _LocalDriver(Driver):
            name = "local"

            def complete(self, messages, model, max_tokens=4096, temperature=0.0, **kwargs):  # noqa: ANN001
                return LLMResponse(
                    text="local-response", model=model, provider="local",
                    input_tokens=10, output_tokens=5, usd_cost=0.0, elapsed_sec=0.001,
                )

        register_driver("local", _LocalDriver)
        client = get_client(provider="local", api_key="dummy")
        resp = client.complete(
            messages=[Message(role="user", content="x")],
            model="local-llama",
        )
        assert resp.text == "local-response"
        assert resp.usd_cost == 0.0  # local model = no cost
