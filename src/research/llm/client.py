"""LLM client abstraction (CL-zhix) — provider-agnostic surface for the
multi-agent research pipeline.

Three providers wired in v1:
  * `claude`   via the official Anthropic SDK (api.anthropic.com)
  * `deepseek` via the OpenAI SDK pointed at api.deepseek.com (it is
                OpenAI-compatible)
  * `grok`     via the OpenAI SDK pointed at api.x.ai (xAI; CL-dpw6
                handles the dedicated cost/limit tuning)

A new provider plugs in by adding a `Driver` subclass and registering it
in ``_DRIVERS``. The agent config layer (CL-rzf7) just references the
provider name; agents stay provider-agnostic.

Cost is tracked per call so the loop runner can surface daily spend.

Tests use the ``MockClient`` in ``tests/unit/test_llm_client.py`` — no
live calls in the test suite.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Per-Mtok prices in USD as of 2026-04 — kept here so cost logging is
# self-contained. Adjust when a provider re-prices. Format: (input, output).
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # Claude
    "claude-opus-4-7":          (15.0, 75.0),
    "claude-sonnet-4-6":        (3.0,  15.0),
    "claude-haiku-4-5-20251001": (1.0,   5.0),
    # DeepSeek
    "deepseek-chat":            (0.27, 1.10),
    "deepseek-reasoner":        (0.55, 2.19),
    # deepseek-v4-pro pricing not yet verified against the rate card —
    # operator should confirm. Using deepseek-reasoner's rate as a
    # conservative placeholder so cost reports don't show $0. If the
    # model ID isn't recognized at the API end, requests will 400
    # cleanly without spending tokens.
    "deepseek-v4-pro":          (0.55, 2.19),
    # Grok (xAI). Tune in CL-dpw6.
    "grok-4":                   (3.0,  15.0),
    "grok-3":                   (3.0,  15.0),
    "grok-3-mini":              (0.30, 0.50),
}


@dataclass
class Message:
    """One conversation turn. ``role`` is 'system' / 'user' / 'assistant'."""

    role: str
    content: str


@dataclass
class LLMResponse:
    """Standardized response shape across providers."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    usd_cost: float
    elapsed_sec: float
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Look up the per-Mtok price and compute cost. 0.0 if model unknown."""
    if model not in _PRICING_USD_PER_MTOK:
        return 0.0
    in_price, out_price = _PRICING_USD_PER_MTOK[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


# =============================================================================
# Driver interface
# =============================================================================


class Driver(ABC):
    """Provider-specific driver. Implementations construct the underlying
    SDK client and translate ``Message`` lists to the provider's payload.
    """

    name: str  # 'claude' | 'deepseek' | 'grok' | future…

    def __init__(self, api_key: str) -> None:
        if not api_key:
            msg = f"{type(self).__name__} requires non-empty API key"
            raise ValueError(msg)
        self.api_key = api_key

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse: ...


class _ClaudeDriver(Driver):
    """Anthropic SDK driver."""

    name = "claude"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from anthropic import Anthropic  # lazy: only import if used
        self._client = Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        # Anthropic separates system from user/assistant messages.
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        chat: Any = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role in ("user", "assistant")
        ]
        t0 = time.time()
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_text if system_text else "",
            messages=chat,
            **kwargs,
        )
        elapsed = time.time() - t0
        # Anthropic returns a union of block types — only TextBlock has .text.
        # We only request plain-text generation here so the first block is
        # always a TextBlock, but mypy can't prove it from the union type.
        first = resp.content[0] if resp.content else None
        text = getattr(first, "text", "") if first is not None else ""
        usage = resp.usage
        in_tok = int(getattr(usage, "input_tokens", 0))
        out_tok = int(getattr(usage, "output_tokens", 0))
        return LLMResponse(
            text=text, model=model, provider=self.name,
            input_tokens=in_tok, output_tokens=out_tok,
            usd_cost=_cost_usd(model, in_tok, out_tok),
            elapsed_sec=elapsed, raw=resp,
        )


class _OpenAICompatDriver(Driver):
    """Shared driver for any OpenAI-API-compatible endpoint.

    DeepSeek and xAI Grok both serve the OpenAI chat-completions API with
    a different ``base_url``, so a single driver covers both. Subclass to
    set ``name`` and ``base_url`` and pass through.
    """

    base_url: str = ""

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from openai import OpenAI  # lazy
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        chat: Any = [{"role": m.role, "content": m.content} for m in messages]
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat,
            **kwargs,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0))
        out_tok = int(getattr(usage, "completion_tokens", 0))
        return LLMResponse(
            text=text, model=model, provider=self.name,
            input_tokens=in_tok, output_tokens=out_tok,
            usd_cost=_cost_usd(model, in_tok, out_tok),
            elapsed_sec=elapsed, raw=resp,
        )


class _DeepSeekDriver(_OpenAICompatDriver):
    name = "deepseek"
    base_url = "https://api.deepseek.com"


# =============================================================================
# Public client
# =============================================================================


# Grok lives in its own module per CL-dpw6 acceptance and registers
# itself at import time via ``register_driver``. Avoid a top-level
# import here (the module's class subclasses ``_OpenAICompatDriver``
# defined above, so importing it before ``register_driver`` is defined
# would either circularly fail or pull the registration too early).
_DRIVERS: dict[str, type[Driver]] = {
    "claude": _ClaudeDriver,
    "deepseek": _DeepSeekDriver,
}


_DEFAULT_API_KEY_ENV: dict[str, str] = {
    "claude": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "grok": "XAI_API_KEY",
}


class LLMClient:
    """Provider-agnostic client. Instantiate via ``get_client(provider)``.

    Tracks cumulative cost + token usage across calls so the loop runner
    can summarize daily spend per provider.
    """

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self.calls: int = 0
        self.input_tokens_total: int = 0
        self.output_tokens_total: int = 0
        self.usd_total: float = 0.0

    @property
    def provider(self) -> str:
        return self.driver.name

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        resp = self.driver.complete(
            messages=messages, model=model,
            max_tokens=max_tokens, temperature=temperature,
            **kwargs,
        )
        self.calls += 1
        self.input_tokens_total += resp.input_tokens
        self.output_tokens_total += resp.output_tokens
        self.usd_total += resp.usd_cost
        logger.debug(
            "llm[%s/%s] in=%d out=%d cost=$%.4f elapsed=%.2fs",
            resp.provider, resp.model,
            resp.input_tokens, resp.output_tokens,
            resp.usd_cost, resp.elapsed_sec,
        )
        return resp

    def usage_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "calls": self.calls,
            "input_tokens": self.input_tokens_total,
            "output_tokens": self.output_tokens_total,
            "total_tokens": self.input_tokens_total + self.output_tokens_total,
            "usd_cost": round(self.usd_total, 6),
        }


def register_driver(name: str, driver_cls: type[Driver]) -> None:
    """Register a new provider driver — extension hook for future providers
    (e.g. local Ollama, Bedrock-hosted Claude). Called once at import time
    by the new driver's module."""
    _DRIVERS[name] = driver_cls


def get_client(
    provider: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> LLMClient:
    """Instantiate an ``LLMClient`` for the given provider.

    If ``api_key`` is None we read it from ``api_key_env`` (or the
    provider's default env var). Raises ``ValueError`` if the provider is
    unknown or the key is missing — fail-loud at construction time, not
    on first call.
    """
    if provider not in _DRIVERS:
        msg = f"unknown LLM provider {provider!r}; available: {sorted(_DRIVERS)}"
        raise ValueError(msg)
    if api_key is None:
        env_var = api_key_env or _DEFAULT_API_KEY_ENV.get(provider, "")
        api_key = os.environ.get(env_var, "")
    if not api_key:
        msg = (
            f"no API key for provider {provider!r} "
            f"(checked env var {api_key_env or _DEFAULT_API_KEY_ENV.get(provider)})"
        )
        raise ValueError(msg)
    driver_cls = _DRIVERS[provider]
    return LLMClient(driver=driver_cls(api_key=api_key))
