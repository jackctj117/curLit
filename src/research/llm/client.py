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
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Per-Mtok prices in USD as of 2026-07 — kept here so cost logging is
# self-contained. Adjust when a provider re-prices. Format: (input, output).
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # Claude
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # DeepSeek (kept for the comparison harness; no longer a pipeline
    # default provider).
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v4-pro": (0.55, 2.19),
    # Grok (xAI). NOTE: the grok-4.5 model id contains a DOT — the
    # dashed form "grok-4-5" is a 404 at the API. $0.50/MTok cached
    # input not modeled here.
    "grok-4.5": (2.0, 6.0),
    "grok-4": (3.0, 15.0),
    "grok-3": (3.0, 15.0),
    "grok-3-mini": (0.30, 0.50),
}

# Claude models that reject sampling parameters (`temperature`, `top_p`,
# `top_k`) with a 400 — Fable 5 / Mythos 5 / Opus 4.7+ removed them
# entirely. The driver drops `temperature` proactively for these
# prefixes instead of relying on the error-message retry.
_CLAUDE_NO_SAMPLING_PREFIXES: tuple[str, ...] = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
)


@dataclass
class Message:
    """One conversation turn. ``role`` is 'system' / 'user' / 'assistant'."""

    role: str
    content: str


# Cost provenance labels (CL-h7c1). ``usd_cost`` alone cannot say whether a
# number is an estimate, and ``None`` alone cannot say why it is unknown —
# every response carries one of these so ledgers stay auditable.
#: tokens x the static ``_PRICING_USD_PER_MTOK`` table — an ESTIMATE of the
#: API invoice, not a billed amount.
COST_PRICE_TABLE_ESTIMATE = "price_table_estimate"
#: model absent from the price table — ``usd_cost`` is None (unknown), never 0.
COST_UNPRICED_MODEL = "unpriced_model"
#: ran on a Claude subscription via the ``claude`` CLI — no per-call USD is
#: metered or invoiced, so ``usd_cost`` is None (unknown), never a measured 0.
COST_SUBSCRIPTION_UNMETERED = "subscription_unmetered"
#: driver did not declare a provenance (test doubles, dry-run fakes).
COST_UNSPECIFIED = "unspecified"

#: ``LLMResponse.model`` value when the provider payload does not prove which
#: model served the call (CL-h7c1). The requested model is kept separately in
#: ``LLMResponse.requested_model``; never substitute it here.
UNVERIFIED_MODEL = "unverified"


@dataclass
class LLMResponse:
    """Standardized response shape across providers.

    Unknown is distinct from zero (CL-h7c1): ``input_tokens`` /
    ``output_tokens`` are None when the provider did not report usage, and
    ``usd_cost`` is None when no USD figure exists (see ``cost_provenance``).
    Aggregate with :func:`sum_or_unknown`, never a bare ``sum``.
    """

    text: str
    #: The SERVING model as proven by the provider payload, or
    #: ``UNVERIFIED_MODEL`` when the payload does not prove it.
    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    usd_cost: float | None
    elapsed_sec: float
    raw: Any = field(default=None, repr=False)
    #: The model the caller asked for (may differ from ``model``).
    requested_model: str | None = None
    #: One of the ``COST_*`` labels above.
    cost_provenance: str = COST_UNSPECIFIED
    #: Provider-reported API-equivalent cost when nothing is actually billed
    #: (claude CLI ``total_cost_usd``) — an ESTIMATE, never summed into spend.
    nominal_usd_cost: float | None = None
    #: Requested parameters the transport could NOT enforce (e.g.
    #: ``("temperature",)`` on the claude CLI). Explicit, never silent.
    unenforced_params: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate cost from the per-Mtok price table.

    None when the model is not in the table — an unpriced model's cost is
    UNKNOWN, not zero (CL-h7c1), so spend totals cannot silently undercount.
    """
    if model not in _PRICING_USD_PER_MTOK:
        return None
    in_price, out_price = _PRICING_USD_PER_MTOK[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


def _cost_provenance(usd_cost: float | None) -> str:
    """Provenance label for a price-table estimate (API drivers)."""
    return COST_PRICE_TABLE_ESTIMATE if usd_cost is not None else COST_UNPRICED_MODEL


def sum_or_unknown(values: Iterable[float | None]) -> float | None:
    """Sum costs where ANY unknown (None) makes the total unknown.

    A total with an unmeasured component is not a measurement (CL-h7c1);
    callers that want a lower bound must compute and label it explicitly.
    """
    total = 0.0
    for v in values:
        if v is None:
            return None
        total += v
    return total


def format_usd(value: float | None) -> str:
    """Render a cost for logs/reports; unknown never prints as $0."""
    return "unknown" if value is None else f"${value:.4f}"


# =============================================================================
# Driver interface
# =============================================================================


class Driver(ABC):
    """Provider-specific driver. Implementations construct the underlying
    SDK client and translate ``Message`` lists to the provider's payload.
    """

    name: str  # 'claude' | 'deepseek' | 'grok' | 'claude-code' | future…

    #: Drivers that authenticate out-of-band (e.g. the claude-code
    #: driver rides the CLI's subscription login) set this False and
    #: get_client skips the key requirement.
    requires_api_key: bool = True

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
        # no_tools is a claude-code TRANSPORT hint (CL-scup) — an API call
        # has no tools unless declared, so the capability is already absent
        # here; forwarding the kwarg would 400.
        kwargs.pop("no_tools", None)
        # Anthropic separates system from user/assistant messages.
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        chat: Any = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        request_args: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_text if system_text else "",
            "messages": chat,
            **kwargs,
        }
        # Fable 5 / Mythos 5 / Opus 4.7+ removed sampling params —
        # sending `temperature` is a 400, so drop it up front for those.
        # The reactive retry below stays as a net for models we haven't
        # listed (CL-xo0t).
        if model.startswith(_CLAUDE_NO_SAMPLING_PREFIXES):
            request_args.pop("temperature", None)
        # Fable-class models run safety classifiers that can decline a
        # request (HTTP 200, stop_reason "refusal"). Opt into the
        # server-side fallback so a false-positive decline is re-served
        # by Opus 4.8 inside the same call instead of failing the phase.
        fable_class = model.startswith(("claude-fable-5", "claude-mythos-5"))
        t0 = time.time()
        try:
            if fable_class:
                # Sent via extra_headers/extra_body so this works on
                # SDK versions predating the typed `fallbacks` param
                # (installed 0.97.0 lacks it).
                resp = self._client.messages.create(
                    extra_headers={
                        "anthropic-beta": "server-side-fallback-2026-06-01",
                    },
                    extra_body={
                        "fallbacks": [{"model": "claude-opus-4-8"}],
                    },
                    **request_args,
                )
            else:
                resp = self._client.messages.create(**request_args)
        except Exception as exc:  # anthropic.BadRequestError + base
            msg = str(exc).lower()
            if "temperature" in msg and ("deprecated" in msg or "not supported" in msg):
                logger.info(
                    "Claude model %s rejects `temperature`; retrying without it (CL-xo0t)",
                    model,
                )
                request_args.pop("temperature", None)
                resp = self._client.messages.create(**request_args)
            else:
                raise
        elapsed = time.time() - t0
        if getattr(resp, "stop_reason", None) == "refusal":
            # Whole fallback chain declined (or non-fallback model
            # refused). Fail loud — the loop treats it as a phase error
            # rather than parsing an empty brief.
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            msg = f"Claude {model} refused the request (stop_reason=refusal, category={category!r})"
            raise RuntimeError(msg)
        # Anthropic returns a union of block types — only TextBlock has
        # .text. Fable-class responses may also carry `thinking` /
        # `fallback` blocks before the text, so take the first text
        # block rather than content[0].
        text = next(
            (b.text for b in (resp.content or []) if getattr(b, "type", "") == "text"),
            "",
        )
        # On a fallback rescue the serving model differs from the
        # requested one — price by what actually ran.
        served_model = getattr(resp, "model", model) or model
        usage = resp.usage
        in_tok = int(getattr(usage, "input_tokens", 0))
        out_tok = int(getattr(usage, "output_tokens", 0))
        cost = _cost_usd(served_model, in_tok, out_tok)
        return LLMResponse(
            text=text,
            model=served_model,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            usd_cost=cost,
            elapsed_sec=elapsed,
            raw=resp,
            requested_model=model,
            cost_provenance=_cost_provenance(cost),
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
        kwargs.pop("no_tools", None)  # claude-code transport hint (CL-scup)
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
        cost = _cost_usd(model, in_tok, out_tok)
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            usd_cost=cost,
            elapsed_sec=elapsed,
            raw=resp,
            requested_model=model,
            cost_provenance=_cost_provenance(cost),
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
    can summarize daily spend per provider. ``*_total`` counters hold the
    sum of KNOWN values only; calls with unknown usage / cost are counted
    separately so a summary never presents a partial sum as the total
    (CL-h7c1).
    """

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self.calls: int = 0
        self.input_tokens_total: int = 0
        self.output_tokens_total: int = 0
        self.usd_total: float = 0.0
        self.usage_unknown_calls: int = 0
        self.usd_unknown_calls: int = 0
        #: Provenance labels of every call — a numeric total is only as
        #: "measured" as these say (all current drivers: estimates/unknown).
        self.cost_provenances: set[str] = set()

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
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        self.calls += 1
        if resp.input_tokens is None or resp.output_tokens is None:
            self.usage_unknown_calls += 1
        else:
            self.input_tokens_total += resp.input_tokens
            self.output_tokens_total += resp.output_tokens
        if resp.usd_cost is None:
            self.usd_unknown_calls += 1
        else:
            self.usd_total += resp.usd_cost
        self.cost_provenances.add(resp.cost_provenance)
        logger.debug(
            "llm[%s/%s requested=%s] in=%s out=%s cost=%s (%s) elapsed=%.2fs",
            resp.provider,
            resp.model,
            resp.requested_model,
            resp.input_tokens,
            resp.output_tokens,
            format_usd(resp.usd_cost),
            resp.cost_provenance,
            resp.elapsed_sec,
        )
        return resp

    def usage_summary(self) -> dict[str, Any]:
        """Totals are None when ANY call's value was unknown (CL-h7c1); the
        ``*_known`` fields carry the partial sums, labelled as such, and
        ``usd_cost_provenance`` says what kind of figure ``usd_cost`` is
        (e.g. ``["price_table_estimate"]`` — an estimate, not an invoice)."""
        usage_known = self.usage_unknown_calls == 0
        known_tokens = self.input_tokens_total + self.output_tokens_total
        return {
            "provider": self.provider,
            "calls": self.calls,
            "input_tokens": self.input_tokens_total if usage_known else None,
            "output_tokens": self.output_tokens_total if usage_known else None,
            "total_tokens": known_tokens if usage_known else None,
            "usage_unknown_calls": self.usage_unknown_calls,
            "usd_cost": round(self.usd_total, 6) if self.usd_unknown_calls == 0 else None,
            "usd_cost_known": round(self.usd_total, 6),
            "usd_unknown_calls": self.usd_unknown_calls,
            "usd_cost_provenance": sorted(self.cost_provenances),
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
    driver_cls = _DRIVERS[provider]
    if not driver_cls.requires_api_key:
        # Out-of-band auth (e.g. claude-code rides the CLI's stored
        # subscription login) — no key wanted, none checked.
        return LLMClient(driver=driver_cls(api_key=""))
    if api_key is None:
        env_var = api_key_env or _DEFAULT_API_KEY_ENV.get(provider, "")
        api_key = os.environ.get(env_var, "")
    if not api_key:
        msg = (
            f"no API key for provider {provider!r} "
            f"(checked env var {api_key_env or _DEFAULT_API_KEY_ENV.get(provider)})"
        )
        raise ValueError(msg)
    return LLMClient(driver=driver_cls(api_key=api_key))
