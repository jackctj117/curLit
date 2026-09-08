"""Kimi's approved-tool research loop (CL-eh28; extends CL-ddzt).

discover_result() returns per-invocation status, provenance, captured sources,
tool trace and usage. Failed calls, malformed output, valid abstention and
budget exhaustion remain distinguishable. discover() is the legacy text
adapter. Claims are validated downstream against tool-captured source records,
not model-provided URLs. Provider failures are logged without account details.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from typing import Any

from src.events.impact_agent import extract_json_object
from src.events.research_evidence import EVIDENCE_INSTRUCTIONS, DiscoveryOutcome, SourceDocument

logger = logging.getLogger(__name__)

#: Moonshot flagship; strong reasoning + tool use. kimi-k3 requires
#: temperature == 1 (a reasoning-model constraint) — that is the default.
DEFAULT_KIMI_MODEL = "kimi-k3"
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MAX_ITERATIONS = 8
#: Cap a tool result fed back to the model (SEC excerpts can be long).
_MAX_TOOL_RESULT_CHARS = 4000

#: OpenAI-format tool schemas the model may call.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_ticker",
            "description": "Verify a US-listed ticker exists and is tradable; "
            "returns its exchange and Robinhood-tradeable flag.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_company",
            "description": "Resolve a company NAME to candidate US-listed tickers "
            "when you're unsure of the exact symbol.",
            "parameters": {
                "type": "object",
                "properties": {"company_name": {"type": "string"}},
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_profile",
            "description": "Sector, industry and business summary for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filing",
            "description": "Excerpt of a company's latest 10-K/10-Q (business + "
            "risk factors) — where real customers, suppliers, "
            "competitors and dependencies are named. Use it to find "
            "the next hop from FACT, not memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "query": {
                        "type": "string",
                        "description": "Relationship or exposure to locate",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


_SYSTEM_PROMPT = EVIDENCE_INSTRUCTIONS + "\nUse the approved ticker, profile and filing tools."


class KimiToolAgent:
    """Runs the agentic tool-loop on the Kimi API and returns the final JSON."""

    def __init__(
        self,
        universe: Any,
        tools: Any,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_iterations: int | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        create_fn: Any = None,
    ) -> None:
        self.universe = universe
        self.tools = tools  # ResearchTools (SEC excerpt + profile)
        self.api_key = (
            api_key
            if api_key is not None
            else os.environ.get(
                "MOONSHOT_API_KEY",
                "",
            )
        )
        self.model = model or os.environ.get("KIMI_MODEL", DEFAULT_KIMI_MODEL)
        self.base_url = base_url or os.environ.get(
            "KIMI_BASE_URL",
            DEFAULT_KIMI_BASE_URL,
        )
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            try:
                self.max_iterations = int(
                    os.environ.get(
                        "KIMI_MAX_ITERATIONS",
                        DEFAULT_MAX_ITERATIONS,
                    )
                )
            except ValueError:
                self.max_iterations = DEFAULT_MAX_ITERATIONS
        self.temperature = temperature
        self.max_tokens = max_tokens
        #: Injectable create fn (model, messages, tools, ...) → response; the
        #: live path builds an OpenAI client lazily against the Moonshot base.
        self._create_fn = create_fn
        self._client: Any = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # -- tools ----------------------------------------------------------

    def _dispatch(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one tool call against the real universe / research tools.
        Always returns a JSON-serialisable dict (never raises)."""
        try:
            if name == "check_ticker":
                t = str(args.get("ticker", "")).strip()
                exists = bool(t) and self.universe.exists(t)
                info = (self.universe.get(t) or {}) if exists else {}
                return {
                    "ticker": t,
                    "exists": exists,
                    "exchange": info.get("exchange"),
                    "robinhood_tradeable": (
                        bool(self.universe.robinhood_tradeable(t)) if exists else False
                    ),
                }
            if name == "resolve_company":
                q = str(args.get("company_name", "")).strip()
                matches = self.universe.resolve_name(q) if q else []
                return {
                    "matches": [
                        {
                            "symbol": m.get("symbol"),
                            "name": m.get("security_name"),
                            "exchange": m.get("exchange"),
                        }
                        for m in matches[:5]
                    ]
                }
            if name == "get_company_profile":
                t = str(args.get("ticker", "")).strip()
                prof = None
                if self.tools is not None:
                    prof = self.tools._profile_fn(t)  # noqa: SLF001
                return prof or {"error": "no profile available"}
            if name == "get_sec_filing":
                t = str(args.get("ticker", "")).strip()
                cik = self.universe.get_cik(t) if hasattr(self.universe, "get_cik") else None
                if cik and callable(getattr(self.tools, "filing_documents", None)):
                    documents = self.tools.filing_documents(cik, t, str(args.get("query", "")))
                    return {"ticker": t, "cik": cik, "sources": [d.to_dict() for d in documents]}
                excerpt = None
                if cik and self.tools is not None:
                    excerpt = self.tools.sec_excerpt(cik)
                return {
                    "ticker": t,
                    "cik": cik,
                    "excerpt": (excerpt or "no filing found")[:_MAX_TOOL_RESULT_CHARS],
                }
        except Exception as exc:
            logger.debug("kimi tool %s failed: %s", name, exc, exc_info=True)
            return {"error": f"tool {name} failed"}
        return {"error": f"unknown tool {name}"}

    # -- loop -----------------------------------------------------------

    def _create(self, **kwargs: Any) -> Any:
        if self._create_fn is not None:
            return self._create_fn(**kwargs)
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415 — lazy

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client.chat.completions.create(**kwargs)

    def _user_prompt(self, event_row: Mapping[str, Any], playbook: Any) -> str:
        pb = ""
        if playbook is not None:
            insts = ", ".join(i.instrument for i in getattr(playbook, "instruments", []))
            pb = (
                f"\nMatched playbook {getattr(playbook, 'key', '')}: "
                f"{getattr(playbook, 'description', '')}\n"
                f"Obvious reachable instruments (go BEYOND these): {insts}"
            )
        return (
            f"EVENT: {event_row.get('headline')}\n"
            f"THEME: {event_row.get('theme') or 'unmatched'}{pb}\n\n"
            "Research and return the niche_ideas JSON. Use the tools."
        )

    def discover(
        self,
        event_row: Mapping[str, Any],
        playbook: Any = None,
    ) -> str:
        """Legacy text adapter. New consumers must use discover_result()."""
        return self.discover_result(event_row, playbook).text

    def discover_result(
        self,
        event_row: Mapping[str, Any],
        playbook: Any = None,
    ) -> DiscoveryOutcome:
        """Per-invocation result: failures never masquerade as abstention."""
        outcome = DiscoveryOutcome("unavailable", provider="moonshot", model=self.model)
        if not self.configured:
            logger.warning("kimi tool agent: no MOONSHOT_API_KEY — skipping")
            outcome.reason = "credentials_missing"
            return outcome
        started = time.monotonic()
        outcome.input_tokens, outcome.output_tokens = 0, 0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(event_row, playbook)},
        ]
        tool_calls_made = 0
        try:
            for _ in range(self.max_iterations):
                logger.info("niche discovery: requesting %s response", self.model)
                resp = self._create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                msg = resp.choices[0].message
                outcome.model = getattr(resp, "model", None) or self.model
                calls = getattr(msg, "tool_calls", None) or []
                messages.append(_assistant_dict(msg, calls))
                usage = getattr(resp, "usage", None)
                for key, attr in (
                    ("input_tokens", "prompt_tokens"),
                    ("output_tokens", "completion_tokens"),
                ):
                    count, total = getattr(usage, attr, None), getattr(outcome, key)
                    setattr(
                        outcome,
                        key,
                        total + count
                        if total is not None and type(count) is int and count >= 0
                        else None,
                    )
                outcome.trace.append(
                    {
                        "model": outcome.model,
                        "prompt": list(messages[:-1]),
                        "response": messages[-1],
                    }
                )
                if not calls:
                    logger.info(
                        "kimi tool agent: event id=%s done after %d tool call(s)",
                        event_row.get("id"),
                        tool_calls_made,
                    )
                    outcome.text = msg.content or ""
                    if getattr(resp.choices[0], "finish_reason", "stop") != "stop":
                        outcome.status, outcome.reason = "partial", "incomplete_generation"
                        return outcome
                    try:
                        ideas = extract_json_object(outcome.text).get("niche_ideas")
                        if not isinstance(ideas, list):
                            raise ValueError("missing ideas")
                        outcome.status = "completed" if ideas else "abstained"
                    except ValueError:
                        outcome.status = "invalid_output"
                        outcome.reason = "invalid_idea_envelope"
                    return outcome
                for tc in calls:
                    if tool_calls_made >= 24:  # Fixed bounded research workload per event.
                        outcome.status, outcome.reason = "budget_exhausted", "tool_call_limit"
                        return outcome
                    tool_calls_made += 1
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except (ValueError, TypeError):
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    result = self._dispatch(tc.function.name, args)
                    for raw in result.get("sources", []):
                        outcome.sources.append(SourceDocument.from_dict(raw))
                    outcome.trace.append(
                        {"tool": tc.function.name, "arguments": args, "result": result}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            # Sources are already excerpt-bounded. Do not truncate
                            # JSON or a source ID/quote after recording its hash.
                            "content": json.dumps(result),
                        }
                    )
            logger.info(
                "kimi tool agent: event id=%s hit max_iterations (%d) — no final answer",
                event_row.get("id"),
                self.max_iterations,
            )
            outcome.status, outcome.reason = "budget_exhausted", "model_call_limit"
            return outcome
        except Exception as exc:
            logger.warning(
                "kimi tool agent failed for event id=%s: %s",
                event_row.get("id"),
                type(exc).__name__,  # Provider errors may contain account/key identifiers.
            )
            outcome.reason = (
                "insufficient_balance"
                if "insufficient balance" in str(exc).lower()
                else type(exc).__name__
            )
            return outcome
        finally:
            outcome.elapsed_sec = time.monotonic() - started


def _assistant_dict(msg: Any, calls: list[Any]) -> dict[str, Any]:
    """Serialise an assistant message (with any tool_calls) back into the
    conversation so the follow-up tool results reference the right ids."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in calls
        ]
    return out
