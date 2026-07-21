"""Agentic Kimi tool-loop for the niche pass (CL-ddzt).

The retrieval-augmented pass (CL-2czc) grounds hops in real data, but WE decide
what to look up. This is the full agentic version: the MODEL drives the tools
mid-generation — it proposes a name, calls a tool to verify the ticker or pull
the company's 10-K, reads the real customers/suppliers/risks in the result,
and decides its next hop from fact. It loops call → tool → call until it emits
the final niche-idea JSON.

Runs on the Kimi (Moonshot AI) API — OpenAI-compatible function calling, cheap,
strong at tool use. This is an INTENTIONAL, opt-in, API-billed path (the
operator supplied a Kimi key specifically for it); the subscription claude-code
cycles remain the default. It reuses the SAME tools as the retrieval pass
(:class:`SymbolUniverse` + :class:`ResearchTools`), so verification and
liquidity gating downstream are unchanged.

``discover()`` returns the model's final JSON text; the caller
(:class:`NicheAgent`) parses it with the existing ``parse_niche_ideas`` and runs
the same verify / score / gate — so no unverified ticker survives regardless of
what the model claims. Fail-soft: any API/loop failure returns ``""``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

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
    {"type": "function", "function": {
        "name": "check_ticker",
        "description": "Verify a US-listed ticker exists and is tradable; "
                       "returns its exchange and Robinhood-tradeable flag.",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "resolve_company",
        "description": "Resolve a company NAME to candidate US-listed tickers "
                       "when you're unsure of the exact symbol.",
        "parameters": {"type": "object", "properties": {
            "company_name": {"type": "string"}}, "required": ["company_name"]}}},
    {"type": "function", "function": {
        "name": "get_company_profile",
        "description": "Sector, industry and business summary for a ticker.",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "get_sec_filing",
        "description": "Excerpt of a company's latest 10-K/10-Q (business + "
                       "risk factors) — where real customers, suppliers, "
                       "competitors and dependencies are named. Use it to find "
                       "the next hop from FACT, not memory.",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string"}}, "required": ["ticker"]}}},
]

_SYSTEM_PROMPT = """\
You are the NICHE OPPORTUNITY researcher for an event desk. Given an event the
desk has already assessed (the obvious liquid trade is known), hunt the
high-torque, UNDER-FOLLOWED second- and third-order names it misses.

You have TOOLS — USE THEM, don't guess:
- check_ticker: confirm any ticker you name is real and tradable.
- resolve_company: turn a company name into its real ticker when unsure.
- get_company_profile: what a company actually does.
- get_sec_filing: a company's real 10-K/10-Q — read the named customers,
  suppliers, competitors and dependencies, and hop from THOSE facts.

Method: reason MULTI-HOP (hop 1 directly affected -> hop 2 suppliers /
customers / competitors / financiers -> hop 3+ niche pure-plays, juniors,
royalty/streaming, equipment, logistics with high operational/financial
leverage). At each hop, pull a filing or profile to find the NEXT real name
rather than recalling one. Verify every ticker with check_ticker or
resolve_company before you rely on it.

Prefer SMALLER, less-covered, non-consensus names with more upside torque than
the obvious large-caps. This is NOT a lottery ticket: high-asymmetry,
~3-10x-if-right with DEFINED risk. Cite the chain (name each hop) in the
rationale. A confident wrong ticker on an illiquid name is real money on
fiction.

When you have finished researching, respond with ONE JSON object and NOTHING
else — no tool call, no prose:
{
  "niche_ideas": [
    {"ticker": "<verified US-listed ticker>",
     "company_name": "<exact company name>",
     "action": "long" | "short" | "buy_calls" | "buy_puts",
     "direction": "bullish" | "bearish",
     "hop_count": <int 1-5, links from the obvious trade>,
     "torque_reason": "<one line: the leverage mechanism>",
     "rationale": "<cite the multi-hop chain, note which filing confirmed it>",
     "confidence": <float 0.0-1.0>}
  ]
}
Return 2-6 ideas, mixing obvious-adjacent (hop 1-2) and niche (hop 3+).
"""


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
        self.api_key = api_key if api_key is not None else os.environ.get(
            "MOONSHOT_API_KEY", "",
        )
        self.model = model or os.environ.get("KIMI_MODEL", DEFAULT_KIMI_MODEL)
        self.base_url = base_url or os.environ.get(
            "KIMI_BASE_URL", DEFAULT_KIMI_BASE_URL,
        )
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            try:
                self.max_iterations = int(os.environ.get(
                    "KIMI_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS,
                ))
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
                    "ticker": t, "exists": exists,
                    "exchange": info.get("exchange"),
                    "robinhood_tradeable": (
                        bool(self.universe.robinhood_tradeable(t)) if exists
                        else False
                    ),
                }
            if name == "resolve_company":
                q = str(args.get("company_name", "")).strip()
                matches = self.universe.resolve_name(q) if q else []
                return {"matches": [
                    {"symbol": m.get("symbol"),
                     "name": m.get("security_name"),
                     "exchange": m.get("exchange")}
                    for m in matches[:5]
                ]}
            if name == "get_company_profile":
                t = str(args.get("ticker", "")).strip()
                prof = None
                if self.tools is not None:
                    prof = self.tools._profile_fn(t)  # noqa: SLF001
                return prof or {"error": "no profile available"}
            if name == "get_sec_filing":
                t = str(args.get("ticker", "")).strip()
                cik = self.universe.get_cik(t) if hasattr(
                    self.universe, "get_cik") else None
                excerpt = None
                if cik and self.tools is not None:
                    excerpt = self.tools.sec_excerpt(cik)
                return {
                    "ticker": t, "cik": cik,
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
            insts = ", ".join(
                i.instrument for i in getattr(playbook, "instruments", [])
            )
            pb = (f"\nMatched playbook {getattr(playbook, 'key', '')}: "
                  f"{getattr(playbook, 'description', '')}\n"
                  f"Obvious reachable instruments (go BEYOND these): {insts}")
        return (
            f"EVENT: {event_row.get('headline')}\n"
            f"THEME: {event_row.get('theme') or 'unmatched'}{pb}\n\n"
            "Research and return the niche_ideas JSON. Use the tools."
        )

    def discover(
        self, event_row: Mapping[str, Any], playbook: Any = None,
    ) -> str:
        """Run the agentic tool-loop; return the model's final JSON text (or
        "" on any failure / no key / iteration budget exhausted)."""
        if not self.configured:
            logger.warning("kimi tool agent: no MOONSHOT_API_KEY — skipping")
            return ""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(event_row, playbook)},
        ]
        tool_calls_made = 0
        try:
            for _ in range(self.max_iterations):
                resp = self._create(
                    model=self.model, messages=messages, tools=TOOL_SCHEMAS,
                    tool_choice="auto", temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                msg = resp.choices[0].message
                calls = getattr(msg, "tool_calls", None) or []
                messages.append(_assistant_dict(msg, calls))
                if not calls:
                    logger.info(
                        "kimi tool agent: event id=%s done after %d tool call(s)",
                        event_row.get("id"), tool_calls_made,
                    )
                    return msg.content or ""
                for tc in calls:
                    tool_calls_made += 1
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except (ValueError, TypeError):
                        args = {}
                    result = self._dispatch(tc.function.name, args)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result)[:_MAX_TOOL_RESULT_CHARS],
                    })
            logger.info(
                "kimi tool agent: event id=%s hit max_iterations (%d) — "
                "no final answer", event_row.get("id"), self.max_iterations,
            )
            return ""
        except Exception as exc:
            logger.warning(
                "kimi tool agent failed for event id=%s: %s",
                event_row.get("id"), str(exc)[:200],
            )
            return ""


def _assistant_dict(msg: Any, calls: list[Any]) -> dict[str, Any]:
    """Serialise an assistant message (with any tool_calls) back into the
    conversation so the follow-up tool results reference the right ids."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name,
                          "arguments": tc.function.arguments}}
            for tc in calls
        ]
    return out
