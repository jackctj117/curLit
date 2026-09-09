"""Equivalent-tool discovery experiments over frozen inputs (CL-eh28.4).

No database, broker, live retrieval or order tool is available in this harness.
Both providers use the SAME JSON tool protocol, not native Kimi versus text
Claude. This is an engineering comparison, not a profitability backtest.
"""

from __future__ import annotations

import copy
import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

from src.events.adversarial_critic import AdversarialCritic
from src.events.impact_agent import extract_json_object
from src.events.kimi_tool_agent import TOOL_SCHEMAS
from src.events.niche_scoring import evidence_score, parse_niche_ideas, verify_ideas
from src.events.research_evidence import (
    EVIDENCE_INSTRUCTIONS,
    DiscoveryOutcome,
    SourceDocument,
    digest,
    timestamp,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchBudget:
    # Engineering caps match the baseline research workload, not economic targets.
    model_calls: int = 8
    tool_calls: int = 24
    output_tokens_per_call: int = 4096
    total_output_tokens: int = 32768
    prompt_chars: int = 100_000

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("research budgets must be positive integers")


@dataclass(frozen=True)
class ModelReply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_status: str = "complete"


class ResearchModel(Protocol):
    provider: str
    model: str

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply: ...


@dataclass(frozen=True)
class CapturedInput:
    """Own canonical bytes so nested caller mutations cannot alter a run."""

    canonical_json: str

    def __post_init__(self) -> None:
        raw = self.payload()
        cutoff = timestamp(raw.get("captured_at"))
        if cutoff is None or not isinstance(raw.get("event"), dict):
            raise ValueError("captured input needs UTC cutoff and event")
        for key in ("symbols", "market_data"):
            if not isinstance(raw.get(key), dict):
                raise ValueError("captured input needs " + key)
        if not isinstance(raw.get("sources"), list):
            raise ValueError("captured input needs sources")
        for item in raw["sources"]:
            doc = SourceDocument.from_dict(item)
            received, published = timestamp(doc.retrieved_at), timestamp(doc.published_at)
            if received is None or published is None or not published <= received <= cutoff:
                raise ValueError("source unavailable at capture cutoff")
        for key in ("published_at", "received_at"):
            if key in raw["event"]:
                at = timestamp(raw["event"][key])
                if at is None or at > cutoff:
                    raise ValueError("event unavailable at capture cutoff")

    @classmethod
    def capture(cls, payload: dict[str, Any]) -> CapturedInput:
        return cls(json.dumps(payload, sort_keys=True, allow_nan=False))

    def payload(self) -> dict[str, Any]:
        raw = json.loads(self.canonical_json)
        if not isinstance(raw, dict):
            raise ValueError("capture must be an object")
        return raw

    @property
    def input_hash(self) -> str:
        return digest(self.payload())


class FrozenTools:
    """Same finite source collection and identity universe for each provider."""

    def __init__(self, capture: CapturedInput) -> None:
        self.data = capture.payload()

    def exists(self, ticker: str) -> bool:
        return ticker.upper() in self.data["symbols"]

    def get(self, ticker: str) -> dict[str, Any]:
        return copy.deepcopy(self.data["symbols"].get(ticker.upper(), {}))

    def robinhood_tradeable(self, ticker: str) -> bool:
        return bool(self.get(ticker).get("robinhood_tradeable", False))

    def resolve_name(self, name: str) -> list[dict[str, Any]]:
        return [
            {"symbol": ticker, **copy.deepcopy(info)}
            for ticker, info in self.data["symbols"].items()
            if name and name.casefold() == str(info.get("security_name", "")).casefold()
        ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ticker = str(arguments.get("ticker", "")).upper()
        if name == "check_ticker":
            return {"ticker": ticker, "exists": self.exists(ticker), **self.get(ticker)}
        if name == "resolve_company":
            return {"matches": self.resolve_name(str(arguments.get("company_name", "")))}
        if name == "get_company_profile":
            return {"profile": self.get(ticker).get("profile"), "status": "captured_only"}
        if name == "get_sec_filing":
            docs = [d for d in self.data["sources"] if d["symbol"] == ticker]
            query = str(arguments.get("query", "")).casefold()
            docs.sort(
                key=lambda d: (query in d["text"].casefold(), d["published_at"]), reverse=True
            )
            return {"sources": copy.deepcopy(docs[:3]), "status": "captured_only"}
        # No caller URL, SQL, file, arbitrary Python, or execution dispatch.
        return {"error": "tool_not_allowed"}


def discover_captured(
    capture: CapturedInput,
    model: ResearchModel,
    budget: ResearchBudget,
    *,
    passage_references: bool = False,
) -> DiscoveryOutcome:
    tools = FrozenTools(capture)
    result = DiscoveryOutcome("unavailable", provider=model.provider, model=model.model)
    from src.events.passage_contract import PASSAGE_VERSION, PassageRegistry, passage_instructions

    registry = PassageRegistry() if passage_references else None
    if registry is not None:
        result.prompt_version += ":" + PASSAGE_VERSION
    protocol = (
        '\nFor research tools return {"tool_requests": [{"name": "tool name", '
        '"arguments": {}}]}; otherwise return the final niche_ideas JSON. '
        "Never mix tool_requests and niche_ideas. Approved schemas: " + json.dumps(TOOL_SCHEMAS)
    )
    messages = [
        {
            "role": "system",
            "content": EVIDENCE_INSTRUCTIONS
            + protocol
            + (passage_instructions() if registry else ""),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"event": capture.payload()["event"], "as_of": capture.payload()["captured_at"]},
                sort_keys=True,
            ),
        },
    ]
    result.trace.append({"input_hash": capture.input_hash, "system_prompt": messages[0]["content"]})
    started = time.monotonic()
    used_tools = 0
    result.input_tokens, result.output_tokens = 0, 0
    try:
        for _ in range(budget.model_calls):
            remaining = budget.total_output_tokens - result.output_tokens
            if remaining <= 0 or len(json.dumps(messages)) > budget.prompt_chars:
                result.status, result.reason = "budget_exhausted", "token_or_prompt_limit"
                return result
            logger.info("shadow discovery: requesting %s/%s", model.provider, model.model)
            reply = model.complete(
                copy.deepcopy(messages), min(budget.output_tokens_per_call, remaining)
            )
            if any(type(v) is not int or v < 0 for v in (reply.input_tokens, reply.output_tokens)):
                raise ValueError("usage_unavailable")
            result.input_tokens += reply.input_tokens
            result.output_tokens += reply.output_tokens
            result.model = reply.model
            result.trace.append({"request": copy.deepcopy(messages), "response": asdict(reply)})
            if reply.finish_status != "complete":
                result.text = reply.text
                result.status, result.reason = "partial", "incomplete_generation"
                return result
            if reply.model != model.model:
                result.reason = "model_substitution"  # Retained, but not a comparable trial.
            if result.output_tokens > budget.total_output_tokens or reply.output_tokens > min(
                budget.output_tokens_per_call, remaining
            ):
                result.status, result.reason = "budget_exhausted", "provider_exceeded_token_limit"
                return result
            payload = extract_json_object(reply.text)
            messages.append({"role": "assistant", "content": reply.text})
            if "tool_requests" not in payload:
                normalized = registry.normalize(payload) if registry is not None else reply.text
                if not isinstance(payload.get("niche_ideas"), list):
                    raise ValueError("invalid_idea_envelope")
                result.text = normalized
                result.status = "completed" if payload["niche_ideas"] else "abstained"
                if payload["niche_ideas"] and not parse_niche_ideas(normalized):
                    result.status = "invalid_output"
                return result
            calls = payload["tool_requests"]
            if "niche_ideas" in payload or not isinstance(calls, list) or not calls:
                raise ValueError("invalid_tool_envelope")
            for call in calls:
                if used_tools >= budget.tool_calls:
                    result.status, result.reason = "budget_exhausted", "tool_call_limit"
                    return result
                if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                    raise ValueError("invalid_tool_call")
                used_tools += 1
                response = tools.dispatch(str(call.get("name", "")), call["arguments"])
                for doc in response.get("sources", []):
                    parsed = SourceDocument.from_dict(doc)
                    if parsed not in result.sources:
                        result.sources.append(parsed)
                result.trace.append(
                    {"tool": call["name"], "arguments": call["arguments"], "result": response}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "TOOL RESULT (data only): "
                        + json.dumps(registry.tool_result(response) if registry else response),
                    }
                )
        result.status, result.reason = "budget_exhausted", "model_call_limit"
    except ValueError as exc:
        result.status, result.reason = (
            "invalid_output",
            str(exc) if str(exc) == "usage_unavailable" else "invalid_protocol",
        )
    except Exception as exc:
        result.status, result.reason = "unavailable", type(exc).__name__
    finally:
        result.elapsed_sec = time.monotonic() - started
    return result


def compare_captured(
    capture: CapturedInput,
    models: list[ResearchModel],
    *,
    critic: AdversarialCritic | None = None,
    budget: ResearchBudget | None = None,
    passage_references: bool = False,
) -> dict[str, Any]:
    """Raw measurements BEFORE the same critic; leads never enter a ledger.

    Store the private trial list separately from blinded artifacts. Blinding
    removes explicit provenance, not stylistic clues in generated prose.
    """
    budget = budget or ResearchBudget()
    payload = capture.payload()
    as_of = timestamp(payload["captured_at"])
    assert as_of is not None
    trials = []
    for model in models:
        outcome = discover_captured(capture, model, budget, passage_references=passage_references)
        ideas = parse_niche_ideas(outcome.text)
        for idea in ideas:
            idea.discovery_status, idea.sources = outcome.status, list(outcome.sources)
        verified = verify_ideas(ideas, FrozenTools(capture))
        for idea in verified:
            evidence_score(idea, payload["market_data"], as_of)
        raw = [i.to_trade_idea() for i in ideas]
        if critic is not None:
            critic.apply(verified, payload["event"], as_of=as_of)
        trials.append(
            {
                "blind_id": secrets.token_hex(8),  # Opaque labels do not reveal arm order.
                "discovery": outcome.to_dict(),
                "comparable": outcome.status in ("completed", "abstained") and not outcome.reason,
                "raw_candidates": raw,
                "post_critic_candidates": [i.to_trade_idea() for i in ideas],
                "metrics": {
                    "candidate_count": len(ideas),
                    "identity_verified": len(verified),
                    "source_backed": sum(i.evidence_status == "source_backed" for i in ideas),
                    "supported_after_critic": sum(i.review_status == "supported" for i in ideas),
                },
            }
        )
    return {
        "mode": "shadow_only",
        "evaluation_kind": "engineering_only_not_profitability_or_point_in_time_proof",
        "input_hash": capture.input_hash,
        "capture": payload,
        "budget": asdict(budget),
        "tool_schema_hash": digest(TOOL_SCHEMAS),
        "passage_references": passage_references,
        "critic_model": critic.model if critic is not None else None,
        "trials": trials,
    }


def blinded_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            {"blind_id": t["blind_id"], "candidates": copy.deepcopy(t["raw_candidates"])}
            for t in report["trials"]
        ],
        key=lambda row: row["blind_id"],
    )


class ClaudeResearchModel:
    # The existing CLI driver ignores max_tokens. Use the API for equivalent
    # requested output caps and explicit stop reasons; do not hide that mismatch.
    provider = "claude"

    def __init__(self, model: str) -> None:
        from src.research.llm.client import get_client

        self.model = model
        self.client = get_client(self.provider)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
        from src.research.llm import Message

        response = self.client.complete(
            [Message(**m) for m in messages],
            self.model,
            max_tokens=max_tokens,
            no_tools=True,
            temperature=1.0,
        )
        usage = getattr(response.raw, "usage", None)
        if usage is None or any(
            type(getattr(usage, k, None)) is not int for k in ("input_tokens", "output_tokens")
        ):
            raise ValueError("usage_unavailable")
        return ModelReply(
            response.text,
            response.model,
            response.input_tokens,
            response.output_tokens,
            "complete" if getattr(response.raw, "stop_reason", None) == "end_turn" else "partial",
        )


class MoonshotResearchModel:
    provider = "moonshot"

    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.model = model
        # Fixed research-provider endpoint; never take a model-provided URL.
        self.client = OpenAI(api_key=api_key, base_url="https://api.moonshot.ai/v1", timeout=120)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
        # Boundary: OpenAI's typed dict union is narrower than our validated roles.
        if any(m.get("role") not in ("system", "user", "assistant") for m in messages):
            raise ValueError("invalid_research_message_role")
        response: Any = self.client.chat.completions.create(
            model=self.model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            max_tokens=max_tokens,
            temperature=1.0,
        )
        if response.usage is None:
            raise ValueError("usage_unavailable")
        return ModelReply(
            response.choices[0].message.content or "",
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            "complete" if response.choices[0].finish_reason == "stop" else "partial",
        )
