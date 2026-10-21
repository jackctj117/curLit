"""Hop-graph vs baseline replay over captured events (CL-ynuh, SHADOW ONLY).

Replays frozen ``niche_shadow.CapturedInput`` snapshots through:

* the HOP-GRAPH arm — :class:`HopGraphTraversal` with an
  :class:`EdgeVerifier` whose retrieval is the capture's frozen source
  collection (``FrozenTools.dispatch("get_sec_filing")``), and
* the existing BASELINE arm — ``niche_shadow.compare_captured`` (the
  equivalent-tool discovery loop) with whatever ResearchModel the operator
  supplies,

then applies the IDENTICAL unchanged gates to both: ``verify_ideas`` against
the same frozen identity universe, ``evidence_score`` with the same captured
market data and cutoff, and the same critic instance. Nothing is merged into
``trade_ideas``, persisted to a ledger, or sent to a broker.

The report is an engineering measurement (pre-registered metrics: candidates,
evidence-gate passes, eligible ideas per event, hop-depth distribution, model
and tool calls, cost provenance). It is not a profitability result, and
"eligible" here means only that the unchanged gates passed on a capture.
Unknown token/cost figures stay None, never zero (CL-h7c1).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from src.events.adversarial_critic import AdversarialCritic
from src.events.hop_graph import EdgeMemory, HopGraphTraversal
from src.events.hop_graph_verify import EdgeVerifier, EntailmentChecker, FilingRetriever
from src.events.niche_scoring import NicheIdea, evidence_score, verify_ideas
from src.events.niche_shadow import (
    CapturedInput,
    FrozenTools,
    ModelReply,
    ResearchBudget,
    ResearchModel,
    compare_captured,
)
from src.events.playbooks import Playbook
from src.events.research_evidence import SourceDocument, digest, timestamp
from src.events.subscription_shadow_budget import (
    SUBSCRIPTION_MODEL,
    EventCallBudget,
    SubscriptionBudgetClient,
    SubscriptionResearchModel,
)
from src.research.llm.client import LLMClient, sum_or_unknown

logger = logging.getLogger(__name__)

REPORT_VERSION = "hop-graph-shadow-v2"
# CL-ynuh predeclared engineering sample, not a profitability/power threshold.
MIN_COMPARISON_EVENTS = 20


def validate_capture_cohort(captures: Sequence[CapturedInput]) -> dict[str, Any]:
    """Reject repeated observations before they consume calls or inflate n.

    A capture is already internally timestamp-validated. IDs are scoped to
    this single event corpus; string/integer aliases are not new observations.
    Reassigning an ID to an otherwise identical event is not a new event either.
    This checks supplied records, not authenticity or statistical independence.
    """
    if not captures:
        raise ValueError("no captures supplied")
    identities: set[str] = set()
    hashes: set[str] = set()
    event_hashes: set[str] = set()
    for capture in captures:
        event = capture.payload()["event"]
        raw_id = event.get("id")
        if type(raw_id) is int and raw_id > 0:
            identity = str(raw_id)
        elif isinstance(raw_id, str) and raw_id.strip():
            identity = raw_id.strip()
            numeric_part = identity[1:] if identity.startswith(("+", "-")) else identity
            if numeric_part.isdecimal():
                identity = str(int(identity))
                if int(identity) <= 0:
                    raise ValueError("event identity must be positive")
        else:
            raise ValueError("capture needs an explicit nonempty event identity")
        if identity in identities:
            raise ValueError(f"duplicate event identity: {identity}")
        if capture.input_hash in hashes:
            raise ValueError("duplicate captured input")
        event_content = {key: value for key, value in event.items() if key != "id"}
        if not event_content:
            raise ValueError("capture needs event content beyond its identity")
        event_hash = digest(event_content)
        if event_hash in event_hashes:
            raise ValueError("duplicate event content under distinct identities")
        identities.add(identity)
        hashes.add(capture.input_hash)
        event_hashes.add(event_hash)
    logger.info("Validated shadow cohort: %d distinct supplied events", len(identities))
    return {
        "capture_count": len(captures),
        "distinct_event_count": len(identities),
        "event_ids": sorted(identities),
        "input_hashes": sorted(hashes),
        "minimum_engineering_events": MIN_COMPARISON_EVENTS,
        "sample_size_met": len(identities) >= MIN_COMPARISON_EVENTS,
        "authenticity_and_independence": "not_established_by_validation",
    }


def _comparison_coverage(
    events: Sequence[Mapping[str, Any]],
    *,
    critic_present: bool,
) -> dict[str, Any]:
    """Describe completed paired coverage without claiming research success.

    Failures remain in the full report and denominator. Enough completed
    pairs is necessary engineering coverage, not equivalent compute budgets,
    semantic source support, economic edge, or approval to wire the feed.
    """
    paired = [row for row in events if row.get("baseline") is not None]
    complete = [
        row
        for row in paired
        if row["hop_graph"].get("discovery_status") in ("completed", "abstained")
        and row["baseline"].get("comparable") is True
    ]
    reasons = []
    if len(events) < MIN_COMPARISON_EVENTS:
        reasons.append("insufficient_distinct_events")
    if len(paired) != len(events) or not paired:
        reasons.append("missing_baseline_arm")
    if not critic_present:
        reasons.append("critic_not_run")
    if len(complete) < MIN_COMPARISON_EVENTS:
        reasons.append("insufficient_completed_pairs")
    return {
        "status": "exploratory" if reasons else "minimum_completed_paired_coverage",
        "minimum_engineering_events": MIN_COMPARISON_EVENTS,
        "distinct_events": len(events),
        "paired_events": len(paired),
        "completed_pairs": len(complete),
        "incomplete_pairs": len(paired) - len(complete),
        "reasons": reasons,
        "budget_equivalence": "not_established",
        "source_support_accuracy": "requires_independent_evaluation",
        "quality_acceptance": "not_determined_by_sample_coverage",
    }


def captured_retriever(capture: CapturedInput) -> FilingRetriever:
    """Targeted retrieval over the capture only (no live fallback)."""
    tools = FrozenTools(capture)

    def retrieve(ticker: str, query: str) -> list[SourceDocument]:
        result = tools.dispatch("get_sec_filing", {"ticker": ticker, "query": query})
        return [SourceDocument.from_dict(d) for d in result.get("sources", [])]

    return retrieve


def _name_lookup(tools: FrozenTools) -> Any:
    def lookup(ticker: str) -> str | None:
        name = tools.get(ticker).get("security_name")
        return str(name) if name else None

    return lookup


def _token_total(calls: Sequence[Mapping[str, Any]], key: str) -> int | None:
    """Traversal + entailment usage; any unknown (or no call) is unknown."""
    values = [c.get(key) for c in calls]
    if not values or any(type(v) is not int for v in values):
        return None
    return sum(v for v in values if isinstance(v, int))


def _hop_depths(hop_counts: Iterable[int]) -> dict[str, int]:
    counts = Counter(hop_counts)
    return {str(k): counts[k] for k in sorted(counts)}


def _gate(
    ideas: list[NicheIdea],
    tools: FrozenTools,
    market_data: Mapping[str, Mapping[str, Any]],
    as_of: datetime,
    event: Mapping[str, Any],
    critic: AdversarialCritic | None,
) -> list[NicheIdea]:
    """The unchanged gate sequence (same order as compare_captured)."""
    verified = verify_ideas(ideas, tools)
    for idea in verified:
        evidence_score(idea, market_data, as_of)
    if critic is not None and verified:
        critic.apply(verified, event, as_of=as_of)
    return verified


def run_hopgraph_arm(
    capture: CapturedInput,
    *,
    traversal_client: Any,
    entailment_client: Any,
    traversal_model: str | None = None,
    entailment_model: str | None = None,
    critic: AdversarialCritic | None = None,
    memory: EdgeMemory | None = None,
    playbooks: Mapping[str, Playbook] | None = None,
) -> dict[str, Any]:
    payload = capture.payload()
    as_of = timestamp(payload["captured_at"])
    assert as_of is not None
    event = payload["event"]
    playbook = (playbooks or {}).get(str(event.get("theme") or ""))
    tools = FrozenTools(capture)
    entailment = EntailmentChecker(entailment_client, entailment_model)
    verifier = EdgeVerifier(
        captured_retriever(capture), as_of, entailment, name_lookup=_name_lookup(tools)
    )
    traversal = HopGraphTraversal(
        verifier, client=traversal_client, model=traversal_model, memory=memory
    )
    logger.info("hop-graph shadow: event %s (input %s)", event.get("id"), capture.input_hash[:12])
    result = traversal.run(event, playbook, as_of=as_of, market_data=payload["market_data"])
    ideas = result.candidates
    verified = _gate(ideas, tools, payload["market_data"], as_of, event, critic)
    # Failed calls stay in: their unknown usage makes the totals unknown.
    calls = list(result.model_calls) + entailment.calls
    usd = [c.get("usd_cost") for c in calls]
    eligible = [i for i in ideas if i.research_eligible]
    return {
        "arm": "hop_graph",
        "discovery_status": result.outcome.status,
        "discovery_reason": result.outcome.reason,
        "traversal_model": traversal.model,
        "entailment_model": entailment.model,
        "metrics": {
            "candidates": len(ideas),
            "identity_verified": len(verified),
            "evidence_gate_passes": sum(i.evidence_status == "source_backed" for i in ideas),
            "eligible": len(eligible),
            "hop_depth_candidates": _hop_depths(i.hop_count for i in ideas),
            "hop_depth_eligible": _hop_depths(i.hop_count for i in eligible),
            "traversal_calls": len(result.model_calls),
            "entailment_calls": len(entailment.calls),
            "model_calls": len(result.model_calls) + len(entailment.calls),
            "tool_calls": verifier.stats.retrievals,
            "edges_proposed": len(result.graph.edges),
            "edges_by_status": dict(sorted(Counter(e.status for e in result.graph.edges).items())),
            "edges_from_memory": sum(e.origin == "memory" for e in result.graph.edges),
            "sourced_paths": len(result.paths),
            "call_budget_reached": result.call_budget_reached,
            "undirected_terminals": result.undirected_terminals,
        },
        "cost": {
            "usd_cost": sum_or_unknown(usd) if usd else None,
            "cost_provenance": sorted({str(c.get("cost_provenance")) for c in calls}),
            "input_tokens": _token_total(calls, "input_tokens"),
            "output_tokens": _token_total(calls, "output_tokens"),
        },
        "candidates": [i.to_trade_idea() for i in ideas],
        "graph": result.to_dict(),
    }


class _CountingModel:
    """Counts baseline call attempts and failures without changing replies."""

    def __init__(self, inner: ResearchModel) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.attempts = 0
        self.failures = 0

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
        self.attempts += 1
        try:
            return self.inner.complete(messages, max_tokens)
        except Exception:
            self.failures += 1
            raise


def run_baseline_arm(
    capture: CapturedInput,
    model: ResearchModel,
    *,
    critic: AdversarialCritic | None = None,
    budget: ResearchBudget | None = None,
) -> dict[str, Any]:
    """The existing equivalent-tool discovery + the same unchanged gates."""
    counted = _CountingModel(model)
    report = compare_captured(capture, [counted], critic=critic, budget=budget)
    trial = report["trials"][0]
    discovery = trial["discovery"]
    candidates = trial["post_critic_candidates"]
    eligible = [c for c in candidates if (c.get("research") or {}).get("eligible")]
    trace = discovery.get("trace") or []
    # Attempts, not only answers: a failed call has unknown usage, so the
    # discovery loop's partial token sums are reported as unknown.
    usage_known = counted.failures == 0
    return {
        "arm": "baseline",
        "provider": discovery.get("provider"),
        "model": discovery.get("model"),
        "discovery_status": discovery.get("status"),
        "discovery_reason": discovery.get("reason"),
        "comparable": trial["comparable"],
        "metrics": {
            "candidates": len(candidates),
            "identity_verified": trial["metrics"]["identity_verified"],
            "evidence_gate_passes": trial["metrics"]["source_backed"],
            "eligible": len(eligible),
            "hop_depth_candidates": _hop_depths(int(c.get("hop_count") or 0) for c in candidates),
            "hop_depth_eligible": _hop_depths(int(c.get("hop_count") or 0) for c in eligible),
            "model_calls": counted.attempts,
            "failed_model_calls": counted.failures,
            "tool_calls": sum(1 for t in trace if "tool" in t),
        },
        "cost": {
            # The shadow models report tokens, not billed USD (niche_shadow).
            "usd_cost": discovery.get("billed_cost_usd"),
            "cost_provenance": ["tokens_only_no_billed_usd"],
            "input_tokens": discovery.get("input_tokens") if usage_known else None,
            "output_tokens": discovery.get("output_tokens") if usage_known else None,
        },
        "candidates": candidates,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    eligible = sum(r["metrics"]["eligible"] for r in rows)
    candidates = sum(r["metrics"]["candidates"] for r in rows)
    passes = sum(r["metrics"]["evidence_gate_passes"] for r in rows)
    depths: Counter[str] = Counter()
    for r in rows:
        depths.update(r["metrics"]["hop_depth_candidates"])
    return {
        "events": n,
        "candidates": candidates,
        "evidence_gate_passes": passes,
        "eligible": eligible,
        "eligible_per_event": eligible / n if n else None,
        "gate_pass_rate": passes / candidates if candidates else None,
        "hop_depth_candidates": dict(sorted(depths.items())),
        "model_calls": sum(r["metrics"]["model_calls"] for r in rows),
        "tool_calls": sum(r["metrics"]["tool_calls"] for r in rows),
        "usd_cost": sum_or_unknown(r["cost"]["usd_cost"] for r in rows) if rows else None,
        "cost_provenance": sorted({p for r in rows for p in r["cost"]["cost_provenance"]}),
    }


def compare_events(
    captures: Sequence[CapturedInput],
    *,
    traversal_client: Any,
    entailment_client: Any,
    baseline_model: ResearchModel | None,
    critic: AdversarialCritic | None,
    traversal_model: str | None = None,
    entailment_model: str | None = None,
    memory: EdgeMemory | None = None,
    playbooks: Mapping[str, Playbook] | None = None,
    budget: ResearchBudget | None = None,
    subscription_call_limit: int | None = None,
    subscription_critic_client: Any = None,
) -> dict[str, Any]:
    """Replay in capture-time order (memory is point-in-time by ``as_of``)."""
    cohort = validate_capture_cohort(captures)
    if subscription_call_limit is not None:
        EventCallBudget(subscription_call_limit)  # Validate before any role runs.
        if any(
            model not in (None, SUBSCRIPTION_MODEL) for model in (traversal_model, entailment_model)
        ):
            raise ValueError("subscription_model_not_approved")
        if baseline_model is not None and not isinstance(baseline_model, SubscriptionResearchModel):
            raise ValueError("subscription_only_baseline_required")
        if critic is not None:
            raise ValueError("strict shadow requires an explicit subscription critic client")
        traversal_model = entailment_model = SUBSCRIPTION_MODEL
    ordered = sorted(captures, key=lambda c: (str(c.payload()["captured_at"]), c.input_hash))
    events = []
    for capture in ordered:
        payload = capture.payload()
        event_budget = (
            EventCallBudget(subscription_call_limit)
            if subscription_call_limit is not None
            else None
        )
        event_traversal, event_entailment = traversal_client, entailment_client
        event_critic, event_baseline = critic, baseline_model
        if event_budget is not None:
            event_traversal = SubscriptionBudgetClient(traversal_client, event_budget, "traversal")
            event_entailment = SubscriptionBudgetClient(
                entailment_client, event_budget, "entailment"
            )
            if subscription_critic_client is not None:
                event_critic = AdversarialCritic(
                    # Critic consumes only complete(); its nominal LLMClient
                    # annotation predates this transparent policy decorator.
                    client=cast(
                        LLMClient,
                        SubscriptionBudgetClient(
                            subscription_critic_client, event_budget, "critic"
                        ),
                    ),
                    model=SUBSCRIPTION_MODEL,
                    enabled=True,
                    subscription_only=True,
                )
            if isinstance(baseline_model, SubscriptionResearchModel):
                event_baseline = baseline_model.bound(event_budget)
        row: dict[str, Any] = {
            "event_id": payload["event"].get("id"),
            "captured_at": payload["captured_at"],
            "input_hash": capture.input_hash,
            "hop_graph": run_hopgraph_arm(
                capture,
                traversal_client=event_traversal,
                entailment_client=event_entailment,
                traversal_model=traversal_model,
                entailment_model=entailment_model,
                critic=event_critic,
                memory=memory,
                playbooks=playbooks,
            ),
            "baseline": None,
        }
        if event_baseline is not None:
            row["baseline"] = run_baseline_arm(
                capture, event_baseline, critic=event_critic, budget=budget
            )
        if event_budget is not None:
            row["subscription_budget"] = event_budget.report()
            reason = row["subscription_budget"]["stop_reason"]
            if reason:
                # Components intentionally catch failures. Their partial paths
                # must never turn a shared event-budget failure into approval.
                for arm in (row["hop_graph"], row["baseline"]):
                    if arm is None:
                        continue
                    arm["shared_stop_reason"] = reason
                    # Keep the component's specific malformed/partial outcome
                    # when the shared transport stop is its consequence. A
                    # true shared call-limit exhaustion remains explicit.
                    if reason == "event_model_call_limit" or arm["discovery_status"] not in (
                        "invalid_output",
                        "partial",
                    ):
                        arm["discovery_status"] = (
                            "budget_exhausted"
                            if reason == "event_model_call_limit"
                            else "unavailable"
                        )
                        arm["discovery_reason"] = reason
                    arm["comparable"] = False
                    arm["metrics"]["eligible"] = 0
                    arm["metrics"]["hop_depth_eligible"] = {}
                    if "graph" in arm:
                        graph = arm["graph"]
                        graph["component_outcome_before_shared_budget"] = dict(graph["outcome"])
                        graph["outcome"].update(
                            status=arm["discovery_status"], reason=arm["discovery_reason"]
                        )
                    for candidate in arm["candidates"]:
                        record = candidate.setdefault("research", {})
                        record.update(
                            discovery_status=arm["discovery_status"],
                            eligible=False,
                            review_status="review_unavailable",
                            review_reason=reason,
                        )
        events.append(row)
    hop_rows = [e["hop_graph"] for e in events]
    base_rows = [e["baseline"] for e in events if e["baseline"] is not None]
    return {
        "mode": "shadow_only",
        "report_version": REPORT_VERSION,
        "evaluation_kind": "engineering_only_not_profitability_or_point_in_time_proof",
        "identical_gates": "verify_ideas -> evidence_score -> same critic instance",
        "critic_model": SUBSCRIPTION_MODEL
        if subscription_critic_client is not None
        else critic.model
        if critic is not None
        else None,
        "baseline_run": baseline_model is not None,
        "cohort": cohort,
        "evidence_coverage": _comparison_coverage(
            events, critic_present=critic is not None or subscription_critic_client is not None
        ),
        "subscription_only": subscription_call_limit is not None,
        "budget_scope": "shared_per_event_both_arms_graph_first"
        if subscription_call_limit is not None
        else "component_local",
        "events": events,
        "summary": {
            "hop_graph": _summary(hop_rows),
            "baseline": _summary(base_rows) if base_rows else None,
        },
    }
