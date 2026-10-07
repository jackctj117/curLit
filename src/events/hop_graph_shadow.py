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
from typing import Any

from src.events.adversarial_critic import AdversarialCritic
from src.events.hop_graph import EdgeMemory, HopGraphTraversal
from src.events.hop_graph_verify import EdgeVerifier, EntailmentChecker, FilingRetriever
from src.events.niche_scoring import NicheIdea, evidence_score, verify_ideas
from src.events.niche_shadow import (
    CapturedInput,
    FrozenTools,
    ResearchBudget,
    ResearchModel,
    compare_captured,
)
from src.events.playbooks import Playbook
from src.events.research_evidence import SourceDocument, timestamp
from src.research.llm.client import sum_or_unknown

logger = logging.getLogger(__name__)

REPORT_VERSION = "hop-graph-shadow-v1"


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
    calls = [c for c in result.model_calls if "response" in c] + entailment.calls
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
            "input_tokens": result.outcome.input_tokens,
            "output_tokens": result.outcome.output_tokens,
        },
        "candidates": [i.to_trade_idea() for i in ideas],
        "graph": result.to_dict(),
    }


def run_baseline_arm(
    capture: CapturedInput,
    model: ResearchModel,
    *,
    critic: AdversarialCritic | None = None,
    budget: ResearchBudget | None = None,
) -> dict[str, Any]:
    """The existing equivalent-tool discovery + the same unchanged gates."""
    report = compare_captured(capture, [model], critic=critic, budget=budget)
    trial = report["trials"][0]
    discovery = trial["discovery"]
    candidates = trial["post_critic_candidates"]
    eligible = [c for c in candidates if (c.get("research") or {}).get("eligible")]
    trace = discovery.get("trace") or []
    model_calls = sum(1 for t in trace if "response" in t)
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
            "model_calls": model_calls,
            "tool_calls": sum(1 for t in trace if "tool" in t),
        },
        "cost": {
            # The shadow models report tokens, not billed USD (niche_shadow).
            "usd_cost": discovery.get("billed_cost_usd"),
            "cost_provenance": ["tokens_only_no_billed_usd"],
            "input_tokens": discovery.get("input_tokens"),
            "output_tokens": discovery.get("output_tokens"),
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
) -> dict[str, Any]:
    """Replay in capture-time order (memory is point-in-time by ``as_of``)."""
    ordered = sorted(captures, key=lambda c: (str(c.payload()["captured_at"]), c.input_hash))
    events = []
    for capture in ordered:
        payload = capture.payload()
        row: dict[str, Any] = {
            "event_id": payload["event"].get("id"),
            "captured_at": payload["captured_at"],
            "input_hash": capture.input_hash,
            "hop_graph": run_hopgraph_arm(
                capture,
                traversal_client=traversal_client,
                entailment_client=entailment_client,
                traversal_model=traversal_model,
                entailment_model=entailment_model,
                critic=critic,
                memory=memory,
                playbooks=playbooks,
            ),
            "baseline": None,
        }
        if baseline_model is not None:
            row["baseline"] = run_baseline_arm(
                capture, baseline_model, critic=critic, budget=budget
            )
        events.append(row)
    hop_rows = [e["hop_graph"] for e in events]
    base_rows = [e["baseline"] for e in events if e["baseline"] is not None]
    return {
        "mode": "shadow_only",
        "report_version": REPORT_VERSION,
        "evaluation_kind": "engineering_only_not_profitability_or_point_in_time_proof",
        "identical_gates": "verify_ideas -> evidence_score -> same critic instance",
        "critic_model": critic.model if critic is not None else None,
        "baseline_run": baseline_model is not None,
        "events": events,
        "summary": {
            "hop_graph": _summary(hop_rows),
            "baseline": _summary(base_rows) if base_rows else None,
        },
    }
