"""Hop-graph traversal, verification and path assembly (CL-ynuh).

Oracles are the hand-written fixture world in ``_hop_graph_fixtures`` (which
filing states which edge is known by construction) and the bead's numeric
bounds. Non-vacuity: the behavioral assertions are also run against stubbed
verifiers and must FAIL there.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.events.adversarial_critic import AdversarialCritic
from src.events.hop_graph import (
    MAX_EDGES_PER_NODE,
    MAX_FRONTIER_NODES,
    MAX_HOPS,
    MAX_MODEL_CALLS,
    Edge,
    HopGraph,
    HopGraphResult,
    HopGraphTraversal,
    Node,
    NodeFact,
    WhereToLook,
    assemble_paths,
    frontier_priority,
    novelty,
    parse_edges,
)
from src.events.hop_graph_verify import (
    EdgeVerifier,
    EntailmentChecker,
    candidate_passages,
    sentences,
)
from src.events.niche_scoring import evidence_score, verify_ideas
from src.events.research_evidence import RelationshipClaim, SourceDocument
from tests.unit._hop_graph_fixtures import (
    ACME_10K,
    CATALYST_PASSAGE,
    FRO_DOC,
    FRO_PASSAGE,
    HOP2_CLAIM,
    MARKET,
    NOW,
    REL_PASSAGE,
    ROUTE_ID,
    FakeEntailment,
    FakeTraversal,
    Retriever,
    SupportiveCritic,
    doc,
    reply,
    world_edges,
    world_entailment,
    world_event,
)


class Universe:
    def __init__(self, tickers: Sequence[str]) -> None:
        self.tickers = set(tickers)

    def exists(self, ticker: str) -> bool:
        return ticker in self.tickers

    def get(self, ticker: str) -> dict[str, Any]:
        return {"exchange": "NYSE"}

    def robinhood_tradeable(self, ticker: str) -> bool:
        return True

    def resolve_name(self, name: str) -> list[dict[str, Any]]:
        return []


def run_world(
    retriever: Retriever | None = None,
    entailment: FakeEntailment | None = None,
    edges: dict[str, list[dict[str, Any]]] | None = None,
    verifier: Any = None,
    memory: Any = None,
) -> tuple[HopGraphResult, Retriever, FakeTraversal]:
    retriever = retriever or Retriever()
    model = FakeTraversal(edges or world_edges())
    verifier = verifier or EdgeVerifier(
        retriever, NOW, EntailmentChecker(entailment or world_entailment(), "fake-haiku")
    )
    traversal = HopGraphTraversal(verifier, client=model, model="fake-trav", memory=memory)
    return traversal.run(world_event(), None, as_of=NOW, market_data=MARKET), retriever, model


def assert_far_node_two_hop(result: HopGraphResult) -> None:
    """The behavioral oracle for the far-node fixture."""
    assert "ACME" in result.candidate_paths
    path = result.candidate_paths["ACME"]
    assert path.hop_count == 2
    assert [e.dst.node_id for e in path.edges] == ["company:FRO", "company:ACME"]
    far = path.edges[1]
    assert far.status == "sourced"
    assert [(c.passage, c.source_id) for c in far.evidence] == [(REL_PASSAGE, ACME_10K.source_id)]
    idea = next(i for i in result.candidates if i.ticker == "ACME")
    assert idea.hop_count == 2
    assert {(c.role, c.passage) for c in idea.claims} >= {
        ("relationship", REL_PASSAGE),
        ("exposure", REL_PASSAGE),
        ("catalyst", CATALYST_PASSAGE),
    }


# ---------------------------------------------------------------------- #
# Far-node evidence, both endpoints, unchanged gates
# ---------------------------------------------------------------------- #


def test_far_node_filing_yields_sourced_two_hop_path() -> None:
    result, retriever, model = run_world()
    assert_far_node_two_hop(result)
    hop1 = result.candidate_paths["ACME"].edges[0]
    assert hop1.evidence[0].passage == FRO_PASSAGE and hop1.sources == [FRO_DOC]
    # The chain (including the FRO-filed hop) is cited in the rationale.
    idea = next(i for i in result.candidates if i.ticker == "ACME")
    assert FRO_DOC.source_id[:12] in idea.rationale and ACME_10K.source_id[:12] in idea.rationale
    # Only the candidate's OWN filings ride in claims/sources (unchanged gate binding).
    assert {s.symbol for s in idea.sources} == {"ACME"}
    assert all(call["no_tools"] for call in model.calls)


class _PointerOnlyVerifier(EdgeVerifier):
    """Stub: retrieves only the model's where-to-look filing (near node)."""

    def _edge_plans(self, edge: Edge) -> list[Any]:
        plans = super()._edge_plans(edge)
        pointer = edge.where_to_look.ticker
        return [p for p in plans if p.ticker == pointer][:1] or plans[:1]


class _RubberStampVerifier:
    """Stub: marks everything sourced without retrieving evidence."""

    def verify_edges(self, edges: Sequence[Edge]) -> None:
        for edge in edges:
            edge.status = "sourced"

    def verify_facts(self, facts: Sequence[NodeFact]) -> None:
        for fact in facts:
            fact.status = "sourced"


@pytest.mark.parametrize(
    "stub",
    [
        lambda r: _PointerOnlyVerifier(r, NOW, EntailmentChecker(world_entailment(), "h")),
        lambda r: _RubberStampVerifier(),
    ],
    ids=["pointer_only_retrieval", "rubber_stamp_no_evidence"],
)
def test_far_node_oracle_is_not_vacuous(stub: Any) -> None:
    retriever = Retriever()
    result, _, _ = run_world(retriever=retriever, verifier=stub(retriever))
    with pytest.raises(AssertionError):
        assert_far_node_two_hop(result)
    assert "ACME" not in result.candidate_paths


def test_both_endpoints_with_tickers_are_retrieved_within_budget() -> None:
    result, retriever, _ = run_world()
    far = result.candidate_paths["ACME"].edges[1]
    assert far.key == ("company:FRO", "company:ACME", "supplies")
    # The FRO->ACME edge consulted FRO's filing (pointer) AND ACME's (far node).
    hop2_queries = [t for t, q in retriever.calls if "coatings" in q.lower()]
    assert sorted(hop2_queries) == ["ACME", "FRO"]
    # Per-target retrieval budget (bead: <= 2 per edge).
    verifier_calls = Counter(t for t, _ in retriever.calls)
    assert sum(verifier_calls.values()) <= 2 * (len(result.graph.edges) + 2)


def test_candidates_pass_only_through_unchanged_gates() -> None:
    result, _, _ = run_world()
    ideas = result.candidates
    verified = verify_ideas(ideas, Universe(["FRO", "ACME"]))
    for idea in verified:
        evidence_score(idea, MARKET, NOW)
    AdversarialCritic(client=SupportiveCritic(), model="c", enabled=True).apply(
        verified, world_event(), as_of=NOW
    )
    by = {i.ticker: i for i in ideas}
    assert by["ACME"].research_eligible
    # FRO's only own-filing claim is the hop-1 relationship: no exposure/catalyst.
    assert by["FRO"].evidence_status == "insufficient_evidence"
    assert not by["FRO"].research_eligible


def test_gates_still_reject_without_critic_or_liquidity() -> None:
    for market, critic_enabled in ((MARKET, False), ({**MARKET, "ACME": {}}, True)):
        result, _, _ = run_world()
        verified = verify_ideas(result.candidates, Universe(["FRO", "ACME"]))
        for idea in verified:
            evidence_score(idea, market, NOW)
        AdversarialCritic(client=SupportiveCritic(), model="c", enabled=critic_enabled).apply(
            verified, world_event(), as_of=NOW
        )
        assert not any(i.research_eligible for i in result.candidates)


# ---------------------------------------------------------------------- #
# Fabricated chains, entailment, contradictions, disconfirming passages
# ---------------------------------------------------------------------- #


def test_fully_fabricated_chain_yields_zero_candidates() -> None:
    result, retriever, _ = run_world(retriever=Retriever({}))
    assert retriever.calls  # it looked
    assert result.candidates == [] and result.paths == []
    assert {e.status for e in result.graph.edges} == {"unverifiable"}
    assert all(e.note == "no_passage" for e in result.graph.edges)


@pytest.mark.parametrize("raw", ['{"answer": "unclear"}', '{"answer": "probably"}', "nope"])
def test_entailment_unclear_or_malformed_is_unverifiable(raw: str) -> None:
    class Fixed:
        def complete(self, **kwargs: Any) -> Any:
            assert kwargs["no_tools"] is True
            return reply(raw)

    result, _, _ = run_world(
        verifier=EdgeVerifier(Retriever(), NOW, EntailmentChecker(Fixed(), "h"))
    )
    assert {e.status for e in result.graph.edges} == {"unverifiable"}
    assert result.candidates == []


def test_entailment_transport_error_is_unverifiable() -> None:
    class Broken:
        def complete(self, **kwargs: Any) -> Any:
            raise TimeoutError

    result, _, _ = run_world(
        verifier=EdgeVerifier(Retriever(), NOW, EntailmentChecker(Broken(), "h"))
    )
    assert {e.status for e in result.graph.edges} == {"unverifiable"}
    assert result.candidates == []


def test_entailment_no_is_contradicted_and_blocks_the_path() -> None:
    ent = world_entailment()
    ent.answers[HOP2_CLAIM] = ("no", "38% of our revenue")
    result, _, _ = run_world(entailment=ent)
    far = next(e for e in result.graph.edges if e.dst.node_id == "company:ACME")
    assert far.status == "contradicted"
    assert far.disconfirming[0].passage == REL_PASSAGE
    assert "ACME" not in result.candidate_paths


def test_disconfirming_passages_are_captured_for_the_critic() -> None:
    text = (
        "Frontline Ltd. accounted for 38% of our revenue in 2025. Our supply contract with "
        "Frontline expires in March 2027 and may not be renewed."
    )
    docs = {"FRO": [FRO_DOC], "ACME": [doc("ACME", text)]}
    result, _, _ = run_world(retriever=Retriever(docs))
    far = next(e for e in result.graph.edges if e.dst.node_id == "company:ACME")
    assert far.status == "sourced"
    assert [c.role for c in far.disconfirming] == ["disconfirming"]
    assert "expires in March 2027" in far.disconfirming[0].passage
    idea = next(i for i in result.candidates if i.ticker == "ACME")
    assert any(c.role == "disconfirming" for c in idea.claims)


# ---------------------------------------------------------------------- #
# Path assembly refuses unsourced edges
# ---------------------------------------------------------------------- #


def _graph(second: Edge) -> HopGraph:
    route = Node.make("route", "Strait of Hormuz")
    fro = Node.make("company", "Frontline", "FRO")
    first = Edge(
        route,
        fro,
        "depends_on_route",
        "c1",
        WhereToLook("FRO", "", ()),
        1,
        status="sourced",
        evidence=[
            RelationshipClaim(
                "documented_fact", "relationship", "c1", FRO_DOC.source_id, FRO_PASSAGE
            )
        ],
        sources=[FRO_DOC],
    )
    graph = HopGraph(seeds=[route])
    graph.edges = [first, second]
    return graph


def _far(**changes: Any) -> Edge:
    fields: dict[str, Any] = {
        "status": "sourced",
        "evidence": [
            RelationshipClaim(
                "documented_fact", "relationship", "c2", ACME_10K.source_id, REL_PASSAGE
            )
        ],
        "sources": [ACME_10K],
    }
    fields.update(changes)
    return Edge(
        Node.make("company", "Frontline", "FRO"),
        Node.make("company", "Acme", "ACME"),
        "supplies",
        "c2",
        WhereToLook(None, "", ()),
        2,
        **fields,
    )


def test_path_assembly_positive_control() -> None:
    paths = assemble_paths(_graph(_far()), NOW)
    assert [[e.dst.ticker for e in p.edges] for p in paths] == [["FRO"], ["FRO", "ACME"]]


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "proposed"},
        {"status": "unverifiable"},
        {"status": "contradicted"},
        {"evidence": []},
        {"sources": []},
        {
            "evidence": [
                RelationshipClaim(
                    "documented_fact", "relationship", "c2", ACME_10K.source_id, "not in the text"
                )
            ]
        },
        {
            "evidence": [
                RelationshipClaim(
                    "inference", "relationship", "c2", ACME_10K.source_id, REL_PASSAGE
                )
            ]
        },
        {"sources": [doc("ACME", ACME_10K.text, published="2026-10-01")]},  # after cutoff
        {"sources": [doc("ACME", ACME_10K.text, published="2023-01-01")]},  # > 730 days old
    ],
)
def test_path_assembly_refuses_any_unsourced_edge(changes: dict[str, Any]) -> None:
    paths = assemble_paths(_graph(_far(**changes)), NOW)
    assert all(p.terminal.ticker != "ACME" for p in paths)


# ---------------------------------------------------------------------- #
# Bounds (property tests) and determinism
# ---------------------------------------------------------------------- #


class _SourceAll:
    """Verifier double that produces REAL captured evidence for every edge."""

    def verify_edges(self, edges: Sequence[Edge]) -> None:
        for edge in edges:
            symbol = edge.dst.ticker or edge.src.ticker or "ZZZ"
            source = doc(symbol, f"{edge.dst.display} relates to {edge.src.display} somehow.")
            edge.status, edge.sources = "sourced", [source]
            edge.evidence = [
                RelationshipClaim(
                    "documented_fact", "relationship", edge.claim, source.source_id, source.text
                )
            ]

    def verify_facts(self, facts: Sequence[NodeFact]) -> None:
        return None


class _Explosive:
    """Proposes ``fanout`` fresh company edges for every frontier node."""

    def __init__(self, fanout: int) -> None:
        self.fanout = fanout
        self.calls = 0

    def complete(self, *, messages: Any, model: str, max_tokens: int, no_tools: bool) -> Any:
        self.calls += 1
        payload = json.loads(messages[1].content)
        edges = []
        for n_index, node in enumerate(payload["frontier"]):
            for i in range(self.fanout):
                ticker = f"T{payload['hop']}{self.calls}{n_index:02d}{i:02d}"[:10]
                edges.append(
                    {
                        "src": node["id"],
                        "dst": {"kind": "company", "name": ticker, "ticker": ticker},
                        "relation": "supplies",
                        "claim": f"{ticker} supplies {node['name']}",
                        "where_to_look": {"ticker": ticker, "section": "", "keywords": []},
                        "direction": "bullish",
                    }
                )
        return reply(json.dumps({"edges": edges}))


@settings(max_examples=40, deadline=None)
@given(
    n_seeds=st.integers(min_value=0, max_value=30),
    fanout=st.integers(min_value=0, max_value=9),
    max_hops=st.integers(min_value=1, max_value=MAX_HOPS),
    max_frontier=st.integers(min_value=1, max_value=MAX_FRONTIER_NODES),
    max_calls=st.integers(min_value=1, max_value=MAX_MODEL_CALLS),
)
def test_frontier_bounds_hold_for_any_reply_volume(
    n_seeds: int, fanout: int, max_hops: int, max_frontier: int, max_calls: int
) -> None:
    model = _Explosive(fanout)
    event = {
        "id": 1,
        "headline": "h",
        "theme": "t",
        "hop_seeds": [{"kind": "commodity", "name": f"seed {i:02d}"} for i in range(n_seeds)],
    }
    result = HopGraphTraversal(
        _SourceAll(),
        client=model,
        model="m",
        max_hops=max_hops,
        max_frontier=max_frontier,
        max_model_calls=max_calls,
    ).run(event, None, as_of=NOW)
    assert model.calls == len(result.model_calls) <= max_calls
    assert len(result.frontiers) <= max_hops
    assert all(len(f) <= max_frontier for f in result.frontiers)
    assert all(1 <= e.hop <= max_hops for e in result.graph.edges)
    per_src = Counter(e.src.node_id for e in result.graph.edges)
    assert all(count <= MAX_EDGES_PER_NODE for count in per_src.values())
    assert all(p.hop_count <= max_hops for p in result.paths)
    assert len(result.candidates) <= 6


@pytest.mark.parametrize(
    "kwargs",
    [{"max_hops": 4}, {"max_frontier": 13}, {"max_model_calls": 7}, {"max_edges_per_node": 5}],
)
def test_hard_caps_cannot_be_raised(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        HopGraphTraversal(_SourceAll(), client=_Explosive(1), model="m", **kwargs)


def test_parse_caps_edges_per_node_in_reply_order() -> None:
    src = Node.make("route", "Suez Canal")
    items = [
        {
            "src": src.node_id,
            "dst": {"kind": "company", "name": f"C{i}", "ticker": f"C{i}"},
            "relation": "depends_on_route",
            "claim": f"claim {i}",
        }
        for i in range(7)
    ]
    items.insert(0, {"src": "company:NOTFRONTIER", "dst": {}, "relation": "x", "claim": "y"})
    edges, _, _, status = parse_edges(json.dumps({"edges": items}), [src], 1, MAX_EDGES_PER_NODE)
    assert status == "ok"
    assert [e.dst.ticker for e in edges] == ["C0", "C1", "C2", "C3"]
    assert parse_edges("not json", [src], 1, 4)[3] == "invalid_output"


def test_traversal_is_deterministic_given_replies() -> None:
    first, _, _ = run_world()
    second, _, _ = run_world()

    def strip(result: HopGraphResult) -> dict[str, Any]:
        out = result.to_dict()
        out.pop("outcome")
        return out

    assert strip(first) == strip(second)


_names = st.text(alphabet="abcdefghij", min_size=1, max_size=6)


@settings(max_examples=100, deadline=None)
@given(
    tickers=st.lists(
        st.from_regex(r"[A-Z]{1,4}", fullmatch=True), min_size=1, max_size=8, unique=True
    ),
    known_mask=st.lists(st.booleans(), min_size=8, max_size=8),
    caps=st.lists(
        st.one_of(st.none(), st.floats(min_value=1e5, max_value=1e12)), min_size=8, max_size=8
    ),
    reasons=st.lists(
        st.sampled_from(["", "sole supplier", "junior miner", "levered"]), min_size=8, max_size=8
    ),
    seed=st.randoms(use_true_random=False),
)
def test_frontier_priority_orders_unseen_tickers_first_and_is_total(
    tickers: list[str],
    known_mask: list[bool],
    caps: list[float | None],
    reasons: list[str],
    seed: Any,
) -> None:
    nodes = [Node.make("company", t, t) for t in tickers]
    known = {t for t, k in zip(tickers, known_mask, strict=False) if k}
    info = {n.node_id: (reasons[i], caps[i]) for i, n in enumerate(nodes)}

    def key(n: Node) -> tuple[float, float, float, str]:
        reason, cap = info[n.node_id]
        k = frontier_priority(n, known_tickers=known, reason_text=reason, market_cap=cap)
        assert all(-1.0 <= c <= 0.0 for c in k[:3])
        return k

    ordered = sorted(nodes, key=key)
    shuffled = list(nodes)
    seed.shuffle(shuffled)
    assert sorted(shuffled, key=key) == ordered  # total, permutation-invariant
    flags = [novelty(n, known) for n in ordered]
    assert flags == sorted(flags, reverse=True)  # every unseen ticker precedes known ones


def test_sentences_are_exact_substrings_and_keep_abbreviations() -> None:
    text = ACME_10K.text + " We sell in the U.S. and Canada."
    pieces = sentences(text)
    assert all(text[o : o + len(s)] == s for o, s in pieces)
    assert REL_PASSAGE in [s for _, s in pieces]
    found = candidate_passages(
        ACME_10K, terms=("FRO", "Frontline"), ticker="FRO", keywords=(), require_mention=True
    )
    assert found == [REL_PASSAGE]


def test_ticker_mention_is_case_sensitive_whole_word() -> None:
    source = SourceDocument(
        "ACME",
        "https://www.sec.gov/x.htm",
        "2026-08-01T23:59:59+00:00",
        "2026-09-08T12:00:00+00:00",
        "We turned on the line in June across all plants. Shipments to ON began in May 2026.",
        "10-K x",
    )
    found = candidate_passages(
        source, terms=("ON",), ticker="ON", keywords=(), require_mention=True
    )
    assert found == ["Shipments to ON began in May 2026."]


def test_memory_outage_never_invents_edges() -> None:
    class Down:
        def load_sourced(self, theme: str, as_of: Any) -> list[Edge]:
            raise ConnectionError

        def load_contradicted(self, as_of: Any) -> set[tuple[str, str, str]]:
            raise ConnectionError

        def upsert(self, edges: Any, theme: str, as_of: Any) -> int:
            raise ConnectionError

    result, _, _ = run_world(memory=Down())
    assert_far_node_two_hop(result)
    assert all(e.origin == "model" for e in result.graph.edges)


def test_partial_traversal_failure_is_not_completed() -> None:
    class FailSecond(FakeTraversal):
        def complete(self, **kwargs: Any) -> Any:
            if self.calls:
                raise TimeoutError
            return super().complete(**kwargs)

    retriever = Retriever()
    model = FailSecond(world_edges())
    verifier = EdgeVerifier(retriever, NOW, EntailmentChecker(world_entailment(), "h"))
    result = HopGraphTraversal(verifier, client=model, model="m").run(
        world_event(), None, as_of=NOW
    )
    assert result.outcome.status == "partial"
    assert all(i.discovery_status == "partial" for i in result.candidates)
    assert not any(i.research_eligible for i in result.candidates)


def test_stale_cutoff_sources_are_not_evidence() -> None:
    later = NOW + timedelta(days=800)
    retriever = Retriever()
    verifier = EdgeVerifier(retriever, later, EntailmentChecker(world_entailment(), "h"))
    result = HopGraphTraversal(verifier, client=FakeTraversal(world_edges()), model="m").run(
        world_event(), None, as_of=later
    )
    assert result.candidates == []
    assert {e.status for e in result.graph.edges} == {"unverifiable"}


def test_seed_frontier_is_sent_to_model() -> None:
    _, _, model = run_world()
    assert model.calls[0]["payload"]["frontier"][0]["id"] == ROUTE_ID
    assert model.calls[0]["payload"]["hop"] == 1


def test_upstream_limiting_disclosures_reach_the_critic_input() -> None:
    hedge = "Our exposure to voyages through the Strait of Hormuz is hedged with war-risk cover."
    docs = {"FRO": [doc("FRO", FRO_DOC.text + " " + hedge)], "ACME": Retriever().docs["ACME"]}
    result, _, _ = run_world(retriever=Retriever(docs))
    hop1 = result.candidate_paths["ACME"].edges[0]
    assert [c.passage for c in hop1.disconfirming] == [hedge]
    idea = next(i for i in result.candidates if i.ticker == "ACME")
    # Not a candidate claim (the gate binds claims to ACME's own filings) ...
    assert all(c.passage != hedge for c in idea.claims)
    # ... but visible in the thesis text the critic is given.
    assert "LIMITING DISCLOSURES" in idea.rationale and hedge in idea.rationale
    prompt = AdversarialCritic(client=SupportiveCritic(), model="c")._user_prompt(
        [idea], world_event()
    )
    assert hedge in prompt
