"""Hop-graph edge memory over migration 025 on sqlite (CL-ynuh).

Schema comes from the real migration file with the documented sqlite shim
(TIMESTAMPTZ -> TEXT, NUMERIC -> FLOAT). Oracles: the 730-day policy from the
bead, the fixture world's known edges, and the stored SHA-256 identities.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from src.events.hop_graph import Edge, HopGraphTraversal
from src.events.hop_graph_memory import NicheEdgeMemory, iso
from src.events.hop_graph_verify import EdgeVerifier, EntailmentChecker
from src.events.research_evidence import SourceDocument
from tests.unit._hop_graph_fixtures import (
    ACME_8K,
    FRO_DOC,
    HOP2_CLAIM,
    MARKET,
    NOW,
    FakeTraversal,
    Retriever,
    world_edges,
    world_entailment,
    world_event,
)


def _engine(tmp_path: Path) -> Any:
    from migrations.run import _strip_sql_comments

    engine = create_engine(f"sqlite:///{tmp_path / 'edges.db'}")
    sql = _strip_sql_comments(Path("migrations", "026_niche_edges.sql").read_text())
    sql = sql.replace("TIMESTAMPTZ", "TEXT").replace("NUMERIC", "FLOAT")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return engine


def _run(
    memory: NicheEdgeMemory,
    edges: dict[str, list[dict[str, Any]]] | None = None,
    entailment: Any = None,
    as_of: Any = NOW,
    docs: dict[str, list[SourceDocument]] | None = None,
    max_edges_per_node: int = 4,
) -> tuple[Any, Retriever, FakeTraversal]:
    retriever = Retriever(docs)
    model = FakeTraversal(edges if edges is not None else world_edges())
    verifier = EdgeVerifier(
        retriever, as_of, EntailmentChecker(entailment or world_entailment(), "h")
    )
    result = HopGraphTraversal(
        verifier, client=model, model="m", memory=memory, max_edges_per_node=max_edges_per_node
    ).run(world_event(), None, as_of=as_of, market_data=MARKET)
    return result, retriever, model


def _rows(engine: Any) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text("SELECT * FROM niche_edges")).mappings()]


def test_upsert_records_hash_locator_and_730_day_expiry(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _run(NicheEdgeMemory(engine))
    rows = {(r["src_id"], r["dst_id"]): r for r in _rows(engine)}
    assert set(rows) == {
        ("route:strait of hormuz", "company:FRO"),
        ("company:FRO", "company:ACME"),
    }
    far = rows[("company:FRO", "company:ACME")]
    record = json.loads(far["source_record"])
    assert far["status"] == "sourced" and far["event_theme"] == "hormuz"
    assert far["source_hash"] == record["source_id"] and record["symbol"] == "ACME"
    assert far["passage"] in record["text"]
    assert far["passage_locator"] == record["locator"]
    assert far["as_of"] == iso(NOW) and far["expires_at"] == iso(NOW + timedelta(days=730))


def test_remembered_sourced_edges_preseed_and_save_retrievals(tmp_path: Path) -> None:
    memory = NicheEdgeMemory(_engine(tmp_path))
    _, first_retriever, _ = _run(memory)
    later = NOW + timedelta(days=1)
    # The model re-proposes the same edges; remembered ones are not re-verified.
    second, second_retriever, model = _run(memory, as_of=later)
    assert {e.origin for e in second.graph.edges} == {"memory"}
    assert second.candidate_paths["ACME"].hop_count == 2
    edge_queries = [q for _, q in second_retriever.calls if "coatings" in q or "voyages" in q]
    assert edge_queries == []  # only ACME's own exposure/catalyst facts were fetched
    assert len(second_retriever.calls) < len(first_retriever.calls)
    assert model.calls[0]["payload"]["known_edges"]


def test_memory_alone_does_not_invent_a_direction(tmp_path: Path) -> None:
    memory = NicheEdgeMemory(_engine(tmp_path))
    _run(memory)
    # The model proposes nothing: the chain is rebuilt from memory, but with no
    # bullish/bearish direction there is no candidate (as in parse_niche_ideas).
    second, _, _ = _run(memory, edges={}, as_of=NOW + timedelta(days=1))
    assert {e.origin for e in second.graph.edges} == {"memory"}
    assert {p.terminal.ticker for p in second.paths} == {"FRO", "ACME"}
    assert second.candidates == []
    assert sorted(second.undirected_terminals) == ["ACME", "FRO"]


def test_contradicted_edges_are_never_reproposed(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    memory = NicheEdgeMemory(engine)
    ent = world_entailment()
    ent.answers[HOP2_CLAIM] = ("no", "38% of our revenue")
    _run(memory, entailment=ent)
    status = {r["dst_id"]: r["status"] for r in _rows(engine)}
    assert status["company:ACME"] == "contradicted"
    # Even a different theme's run re-proposing it drops the edge before verification.
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE niche_edges SET event_theme = 'other' WHERE status='contradicted'")
        )
    second, retriever, model = _run(memory, as_of=NOW + timedelta(days=1))
    keys = {e.key for e in second.graph.edges}
    assert ("company:FRO", "company:ACME", "supplies") not in keys
    assert "ACME" not in second.candidate_paths
    assert ["company:FRO", "company:ACME", "supplies"] in model.calls[-1]["payload"][
        "do_not_propose"
    ]
    assert not any("coatings" in q for _, q in retriever.calls)


def test_expired_and_future_edges_are_ignored(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    memory = NicheEdgeMemory(engine)
    ent = world_entailment()
    ent.answers[HOP2_CLAIM] = ("no", "38% of our revenue")
    _run(memory, entailment=ent)
    assert memory.load_contradicted(NOW)
    assert memory.load_sourced("hormuz", NOW)
    expired = NOW + timedelta(days=730)
    assert memory.load_contradicted(expired) == set()
    assert memory.load_sourced("hormuz", expired) == []
    before = NOW - timedelta(seconds=1)  # recorded after this cutoff: not point-in-time
    assert memory.load_contradicted(before) == set()
    assert memory.load_sourced("hormuz", before) == []


def test_tampered_source_record_fails_loud_and_is_not_reused(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    memory = NicheEdgeMemory(engine)
    _run(memory)
    with engine.begin() as conn:
        conn.execute(text("UPDATE niche_edges SET passage = 'invented passage text here'"))
    with pytest.raises(ValueError):
        memory.load_sourced("hormuz", NOW)
    # The traversal treats the failure as "no memory", never as evidence.
    result, _, _ = _run(memory, edges={}, as_of=NOW + timedelta(days=1))
    assert result.graph.edges == [] and result.candidates == []


def test_only_evidenced_statuses_are_persisted(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    unclear = world_entailment()
    unclear.answers.clear()  # every passage -> "unclear" -> unverifiable
    result, _, _ = _run(NicheEdgeMemory(engine), entailment=unclear)
    assert {e.status for e in result.graph.edges} == {"unverifiable"}
    assert _rows(engine) == []  # unverifiable edges carry no evidence


def test_schema_rejects_unknown_status_and_bad_expiry(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    row = {
        "src_id": "a:b",
        "src_kind": "route",
        "src_label": "b",
        "dst_id": "company:X",
        "dst_kind": "company",
        "dst_label": "X",
        "relation": "supplies",
        "status": "unverifiable",
        "claim": "c",
        "source_hash": "h",
        "passage": "p",
        "passage_locator": "l",
        "source_record": "{}",
        "evidence_bundle": "{}",
        "event_theme": "t",
        "as_of": iso(NOW),
        "expires_at": iso(NOW + timedelta(days=730)),
        "recorded_at": iso(NOW),
    }
    insert = text(
        "INSERT INTO niche_edges ("
        + ",".join(row)
        + ") VALUES ("
        + ",".join(":" + k for k in row)
        + ")"
    )
    with engine.begin() as conn:  # positive control: a valid status inserts
        conn.execute(insert, {**row, "status": "sourced", "event_theme": "ok"})
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert, row)
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert, {**row, "status": "sourced", "expires_at": iso(NOW)})


def test_edge_type_round_trips_through_memory(tmp_path: Path) -> None:
    memory = NicheEdgeMemory(_engine(tmp_path))
    _run(memory)
    loaded = memory.load_sourced("hormuz", NOW)
    assert all(isinstance(e, Edge) and e.is_sourced(NOW) for e in loaded)
    assert {e.dst.node_id for e in loaded} == {"company:FRO", "company:ACME"}


def test_contradiction_oracle_is_not_vacuous(tmp_path: Path) -> None:
    """Without the remembered contradiction the same re-proposal IS kept and
    verified, so the never-re-proposed assertion above tests the memory."""
    second, retriever, _ = _run(NicheEdgeMemory(_engine(tmp_path)), as_of=NOW + timedelta(days=1))
    assert ("company:FRO", "company:ACME", "supplies") in {e.key for e in second.graph.edges}
    assert any("coatings" in q for _, q in retriever.calls)


# ---------------------------------------------------------------------- #
# Codex round-1 regressions
# ---------------------------------------------------------------------- #

FAR_KEY = ("company:FRO", "company:ACME", "supplies")


def test_contradiction_from_any_theme_vetoes_a_remembered_sourced_edge(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    memory = NicheEdgeMemory(engine)
    _run(memory)  # FRO->ACME sourced under "hormuz"
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO niche_edges SELECT src_id, src_kind, src_ticker, src_label, dst_id, "
                "dst_kind, dst_ticker, dst_label, relation, 'contradicted', claim, source_hash, "
                "passage, passage_locator, source_record, evidence_bundle, 'other', as_of, "
                "expires_at, recorded_at FROM niche_edges WHERE dst_id = 'company:ACME'"
            )
        )
    later = NOW + timedelta(days=1)
    assert any(e.key == FAR_KEY for e in memory.load_sourced("hormuz", later))
    assert FAR_KEY in memory.load_contradicted(later)
    second, _, _ = _run(memory, as_of=later)
    assert FAR_KEY not in {e.key for e in second.graph.edges}
    assert "ACME" not in second.candidate_paths


BOTH_FRO = FRO_DOC.text + " Acme Marine Coatings supplies hull coatings to our fleet."
BOTH_ACME = (
    "Frontline Ltd. accounted for 38% of our revenue in 2025. Our supply contract with "
    "Frontline expires in March 2027 and may not be renewed. Our coatings are applied "
    "during scheduled drydock periods."
)


def _both_docs() -> dict[str, list[SourceDocument]]:
    from tests.unit._hop_graph_fixtures import doc

    return {"FRO": [doc("FRO", BOTH_FRO)], "ACME": [ACME_8K, doc("ACME", BOTH_ACME)]}


def test_memory_round_trips_every_supporting_and_limiting_claim(tmp_path: Path) -> None:
    memory = NicheEdgeMemory(_engine(tmp_path))
    ent = world_entailment()
    ent.answers[HOP2_CLAIM] = ("yes", "")  # both filings state the edge
    first, _, _ = _run(memory, entailment=ent, docs=_both_docs())
    cold = next(e for e in first.graph.edges if e.key == FAR_KEY)
    assert sorted(s.symbol for s in cold.sources) == ["ACME", "FRO"]
    assert len(cold.evidence) == 2 and len(cold.disconfirming) == 1
    [warm] = [e for e in memory.load_sourced("hormuz", NOW) if e.key == FAR_KEY]
    assert sorted(vars(c).items() for c in warm.evidence) == sorted(
        vars(c).items() for c in cold.evidence
    )
    assert warm.disconfirming == cold.disconfirming
    assert {s.source_id for s in warm.sources} == {s.source_id for s in cold.sources}
    # The reused edge still gives the candidate its OWN relationship + limitation.
    second, _, _ = _run(memory, entailment=ent, docs=_both_docs(), as_of=NOW + timedelta(days=1))
    acme = next(i for i in second.candidates if i.ticker == "ACME")
    roles = sorted(c.role for c in acme.claims if c.source_id != ACME_8K.source_id)
    assert "relationship" in roles and "disconfirming" in roles


def test_tampered_bundle_claim_fails_loud(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    memory = NicheEdgeMemory(engine)
    _run(memory)
    with engine.begin() as conn:
        bundle = json.loads(
            conn.execute(
                text("SELECT evidence_bundle FROM niche_edges WHERE dst_id='company:ACME'")
            ).scalar_one()
        )
        bundle["evidence"][0]["passage"] = "Frontline is our only customer."
        conn.execute(
            text("UPDATE niche_edges SET evidence_bundle = :b WHERE dst_id='company:ACME'"),
            {"b": json.dumps(bundle)},
        )
    with pytest.raises(ValueError):
        memory.load_sourced("hormuz", NOW)


def _two_customer_world() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from tests.unit._hop_graph_fixtures import doc

    edges = world_edges()
    beta = json.loads(json.dumps(edges["company:FRO"][0]))
    beta["dst"] = {"kind": "company", "name": "Beta Paints Inc", "ticker": "BETA"}
    beta["claim"] = "Beta Paints supplies paint to Frontline"
    beta.pop("exposure"), beta.pop("catalyst")
    edges["company:FRO"].append(beta)
    docs: dict[str, Any] = {
        **_both_docs(),
        "BETA": [doc("BETA", "Frontline Ltd. accounted for 22% of our paint revenue in 2025.")],
    }
    return edges, docs


def test_memory_respects_the_edge_cap_and_still_asks_for_directions(tmp_path: Path) -> None:
    memory = NicheEdgeMemory(_engine(tmp_path))
    edges, docs = _two_customer_world()
    ent = world_entailment()
    ent.answers[HOP2_CLAIM] = ("yes", "")
    ent.answers["Beta Paints supplies paint to Frontline"] = ("yes", "22%")
    cold, _, _ = _run(memory, edges=edges, entailment=ent, docs=docs)
    assert {"ACME", "BETA"} <= set(cold.candidate_paths)
    warm, _, model = _run(
        memory,
        edges=edges,
        entailment=ent,
        docs=docs,
        as_of=NOW + timedelta(days=1),
        max_edges_per_node=1,
    )
    per_src: dict[str, int] = {}
    for e in warm.graph.edges:
        per_src[e.src.node_id] = per_src.get(e.src.node_id, 0) + 1
    assert max(per_src.values()) <= 1
    # FRO's cap is full from memory, yet FRO is still asked, so the reused
    # terminal gets an event-specific direction and is a candidate.
    asked = [n["id"] for c in model.calls for n in c["payload"]["frontier"]]
    assert "company:FRO" in asked
    reused = [e for e in warm.graph.edges if e.src.node_id == "company:FRO"]
    assert [e.origin for e in reused] == ["memory"]
    assert reused[0].dst.ticker in {i.ticker for i in warm.candidates}
    assert warm.undirected_terminals == []


def test_corrupt_sourced_row_does_not_lift_contradiction_vetoes(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    memory = NicheEdgeMemory(engine)
    _run(memory)  # FRO->ACME (and route->FRO) sourced under "hormuz"
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO niche_edges SELECT src_id, src_kind, src_ticker, src_label, dst_id, "
                "dst_kind, dst_ticker, dst_label, relation, 'contradicted', claim, source_hash, "
                "passage, passage_locator, source_record, evidence_bundle, 'other', as_of, "
                "expires_at, recorded_at FROM niche_edges WHERE dst_id = 'company:ACME'"
            )
        )
        # Corrupt an UNRELATED sourced row so load_sourced() raises.
        conn.execute(
            text(
                "UPDATE niche_edges SET passage = 'invented passage text here' "
                "WHERE dst_id = 'company:FRO' AND status = 'sourced'"
            )
        )
    later = NOW + timedelta(days=1)
    with pytest.raises(ValueError):
        memory.load_sourced("hormuz", later)
    assert FAR_KEY in memory.load_contradicted(later)
    second, retriever, model = _run(memory, as_of=later)
    assert FAR_KEY not in {e.key for e in second.graph.edges}
    assert "ACME" not in second.candidate_paths
    assert all(e.origin == "model" for e in second.graph.edges)  # no reuse
    assert list(FAR_KEY) in model.calls[-1]["payload"]["do_not_propose"]
