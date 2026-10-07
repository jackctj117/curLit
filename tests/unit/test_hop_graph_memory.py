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
from tests.unit._hop_graph_fixtures import (
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
    sql = _strip_sql_comments(Path("migrations", "025_niche_edges.sql").read_text())
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
) -> tuple[Any, Retriever, FakeTraversal]:
    retriever = Retriever()
    model = FakeTraversal(edges if edges is not None else world_edges())
    verifier = EdgeVerifier(
        retriever, as_of, EntailmentChecker(entailment or world_entailment(), "h")
    )
    result = HopGraphTraversal(verifier, client=model, model="m", memory=memory).run(
        world_event(), None, as_of=as_of, market_data=MARKET
    )
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
