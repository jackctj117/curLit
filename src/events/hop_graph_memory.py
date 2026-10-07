"""Persistent hop-graph edge memory over ``niche_edges`` (CL-ynuh, mig 025).

SHADOW ONLY. Read/written through an INJECTED SQLAlchemy engine (unit tests
use sqlite with the TIMESTAMPTZ -> TEXT shim); nothing in the trading path
reads this table.

* :meth:`NicheEdgeMemory.load_sourced` — non-expired sourced edges for a
  theme, recorded at or before the cutoff, whose stored source record still
  hashes to ``source_hash``, still contains the passage and is still usable
  at the cutoff. They pre-seed the graph (reused hop facts).
* :meth:`NicheEdgeMemory.load_contradicted` — non-expired contradicted edge
  keys across ALL themes (a contradicted company fact is not theme-bound);
  traversal drops any re-proposal of them.
* :meth:`NicheEdgeMemory.upsert` — persists sourced/contradicted edges with
  source hash, passage locator, ``as_of`` and ``expires_at = as_of + 730d``.
  ``proposed``/``unverifiable`` edges carry no evidence and are not stored.

A corrupt row (hash mismatch, missing passage) raises; the traversal treats
memory failure as "no memory", never as evidence.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from src.events.hop_graph import Edge, Node, WhereToLook
from src.events.research_evidence import MAX_SOURCE_AGE_DAYS, RelationshipClaim, SourceDocument

logger = logging.getLogger(__name__)

#: Same two-annual-filing-cycle freshness policy as source documents.
EDGE_TTL = timedelta(days=MAX_SOURCE_AGE_DAYS)

_SELECT = """
SELECT src_id, src_kind, src_ticker, src_label, dst_id, dst_kind, dst_ticker, dst_label,
       relation, status, claim, source_hash, passage, passage_locator, source_record,
       evidence_bundle
FROM niche_edges
WHERE status = :status AND as_of <= :as_of AND expires_at > :as_of
"""

_UPSERT = """
INSERT INTO niche_edges (
    src_id, src_kind, src_ticker, src_label, dst_id, dst_kind, dst_ticker, dst_label,
    relation, status, claim, source_hash, passage, passage_locator, source_record,
    evidence_bundle, event_theme, as_of, expires_at, recorded_at
) VALUES (
    :src_id, :src_kind, :src_ticker, :src_label, :dst_id, :dst_kind, :dst_ticker, :dst_label,
    :relation, :status, :claim, :source_hash, :passage, :passage_locator, :source_record,
    :evidence_bundle, :event_theme, :as_of, :expires_at, :recorded_at
)
ON CONFLICT (src_id, dst_id, relation, event_theme) DO UPDATE SET
    status = excluded.status, claim = excluded.claim, source_hash = excluded.source_hash,
    passage = excluded.passage, passage_locator = excluded.passage_locator,
    source_record = excluded.source_record, evidence_bundle = excluded.evidence_bundle,
    as_of = excluded.as_of,
    expires_at = excluded.expires_at, recorded_at = excluded.recorded_at
"""


def iso(value: datetime) -> str:
    """Fixed-width UTC text so sqlite TEXT comparison orders like time."""
    assert value.tzinfo is not None, "memory timestamps must be aware"
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _node(node_id: str, kind: str, ticker: str | None, label: str) -> Node:
    id_kind, _, key = node_id.partition(":")
    if id_kind != kind:
        raise ValueError("niche_edges node id/kind mismatch")
    return Node(kind, key, ticker or None, label)


def _claims(raw: object) -> list[RelationshipClaim]:
    if not isinstance(raw, list):
        raise ValueError("niche_edges evidence bundle is malformed")
    keys = ("kind", "role", "statement", "source_id", "passage")
    out = []
    for item in raw:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in keys):
            raise ValueError("niche_edges evidence bundle claim is malformed")
        out.append(RelationshipClaim(**{k: item[k] for k in keys}))
    return out


def _bundle(
    raw: str,
) -> tuple[list[RelationshipClaim], list[RelationshipClaim], list[SourceDocument]]:
    """Every supporting AND limiting claim with its source (hash-verified)."""
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError("niche_edges evidence bundle is malformed")
    sources = [SourceDocument.from_dict(d) for d in data["sources"]]  # re-hashes each
    evidence, contrary = _claims(data.get("evidence")), _claims(data.get("disconfirming"))
    by_id = {d.source_id: d for d in sources}
    for claim in evidence + contrary:
        doc = by_id.get(claim.source_id)
        if doc is None or not claim.passage or claim.passage not in doc.text:
            raise ValueError("niche_edges bundle claim does not match a stored source")
    return evidence, contrary, sources


def _bundle_json(edge: Edge) -> str:
    return json.dumps(
        {
            "evidence": [vars(c) for c in edge.evidence],
            "disconfirming": [vars(c) for c in edge.disconfirming],
            "sources": [d.to_dict() for d in edge.sources],
        },
        sort_keys=True,
    )


class NicheEdgeMemory:
    """Implements hop_graph.EdgeMemory over an injected SQLAlchemy engine."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _rows(self, status: str, as_of: datetime) -> list[Any]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(_SELECT), {"status": status, "as_of": iso(as_of)})
            return list(rows.mappings())

    def load_sourced(self, theme: str, as_of: datetime) -> list[Edge]:
        with self.engine.connect() as conn:
            rows = list(
                conn.execute(
                    text(_SELECT + " AND event_theme = :theme ORDER BY src_id, dst_id, relation"),
                    {"status": "sourced", "as_of": iso(as_of), "theme": theme},
                ).mappings()
            )
        edges: list[Edge] = []
        for row in rows:
            doc = SourceDocument.from_dict(json.loads(row["source_record"]))
            if doc.source_id != row["source_hash"] or row["passage"] not in doc.text:
                raise ValueError("niche_edges source record does not match its hash/passage")
            if not doc.usable(as_of, doc.symbol):
                logger.info(
                    "hop-graph memory: stale/unusable source for %s; skipped", row["dst_id"]
                )
                continue
            evidence, contrary, sources = _bundle(row["evidence_bundle"])
            if not any(
                c.source_id == doc.source_id and c.passage == row["passage"] for c in evidence
            ):
                raise ValueError("niche_edges primary proof missing from its evidence bundle")
            edges.append(
                Edge(
                    src=_node(row["src_id"], row["src_kind"], row["src_ticker"], row["src_label"]),
                    dst=_node(row["dst_id"], row["dst_kind"], row["dst_ticker"], row["dst_label"]),
                    relation=row["relation"],
                    claim=row["claim"],
                    where_to_look=WhereToLook(doc.symbol, "", ()),
                    hop=1,
                    status="sourced",
                    evidence=evidence,
                    sources=sources,
                    disconfirming=contrary,
                    origin="memory",
                    note="remembered",
                )
            )
        logger.info("hop-graph memory: %d sourced edges for theme %s", len(edges), theme)
        return edges

    def load_contradicted(self, as_of: datetime) -> set[tuple[str, str, str]]:
        keys = {
            (r["src_id"], r["dst_id"], r["relation"]) for r in self._rows("contradicted", as_of)
        }
        logger.info("hop-graph memory: %d contradicted edges (never re-proposed)", len(keys))
        return keys

    def upsert(self, edges: Sequence[Edge], theme: str, as_of: datetime) -> int:
        rows = []
        for edge in edges:
            proof: RelationshipClaim | None = None
            if edge.status == "sourced" and edge.is_sourced(as_of):
                proof = edge.evidence[0]
            elif edge.status == "contradicted":
                proof = next(
                    (c for c in edge.disconfirming if c.statement.startswith("Contradicts:")),
                    None,
                )
                if proof is None:
                    continue
            else:
                continue
            doc = next(s for s in edge.sources if s.source_id == proof.source_id)
            rows.append(
                {
                    "src_id": edge.src.node_id,
                    "src_kind": edge.src.kind,
                    "src_ticker": edge.src.ticker,
                    "src_label": edge.src.display,
                    "dst_id": edge.dst.node_id,
                    "dst_kind": edge.dst.kind,
                    "dst_ticker": edge.dst.ticker,
                    "dst_label": edge.dst.display,
                    "relation": edge.relation,
                    "status": edge.status,
                    "claim": edge.claim,
                    "source_hash": doc.source_id,
                    "passage": proof.passage,
                    "passage_locator": doc.locator,
                    "source_record": json.dumps(doc.to_dict(), sort_keys=True),
                    "evidence_bundle": _bundle_json(edge),
                    "event_theme": theme,
                    "as_of": iso(as_of),
                    "expires_at": iso(as_of + EDGE_TTL),
                    "recorded_at": iso(datetime.now(UTC)),
                }
            )
        if not rows:
            return 0
        logger.info("hop-graph memory: upserting %d edges for theme %s", len(rows), theme)
        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(text(_UPSERT), row)
        return len(rows)
