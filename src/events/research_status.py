"""Write-once research status for executable ideas (CL-7kuu).

The Alpaca executors used to decide "niche, red-team-survived" by matching the
free text ``trade_ideas.notes`` (``LIKE '%niche%'`` / ``'%red-team%'``). Any
writer of notes could satisfy that, including an LLM-authored impact idea, and
it was uncoupled from the evidence/review statuses the niche pipeline records.

This module is the single owner of the ``idea_research_status`` table
(migration 025):

* :func:`status_from_niche_idea` snapshots a research-eligible
  :class:`~src.events.niche_scoring.NicheIdea` — an in-process object from the
  niche merge, never a dict read back from an assessment — into an immutable
  :class:`ResearchStatus`.
* :func:`insert_research_status` writes it ONCE, inside the idea ledger's
  insert transaction. A second insert for the same idea_id is ignored and
  logged; there is no UPDATE or DELETE path, and migration 025 rules discard both.
* :func:`executable_research_predicate` is the SQL the executors use when their
  policy flags are on. It only READS the table.

The legacy note-substring filter survives only behind the explicit, logged
``ALPACA_LEGACY_NOTE_MATCH=1`` transition shim (:func:`legacy_note_predicate`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from src.events.niche_scoring import NicheIdea

logger = logging.getLogger(__name__)

#: The exact status values that make a NicheIdea research-eligible
#: (``NicheIdea.research_eligible``). Repeated here as SQL literals so the
#: executor predicate checks the recorded statuses, not just the boolean.
DISCOVERY_OK = "completed"
EVIDENCE_OK = "source_backed"
REVIEW_OK = "supported"
LIQUIDITY_OK = "sufficient"

_EXISTS_SQL = text("SELECT 1 FROM idea_research_status WHERE idea_id = :idea_id")
_INSERT_SQL = text(
    "INSERT INTO idea_research_status "
    "(idea_id, discovery_status, evidence_status, review_status, liquidity_status, "
    " research_eligible, source_hashes, score_version, research_invocation_id, recorded_at) "
    "VALUES (:idea_id, :discovery_status, :evidence_status, :review_status, "
    " :liquidity_status, :research_eligible, :source_hashes, :score_version, "
    " :research_invocation_id, :recorded_at)"
)
# NOT ``ON CONFLICT``: Postgres rejects ON CONFLICT on a table that has
# UPDATE rules, and migration 025's no-update/no-delete rules are what make
# rows immutable. Existence is checked first on the caller's transaction; a
# concurrent duplicate would hit the primary key and raise (fail loud). The
# ledger only reaches this after winning the trade_ideas insert, so a race for
# the same idea_id is already serialized by trade_ideas' unique index.


@dataclass(frozen=True)
class ResearchStatus:
    """Immutable snapshot of one eligible niche idea's research outcome."""

    discovery_status: str
    evidence_status: str
    review_status: str
    liquidity_status: str
    research_eligible: bool
    source_hashes: tuple[str, ...]
    score_version: str
    research_invocation_id: str | None = None

    def __post_init__(self) -> None:
        # The eligible flag must agree with the statuses it summarizes; a
        # snapshot claiming eligibility with any other status is a bug.
        consistent = (
            self.discovery_status == DISCOVERY_OK
            and self.evidence_status == EVIDENCE_OK
            and self.review_status == REVIEW_OK
            and self.liquidity_status == LIQUIDITY_OK
        )
        if self.research_eligible and not consistent:
            raise ValueError("research_eligible contradicts the recorded statuses")


def research_key(ticker: str, action: str) -> tuple[str, str]:
    """The (ticker, action) normalization the idea ledger derives idea_id from."""
    return (str(ticker or "").strip(), str(action or "").strip().lower())


def status_from_niche_idea(idea: NicheIdea, invocation_id: str | None = None) -> ResearchStatus:
    """Snapshot ``idea`` (must be research-eligible) into a ResearchStatus."""
    if not idea.research_eligible:
        raise ValueError("only research-eligible niche ideas receive a status row")
    hashes = tuple(sorted({doc.source_id for doc in idea.sources}))
    status = ResearchStatus(
        discovery_status=idea.discovery_status,
        evidence_status=idea.evidence_status,
        review_status=idea.review_status,
        liquidity_status=idea.liquidity_status,
        research_eligible=True,
        source_hashes=hashes,
        score_version=idea.score_version,
        research_invocation_id=invocation_id,
    )
    assert status.research_eligible
    return status


def insert_research_status(
    conn: Any,
    idea_id: str,
    status: ResearchStatus,
    now: datetime | None = None,
) -> bool:
    """Insert the status row for ``idea_id`` on the caller's transaction.

    Returns True when written. An existing row is NEVER replaced: the conflict
    is ignored and logged at WARNING, and False is returned.
    """
    if not idea_id:
        raise ValueError("idea_id is required")
    params = {
        "idea_id": idea_id,
        "discovery_status": status.discovery_status,
        "evidence_status": status.evidence_status,
        "review_status": status.review_status,
        "liquidity_status": status.liquidity_status,
        "research_eligible": bool(status.research_eligible),
        "source_hashes": json.dumps(list(status.source_hashes)),
        "score_version": status.score_version,
        "research_invocation_id": status.research_invocation_id,
        "recorded_at": now or datetime.now(UTC),
    }
    logger.info(
        "research status: recording idea=%s eligible=%s review=%s sources=%d",
        idea_id,
        status.research_eligible,
        status.review_status,
        len(status.source_hashes),
    )
    if conn.execute(_EXISTS_SQL, {"idea_id": idea_id}).first() is not None:
        logger.warning(
            "research status: row for idea=%s already exists; second write ignored "
            "(status rows are write-once)",
            idea_id,
        )
        return False
    result = conn.execute(_INSERT_SQL, params)
    written = (result.rowcount or 0) == 1
    if not written:  # never claim a write that did not happen
        logger.warning("research status: insert for idea=%s wrote no row", idea_id)
    return written


def executable_research_predicate(
    *,
    require_niche: bool,
    require_red_team: bool,
    idea_alias: str = "ti",
) -> list[str]:
    """SQL WHERE clauses (literals only) for the executors' policy flags.

    Each flag requires a recorded, research-eligible status row; the niche
    flag additionally pins the discovery/evidence/liquidity statuses and the
    red-team flag pins a ``supported`` review. Both off: no clause at all, so
    the candidate set is exactly the pre-CL-7kuu one.
    """
    if not (require_niche or require_red_team):
        return []
    conds = [
        f"rs.idea_id = {idea_alias}.idea_id",
        "rs.research_eligible = TRUE",
    ]
    if require_niche:
        conds += [
            f"rs.discovery_status = '{DISCOVERY_OK}'",
            f"rs.evidence_status = '{EVIDENCE_OK}'",
            f"rs.liquidity_status = '{LIQUIDITY_OK}'",
        ]
    if require_red_team:
        conds.append(f"rs.review_status = '{REVIEW_OK}'")
    return [
        "EXISTS (SELECT 1 FROM idea_research_status rs WHERE " + " AND ".join(conds) + ")",
    ]


def legacy_note_predicate(
    *,
    require_niche: bool,
    require_red_team: bool,
    notes_column: str,
) -> list[str]:
    """The pre-CL-7kuu note-substring filter, for the explicit transition shim."""
    out: list[str] = []
    if require_niche:
        out.append(f"lower({notes_column}) LIKE '%niche%'")
    if require_red_team:
        out.append(f"lower({notes_column}) LIKE '%red-team%'")
    return out


def eligibility_clauses(
    *,
    require_niche: bool,
    require_red_team: bool,
    legacy_note_match: bool,
    notes_column: str,
    idea_alias: str,
    book: str,
) -> list[str]:
    """Pick the executors' research-eligibility clauses, logging the shim."""
    if legacy_note_match and (require_niche or require_red_team):
        logger.warning(
            "%s: ALPACA_LEGACY_NOTE_MATCH=1 — selecting ideas by NOTE SUBSTRINGS, not "
            "validated research status (CL-7kuu transition shim; disable when done)",
            book,
        )
        return legacy_note_predicate(
            require_niche=require_niche,
            require_red_team=require_red_team,
            notes_column=notes_column,
        )
    return executable_research_predicate(
        require_niche=require_niche,
        require_red_team=require_red_team,
        idea_alias=idea_alias,
    )
