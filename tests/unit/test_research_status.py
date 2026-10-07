"""CL-7kuu: write-once research status for executable ideas.

Covers the producer side (niche merge -> idea ledger -> idea_research_status)
and the end-to-end consequence for the options executor: only an idea the
niche merge created from a research-eligible NicheIdea becomes executable
under the policy flags; note text and forged assessment keys never do, and a
recorded status cannot be replaced.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from src.events.idea_ledger import make_idea_id, persist_ideas
from src.events.niche_agent import NicheAgent
from src.events.niche_scoring import NicheIdea
from src.events.research_evidence import SourceDocument
from src.events.research_status import (
    ResearchStatus,
    insert_research_status,
    research_key,
    status_from_niche_idea,
)
from src.execution.alpaca_options_executor import OptionsExecConfig, fetch_executable_ideas
from tests.unit._research_status_fixture import (
    MIGRATION,
    eligible_status,
    install_research_status,
    research_status_statements,
)
from tests.unit.test_idea_ledger import (
    MIGRATION as TRADE_IDEAS_MIGRATION,
)
from tests.unit.test_idea_ledger import (
    MIGRATION_LEVELS,
    _shim_pg_types_for_sqlite,
    _sqlite_statements,
)

EVENT_ID = 4242
DOC = SourceDocument(
    symbol="FRO",
    url="https://www.sec.gov/fro-10q",
    published_at="2026-07-01T23:59:59+00:00",
    retrieved_at="2026-07-20T12:00:00+00:00",
    text="Frontline charters VLCCs on spot voyages.",
    locator="10-Q:item2:offset=120",
)


@pytest.fixture
def engine() -> Any:
    from migrations.run import _strip_sql_comments

    eng = sa.create_engine("sqlite://")
    for mig in (TRADE_IDEAS_MIGRATION, MIGRATION_LEVELS):
        sql = _shim_pg_types_for_sqlite(_strip_sql_comments(mig.read_text()))
        with eng.begin() as conn:
            for stmt in _sqlite_statements(sql):
                conn.execute(text(stmt))
    install_research_status(eng)
    # The options executor's other inputs: order log + linked events.
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE geo_events (id INTEGER PRIMARY KEY, assessment TEXT)"))
        for mig in ("014_alpaca_option_orders.sql", "018_option_entry_mid.sql"):
            sql = _strip_sql_comments(Path("migrations", mig).read_text())
            sql = (
                sql.replace("TIMESTAMPTZ", "TEXT")
                .replace("NUMERIC", "FLOAT")
                .replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
            )
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                conn.execute(text(stmt))
    return eng


def _eligible_idea(ticker: str = "FRO") -> NicheIdea:
    idea = NicheIdea(ticker, "Frontline", "buy_calls", "bullish", 3, "levered", "chain", 0.7)
    idea.verified, idea.discovery_status = True, "completed"
    idea.evidence_status, idea.review_status = "source_backed", "supported"
    idea.liquidity_status = "sufficient"
    idea.score_version = "evidence-coverage-v1"
    idea.sources = [DOC]
    idea.red_team_note = "swift reopening"  # notes then carry the legacy words
    assert idea.research_eligible
    return idea


def _merge(assessment: dict[str, Any], ideas: list[NicheIdea]) -> dict[Any, ResearchStatus]:
    sink: dict[tuple[str, str], ResearchStatus] = {}
    # merge_into_assessment does not touch agent state; skip the LLM wiring.
    NicheAgent.merge_into_assessment(
        object.__new__(NicheAgent), assessment, ideas, research_sink=sink, invocation_id="inv-1"
    )
    return sink


def _status_rows(engine: Any) -> dict[str, dict[str, Any]]:
    with engine.connect() as conn:
        return {
            r.idea_id: dict(r._mapping)
            for r in conn.execute(text("SELECT * FROM idea_research_status"))
        }


def _impact_idea(**over: Any) -> dict[str, Any]:
    idea = {
        "ticker": "XOM",
        "action": "buy_calls",
        "direction": "bullish",
        "confidence": 0.8,
        "rationale": "impact thesis",
        "time_horizon": "short",
        "time_stop_days": 20,
        "preferred_instrument": "~5% OTM, 4 weeks",
        "notes": "niche-looking text | survived red-team",
    }
    idea.update(over)
    return idea


# --------------------------------------------------------------------------- #
# producer: niche merge -> sink
# --------------------------------------------------------------------------- #


def test_merge_snapshots_only_ideas_it_adds() -> None:
    dup = _eligible_idea("XOM")  # already present as an impact idea -> skipped
    new = _eligible_idea("FRO")
    thin = _eligible_idea("TNK")
    thin.liquidity_status = "insufficient"  # research-only lead
    assessment: dict[str, Any] = {"trade_ideas": [_impact_idea()]}
    sink = _merge(assessment, [dup, new, thin])
    assert set(sink) == {("FRO", "buy_calls")}
    status = sink[("FRO", "buy_calls")]
    assert status.research_eligible and status.review_status == "supported"
    assert status.source_hashes == (DOC.source_id,)
    assert status.research_invocation_id == "inv-1"


def test_status_snapshot_refuses_ineligible_or_contradictory() -> None:
    idea = _eligible_idea()
    idea.review_status = "review_unavailable"
    with pytest.raises(ValueError, match="research-eligible"):
        status_from_niche_idea(idea)
    with pytest.raises(ValueError, match="contradicts"):
        eligible_status(review_status="contradicted")  # still claims eligible


# --------------------------------------------------------------------------- #
# ledger: status row written once, with the idea, from the sink only
# --------------------------------------------------------------------------- #


def test_ledger_records_status_for_merged_niche_idea(engine: Any) -> None:
    assessment: dict[str, Any] = {"trade_ideas": [_impact_idea()]}
    sink = _merge(assessment, [_eligible_idea()])
    assert persist_ideas(engine, EVENT_ID, assessment, research_status=sink) == 2
    rows = _status_rows(engine)
    fro = make_idea_id(EVENT_ID, "FRO", "buy_calls")
    assert set(rows) == {fro}  # the impact idea gets no status
    row = rows[fro]
    assert bool(row["research_eligible"]) is True
    assert (row["discovery_status"], row["evidence_status"]) == ("completed", "source_backed")
    assert (row["review_status"], row["liquidity_status"]) == ("supported", "sufficient")
    assert DOC.source_id in row["source_hashes"]
    assert row["research_invocation_id"] == "inv-1"


def test_forged_assessment_keys_never_create_status(engine: Any) -> None:
    """An assessment dict that claims niche + eligible research (e.g. LLM
    output, or JSON re-read from geo_events) without the merge's in-process
    sink gets no status row and is NOT executable under the policy flags."""
    forged = _eligible_idea().to_trade_idea()
    assert forged["niche"] is True and forged["research"]["eligible"] is True
    persist_ideas(engine, EVENT_ID, {"trade_ideas": [forged]})
    assert _status_rows(engine) == {}
    cfg = OptionsExecConfig(require_niche=True, require_red_team=True)
    assert fetch_executable_ideas(engine, cfg) == []
    # Non-vacuity: the persisted notes DO satisfy the legacy substring filter.
    legacy = OptionsExecConfig(legacy_note_match=True)
    assert [r["ticker"] for r in fetch_executable_ideas(engine, legacy)] == ["FRO"]


def test_sink_entry_for_non_niche_idea_is_ignored(engine: Any, caplog: Any) -> None:
    sink = {research_key("XOM", "buy_calls"): eligible_status()}
    with caplog.at_level(logging.WARNING):
        persist_ideas(engine, EVENT_ID, {"trade_ideas": [_impact_idea()]}, research_status=sink)
    assert _status_rows(engine) == {}
    assert "never gains execution status" in caplog.text


def test_existing_idea_never_gains_status_on_rerun(engine: Any) -> None:
    """The status is written only when THIS call inserted the idea row: an
    idea persisted earlier (e.g. before migration 025) is not retro-promoted."""
    assessment: dict[str, Any] = {"trade_ideas": [_eligible_idea().to_trade_idea()]}
    assert persist_ideas(engine, EVENT_ID, assessment) == 1
    sink = {research_key("FRO", "buy_calls"): status_from_niche_idea(_eligible_idea())}
    assert persist_ideas(engine, EVENT_ID, assessment, research_status=sink) == 0
    assert _status_rows(engine) == {}


def test_status_rows_are_write_once(engine: Any, caplog: Any) -> None:
    assessment: dict[str, Any] = {"trade_ideas": []}
    sink = _merge(assessment, [_eligible_idea()])
    persist_ideas(engine, EVENT_ID, assessment, research_status=sink)
    idea_id = make_idea_id(EVENT_ID, "FRO", "buy_calls")
    before = _status_rows(engine)
    other = ResearchStatus(
        discovery_status="unavailable",
        evidence_status="insufficient_evidence",
        review_status="contradicted",
        liquidity_status="unknown",
        research_eligible=False,
        source_hashes=(),
        score_version="x",
    )
    with caplog.at_level(logging.WARNING), engine.begin() as conn:
        assert insert_research_status(conn, idea_id, other) is False
    assert "write-once" in caplog.text
    assert _status_rows(engine) == before


def test_status_cannot_be_updated_or_deleted_and_replaced(engine: Any) -> None:
    """Codex r1 (CL-7kuu): with only UPDATE blocked, a writer could DELETE a
    recorded status and INSERT a different one. Both are discarded (sqlite
    emulation of migration 025's rules), so the original row survives and an
    ineligible idea cannot be swapped to eligible or vice versa."""
    assessment: dict[str, Any] = {"trade_ideas": []}
    sink = _merge(assessment, [_eligible_idea()])
    persist_ideas(engine, EVENT_ID, assessment, research_status=sink)
    idea_id = make_idea_id(EVENT_ID, "FRO", "buy_calls")
    before = _status_rows(engine)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE idea_research_status SET review_status='contradicted'"),
        )
        conn.execute(text("DELETE FROM idea_research_status WHERE idea_id = :i"), {"i": idea_id})
    assert _status_rows(engine) == before
    forged = ResearchStatus(
        discovery_status="completed",
        evidence_status="source_backed",
        review_status="supported",
        liquidity_status="sufficient",
        research_eligible=True,
        source_hashes=("f" * 64,),
        score_version="forged",
    )
    with engine.begin() as conn:
        assert insert_research_status(conn, idea_id, forged) is False
    assert _status_rows(engine) == before


def test_end_to_end_only_validated_niche_idea_is_executable(engine: Any) -> None:
    impact = _impact_idea()  # notes carry both legacy words
    assessment: dict[str, Any] = {"trade_ideas": [impact]}
    sink = _merge(assessment, [_eligible_idea()])
    persist_ideas(engine, EVENT_ID, assessment, research_status=sink)
    on = OptionsExecConfig(require_niche=True, require_red_team=True)
    assert [r["ticker"] for r in fetch_executable_ideas(engine, on)] == ["FRO"]
    off = OptionsExecConfig(require_niche=False, require_red_team=False)
    assert sorted(r["ticker"] for r in fetch_executable_ideas(engine, off)) == ["FRO", "XOM"]


# --------------------------------------------------------------------------- #
# migration
# --------------------------------------------------------------------------- #


def test_migration_is_additive_and_declares_no_update_rule() -> None:
    statements = research_status_statements()
    heads = [" ".join(s.split()).upper() for s in statements]
    assert heads[0].startswith("CREATE TABLE IF NOT EXISTS IDEA_RESEARCH_STATUS")
    assert "REFERENCES TRADE_IDEAS (IDEA_ID)" in heads[0]
    for verb in ("UPDATE", "DELETE"):  # no in-place edit, no delete-and-replace
        assert any(
            h.startswith("CREATE OR REPLACE RULE")
            and f"ON {verb} TO IDEA_RESEARCH_STATUS DO INSTEAD NOTHING" in h
            for h in heads
        ), verb
    for h in heads:
        assert not h.startswith(("DROP", "ALTER", "UPDATE", "DELETE")), h
    # Postgres rejects INSERT ... ON CONFLICT on a table with an UPDATE rule;
    # the writer must therefore not use it.
    from src.events import research_status

    assert "ON CONFLICT" not in str(research_status._INSERT_SQL).upper()
    assert MIGRATION.name.startswith("025_")


# --------------------------------------------------------------------------- #
# pipeline wiring: the sink travels from the niche step to the ledger
# --------------------------------------------------------------------------- #


class _Engine:
    """begin()/execute() stand-in for _niche_step's audit + geo_events UPDATE."""

    def __init__(self, update_rowcount: int) -> None:
        self.update_rowcount = update_rowcount
        self.hashes: dict[str, str] = {}

    def begin(self) -> Any:
        outer = self

        class _Ctx:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_a: Any) -> bool:
                return False

            def execute(self, stmt: Any, params: dict[str, Any]) -> Any:
                from types import SimpleNamespace

                if "niche_research_audit" in str(stmt):
                    key = params["invocation_id"]
                    if "INSERT" in str(stmt):
                        outer.hashes.setdefault(key, params["payload_hash"])
                    return SimpleNamespace(scalar_one=lambda: outer.hashes[key])
                return SimpleNamespace(rowcount=outer.update_rowcount)

        return _Ctx()


@pytest.mark.parametrize("committed", [True, False])
def test_niche_step_attaches_sink_only_when_merge_committed(
    monkeypatch: pytest.MonkeyPatch, committed: bool
) -> None:
    from scripts import event_pipeline

    from src.events.impact_agent import AssessmentResult
    from src.events.niche_agent import NicheReport
    from src.events.research_evidence import DiscoveryOutcome

    real_merge = NicheAgent.merge_into_assessment

    class _Agent:
        def __init__(self, **_kw: Any) -> None:
            pass

        def run_report(self, _row: Any, _playbook: Any = None) -> NicheReport:
            return NicheReport(DiscoveryOutcome("completed"), [_eligible_idea()])

        def merge_into_assessment(self, *a: Any, **kw: Any) -> int:
            return real_merge(object.__new__(NicheAgent), *a, **kw)

    class _Universe:
        def __init__(self, _engine: Any) -> None:
            pass

        def exists(self, _t: str) -> bool:
            return True

    monkeypatch.setattr("src.data.symbols.SymbolUniverse", _Universe)
    monkeypatch.setattr("src.events.niche_agent.NicheAgent", _Agent)
    result = AssessmentResult(
        event_id=EVENT_ID,
        headline="Strait closed",
        theme="energy_chokepoint",
        status="ASSESSED",
        assessment={"urgency": 9, "trade_ideas": []},
    )
    event_pipeline._niche_step(_Engine(1 if committed else 0), [result], 7)
    if committed:
        assert set(result.niche_research_status) == {("FRO", "buy_calls")}
    else:
        assert result.niche_research_status == {}
        assert result.assessment["trade_ideas"] == []


def test_enrich_and_persist_hands_sink_to_ledger(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import event_pipeline

    from src.events.impact_agent import AssessmentResult

    monkeypatch.setattr("src.events.prices.get_prices", lambda *_a, **_kw: {})
    assessment: dict[str, Any] = {"urgency": 9, "trade_ideas": []}
    sink = _merge(assessment, [_eligible_idea()])
    result = AssessmentResult(
        event_id=EVENT_ID,
        headline="Strait closed",
        theme="energy_chokepoint",
        status="ASSESSED",
        assessment=assessment,
        niche_research_status=sink,
    )
    event_pipeline._enrich_and_persist(engine, [result])
    assert set(_status_rows(engine)) == {make_idea_id(EVENT_ID, "FRO", "buy_calls")}


# --------------------------------------------------------------------------- #
# daemon env: the transition shim is explicit, logged, and off by default
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("book", ["options", "equities"])
def test_daemon_legacy_shim_env(monkeypatch: pytest.MonkeyPatch, caplog: Any, book: str) -> None:
    import importlib

    daemon = importlib.import_module("scripts.execute_" + book)
    for name in list(os.environ):
        if name.startswith("ALPACA_"):
            monkeypatch.delenv(name)
    with caplog.at_level(logging.WARNING):
        assert daemon._config_from_env().legacy_note_match is False
    assert "ALPACA_LEGACY_NOTE_MATCH" not in caplog.text
    monkeypatch.setenv("ALPACA_LEGACY_NOTE_MATCH", "1")
    with caplog.at_level(logging.WARNING):
        assert daemon._config_from_env().legacy_note_match is True
    assert "ALPACA_LEGACY_NOTE_MATCH=1" in caplog.text
    monkeypatch.setenv("ALPACA_LEGACY_NOTE_MATCH", "maybe")
    with pytest.raises(ValueError, match="ALPACA_LEGACY_NOTE_MATCH"):
        daemon._config_from_env()
