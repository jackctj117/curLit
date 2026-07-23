"""Tests for the X → geo_events bridge (CL-esyo).

match_theme is pure (no DB). ingest_posts is exercised against a real
sqlite engine (the migration 005 schema, type-shimmed like the impact
agent tests) so INSERT / ON CONFLICT / cap behavior is real.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.data.x_monitor import Post, WatchAccount
from src.events.playbooks import Playbook, PlaybookInstrument, load_playbooks
from src.events.x_ingest import (
    DEFAULT_INGEST_CAP,
    IngestResult,
    _parse_created_at,
    ingest_posts,
    match_theme,
    source_credibility_note,
)

PLAYBOOKS = load_playbooks("configs/event_playbooks.yaml")


def _account(
    handle: str = "DeItaone",
    category: str = "financial_flow",
) -> WatchAccount:
    return WatchAccount(
        handle=handle,
        category=category,
        note="test",
        priority="high",
    )


def _post(pid: str, text_out: str, created_at: str = "") -> Post:
    return Post(id=pid, text=text_out, created_at=created_at)


# --------------------------------------------------------------------- #
# match_theme (pure)
# --------------------------------------------------------------------- #


class TestMatchTheme:
    def test_war_term_matches_war_theme(self) -> None:
        theme = match_theme(
            "BREAKING: Country X declares war on its neighbor",
            PLAYBOOKS,
        )
        assert theme == "war_escalation"

    def test_africa_term_matches_africa_theme(self) -> None:
        theme = match_theme(
            "Military coup in Mali as the junta seizes power",
            PLAYBOOKS,
        )
        # africa_power_shift owns 'military coup', 'junta', 'seizes power'.
        assert theme == "africa_power_shift"

    def test_hormuz_phrase_matches_energy_chokepoint(self) -> None:
        theme = match_theme(
            "Iran threatens to close the Strait of Hormuz",
            PLAYBOOKS,
        )
        assert theme == "energy_chokepoint"

    def test_no_match_returns_none(self) -> None:
        assert match_theme("I just ate a great sandwich", PLAYBOOKS) is None

    def test_empty_text_returns_none(self) -> None:
        assert match_theme("", PLAYBOOKS) is None
        assert match_theme("   ", PLAYBOOKS) is None

    def test_case_insensitive(self) -> None:
        assert match_theme("HOUTHI ATTACK IN THE RED SEA", PLAYBOOKS) == ("red_sea_shipping")

    def test_strongest_match_wins_by_count(self) -> None:
        # Text hits three africa_power_shift terms but only one for any
        # other theme → africa_power_shift wins on count.
        text_in = "Military coup: the junta seizes power and announces an export ban on cobalt"
        assert match_theme(text_in, PLAYBOOKS) == "africa_power_shift"

    def test_specific_beats_generic_on_equal_count(self) -> None:
        # CL-gn6k: same single matched term → equal count+longest; the
        # SPECIFIC theme wins the tie regardless of insertion order.
        inst = (PlaybookInstrument("EUR_USD", "fx", "long", "r"),)
        generic = Playbook("g", "G", "", ("border clash",), inst, tier="generic")
        specific = Playbook("s", "S", "", ("border clash",), inst, tier="specific")
        assert match_theme("a border clash erupted", {"g": generic, "s": specific}) == "s"
        assert match_theme("a border clash erupted", {"s": specific, "g": generic}) == "s"

    def test_generic_still_wins_on_higher_count(self) -> None:
        # CL-gn6k: specific preference is a TIE-break only — a stronger
        # (higher term count) generic match still wins, so a headline that is
        # genuinely about the catch-all is still attributed to it.
        inst = (PlaybookInstrument("EUR_USD", "fx", "long", "r"),)
        generic = Playbook("g", "G", "", ("border clash", "troops massing"), inst, tier="generic")
        specific = Playbook("s", "S", "", ("border clash",), inst, tier="specific")
        text_in = "border clash as troops massing on the line"
        assert match_theme(text_in, {"s": specific, "g": generic}) == "g"

    def test_single_word_boundary_not_substring(self) -> None:
        # 'coup' must not fire inside 'couple'; 'junta' isn't present.
        assert match_theme("A couple went to the warehouse", PLAYBOOKS) is None

    def test_phrase_term_is_substring(self) -> None:
        # Multi-word watch terms match as plain substrings.
        assert (
            match_theme(
                "reports of a taiwan blockade emerging",
                PLAYBOOKS,
            )
            == "taiwan_semiconductor"
        )


# --------------------------------------------------------------------- #
# source_credibility_note
# --------------------------------------------------------------------- #


class TestSourceCredibility:
    def test_known_handle_specific_note(self) -> None:
        note = source_credibility_note("x:DeItaone")
        assert note is not None
        assert "X/@DeItaone" in note
        assert "unconfirmed" in note.lower()

    def test_osint_handle_note(self) -> None:
        note = source_credibility_note("x:sentDefender")
        assert note is not None
        assert "cross-check" in note.lower()

    def test_unknown_handle_with_category(self) -> None:
        note = source_credibility_note("x:NewFlowGuy", category="financial_flow")
        assert note is not None
        assert "X/@NewFlowGuy" in note
        assert "unconfirmed" in note.lower()

    def test_unknown_handle_no_category_generic(self) -> None:
        note = source_credibility_note("x:Mystery")
        assert note is not None
        assert "single unverified source" in note.lower()

    def test_gdelt_source_returns_none(self) -> None:
        assert source_credibility_note("gdelt") is None

    def test_empty_source_returns_none(self) -> None:
        assert source_credibility_note("") is None


# --------------------------------------------------------------------- #
# _parse_created_at
# --------------------------------------------------------------------- #


class TestParseCreatedAt:
    def test_iso8601_z(self) -> None:
        dt = _parse_created_at("2026-07-20T21:00:54.000Z")
        assert dt.tzinfo is not None
        assert dt.year == 2026 and dt.hour == 21 and dt.minute == 0

    def test_classic_twitter_format(self) -> None:
        dt = _parse_created_at("Mon Jul 20 21:00:54 +0000 2026")
        assert dt.tzinfo is not None
        assert dt.year == 2026 and dt.month == 7 and dt.day == 20
        assert dt.hour == 21

    def test_offset_converted_to_utc(self) -> None:
        dt = _parse_created_at("2026-07-20T23:00:54+02:00")
        assert dt.utcoffset().total_seconds() == 0
        assert dt.hour == 21

    def test_empty_falls_back_to_now(self) -> None:
        before = datetime.now(UTC)
        dt = _parse_created_at("")
        assert dt.tzinfo is not None
        assert dt >= before.replace(microsecond=0)

    def test_garbage_falls_back_to_now(self) -> None:
        dt = _parse_created_at("not a date at all")
        assert dt.tzinfo is not None


# --------------------------------------------------------------------- #
# ingest_posts (real sqlite engine)
# --------------------------------------------------------------------- #


def _shim(sql: str) -> str:
    return (
        sql.replace("TIMESTAMPTZ", "TEXT").replace("JSONB", "TEXT").replace("BIGSERIAL", "INTEGER")
    )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'geo.db'}")
    sql = _shim(
        _strip_sql_comments(
            Path("migrations/005_geo_events.sql").read_text(),
        )
    )
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


def _rows(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        return [
            dict(r._mapping)
            for r in conn.execute(
                text(
                    "SELECT seen_at, source, external_id, headline, url, "
                    "theme, status FROM geo_events ORDER BY external_id",
                )
            )
        ]


class TestIngestPosts:
    def test_theme_matched_post_inserted(self, engine: Engine) -> None:
        post = _post(
            "555",
            "Iran moves to close the Strait of Hormuz today",
            created_at="2026-07-20T21:00:54.000Z",
        )
        result = ingest_posts(engine, _account(), [post], PLAYBOOKS)
        assert result == IngestResult(ingested=1)
        rows = _rows(engine)
        assert len(rows) == 1
        row = rows[0]
        assert row["source"] == "x:DeItaone"
        assert row["external_id"] == "x:555"
        assert row["url"] == "https://x.com/DeItaone/status/555"
        assert row["theme"] == "energy_chokepoint"
        assert row["status"] == "NEW"
        assert "Strait of Hormuz" in row["headline"]
        assert str(row["seen_at"]).startswith("2026-07-20 21:00")

    def test_no_theme_post_skipped(self, engine: Engine) -> None:
        post = _post("1", "Good morning everyone, coffee time")
        result = ingest_posts(engine, _account(), [post], PLAYBOOKS)
        assert result.ingested == 0
        assert result.skipped_no_theme == 1
        assert _rows(engine) == []

    def test_small_traders_category_skipped_entirely(
        self,
        engine: Engine,
    ) -> None:
        # Even a war-themed post is dropped for small_traders.
        post = _post("9", "Country X declares war on its neighbor")
        acct = _account(handle="daytrader", category="small_traders")
        result = ingest_posts(engine, acct, [post], PLAYBOOKS)
        assert result.ingested == 0
        assert result.skipped_category == 1
        assert _rows(engine) == []

    def test_dedup_on_conflict(self, engine: Engine) -> None:
        post = _post("77", "Houthi attack in the Red Sea shipping lane")
        first = ingest_posts(engine, _account(), [post], PLAYBOOKS)
        assert first.ingested == 1
        # Same post id again → ON CONFLICT DO NOTHING.
        second = ingest_posts(engine, _account(), [post], PLAYBOOKS)
        assert second.ingested == 0
        assert second.deduped == 1
        assert len(_rows(engine)) == 1

    def test_cap_enforced(self, engine: Engine) -> None:
        # 5 relevant posts, cap 2 → only 2 inserted.
        posts = [_post(str(i), "Iran threatens the Strait of Hormuz again") for i in range(5)]
        result = ingest_posts(engine, _account(), posts, PLAYBOOKS, cap=2)
        assert result.ingested == 2
        assert len(_rows(engine)) == 2

    def test_default_cap(self, engine: Engine) -> None:
        posts = [
            _post(str(i), "Houthi vessel attacked in the Red Sea")
            for i in range(DEFAULT_INGEST_CAP + 5)
        ]
        result = ingest_posts(engine, _account(), posts, PLAYBOOKS)
        assert result.ingested == DEFAULT_INGEST_CAP

    def test_mixed_batch_counts(self, engine: Engine) -> None:
        posts = [
            _post("a", "Iran closes the Strait of Hormuz"),  # theme
            _post("b", "just had lunch"),  # no theme
            _post("c", "Houthi attack in the Red Sea"),  # theme
        ]
        result = ingest_posts(engine, _account(), posts, PLAYBOOKS)
        assert result.ingested == 2
        assert result.skipped_no_theme == 1
        assert len(_rows(engine)) == 2

    def test_bad_created_at_uses_now_fallback(self, engine: Engine) -> None:
        before = datetime.now(UTC)
        post = _post(
            "12",
            "Iran closes the Strait of Hormuz",
            created_at="totally-bogus-timestamp",
        )
        ingest_posts(engine, _account(), [post], PLAYBOOKS)
        row = _rows(engine)[0]
        # seen_at fell back to ~now (parse-safe), not empty/null.
        assert row["seen_at"]
        parsed = datetime.fromisoformat(str(row["seen_at"]))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        assert parsed >= before.replace(microsecond=0)

    def test_db_error_tolerated(self) -> None:
        # An engine whose begin() raises must not propagate — the monitor
        # loop must survive a DB outage. Result reports zero ingested.
        class BoomEngine:
            def begin(self) -> None:
                raise RuntimeError("db is down")

        post = _post("1", "Iran closes the Strait of Hormuz")
        result = ingest_posts(
            BoomEngine(),
            _account(),
            [post],
            PLAYBOOKS,  # type: ignore[arg-type]
        )
        assert result.ingested == 0
        # The relevance gate still counted correctly before the write.
        assert result.skipped_no_theme == 0

    def test_headline_whitespace_normalised(self, engine: Engine) -> None:
        post = _post(
            "42",
            "Iran   closes\n\n the  Strait of Hormuz\t now",
        )
        ingest_posts(engine, _account(), [post], PLAYBOOKS)
        row = _rows(engine)[0]
        assert row["headline"] == "Iran closes the Strait of Hormuz now"
