"""Unit tests — trade-idea ledger (CL-mgcp).

The REAL table is owned by migration 007; the sqlite fixture applies
that exact SQL (Postgres types shimmed, same pattern as
test_new_migrations.py) so the schema under test is the schema that
ships. Covers: upsert dedup, selector gap-fill + disagreement
annotation, price_at_signal, expiry math, guarded status transitions,
and the open-ideas listing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from src.events.idea_ledger import (
    expire_stale,
    list_open,
    make_idea_id,
    persist_ideas,
)

MIGRATION = Path("migrations/007_trade_ideas.sql")
#: CL-jiqq — concrete trade-card level columns (stop_price, targets, R:R,
#: entry_trigger, invalidation, dte_window, suggested_strike) live in 008;
#: the ledger now writes them, so the test schema must include both.
MIGRATION_LEVELS = Path("migrations/008_trade_idea_levels.sql")


def _shim_pg_types_for_sqlite(sql: str) -> str:
    return (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("DOUBLE PRECISION", "REAL")
        .replace("BIGSERIAL", "INTEGER")
        .replace("BIGINT", "INTEGER")
        # sqlite lacks JSONB and ALTER ... ADD COLUMN IF NOT EXISTS; store
        # the JSON list as TEXT and drop the guard (fresh table per test).
        .replace("JSONB", "TEXT")
        .replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
    )


def _sqlite_statements(sql: str) -> list[str]:
    """Split shimmed SQL into sqlite-executable statements. sqlite only
    accepts ONE ``ADD COLUMN`` per ``ALTER TABLE``; migration 008 batches
    seven, so fan a multi-add ALTER out into one ALTER per column."""
    import re  # noqa: PLC0415

    out: list[str] = []
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        # Collapse whitespace so the multi-add ALTER is easy to split.
        flat = " ".join(stmt.split())
        if flat.upper().startswith("ALTER TABLE") and flat.count("ADD COLUMN") > 1:
            head, _, rest = flat.partition("ADD COLUMN")
            table_prefix = head.strip()  # "ALTER TABLE trade_ideas"
            for col in re.split(r",\s*ADD COLUMN", "ADD COLUMN" + rest):
                col = re.sub(r"^ADD COLUMN\s*", "", col.strip())
                out.append(f"{table_prefix} ADD COLUMN {col}")
        else:
            out.append(stmt)
    return out


@pytest.fixture
def engine() -> Any:
    from migrations.run import _strip_sql_comments

    eng = sa.create_engine("sqlite://")
    for mig in (MIGRATION, MIGRATION_LEVELS):
        sql = _shim_pg_types_for_sqlite(_strip_sql_comments(mig.read_text()))
        with eng.begin() as conn:
            for stmt in _sqlite_statements(sql):
                conn.execute(text(stmt))
    return eng


def _idea(**overrides: Any) -> dict[str, Any]:
    idea = {
        "ticker": "TSM",
        "action": "buy_puts",
        "direction": "bearish",
        "confidence": 0.7,
        "rationale": "advanced-node concentration in Taiwan",
        "time_horizon": "short",
        "holding_period_days": "2-6",
        "time_stop_days": 5,
        "suggested_entry": "on strength",
        "preferred_instrument": "puts, 2-4 weeks out",
        "notes": "vol crush risk",
    }
    idea.update(overrides)
    return idea


def _assessment(*ideas: dict[str, Any]) -> dict[str, Any]:
    return {"trade_ideas": list(ideas), "fade_candidates": []}


def _all_rows(engine: Any) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [
            dict(r._mapping)
            for r in conn.execute(text("SELECT * FROM trade_ideas ORDER BY id"))
        ]


NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------- #
# Migration smoke (sqlite parse of the real 007 SQL)
# --------------------------------------------------------------------- #


class TestMigrationSmoke:
    def test_table_and_indexes_created(self, engine: Any) -> None:
        insp = sa.inspect(engine)
        assert "trade_ideas" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("trade_ideas")}
        assert {
            "idea_id", "geo_event_id", "ticker", "action", "direction",
            "confidence", "time_horizon", "holding_period_days",
            "time_stop_days", "stop_loss_pct", "preferred_instrument",
            "instrument_reason", "rationale", "suggested_entry", "notes",
            "price_at_signal", "created_at", "status", "status_updated_at",
        } <= cols
        idx = {i["name"] for i in insp.get_indexes("trade_ideas")}
        assert "idx_trade_ideas_status_created" in idx
        assert "idx_trade_ideas_ticker" in idx

    def test_status_check_constraint_enforced(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), now=NOW)
        with pytest.raises(Exception, match="(?i)constraint"), engine.begin() as conn:
            conn.execute(text(
                "UPDATE trade_ideas SET status = 'bogus'",
            ))


# --------------------------------------------------------------------- #
# persist_ideas
# --------------------------------------------------------------------- #


class TestPersist:
    def test_inserts_row_with_fields(self, engine: Any) -> None:
        n = persist_ideas(engine, 42, _assessment(_idea()), now=NOW)
        assert n == 1
        row = _all_rows(engine)[0]
        assert row["idea_id"] == make_idea_id(42, "TSM", "buy_puts")
        assert row["geo_event_id"] == 42
        assert row["ticker"] == "TSM"
        assert row["action"] == "buy_puts"
        assert row["status"] == "pending"
        assert row["time_stop_days"] == 5
        # LLM-provided preferred_instrument wins — no selector override.
        assert row["preferred_instrument"] == "puts, 2-4 weeks out"

    def test_dedup_on_conflict_do_nothing(self, engine: Any) -> None:
        assert persist_ideas(engine, 42, _assessment(_idea()), now=NOW) == 1
        # Re-assessing the same event later must not duplicate the row
        # nor clobber the original created_at.
        later = NOW + timedelta(hours=3)
        assert persist_ideas(engine, 42, _assessment(_idea()), now=later) == 0
        rows = _all_rows(engine)
        assert len(rows) == 1
        assert "12:00" in str(rows[0]["created_at"])

    def test_same_ticker_different_event_is_new_row(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), now=NOW)
        persist_ideas(engine, 2, _assessment(_idea()), now=NOW)
        assert len(_all_rows(engine)) == 2

    def test_price_at_signal_recorded_when_available(self, engine: Any) -> None:
        prices = {"TSM": {"price": 172.40, "change_pct": -1.8}}
        persist_ideas(engine, 1, _assessment(_idea()), prices=prices, now=NOW)
        assert _all_rows(engine)[0]["price_at_signal"] == pytest.approx(172.40)

    def test_price_absent_is_null_not_fake(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), prices={}, now=NOW)
        assert _all_rows(engine)[0]["price_at_signal"] is None

    def test_no_ideas_is_zero(self, engine: Any) -> None:
        assert persist_ideas(engine, 1, {"trade_ideas": []}, now=NOW) == 0
        assert persist_ideas(engine, 1, {}, now=NOW) == 0
        assert persist_ideas(engine, 1, {"trade_ideas": "junk"}, now=NOW) == 0

    def test_malformed_idea_dropped_individually(self, engine: Any) -> None:
        n = persist_ideas(
            engine, 1,
            _assessment({"ticker": "", "action": "long"}, _idea()),
            now=NOW,
        )
        assert n == 1


class TestTradeCardPersistence:
    """CL-jiqq — the grounded trade-card columns (migration 008) are
    computed from the LLM percentages + the cycle price and persisted."""

    def test_card_levels_persisted_with_price(self, engine: Any) -> None:
        idea = _idea(
            action="buy_puts", direction="bearish",
            stop_loss_pct=0.40, target_pct=[0.10, 0.18],
            entry_trigger="on confirmed blockade language",
            invalidation="official denial",
        )
        prices = {"TSM": {"price": 172.4, "change_pct": -1.8}}
        persist_ideas(engine, 1, _assessment(idea), prices=prices, now=NOW)
        row = _all_rows(engine)[0]
        # buy_puts is bearish → stop ABOVE spot (option underlying default),
        # rounded to the name's tick (>= 100 → 1dp).
        assert row["stop_price"] == pytest.approx(186.2)   # 172.4 × 1.08
        # targets DOWN, stored as a JSON list.
        import json
        assert json.loads(row["target_prices"]) == [155.2, 141.4]
        assert row["risk_reward"] is not None
        assert row["entry_trigger"] == "on confirmed blockade language"
        assert row["invalidation"] == "official denial"
        assert row["dte_window"] == "1-3 weeks"          # short horizon
        assert row["suggested_strike"] == pytest.approx(163.8)  # 172.4 × 0.95

    def test_dollar_levels_null_without_price(self, engine: Any) -> None:
        idea = _idea(
            entry_trigger="on strength", invalidation="denial",
            target_pct=[0.10],
        )
        persist_ideas(engine, 1, _assessment(idea), prices={}, now=NOW)
        row = _all_rows(engine)[0]
        # No live price → no dollar levels, but the TEXT fields persist.
        assert row["stop_price"] is None
        assert row["target_prices"] is None
        assert row["risk_reward"] is None
        assert row["suggested_strike"] is None
        assert row["entry_trigger"] == "on strength"
        assert row["invalidation"] == "denial"
        assert row["dte_window"] == "1-3 weeks"          # DTE is price-free

    def test_stock_stop_uses_llm_pct(self, engine: Any) -> None:
        # A STOCK idea's stop_loss_pct is a real share move → dollar stop.
        idea = _idea(
            action="short", direction="bearish", stop_loss_pct=0.09,
            time_horizon="structural", holding_period_days="60",
        )
        prices = {"TSM": {"price": 100.0, "change_pct": 0.0}}
        persist_ideas(engine, 1, _assessment(idea), prices=prices, now=NOW)
        row = _all_rows(engine)[0]
        # short → stop above; stock → uses the 9% the LLM gave.
        assert row["stop_price"] == pytest.approx(109.0)
        assert row["suggested_strike"] is None            # not an option

    def test_list_open_parses_target_prices_to_list(self, engine: Any) -> None:
        idea = _idea(target_pct=[0.10, 0.18])
        prices = {"TSM": {"price": 172.4, "change_pct": -1.8}}
        persist_ideas(engine, 1, _assessment(idea), prices=prices, now=NOW)
        row = list_open(engine)[0]
        assert isinstance(row["target_prices"], list)
        assert all(isinstance(t, float) for t in row["target_prices"])

    def test_get_idea_by_prefix(self, engine: Any) -> None:
        from src.events.idea_ledger import get_idea, make_idea_id

        prices = {"TSM": {"price": 172.4, "change_pct": -1.8}}
        persist_ideas(engine, 1, _assessment(_idea()), prices=prices, now=NOW)
        idea_id = make_idea_id(1, "TSM", "buy_puts")
        row = get_idea(engine, idea_id[:6])
        assert row is not None
        assert row["ticker"] == "TSM"
        assert isinstance(row["target_prices"], list)
        assert get_idea(engine, "nomatch") is None


class TestSelectorGapFill:
    def test_missing_preferred_instrument_filled(self, engine: Any) -> None:
        idea = _idea(preferred_instrument="", notes="")
        persist_ideas(engine, 1, _assessment(idea), now=NOW)
        row = _all_rows(engine)[0]
        # 2-6d midpoint = 4d → short band → puts 1-3 weeks.
        assert row["preferred_instrument"] == "puts, 1-3 weeks to expiry"
        assert "short horizon" in row["instrument_reason"]
        # Selector agrees with buy_puts — no disagreement annotation.
        assert row["notes"] is None

    def test_stop_loss_filled_from_selector(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), now=NOW)
        assert _all_rows(engine)[0]["stop_loss_pct"] == pytest.approx(0.40)

    def test_disagreement_annotated_never_overridden(self, engine: Any) -> None:
        # LLM says short the stock on a 2-6d horizon; selector would
        # rather own puts. Action stays `short`, notes call it out.
        idea = _idea(action="short", direction="bearish")
        persist_ideas(engine, 1, _assessment(idea), now=NOW)
        row = _all_rows(engine)[0]
        assert row["action"] == "short"
        assert "selector prefers buy_puts (short horizon)" in row["notes"]
        assert row["notes"].startswith("vol crush risk; ")

    def test_agreement_leaves_notes_untouched(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), now=NOW)
        assert _all_rows(engine)[0]["notes"] == "vol crush risk"


# --------------------------------------------------------------------- #
# expire_stale
# --------------------------------------------------------------------- #


class TestExpiry:
    def test_elapsed_time_stop_expires(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea(time_stop_days=5)), now=NOW)
        assert expire_stale(engine, now=NOW + timedelta(days=5)) == 1
        row = _all_rows(engine)[0]
        assert row["status"] == "expired"
        assert row["status_updated_at"] != row["created_at"]

    def test_not_yet_elapsed_stays_pending(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea(time_stop_days=5)), now=NOW)
        assert expire_stale(
            engine, now=NOW + timedelta(days=4, hours=23),
        ) == 0
        assert _all_rows(engine)[0]["status"] == "pending"

    def test_only_stale_rows_expire(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(
            _idea(time_stop_days=3),
            _idea(ticker="RTX", action="long", direction="bullish",
                  time_stop_days=20, time_horizon="medium",
                  holding_period_days="10-20"),
        ), now=NOW)
        assert expire_stale(engine, now=NOW + timedelta(days=4)) == 1
        by_ticker = {r["ticker"]: r["status"] for r in _all_rows(engine)}
        assert by_ticker == {"TSM": "expired", "RTX": "pending"}

    def test_guarded_update_skips_non_pending(self, engine: Any) -> None:
        # A (future) operator transition to `taken` must never be
        # clobbered back to `expired` by the auto-sweep.
        persist_ideas(engine, 1, _assessment(_idea(time_stop_days=1)), now=NOW)
        with engine.begin() as conn:
            conn.execute(text("UPDATE trade_ideas SET status = 'taken'"))
        assert expire_stale(engine, now=NOW + timedelta(days=30)) == 0
        assert _all_rows(engine)[0]["status"] == "taken"

    def test_idempotent(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea(time_stop_days=1)), now=NOW)
        later = NOW + timedelta(days=2)
        assert expire_stale(engine, now=later) == 1
        assert expire_stale(engine, now=later) == 0

    def test_null_time_stop_never_auto_expires(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), now=NOW)
        with engine.begin() as conn:
            conn.execute(text("UPDATE trade_ideas SET time_stop_days = NULL"))
        assert expire_stale(engine, now=NOW + timedelta(days=365)) == 0

    def test_empty_table_noop(self, engine: Any) -> None:
        assert expire_stale(engine, now=NOW) == 0


# --------------------------------------------------------------------- #
# list_open
# --------------------------------------------------------------------- #


class TestListOpen:
    def test_pending_only_newest_first(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), now=NOW)
        persist_ideas(
            engine, 2,
            _assessment(_idea(ticker="RTX", action="long", direction="bullish")),
            now=NOW + timedelta(hours=2),
        )
        persist_ideas(
            engine, 3,
            _assessment(_idea(ticker="LMT", action="long", time_stop_days=1)),
            now=NOW - timedelta(days=5),
        )
        expire_stale(engine, now=NOW)  # LMT expires
        rows = list_open(engine)
        assert [r["ticker"] for r in rows] == ["RTX", "TSM"]

    def test_created_at_parsed_to_aware_datetime(self, engine: Any) -> None:
        persist_ideas(engine, 1, _assessment(_idea()), now=NOW)
        created = list_open(engine)[0]["created_at"]
        assert isinstance(created, datetime)
        assert created.tzinfo is not None
        assert created == NOW

    def test_empty_table(self, engine: Any) -> None:
        assert list_open(engine) == []

    def test_missing_table_raises(self) -> None:
        bare = sa.create_engine("sqlite://")
        with pytest.raises(Exception, match="(?i)no such table"):
            list_open(bare)
