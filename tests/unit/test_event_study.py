"""Tests for the event study (CL-z95p).

sqlite fixtures built from the real migrations (005 geo_events, 007
trade_ideas, 012 intraday_quotes, 014/016/018 alpaca_option_orders), following
the pattern in test_alpaca_options_exit.py.

Every return/MFE/MAE assertion is HAND-COMPUTED from the seeded price path —
these are not smoke tests. The seeded paths are chosen so each horizon lands
on an exact quote, so the expected numbers are exact rationals.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.events.playbooks import Playbook, PlaybookInstrument
from src.research.event_study import (
    LEGACY_ERA,
    POST_018_ERA,
    EventStudyConfig,
    Quote,
    QuoteSeries,
    aggregate,
    aggregate_options,
    bucket_confidence,
    bucket_spread,
    bucket_urgency,
    build_report,
    parse_assessment,
    run_event_study,
    run_options_attribution,
    summarize,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 21, 0, 0, tzinfo=UTC)
SEEN = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _ts(dt: datetime) -> str:
    """sqlite storage format matching SQLAlchemy's sqlite DATETIME bind
    rendering, so `ts >= :bound` comparisons are lexicographically correct."""
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'study.db'}")
    with eng.begin() as conn:
        for mig in (
            "005_geo_events.sql",
            "007_trade_ideas.sql",
            "012_intraday_quotes.sql",
            "014_alpaca_option_orders.sql",
            "016_alpaca_option_exits.sql",
            "018_option_entry_mid.sql",
        ):
            sql = _strip_sql_comments(Path("migrations", mig).read_text())
            sql = (
                sql.replace("TIMESTAMPTZ", "TEXT")
                .replace("NUMERIC", "FLOAT")
                .replace("JSONB", "TEXT")
                .replace("BIGSERIAL", "INTEGER")
                .replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
            )
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                # Timescale-only; sqlite has no hypertables.
                if "create_hypertable" in stmt:
                    continue
                conn.execute(text(stmt))
    return eng


def _seed_event(
    engine,  # type: ignore[no-untyped-def]
    *,
    seen_at: datetime,
    status: str = "ASSESSED",
    theme: str = "energy_chokepoint",
    urgency: int | None = 8,
    confidence: float | None = 0.8,
    affected: list[dict[str, str]] | None = None,
    status_updated_at: datetime | None = None,
    assessment: Any = "__default__",
    external_id: str | None = None,
) -> int:
    if assessment == "__default__":
        body: dict[str, Any] = {"affected": affected if affected is not None else []}
        if urgency is not None:
            body["urgency"] = urgency
        if confidence is not None:
            body["confidence"] = confidence
        payload = json.dumps(body)
    else:
        payload = assessment  # type: ignore[assignment]
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO geo_events (seen_at, source, external_id, headline, url, "
                "theme, assessment, status, status_updated_at) "
                "VALUES (:seen, 'test', :ext, 'h', NULL, :theme, :a, :st, :sua)"
            ),
            {
                "seen": _ts(seen_at),
                "ext": external_id or f"ext-{seen_at.isoformat()}-{status}-{theme}",
                "theme": theme,
                "a": payload,
                "st": status,
                "sua": _ts(status_updated_at or seen_at),
            },
        )
        row = conn.execute(text("SELECT max(id) FROM geo_events")).scalar()
    return int(row or 0)


def _seed_quotes(
    engine,  # type: ignore[no-untyped-def]
    symbol: str,
    points: list[tuple[datetime, float]],
    *,
    spread_frac: float | None = None,
) -> None:
    with engine.begin() as conn:
        for ts, mid in points:
            bid = ask = None
            if spread_frac is not None:
                bid = mid * (1.0 - spread_frac)
                ask = mid * (1.0 + spread_frac)
            conn.execute(
                text(
                    "INSERT INTO intraday_quotes (ts, symbol, source, bid, ask, mid) "
                    "VALUES (:ts, :sym, 'oanda', :bid, :ask, :mid)"
                ),
                {"ts": _ts(ts), "sym": symbol, "bid": bid, "ask": ask, "mid": mid},
            )


#: The hand-built price path. Entry 12:00 = 1.0; every study horizon except
#: 1440m lands on an exact quote.
PATH = [
    (SEEN, 1.0),  # 12:00 entry
    (SEEN + timedelta(minutes=30), 1.001),  # +10 bps
    (SEEN + timedelta(minutes=60), 1.002),  # +20 bps
    (SEEN + timedelta(minutes=120), 0.999),  # -10 bps
    (SEEN + timedelta(minutes=240), 1.005),  # +50 bps
]

_PLAYBOOKS = {
    "energy_chokepoint": Playbook(
        key="energy_chokepoint",
        name="Energy chokepoint",
        description="",
        watch_terms=("hormuz",),
        instruments=(PlaybookInstrument("BCO_USD", "oanda", "long", ""),),
        tier="specific",
    ),
    "war_escalation": Playbook(
        key="war_escalation",
        name="War escalation",
        description="",
        watch_terms=("war",),
        instruments=(PlaybookInstrument("XAU_USD", "oanda", "long", ""),),
        tier="generic",
    ),
}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_parse_assessment_dict_str_and_garbage():
    assert parse_assessment({"urgency": 8}) == {"urgency": 8}
    assert parse_assessment('{"urgency": 8}') == {"urgency": 8}
    assert parse_assessment("not json") is None
    assert parse_assessment("[1,2]") is None
    assert parse_assessment(None) is None


def test_bucket_urgency_edges():
    assert bucket_urgency(10) == ">=7"
    assert bucket_urgency(7) == ">=7"
    assert bucket_urgency(6) == "5-6"
    assert bucket_urgency(5) == "5-6"
    assert bucket_urgency(4) == "<5"
    assert bucket_urgency(1) == "<5"
    assert bucket_urgency(None) == "unknown"


def test_bucket_confidence_edges():
    assert bucket_confidence(0.9) == ">=0.75"
    assert bucket_confidence(0.75) == ">=0.75"
    assert bucket_confidence(0.7499) == "0.55-0.749"
    assert bucket_confidence(0.55) == "0.55-0.749"
    assert bucket_confidence(0.5499) == "<0.55"
    assert bucket_confidence(None) == "unknown"


def test_bucket_spread_edges():
    assert bucket_spread(None) == "null"
    assert bucket_spread(0.199) == "<0.2"
    assert bucket_spread(0.2) == "0.2-0.35"
    assert bucket_spread(0.349) == "0.2-0.35"
    assert bucket_spread(0.35) == "0.35-0.5"
    assert bucket_spread(0.5) == "0.35-0.5"
    assert bucket_spread(0.501) == ">0.5"


def test_summarize_exact_hand_computed():
    cell = summarize([0.001, 0.002, -0.001])
    assert cell.n == 3
    assert cell.hit_rate == pytest.approx(2 / 3)
    assert cell.mean == pytest.approx(0.002 / 3)
    assert cell.median == pytest.approx(0.001)
    # stdev of {0.001, 0.002, -0.001}: sum sq dev = 4.666667e-6, /2, sqrt
    assert cell.std == pytest.approx(0.0015275252316519468, rel=1e-9)
    # t = mean / (std / sqrt(3))
    assert cell.t_stat == pytest.approx(0.7559289460184544, rel=1e-9)


def test_summarize_degenerate_samples_do_not_fabricate():
    empty = summarize([])
    assert (empty.n, empty.mean, empty.std, empty.t_stat) == (0, None, None, None)
    single = summarize([0.01])
    assert single.n == 1
    assert single.mean == pytest.approx(0.01)
    assert single.std is None and single.t_stat is None  # never "certain" on n=1


def test_quote_series_entry_is_never_before_target():
    before = Quote(SEEN - timedelta(seconds=1), 2.0, None, None)
    at = Quote(SEEN, 1.0, None, None)
    series = QuoteSeries([before, at])
    assert series.first_at_or_after(SEEN, 15.0) is at
    # nothing at/after the target within tolerance -> None (never the earlier one)
    assert QuoteSeries([before]).first_at_or_after(SEEN, 15.0) is None


def test_quote_half_spread_fraction():
    q = Quote(
        SEEN,
        100.0,
        bid=99.0,
        ask=101.0,
    )
    assert q.half_spread_frac == pytest.approx(0.01)
    assert Quote(SEEN, 100.0, None, 101.0).half_spread_frac is None
    assert Quote(SEEN, 0.0, 1.0, 2.0).half_spread_frac is None


# --------------------------------------------------------------------------- #
# Signed return math — long AND short, exact
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("direction", "sign"),
    [("long", 1.0), ("short", -1.0)],
)
def test_signed_returns_exact(engine, direction, sign):
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": direction}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)

    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert len(result.headline_legs) == 1
    leg = result.headline_legs[0]
    assert leg.direction == direction
    assert leg.symbol == "BCO_USD"
    assert leg.entry_price == pytest.approx(1.0)
    assert leg.entry_ts == SEEN

    assert leg.returns[30] == pytest.approx(sign * 0.001)
    assert leg.returns[60] == pytest.approx(sign * 0.002)
    assert leg.returns[120] == pytest.approx(sign * -0.001)
    assert leg.returns[240] == pytest.approx(sign * 0.005)
    # 1440m is beyond the seeded path -> dropped, and COUNTED.
    assert 1440 not in leg.returns
    assert leg.missing_horizons == (1440,)
    assert result.missing_horizon_counts == {1440: 1}


@pytest.mark.parametrize(
    ("direction", "expected_mfe", "expected_mae"),
    [("long", 0.005, -0.001), ("short", 0.001, -0.005)],
)
def test_mfe_mae_on_crafted_path(engine, direction, expected_mfe, expected_mae):
    """Excursions over (entry, entry+240m]: mids 1.001/1.002/0.999/1.005 →
    signed long path {+10, +20, -10, +50} bps; short is its negation."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": direction}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)

    leg = run_event_study(engine, EventStudyConfig(), days=30, now=NOW).headline_legs[0]
    assert leg.mfe == pytest.approx(expected_mfe)
    assert leg.mae == pytest.approx(expected_mae)


def test_entry_excluded_from_excursion_window(engine):
    """The entry quote itself is not an excursion (a 0.0 would drag MAE to 0
    on every leg and make every trade look like it never went underwater)."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", [(SEEN, 1.0), (SEEN + timedelta(minutes=30), 1.001)])
    leg = run_event_study(engine, EventStudyConfig(), days=30, now=NOW).headline_legs[0]
    assert leg.mfe == pytest.approx(0.001)
    assert leg.mae == pytest.approx(0.001)  # never touched 0 → MAE is the only point


# --------------------------------------------------------------------------- #
# No lookahead
# --------------------------------------------------------------------------- #


def test_no_lookahead_pre_headline_quote_is_never_the_entry(engine):
    """A quote one second BEFORE seen_at must not become the entry. If it did,
    the entry price would be 2.0 and the 30m return would be a fictitious
    -49.95% instead of the true +10 bps."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", [(SEEN - timedelta(seconds=1), 2.0), *PATH])

    leg = run_event_study(engine, EventStudyConfig(), days=30, now=NOW).headline_legs[0]
    assert leg.entry_ts == SEEN
    assert leg.entry_price == pytest.approx(1.0)
    assert leg.returns[30] == pytest.approx(0.001)


def test_no_quote_at_or_after_headline_is_excluded_not_backfilled(engine):
    """Only pre-headline quotes exist → the leg is UNMEASURABLE. It must be
    excluded and counted, never priced off the stale pre-event quote."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(
        engine,
        "BCO_USD",
        [(SEEN - timedelta(minutes=10), 1.0), (SEEN - timedelta(minutes=2), 1.5)],
    )
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.headline_legs == ()
    # The anchor is past the LAST quote this symbol has → a coverage-end limit.
    assert result.exclusions["legs_anchor_after_quote_buffer"] == 1


# --------------------------------------------------------------------------- #
# Tolerance exclusions — counted, never imputed
# --------------------------------------------------------------------------- #


def test_entry_tolerance_exclusion_is_counted(engine):
    """Anchor sits INSIDE the symbol's quote coverage but the next quote is 20
    min away (tolerance 15) → a genuine feed gap, counted as such."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(
        engine,
        "BCO_USD",
        [
            (SEEN - timedelta(minutes=30), 0.99),
            (SEEN + timedelta(minutes=20), 1.0),
            (SEEN + timedelta(minutes=50), 1.01),
        ],
    )
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.headline_legs == ()
    assert result.exclusions["legs_no_entry_quote_in_tolerance"] == 1
    assert result.events_with_legs == 0


def test_anchor_before_quote_buffer_is_counted_separately(engine):
    """The rolling buffer had not started yet — a COVERAGE limit, not a feed
    gap. Conflating the two would hide why the sample is small."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(
        engine,
        "BCO_USD",
        [(SEEN + timedelta(minutes=20), 1.0), (SEEN + timedelta(minutes=50), 1.01)],
    )
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.headline_legs == ()
    assert result.exclusions["legs_anchor_before_quote_buffer"] == 1
    assert "legs_no_entry_quote_in_tolerance" not in result.exclusions


def test_entry_tolerance_boundary_is_inclusive(engine):
    """A quote exactly at the tolerance edge (15 min) still matches."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(
        engine,
        "BCO_USD",
        [(SEEN + timedelta(minutes=15), 1.0), (SEEN + timedelta(minutes=30), 1.002)],
    )
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert len(result.headline_legs) == 1
    leg = result.headline_legs[0]
    assert leg.entry_ts == SEEN + timedelta(minutes=15)
    # 30m target = 12:30; nearest quote is the 12:30 one (exact).
    assert leg.returns[30] == pytest.approx(0.002)


def test_horizon_tolerance_exclusion_is_counted(engine):
    """Entry + a 30m quote, then a gap: the 60m and 120m targets have no quote
    within 15 min, so those horizons are dropped for this leg and counted."""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(
        engine,
        "BCO_USD",
        [
            (SEEN, 1.0),
            (SEEN + timedelta(minutes=30), 1.001),
            (SEEN + timedelta(minutes=90), 1.004),  # 30m from the 60m target
            (SEEN + timedelta(minutes=240), 1.005),
        ],
    )
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    leg = result.headline_legs[0]
    assert set(leg.returns) == {30, 240}
    assert leg.missing_horizons == (60, 120, 1440)
    assert result.missing_horizon_counts == {60: 1, 120: 1, 1440: 1}


def test_unmapped_instrument_excluded_and_counted(engine):
    """An assessment leg whose instrument is NOT in the strategy map must be
    excluded and counted — the study never silently invents a mapping.

    (Originally exercised with WHEAT_USD, which the study's own first live
    run exposed as a REAL map gap; that gap is now fixed and enforced by
    TestInstrumentMapCoverage, so this uses a deliberately fictional id.)"""
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[
            {"instrument": "SOYBN_USD", "kind": "oanda", "direction": "long"},
            {"instrument": "BCO_USD", "kind": "oanda", "direction": "long"},
        ],
    )
    _seed_quotes(engine, "BCO_USD", PATH)
    _seed_quotes(engine, "SOYBN_USD", PATH)

    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert len(result.headline_legs) == 1
    assert result.headline_legs[0].symbol == "BCO_USD"
    assert result.exclusions["legs_unmapped_instrument"] == 1
    assert result.unmapped_instruments == {"SOYBN_USD": 1}


def test_symbol_absent_from_quote_feed_is_counted(engine):
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "USD_MXN", "kind": "fx", "direction": "short"}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)  # a different symbol entirely
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.headline_legs == ()
    assert result.exclusions["legs_symbol_absent_from_quote_feed"] == 1


def test_non_tradable_legs_and_watch_direction_are_not_traded(engine):
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[
            {"instrument": "RTX", "kind": "equity_watch", "direction": "watch"},
            {"instrument": "BCO_USD", "kind": "oanda", "direction": "watch"},
            {"instrument": "SOMETHING", "kind": "polymarket", "direction": "long"},
        ],
    )
    _seed_quotes(engine, "BCO_USD", PATH)
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.headline_legs == ()
    assert result.exclusions["events_no_tradable_legs"] == 1


def test_dismissed_counted_but_never_scored(engine):
    _seed_event(
        engine,
        seen_at=SEEN,
        status="DISMISSED",
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.status_counts["DISMISSED"] == 1
    assert result.headline_legs == ()


def test_missing_assessment_counted(engine):
    _seed_event(engine, seen_at=SEEN, assessment=None)
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.exclusions["events_no_assessment"] == 1


def test_events_outside_window_are_not_fetched(engine):
    _seed_event(
        engine,
        seen_at=NOW - timedelta(days=45),
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
        external_id="old",
    )
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert sum(result.status_counts.values()) == 0


# --------------------------------------------------------------------------- #
# Confirmation entry vs headline entry
# --------------------------------------------------------------------------- #


def test_confirmation_entry_diverges_from_headline_entry(engine):
    """Crafted event: the whole move happens in the first 30 min. Entering at
    the headline captures +100 bps at 30m; entering at the confirmation stamp
    (60 min later, after the move) captures exactly 0 — the gate paid away the
    move it selected for."""
    confirmed_at = SEEN + timedelta(minutes=60)
    _seed_event(
        engine,
        seen_at=SEEN,
        status="CONFIRMED",
        status_updated_at=confirmed_at,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(
        engine,
        "BCO_USD",
        [
            (SEEN, 1.0),
            (SEEN + timedelta(minutes=30), 1.01),  # the whole move
            (SEEN + timedelta(minutes=60), 1.01),  # confirmation entry
            (SEEN + timedelta(minutes=90), 1.01),  # +30m from confirmation
            (SEEN + timedelta(minutes=120), 1.01),
            (SEEN + timedelta(minutes=180), 1.01),
            (SEEN + timedelta(minutes=240), 1.01),
            (SEEN + timedelta(minutes=300), 1.01),
        ],
    )
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)

    assert len(result.headline_legs) == 1
    assert len(result.confirmation_legs) == 1
    head = result.headline_legs[0]
    conf = result.confirmation_legs[0]

    assert head.entry_ts == SEEN
    assert head.entry_price == pytest.approx(1.0)
    assert head.returns[30] == pytest.approx(0.01)
    assert head.returns[60] == pytest.approx(0.01)

    assert conf.entry_ts == confirmed_at
    assert conf.entry_price == pytest.approx(1.01)
    assert conf.returns[30] == pytest.approx(0.0)
    assert conf.returns[60] == pytest.approx(0.0)

    # Latency distribution captured from status_updated_at - seen_at.
    assert result.latency_minutes == (60.0,)
    assert result.events_with_confirmation_legs == 1


def test_non_confirmed_status_has_no_confirmation_leg(engine):
    _seed_event(
        engine,
        seen_at=SEEN,
        status="EXPIRED",
        status_updated_at=SEEN + timedelta(minutes=120),
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert len(result.headline_legs) == 1
    assert result.confirmation_legs == ()
    assert result.latency_minutes == ()


def test_confirmation_stamp_not_after_headline_is_counted(engine):
    """A CONFIRMED row whose status_updated_at == seen_at carries no usable
    confirmation time — no confirmation leg, and the gap is COUNTED."""
    _seed_event(
        engine,
        seen_at=SEEN,
        status="CONFIRMED",
        status_updated_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.confirmation_legs == ()
    assert result.exclusions["confirmation_anchor_not_after_headline"] == 1


# --------------------------------------------------------------------------- #
# Aggregation / tiers / costs
# --------------------------------------------------------------------------- #


def test_aggregation_buckets_and_tier_mapping(engine):
    """Two events: a specific-tier urgency-8/conf-0.8 long and a generic-tier
    urgency-5/conf-0.6 short. Buckets and tiers must land exactly."""
    _seed_event(
        engine,
        seen_at=SEEN,
        theme="energy_chokepoint",
        urgency=8,
        confidence=0.8,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
        external_id="a",
    )
    _seed_event(
        engine,
        seen_at=SEEN,
        theme="war_escalation",
        urgency=5,
        confidence=0.6,
        affected=[{"instrument": "XAU_USD", "kind": "oanda", "direction": "short"}],
        external_id="b",
    )
    _seed_event(
        engine,
        seen_at=SEEN,
        theme="no_such_theme",
        urgency=2,
        confidence=0.3,
        affected=[{"instrument": "USD_JPY", "kind": "fx", "direction": "long"}],
        external_id="c",
    )
    for sym in ("BCO_USD", "XAU_USD", "USD_JPY"):
        _seed_quotes(engine, sym, PATH)

    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW, playbooks=_PLAYBOOKS)
    assert len(result.headline_legs) == 3
    tiers = {leg.symbol: leg.tier for leg in result.headline_legs}
    assert tiers == {"BCO_USD": "specific", "XAU_USD": "generic", "USD_JPY": "unmatched"}

    by_urgency = aggregate(result.headline_legs, "urgency", (30,))
    assert by_urgency[">=7"][30].n == 1
    assert by_urgency["5-6"][30].n == 1
    assert by_urgency["<5"][30].n == 1

    by_conf = aggregate(result.headline_legs, "confidence", (30,))
    assert by_conf[">=0.75"][30].n == 1
    assert by_conf["0.55-0.749"][30].n == 1
    assert by_conf["<0.55"][30].n == 1

    by_dir = aggregate(result.headline_legs, "direction", (30,))
    assert by_dir["long"][30].n == 2
    assert by_dir["short"][30].n == 1
    # long legs both +10 bps; short leg -10 bps
    assert by_dir["long"][30].mean == pytest.approx(0.001)
    assert by_dir["short"][30].mean == pytest.approx(-0.001)
    assert by_dir["long"][30].hit_rate == pytest.approx(1.0)
    assert by_dir["short"][30].hit_rate == pytest.approx(0.0)

    overall = aggregate(result.headline_legs, "overall", (30,))
    assert overall["ALL"][30].n == 3
    assert overall["ALL"][30].mean == pytest.approx(0.001 / 3)


def test_half_spread_cost_is_measured_when_bid_ask_present(engine):
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", PATH, spread_frac=0.0005)
    leg = run_event_study(engine, EventStudyConfig(), days=30, now=NOW).headline_legs[0]
    # bid = mid*(1-f), ask = mid*(1+f) → (ask-bid)/2/mid = f
    assert leg.half_spread_frac == pytest.approx(0.0005)


def test_half_spread_unavailable_when_no_two_sided_quote(engine):
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)  # bid/ask NULL
    leg = run_event_study(engine, EventStudyConfig(), days=30, now=NOW).headline_legs[0]
    assert leg.half_spread_frac is None
    report = build_report(run_event_study(engine, EventStudyConfig(), days=30, now=NOW))
    assert "Cost estimate **unavailable**" in report


def test_quote_coverage_reported(engine):
    _seed_quotes(engine, "BCO_USD", PATH)
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    assert result.coverage.rows == len(PATH)
    assert result.coverage.symbols == 1
    assert result.coverage.min_ts == SEEN
    assert result.coverage.max_ts == SEEN + timedelta(minutes=240)
    assert result.coverage.span_hours == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Options attribution
# --------------------------------------------------------------------------- #


def _seed_idea(
    engine,  # type: ignore[no-untyped-def]
    idea_id: str,
    geo_event_id: int,
    *,
    ticker: str = "DHT",
    confidence: float = 0.72,
    notes: str = "",
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO trade_ideas (idea_id, geo_event_id, ticker, action, "
                "confidence, notes, created_at, status, status_updated_at) "
                "VALUES (:i, :g, :t, 'buy_calls', :c, :n, :ts, 'pending', :ts)"
            ),
            {
                "i": idea_id,
                "g": geo_event_id,
                "t": ticker,
                "c": confidence,
                "n": notes,
                "ts": _ts(SEEN),
            },
        )


def _seed_order(
    engine,  # type: ignore[no-untyped-def]
    idea_id: str,
    *,
    ticker: str = "DHT",
    pnl_pct: float | None = -0.5,
    entry_mid: float | None = None,
    entry_spread_pct: float | None = None,
    exit_reason: str = "stop_loss",
    submitted_at: datetime = SEEN,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO alpaca_option_orders (idea_id, ticker, occ_symbol, "
                "opt_type, qty, premium_est, status, submitted_at, exit_status, "
                "exit_reason, pnl_pct, entry_mid, entry_spread_pct) "
                "VALUES (:i, :t, 'X', 'call', 1, 100, 'submitted', :sa, 'closed', "
                ":er, :p, :em, :es)"
            ),
            {
                "i": idea_id,
                "t": ticker,
                "sa": _ts(submitted_at),
                "er": exit_reason,
                "p": pnl_pct,
                "em": entry_mid,
                "es": entry_spread_pct,
            },
        )


def test_options_attribution_legacy_split_and_orphans(engine):
    legacy_event = _seed_event(
        engine, seen_at=SEEN, theme="energy_chokepoint", urgency=8, external_id="opt-a"
    )
    post_event = _seed_event(
        engine, seen_at=SEEN, theme="sanctions_trade", urgency=6, external_id="opt-b"
    )
    _seed_idea(engine, "idea-legacy", legacy_event, notes="[niche 3hop] red-team survived")
    _seed_idea(engine, "idea-post", post_event, confidence=0.6, notes="plain note")

    _seed_order(engine, "idea-legacy", pnl_pct=-0.5, entry_mid=None)  # pre-018
    _seed_order(engine, "idea-post", pnl_pct=0.2, entry_mid=1.5, entry_spread_pct=0.25)  # post-018
    _seed_order(engine, "orphan", pnl_pct=-0.3, entry_mid=None)  # no idea/event
    _seed_order(engine, "still-open", pnl_pct=None, entry_mid=1.2)  # unrealized

    result = run_options_attribution(engine, days=30, now=NOW, min_bucket_n=20)

    assert result.total_rows == 4
    assert len(result.outcomes) == 3
    assert result.unrealized_rows == 1
    assert result.unattributed == 1

    by_id = {o.idea_id: o for o in result.outcomes}
    assert by_id["idea-legacy"].era == LEGACY_ERA
    assert by_id["idea-post"].era == POST_018_ERA
    assert by_id["orphan"].era == LEGACY_ERA  # entry_mid NULL → pre-018 cohort
    assert by_id["orphan"].attributed is False
    assert by_id["orphan"].theme == "(unattributed)"

    assert by_id["idea-legacy"].niche is True
    assert by_id["idea-legacy"].red_team is True
    assert by_id["idea-legacy"].theme == "energy_chokepoint"
    assert by_id["idea-legacy"].urgency == 8
    assert by_id["idea-legacy"].confidence == pytest.approx(0.72)

    assert by_id["idea-post"].niche is False
    assert by_id["idea-post"].red_team is False
    assert by_id["idea-post"].urgency == 6
    assert by_id["idea-post"].entry_spread_pct == pytest.approx(0.25)

    # The two eras are never pooled.
    legacy = [o for o in result.outcomes if o.era == LEGACY_ERA]
    post = [o for o in result.outcomes if o.era == POST_018_ERA]
    assert len(legacy) == 2 and len(post) == 1
    assert aggregate_options(legacy, "overall")["ALL"].mean == pytest.approx(-0.4)
    assert aggregate_options(post, "overall")["ALL"].mean == pytest.approx(0.2)
    assert set(aggregate_options(legacy, "theme")) == {
        "energy_chokepoint",
        "(unattributed)",
    }
    assert aggregate_options(post, "spread")["0.2-0.35"].n == 1
    assert aggregate_options(legacy, "spread")["null"].n == 2
    assert aggregate_options(legacy, "niche")["niche"].n == 1
    assert aggregate_options(legacy, "niche")["not-niche"].n == 1
    assert aggregate_options(legacy, "red_team")["red-teamed"].n == 1


def test_options_attribution_window_excludes_old_orders(engine):
    _seed_order(engine, "old", pnl_pct=-0.5, submitted_at=NOW - timedelta(days=60))
    result = run_options_attribution(engine, days=30, now=NOW)
    assert result.total_rows == 0
    assert result.outcomes == ()


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def test_report_renders_with_low_n_flags_and_exact_counts(engine):
    _seed_event(
        engine,
        seen_at=SEEN,
        status="CONFIRMED",
        status_updated_at=SEEN + timedelta(minutes=60),
        theme="energy_chokepoint",
        urgency=8,
        confidence=0.8,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
        external_id="r1",
    )
    _seed_event(
        engine,
        seen_at=SEEN,
        status="DISMISSED",
        theme="cb_surprise",
        assessment=json.dumps({"triaged": True}),
        external_id="r2",
    )
    _seed_quotes(engine, "BCO_USD", PATH, spread_frac=0.0002)
    opt_event = _seed_event(
        engine, seen_at=SEEN, theme="energy_chokepoint", urgency=8, external_id="r3"
    )
    _seed_idea(engine, "idea-legacy", opt_event, notes="[niche 2hop]")
    _seed_order(engine, "idea-legacy", pnl_pct=-0.5, entry_mid=None)
    _seed_order(engine, "idea-post", pnl_pct=0.2, entry_mid=1.5, entry_spread_pct=0.4)

    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW, playbooks=_PLAYBOOKS)
    options = run_options_attribution(engine, days=30, now=NOW)
    report = build_report(result, options)

    assert "# curLit event study (CL-z95p)" in report
    assert "LOW-N" in report  # n=1 cells are all below the default min of 20
    assert "no lookahead" in report
    assert "Data coverage — READ THIS FIRST" in report
    assert "WARNING — the quote buffer" in report  # 4h buffer vs 30d request
    assert "Horizon feasibility given this buffer" in report
    assert "INDEPENDENCE CAVEAT" in report
    assert "headline entry vs confirmation entry" in report
    assert "legs_symbol_absent_from_quote_feed" not in report  # nothing hit it

    # Exact census numbers must appear.
    assert "| ASSESSED | 1 |" in report  # the options-only event, no tradable legs
    assert "| CONFIRMED | 1 |" in report
    assert "| DISMISSED | 1 |" in report
    assert "**total in window**" in report
    assert "measured headline legs: **1**" in report
    assert "measured confirmation legs: **1**" in report

    # Options section with the era labels.
    assert "Options attribution" in report
    assert LEGACY_ERA in report
    assert POST_018_ERA in report
    assert "CL-d44a" in report


def test_report_renders_on_empty_database(engine):
    """A totally empty DB must render an honest, non-crashing report."""
    result = run_event_study(engine, EventStudyConfig(), days=30, now=NOW)
    report = build_report(result, run_options_attribution(engine, days=30, now=NOW))
    assert "**EMPTY** — nothing is measurable" in report
    assert "_No realized option P&L in this window._" in report
    assert "No CONFIRMED/TRADED rows in the window" in report


# --------------------------------------------------------------------------- #
# placebo baseline (CL-2hav) — day-shifted permutation null
# --------------------------------------------------------------------------- #


def _obs(symbol: str, anchor: datetime, direction: str, returns: dict[int, float]):
    from src.research.event_study import LegObservation

    return LegObservation(
        event_id=1,
        status="ASSESSED",
        theme="energy_chokepoint",
        tier="specific",
        urgency=8,
        confidence=0.8,
        instrument=symbol,
        symbol=symbol,
        direction=direction,
        entry_basis="headline",
        anchor=anchor,
        entry_ts=anchor,
        entry_price=100.0,
        half_spread_frac=None,
        returns=returns,
        missing_horizons=(),
        mfe=None,
        mae=None,
    )


def _series(quotes: list[tuple[datetime, float]]) -> QuoteSeries:
    return QuoteSeries(Quote(ts=ts, mid=mid, bid=None, ask=None) for ts, mid in quotes)


def _flat_days(base: datetime, days: int, mid: float = 100.0) -> list[tuple[datetime, float]]:
    out = []
    for d in range(days):
        for m in range(0, 24 * 60, 5):
            out.append((base + timedelta(days=d, minutes=m), mid))
    return out


def test_placebo_pure_drift_is_inside_the_band():
    """A constant-rate drift produces the SAME horizon return everywhere, so
    the actual mean must sit exactly at the null median — the placebo must
    NOT call market drift 'edge'."""
    from src.research.event_study import run_placebo

    base = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    rate_per_min = 0.0001 / 60
    quotes = [
        (ts, 100.0 * (1 + rate_per_min * ((ts - base).total_seconds() / 60)))
        for ts, _ in _flat_days(base, 11)
    ]
    series = {"BCO_USD": _series(quotes)}
    anchor = base + timedelta(days=5, hours=12)
    cfg = EventStudyConfig(
        horizons_minutes=(30,), placebo_draws=50, placebo_seed=3, placebo_max_day_shift=4
    )
    # the actual leg measured under the same drift:
    r30 = 100.0 * (
        1 + rate_per_min * ((anchor + timedelta(minutes=30) - base).total_seconds() / 60)
    )
    r0 = 100.0 * (1 + rate_per_min * ((anchor - base).total_seconds() / 60))
    leg = _obs("BCO_USD", anchor, "long", {30: r30 / r0 - 1})

    pr = run_placebo([leg], series, cfg, label="headline", seed=3)

    nulls = pr.null_means[30]
    assert len(nulls) == 50
    med = nulls[len(nulls) // 2]
    assert pr.actual_mean[30] == pytest.approx(med, rel=0.05)
    lo, hi = nulls[0], nulls[-1]
    assert lo <= pr.actual_mean[30] <= hi  # inside the band — drift-consistent


def test_placebo_detects_event_timed_move():
    """Price jumps only in the 30 minutes AFTER the actual anchor; every other
    day at that clock time is flat. The actual mean must escape the null."""
    from src.research.event_study import run_placebo

    base = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    anchor = base + timedelta(days=5, hours=12)
    quotes = []
    for ts, mid in _flat_days(base, 11):
        in_jump = anchor < ts <= anchor + timedelta(minutes=35)
        quotes.append((ts, 101.0 if in_jump else mid))
    series = {"BCO_USD": _series(quotes)}
    cfg = EventStudyConfig(
        horizons_minutes=(30,), placebo_draws=100, placebo_seed=11, placebo_max_day_shift=4
    )
    leg = _obs("BCO_USD", anchor, "long", {30: 0.01})  # +1% actual

    pr = run_placebo([leg], series, cfg, label="headline", seed=11)

    nulls = pr.null_means[30]
    assert len(nulls) == 100
    assert all(abs(v) < 0.001 for v in nulls)  # shifted days are flat
    assert pr.actual_mean[30] == pytest.approx(0.01)
    assert pr.actual_mean[30] > nulls[-1]  # EXCEEDS the whole null


def test_placebo_same_seed_reproduces():
    from src.research.event_study import run_placebo

    base = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    series = {"BCO_USD": _series(_flat_days(base, 8))}
    leg = _obs("BCO_USD", base + timedelta(days=4, hours=9), "short", {30: 0.0})
    cfg = EventStudyConfig(horizons_minutes=(30,), placebo_draws=25, placebo_seed=5)

    a = run_placebo([leg], series, cfg, label="headline", seed=5)
    b = run_placebo([leg], series, cfg, label="headline", seed=5)
    assert a.null_means == b.null_means


def test_placebo_thin_coverage_drops_all_draws():
    """Coverage shorter than a day leaves NO valid whole-day shift — every
    leg-draw must be dropped and counted, never silently measured at the
    true anchor (offset 0 is excluded by construction)."""
    from src.research.event_study import run_placebo

    base = datetime(2026, 7, 20, 10, 0, tzinfo=UTC)
    quotes = [(base + timedelta(minutes=5 * i), 100.0) for i in range(60)]  # 5h
    series = {"BCO_USD": _series(quotes)}
    leg = _obs("BCO_USD", base + timedelta(hours=1), "long", {30: 0.0})
    cfg = EventStudyConfig(horizons_minutes=(30,), placebo_draws=10)

    pr = run_placebo([leg], series, cfg, label="headline", seed=1)

    assert pr.null_means[30] == ()
    assert pr.dropped_leg_draws == 10


def test_placebo_section_renders_in_report(engine):
    # Thin sqlite fixture coverage (a 4h path → no valid whole-day shift) →
    # the section renders with the honest "insufficient null draws" row
    # rather than a fabricated band.
    _seed_event(
        engine,
        seen_at=SEEN,
        affected=[{"instrument": "BCO_USD", "kind": "oanda", "direction": "long"}],
    )
    _seed_quotes(engine, "BCO_USD", PATH)
    result = run_event_study(engine, EventStudyConfig(placebo_draws=10), days=30, now=NOW)
    report = build_report(result)
    assert "PLACEBO BASELINE" in report
    assert "insufficient null draws" in report


def test_placebo_shifts_whole_events_together():
    """Legs of ONE event must share each draw's day shift (cluster-preserving
    null). Encode the day in the price path — a per-day step 30m after noon of
    size day_index bps — so two same-event legs measured on the same shifted
    day produce IDENTICAL returns every draw; independent shifts would
    diverge with overwhelming probability over 40 draws."""
    from src.research.event_study import run_placebo

    base = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    quotes = []
    for d in range(11):
        for m in range(0, 24 * 60, 5):
            ts = base + timedelta(days=d, minutes=m)
            noon = base + timedelta(days=d, hours=12)
            stepped = ts > noon + timedelta(minutes=15)
            quotes.append((ts, 100.0 * (1 + (0.0001 * d if stepped else 0.0))))
    series = {"BCO_USD": _series(quotes), "XAU_USD": _series(quotes)}
    anchor = base + timedelta(days=5, hours=12)
    leg_a = _obs("BCO_USD", anchor, "long", {30: 0.0})
    leg_b = _obs("XAU_USD", anchor, "long", {30: 0.0})  # same event_id=1
    cfg = EventStudyConfig(
        horizons_minutes=(30,), placebo_draws=40, placebo_seed=9, placebo_max_day_shift=5
    )

    pr = run_placebo([leg_a, leg_b], series, cfg, label="headline", seed=9)

    # Same shift per draw → both legs land on the same day → identical
    # returns → every per-draw mean is EXACTLY one day's step value. An
    # independent-shift null would average TWO different days' steps, which
    # (for distinct days) is never itself a step value.
    day_steps = [0.0001 * d for d in range(11)]
    for mean in pr.null_means[30]:
        assert any(mean == pytest.approx(step, abs=1e-9) for step in day_steps), mean
