"""Tests for the Alpaca paper options executor (CL-ldd2).

Sqlite trade_ideas + alpaca_option_orders; fake Alpaca client + price fn.
Covers the policy filter (niche + red-team + confidence), the premium cap, the
daily cap, dedup, and transient-miss retry behavior.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.execution.alpaca_options_executor import (
    OptionsExecConfig,
    execute_pending_options,
    fetch_executable_ideas,
)

# Midday ET — outside the open-spread entry-delay window, so the delay
# gate is inert everywhere except the tests that exercise it.
NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)  # 12:00 ET
NOW_AT_OPEN = datetime(2026, 7, 21, 13, 35, tzinfo=UTC)  # 9:35 ET


def _no_tech(_t: str):  # unit tests: no live yfinance — fail-open path
    return None


_CONTRACT = {"symbol": "RTX260821C00105000", "strike_price": "105", "expiration_date": "2026-08-18"}


class _FakeClient:
    def __init__(
        self, contract: Any = _CONTRACT, ask: float | None = 2.0, market_open: bool = True
    ) -> None:
        self._contract = contract
        self._ask = ask
        self._market_open = market_open
        self.orders: list[Any] = []

    def is_market_open(self) -> bool:
        return self._market_open

    def find_contracts(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        return [self._contract] if self._contract else []

    def get_option_ask(self, occ: str) -> float | None:
        return self._ask

    def submit_option_order(
        self, occ: str, qty: int, side: str = "buy", client_order_id: str | None = None
    ) -> dict[str, Any]:
        self.orders.append((occ, qty, side))
        return {"id": f"ord-{len(self.orders)}", "status": "accepted"}


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'a.db'}")
    with eng.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE trade_ideas (
                idea_id TEXT PRIMARY KEY, ticker TEXT, action TEXT,
                confidence FLOAT, preferred_instrument TEXT, notes TEXT,
                status TEXT DEFAULT 'pending', created_at TEXT,
                time_stop_days INTEGER, geo_event_id INTEGER)
        """)
        )
        conn.execute(
            text("CREATE TABLE geo_events (id INTEGER PRIMARY KEY, status TEXT, assessment TEXT)")
        )
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


def _seed(
    engine,
    idea_id,
    *,
    ticker="RTX",
    action="buy_calls",
    conf=0.6,
    notes="[niche 3hop asym0.6] torque | survived red-team; top risk: x",
    pref="~5% OTM, 4 weeks",
    time_stop=None,
    urgency=None,
):
    with engine.begin() as conn:
        gid = None
        if urgency is not None:
            conn.execute(
                text("INSERT INTO geo_events (status, assessment) VALUES ('ASSESSED', :a)"),
                {"a": json.dumps({"urgency": urgency})},
            )
            gid = conn.execute(text("SELECT max(id) FROM geo_events")).scalar()
        conn.execute(
            text("""
            INSERT INTO trade_ideas (idea_id, ticker, action, confidence,
                preferred_instrument, notes, status, created_at, time_stop_days,
                geo_event_id)
            VALUES (:i,:t,:a,:c,:p,:n,'pending','2026-07-21',:ts,:g)
        """),
            {
                "i": idea_id,
                "t": ticker,
                "a": action,
                "c": conf,
                "p": pref,
                "n": notes,
                "ts": time_stop,
                "g": gid,
            },
        )


def _price(_t: str) -> float:
    return 100.0


# --------------------------------------------------------------------------- #
# filter
# --------------------------------------------------------------------------- #


def test_filter_requires_niche_red_team_and_confidence(engine):
    _seed(engine, "ok")  # niche + red-team + conf 0.6
    _seed(engine, "lowconf", conf=0.4)
    _seed(engine, "notniche", notes="plain idea | survived red-team")
    _seed(engine, "nocritic", notes="[niche 3hop] torque")  # no red-team
    _seed(engine, "stock", action="long")  # not an option
    got = {i["idea_id"] for i in fetch_executable_ideas(engine, OptionsExecConfig())}
    assert got == {"ok"}


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def test_submits_eligible_idea(engine):
    _seed(engine, "ok")
    client = _FakeClient(ask=2.0)  # premium = 2*100 = $200 < $500
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1
    assert client.orders == [("RTX260821C00105000", 1, "buy")]
    with engine.connect() as c:
        row = c.execute(
            text(
                "SELECT status, opt_type, premium_est FROM alpaca_option_orders WHERE idea_id='ok'"
            )
        ).one()
    assert row[0] == "submitted" and row[1] == "call" and row[2] == pytest.approx(200.0)


class TestUrgencyFloor:
    """CL-khf7: with min_urgency set, options become the restricted channel —
    only ideas whose source event assessed at/above the floor may buy
    (shares are the primary expression, CL-ncbq)."""

    def test_floor_admits_urgent_and_drops_the_rest(self, engine):
        _seed(engine, "hot", urgency=9)
        _seed(engine, "warm", urgency=6)
        _seed(engine, "orphan")  # no linked event — excluded under a floor
        cfg = OptionsExecConfig(min_urgency=8)
        got = {i["idea_id"] for i in fetch_executable_ideas(engine, cfg)}
        assert got == {"hot"}

    def test_default_floor_is_off(self, engine):
        _seed(engine, "warm", urgency=6)
        _seed(engine, "orphan")
        got = {i["idea_id"] for i in fetch_executable_ideas(engine, OptionsExecConfig())}
        assert got == {"warm", "orphan"}

    def test_garbage_assessment_counts_as_zero(self, engine):
        _seed(engine, "hot", urgency=9)
        with engine.begin() as c:
            c.execute(
                text("INSERT INTO geo_events (status, assessment) VALUES ('ASSESSED','not json')")
            )
            gid = c.execute(text("SELECT max(id) FROM geo_events")).scalar()
            c.execute(
                text("""INSERT INTO trade_ideas (idea_id, ticker, action, confidence,
                        preferred_instrument, notes, status, created_at, geo_event_id)
                        VALUES ('garbled','RTX','buy_calls',0.9,'~5% OTM',
                        '[niche] x | survived red-team','pending','2026-07-21',:g)"""),
                {"g": gid},
            )
        got = {i["idea_id"] for i in fetch_executable_ideas(engine, OptionsExecConfig(min_urgency=8))}
        assert got == {"hot"}


def test_skips_idea_about_to_expire(engine):
    """CL-v2m9: RTX/FLNG/ASC 2026-07-31 — contracts bought 1-2 days before
    their ideas auto-expired were force-closed the next morning by the
    'idea auto-expired' exit rule. Under the life floor -> terminal skip."""
    _seed(engine, "dying", time_stop=2)  # created 07-21, NOW 07-21 16:00 -> ~1.3d left
    client = _FakeClient(ask=2.0)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_expiring"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    with engine.connect() as c:
        assert (
            c.execute(
                text("SELECT status FROM alpaca_option_orders WHERE idea_id='dying'")
            ).scalar()
            == "skipped_expiring"
        )
    # Terminal: the recorded row removes the idea from the next fetch.
    counts2 = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts2["skipped_expiring"] == 0 and client.orders == []


def test_enters_when_idea_has_life(engine):
    _seed(engine, "alive", time_stop=10)  # ~9.3d left >> 3d floor
    client = _FakeClient(ask=2.0)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1 and counts["skipped_expiring"] == 0


def test_idea_life_floor_disabled(engine):
    _seed(engine, "dying", time_stop=2)
    client = _FakeClient(ask=2.0)
    cfg = OptionsExecConfig(min_idea_life_days=0)
    counts = execute_pending_options(engine, client, _price, cfg=cfg, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1 and counts["skipped_expiring"] == 0


def test_skips_when_premium_over_cap(engine):
    _seed(engine, "pricey")
    client = _FakeClient(ask=7.0)  # 7*100 = $700 > $500 cap
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_premium"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    with engine.connect() as c:
        assert (
            c.execute(
                text("SELECT status FROM alpaca_option_orders WHERE idea_id='pricey'")
            ).scalar()
            == "skipped_premium"
        )


def test_daily_cap_enforced(engine):
    for i in range(4):
        _seed(engine, f"i{i}")
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(
        engine,
        client,
        _price,
        cfg=OptionsExecConfig(max_per_day=2),
        now=NOW,
        technicals_fn=_no_tech,
    )
    assert counts["submitted"] == 2
    assert len(client.orders) == 2


def test_hourly_pace_caps_this_cycle(engine):
    # CL-h02l: even with daily room to spare, no more than max_per_hour buy
    # in one cycle — the rest spread to later cycles.
    for i in range(6):
        _seed(engine, f"h{i}")
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(
        engine,
        client,
        _price,
        cfg=OptionsExecConfig(max_per_day=10, max_per_hour=2),
        now=NOW,
        technicals_fn=_no_tech,
    )
    assert counts["submitted"] == 2  # hourly pace, not the 10/day room
    assert len(client.orders) == 2


def test_dedup_not_reexecuted(engine):
    _seed(engine, "once")
    client = _FakeClient(ask=1.0)
    execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    # second run: the idea is already recorded → not fetched again.
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 0
    assert len(client.orders) == 1  # only the first run bought


def test_transient_no_contract_not_recorded_and_retries(engine):
    _seed(engine, "nc")
    counts = execute_pending_options(
        engine, _FakeClient(contract=None), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["no_contract"] == 1 and counts["submitted"] == 0
    # NOT recorded → still fetchable next cycle.
    assert {i["idea_id"] for i in fetch_executable_ideas(engine, OptionsExecConfig())} == {"nc"}


def test_no_quote_is_transient(engine):
    _seed(engine, "nq")
    counts = execute_pending_options(
        engine, _FakeClient(ask=None), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["no_quote"] == 1


def test_no_price_skips(engine):
    _seed(engine, "np")
    counts = execute_pending_options(
        engine, _FakeClient(), lambda t: None, now=NOW, technicals_fn=_no_tech
    )
    assert counts["no_price"] == 1 and counts["submitted"] == 0


def test_market_closed_skips_all(engine):
    _seed(engine, "closed")
    client = _FakeClient(ask=1.0, market_open=False)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["market_closed"] == 1 and counts["submitted"] == 0
    assert client.orders == []  # nothing attempted


# --------------------------------------------------------------------------- #
# open-spread entry delay (CL-3rho)
# --------------------------------------------------------------------------- #


def test_entry_delayed_in_first_15_minutes(engine):
    # 9:35 ET: spreads still wide — ordinary-confidence ideas wait.
    _seed(engine, "early")
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(
        engine, client, _price, now=NOW_AT_OPEN, technicals_fn=_no_tech
    )
    assert counts["entry_delayed"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    # transient: still fetchable on the next 5-min cycle
    assert {i["idea_id"] for i in fetch_executable_ideas(engine, OptionsExecConfig())} == {"early"}


def test_entry_delay_override_for_high_confidence(engine):
    # conf 0.85 >= 0.80 override: extremely strong signal enters at 9:35.
    _seed(engine, "hot", conf=0.85)
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(
        engine, client, _price, now=NOW_AT_OPEN, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1 and counts["entry_delayed"] == 0


def test_entry_allowed_after_delay_window(engine):
    # 9:50 ET (> 9:45) — normal entry resumes.
    _seed(engine, "later")
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(
        engine,
        client,
        _price,
        now=datetime(2026, 7, 21, 13, 50, tzinfo=UTC),  # 9:50 ET
        technicals_fn=_no_tech,
    )
    assert counts["submitted"] == 1 and counts["entry_delayed"] == 0


# --------------------------------------------------------------------------- #
# technical-alignment gate (CL-3xoj)
# --------------------------------------------------------------------------- #

from src.events.technical_context import TechnicalContext  # noqa: E402


def _ctx(trend: str, breakout: str) -> TechnicalContext:
    return TechnicalContext(
        ticker="RTX",
        last_close=100.0,
        sma20=95.0,
        sma50=90.0,
        trend=trend,
        pct_from_20d_high=-0.05,
        pct_from_20d_low=0.05,
        support=90.0,
        resistance=105.0,
        breakout_state=breakout,
        volume_ratio=1.0,
    )


def test_misaligned_calls_skipped(engine):
    # buy_calls into a confirmed downtrend at the lows -> alignment -1.0.
    _seed(engine, "mis")
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(
        engine, client, _price, now=NOW, technicals_fn=lambda t: _ctx("downtrend", "at_lows")
    )
    assert counts["misaligned"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    # transient: still fetchable next cycle
    assert {i["idea_id"] for i in fetch_executable_ideas(engine, OptionsExecConfig())} == {"mis"}


def test_aligned_calls_submitted(engine):
    _seed(engine, "al")
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(
        engine, client, _price, now=NOW, technicals_fn=lambda t: _ctx("uptrend", "at_highs")
    )
    assert counts["submitted"] == 1


def test_no_context_fails_open(engine):
    _seed(engine, "noctx")
    client = _FakeClient(ask=1.0)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=lambda t: None)
    assert counts["submitted"] == 1  # gate can't judge -> allow


# --------------------------------------------------------------------------- #
# double-buy protection via client_order_id (review P1)
# --------------------------------------------------------------------------- #


def test_submit_passes_idea_id_as_client_order_id(engine):
    _seed(engine, "cid1")

    class _CidClient(_FakeClient):
        def __init__(self):
            super().__init__(ask=1.0)
            self.cids = []

        def submit_option_order(self, occ, qty, side="buy", client_order_id=None):
            self.cids.append(client_order_id)
            return super().submit_option_order(occ, qty, side)

    client = _CidClient()
    execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert client.cids == ["curlit-cid1"]


def test_duplicate_client_order_id_recovers_not_rebuys(engine):
    """Crash-after-fill-before-record: the resubmit hits Alpaca's uniqueness
    check; we must record the idea as already-executed, never buy again."""
    _seed(engine, "dup1")

    class _DupClient(_FakeClient):
        def submit_option_order(self, occ, qty, side="buy", client_order_id=None):
            raise RuntimeError("422 client_order_id must be unique: order already exists")

    client = _DupClient(ask=1.0)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts.get("recovered") == 1
    assert counts["submitted"] == 0 and counts["error"] == 0
    with engine.connect() as c:
        row = c.execute(
            text("SELECT status, detail FROM alpaca_option_orders WHERE idea_id='dup1'")
        ).one()
    assert row[0] == "submitted" and "recovered" in row[1]
    # dedup restored: not fetchable next cycle
    assert fetch_executable_ideas(engine, OptionsExecConfig()) == []


# --------------------------------------------------------------------------- #
# concentration caps (CL-3nfm)
# --------------------------------------------------------------------------- #


class _PositionedClient(_FakeClient):
    """FakeClient that also reports an existing Alpaca options book."""

    def __init__(self, positions: list[dict[str, Any]], **kw: Any) -> None:
        super().__init__(**kw)
        self._positions = positions

    def list_option_positions(self) -> list[dict[str, Any]]:
        return self._positions


def _opt_pos(symbol: str, qty: int = 1) -> dict[str, Any]:
    return {"symbol": symbol, "qty": str(qty), "asset_class": "us_option"}


def test_skips_when_same_contract_already_held(engine):
    """CL-3nfm: idea_id dedup can't see this — a DIFFERENT idea naming the
    same contract passed it, and FRO reached qty=3 on one contract."""
    _seed(engine, "ok")
    client = _PositionedClient([_opt_pos("RTX260821C00105000", 1)], ask=2.0)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_concentration"] == 1
    assert counts["submitted"] == 0
    assert client.orders == []


def test_skips_when_underlying_cap_reached(engine):
    # Two RTX contracts already open (different strikes) → cap on the name.
    _seed(engine, "ok")
    client = _PositionedClient(
        [_opt_pos("RTX260821C00200000", 1), _opt_pos("RTX260828C00210000", 1)],
        ask=2.0,
    )
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_concentration"] == 1
    assert client.orders == []


def test_allows_when_book_is_clear(engine):
    _seed(engine, "ok")
    client = _PositionedClient([_opt_pos("AAPL260821C00200000", 1)], ask=2.0)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1


def test_underlying_match_is_root_exact_not_prefix(engine):
    """ASC/ASTL both start with 'AS' — a prefix match would wrongly collide
    and block unrelated names."""
    from src.execution.alpaca_options_executor import _held_contract_counts

    class _C:
        def list_option_positions(self):  # type: ignore[no-untyped-def]
            return [_opt_pos("ASTL260821P00004000", 1)]

    same_sym, same_under = _held_contract_counts(_C(), "ASC260821C00017500", "ASC")
    assert same_sym == 0
    assert same_under == 0  # ASTL must NOT count against ASC


def test_position_lookup_failure_fails_open(engine):
    # A broker blip must not block entries — the cap is a ceiling, not an
    # interlock (premium + daily/hourly caps still bound the damage).
    _seed(engine, "ok")

    class _Boom(_FakeClient):
        def list_option_positions(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("alpaca down")

    counts = execute_pending_options(
        engine, _Boom(ask=2.0), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1


def test_client_without_position_listing_fails_open(engine):
    _seed(engine, "ok")
    counts = execute_pending_options(
        engine, _FakeClient(ask=2.0), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1


# --------------------------------------------------------------------------- #
# entry spread filter + entry-mid baseline (CL-d44a)
# --------------------------------------------------------------------------- #


class _QuoteClient(_FakeClient):
    def __init__(self, bid: float | None, ask: float | None, **kw: Any) -> None:
        super().__init__(ask=ask, **kw)
        self._q = (bid, ask)

    def get_option_quote(self, occ: str) -> tuple[float | None, float | None]:
        return self._q


def test_skips_contract_with_a_wide_spread(engine):
    """CL-d44a: 0.13/0.42 is a 69% spread — buying the ask and marking the
    bid is an instant -69%, and it needs a ~69% move just to break even."""
    _seed(engine, "ok")
    client = _QuoteClient(bid=0.13, ask=0.42)
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_spread"] == 1
    assert counts["submitted"] == 0
    assert client.orders == []


def test_tight_spread_is_bought_and_records_entry_mid(engine):
    _seed(engine, "ok")
    client = _QuoteClient(bid=1.90, ask=2.00)  # 5% spread
    counts = execute_pending_options(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1
    with engine.connect() as c:
        mid, spread = c.execute(
            text("SELECT entry_mid, entry_spread_pct FROM alpaca_option_orders WHERE idea_id='ok'")
        ).one()
    assert mid == pytest.approx(1.95)
    assert spread == pytest.approx(0.05)


def test_wide_spread_skip_is_transient_not_recorded(engine):
    # Must NOT write a terminal row — the idea retries when the market tightens.
    _seed(engine, "ok")
    execute_pending_options(
        engine, _QuoteClient(bid=0.13, ask=0.42), _price, now=NOW, technicals_fn=_no_tech
    )
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM alpaca_option_orders")).scalar_one() == 0
    # Market tightens on a later cycle → it buys.
    counts = execute_pending_options(
        engine, _QuoteClient(bid=1.90, ask=2.00), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1


def test_client_without_quote_support_still_buys(engine):
    # Fail-open: a client exposing only get_option_ask keeps working, with a
    # NULL entry_mid (the exit path falls back to the legacy mark + guard).
    _seed(engine, "ok")
    counts = execute_pending_options(
        engine, _FakeClient(ask=2.0), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1
    with engine.connect() as c:
        assert (
            c.execute(
                text("SELECT entry_mid FROM alpaca_option_orders WHERE idea_id='ok'")
            ).scalar_one()
            is None
        )
