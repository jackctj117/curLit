"""Tests for the Alpaca paper options executor (CL-ldd2).

Sqlite trade_ideas + alpaca_option_orders; fake Alpaca client + price fn.
Covers the policy filter (niche + red-team + confidence), the premium cap, the
daily cap, dedup, and transient-miss retry behavior.
"""

from __future__ import annotations

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
                status TEXT DEFAULT 'pending', created_at TEXT)
        """)
        )
        sql = _strip_sql_comments(Path("migrations/014_alpaca_option_orders.sql").read_text())
        sql = sql.replace("TIMESTAMPTZ", "TEXT").replace("NUMERIC", "FLOAT")
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
):
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO trade_ideas (idea_id, ticker, action, confidence,
                preferred_instrument, notes, status, created_at)
            VALUES (:i,:t,:a,:c,:p,:n,'pending','2026-07-21')
        """),
            {"i": idea_id, "t": ticker, "a": action, "c": conf, "p": pref, "n": notes},
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
