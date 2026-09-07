"""Tests for the Alpaca paper EQUITY (shares) executor (CL-ncbq).

Sqlite trade_ideas + alpaca_equity_orders (mig 019); fake Alpaca client +
price fn. Covers the long/short direction mapping, the notional→qty sizing,
the terminal skips (short disabled, price too high, idea about to expire),
dedup, the daily/hourly caps, the concentration cap and the market-hours gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.execution.alpaca_equity_executor import (
    EquityExecConfig,
    execute_pending_equities,
    fetch_executable_ideas,
)

# Midday ET — outside the opening entry-delay window, so the delay gate is
# inert everywhere except the tests that exercise it.
NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)  # 12:00 ET
NOW_AT_OPEN = datetime(2026, 7, 21, 13, 35, tzinfo=UTC)  # 9:35 ET


def _no_tech(_t: str):  # unit tests: no live yfinance — fail-open path
    return None


class _FakeClient:
    def __init__(
        self,
        market_open: bool = True,
        positions: list[dict[str, Any]] | None = None,
        fill: float | None = None,
    ) -> None:
        self._market_open = market_open
        self._positions = positions or []
        self._fill = fill
        self.orders: list[Any] = []
        self.cids: list[str | None] = []

    def is_market_open(self, **kw: Any) -> bool:
        return self._market_open

    def list_equity_positions(self, **kw: Any) -> list[dict[str, Any]]:
        return self._positions

    def submit_equity_order(
        self,
        symbol: str,
        qty: int,
        side: str = "buy",
        client_order_id: str | None = None,
        **kw: Any,
    ) -> dict[str, Any]:
        self.orders.append((symbol, qty, side))
        self.cids.append(client_order_id)
        return {
            "id": f"ord-{len(self.orders)}",
            "status": "accepted",
            "filled_avg_price": self._fill,
        }

    def get_order(self, order_id: str, **kw: Any) -> dict[str, Any] | None:
        return {"id": order_id, "status": "filled", "filled_avg_price": self._fill}


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'eq.db'}")
    with eng.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE trade_ideas (
                idea_id TEXT PRIMARY KEY, ticker TEXT, action TEXT,
                confidence FLOAT, preferred_instrument TEXT, notes TEXT,
                status TEXT DEFAULT 'pending', created_at TEXT,
                time_stop_days INTEGER)
        """)
        )
        sql = _strip_sql_comments(Path("migrations", "019_alpaca_equity_orders.sql").read_text())
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
):
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO trade_ideas (idea_id, ticker, action, confidence,
                preferred_instrument, notes, status, created_at, time_stop_days)
            VALUES (:i,:t,:a,:c,:p,:n,'pending','2026-07-21',:ts)
        """),
            {
                "i": idea_id,
                "t": ticker,
                "a": action,
                "c": conf,
                "p": pref,
                "n": notes,
                "ts": time_stop,
            },
        )


def _price(_t: str) -> float:
    return 95.0


def _row(engine, idea_id):
    with engine.connect() as c:
        return dict(
            c.execute(text("SELECT * FROM alpaca_equity_orders WHERE idea_id=:i"), {"i": idea_id})
            .one()
            ._mapping
        )


# --------------------------------------------------------------------------- #
# eligibility — the SAME candidate set the options book sees
# --------------------------------------------------------------------------- #


def test_filter_requires_niche_red_team_and_confidence(engine):
    _seed(engine, "ok")  # niche + red-team + conf 0.6
    _seed(engine, "lowconf", conf=0.4)
    _seed(engine, "notniche", notes="plain idea | survived red-team")
    _seed(engine, "nocritic", notes="[niche 3hop] torque")
    _seed(engine, "fx", action="long")  # not a directional equity idea
    got = {i["idea_id"] for i in fetch_executable_ideas(engine, EquityExecConfig())}
    assert got == {"ok"}


# --------------------------------------------------------------------------- #
# entries
# --------------------------------------------------------------------------- #


def test_long_entry_records_row_and_sizes_by_notional(engine):
    """buy_calls -> long shares; qty = floor($1000 / $95) = 10."""
    _seed(engine, "ok")
    client = _FakeClient(fill=95.10)
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1
    assert client.orders == [("RTX", 10, "buy")]
    assert client.cids == ["curlit-eq-ok"]
    row = _row(engine, "ok")
    assert row["status"] == "submitted"
    assert row["side"] == "buy"
    assert row["qty"] == 10
    assert row["notional_est"] == pytest.approx(950.0)
    # The FILL is the basis (CL-ncbq): shares need no entry_mid machinery.
    assert row["entry_price"] == pytest.approx(95.10)


def test_short_entry_from_buy_puts(engine):
    """buy_puts -> short shares, submitted to Alpaca as a plain 'sell'."""
    _seed(engine, "bear", action="buy_puts")
    client = _FakeClient(fill=95.0)
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1
    assert client.orders == [("RTX", 10, "sell")]
    assert _row(engine, "bear")["side"] == "sell_short"


def test_short_disabled_is_a_terminal_skip(engine):
    _seed(engine, "bear", action="buy_puts")
    client = _FakeClient()
    cfg = EquityExecConfig(allow_short=False)
    counts = execute_pending_equities(
        engine, client, _price, cfg=cfg, now=NOW, technicals_fn=_no_tech
    )
    assert counts["skipped_short_disabled"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    assert _row(engine, "bear")["status"] == "skipped_short_disabled"
    # Terminal: the recorded row removes the idea from the next fetch.
    assert fetch_executable_ideas(engine, cfg) == []


def test_share_price_above_the_sleeve_is_a_terminal_skip(engine):
    """A $900 share against a $500 sleeve rounds to zero shares. Rounding UP
    would silently take a position ~2x the intended size."""
    _seed(engine, "spendy")
    client = _FakeClient()
    cfg = EquityExecConfig(notional_usd=500.0)
    counts = execute_pending_equities(
        engine, client, lambda t: 900.0, cfg=cfg, now=NOW, technicals_fn=_no_tech
    )
    assert counts["skipped_price_too_high"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    assert _row(engine, "spendy")["status"] == "skipped_price_too_high"


def test_dedup_not_reexecuted(engine):
    _seed(engine, "once")
    client = _FakeClient(fill=95.0)
    execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 0
    assert len(client.orders) == 1  # only the first run traded


def test_skips_idea_about_to_expire(engine):
    """CL-v2m9: an idea with less life than the floor would be force-closed by
    the time-stop exit rule almost immediately — churn, not a thesis."""
    _seed(engine, "dying", time_stop=2)  # created 07-21, NOW 07-21 -> ~1.3d left
    client = _FakeClient()
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_expiring"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    assert _row(engine, "dying")["status"] == "skipped_expiring"
    counts2 = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts2["skipped_expiring"] == 0 and client.orders == []


def test_enters_when_idea_has_life(engine):
    _seed(engine, "alive", time_stop=10)  # ~9.3d left >> 3d floor
    counts = execute_pending_equities(
        engine, _FakeClient(fill=95.0), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1 and counts["skipped_expiring"] == 0


def test_daily_cap_enforced(engine):
    for i in range(4):
        _seed(engine, f"i{i}", ticker=f"T{i}")
    client = _FakeClient(fill=95.0)
    counts = execute_pending_equities(
        engine,
        client,
        _price,
        cfg=EquityExecConfig(max_per_day=2, max_per_hour=5),
        now=NOW,
        technicals_fn=_no_tech,
    )
    assert counts["submitted"] == 2
    assert len(client.orders) == 2


def test_hourly_pace_caps_this_cycle(engine):
    for i in range(6):
        _seed(engine, f"h{i}", ticker=f"T{i}")
    client = _FakeClient(fill=95.0)
    counts = execute_pending_equities(
        engine,
        client,
        _price,
        cfg=EquityExecConfig(max_per_day=10, max_per_hour=2),
        now=NOW,
        technicals_fn=_no_tech,
    )
    assert counts["submitted"] == 2  # hourly pace, not the 10/day room
    assert len(client.orders) == 2


def test_no_price_is_transient(engine):
    _seed(engine, "np")
    counts = execute_pending_equities(
        engine, _FakeClient(), lambda t: None, now=NOW, technicals_fn=_no_tech
    )
    assert counts["no_price"] == 1 and counts["submitted"] == 0
    # NOT recorded -> still fetchable next cycle.
    assert {i["idea_id"] for i in fetch_executable_ideas(engine, EquityExecConfig())} == {"np"}


def test_market_closed_skips_all(engine):
    _seed(engine, "closed")
    client = _FakeClient(market_open=False)
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["market_closed"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM alpaca_equity_orders")).scalar_one() == 0


# --------------------------------------------------------------------------- #
# concentration (CL-3nfm) — idea_id dedup cannot see the live book
# --------------------------------------------------------------------------- #


def _eq_pos(symbol: str, qty: int = 10) -> dict[str, Any]:
    return {"symbol": symbol, "qty": str(qty), "asset_class": "us_equity"}


def test_skips_when_ticker_already_held(engine):
    _seed(engine, "ok")
    client = _FakeClient(positions=[_eq_pos("RTX")])
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_concentration"] == 1 and counts["submitted"] == 0
    assert client.orders == []


def test_allows_when_book_is_clear(engine):
    _seed(engine, "ok")
    client = _FakeClient(positions=[_eq_pos("AAPL")], fill=95.0)
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1


def test_position_lookup_failure_blocks_entry_and_retries(engine):
    # CL-0deu.1.1 deliberately reverses the legacy fail-open policy.
    _seed(engine, "ok")

    class _Boom(_FakeClient):
        def list_equity_positions(self, **kw: Any) -> list[dict[str, Any]]:
            raise RuntimeError("alpaca down")

    client = _Boom(fill=95.0)
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 0 and counts["blocked_exposure"] == 1
    assert client.orders == []
    assert {i["idea_id"] for i in fetch_executable_ideas(engine, EquityExecConfig())} == {"ok"}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM alpaca_equity_orders")).scalar_one() == 0
    counts = execute_pending_equities(
        engine, _FakeClient(fill=95.0), _price, now=NOW, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1


def test_missing_position_capability_blocks_entry(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(engine, "unknown")
    monkeypatch.delattr(_FakeClient, "list_equity_positions")
    client = _FakeClient()
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["blocked_exposure"] == 1 and counts["submitted"] == 0
    assert client.orders == []


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [None],
        *[
            [{"symbol": "RTX", "asset_class": "us_equity", "qty": qty}]
            for qty in (None, "", "bad", "NaN", "Infinity", "-Infinity", True)
        ],
    ],
)
def test_malformed_exposure_blocks_entry(
    engine: Any, payload: object, caplog: pytest.LogCaptureFixture
) -> None:
    _seed(engine, "unknown")

    class _Invalid(_FakeClient):
        def list_equity_positions(self, **kw: Any) -> Any:
            return payload

    client = _Invalid()
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["blocked_exposure"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    assert {i["idea_id"] for i in fetch_executable_ideas(engine, EquityExecConfig())} == {"unknown"}
    blocked = [r for r in caplog.records if hasattr(r, "extra_data")]
    assert blocked[-1].extra_data["reason"] == "blocked_exposure"


@pytest.mark.parametrize("quantity", ["0.001", "-0.001", "0.5", "-0.5"])
def test_fractional_external_equity_position_counts_as_held(engine: Any, quantity: str) -> None:
    _seed(engine, "fractional")
    client = _FakeClient(positions=[{"symbol": "RTX", "asset_class": "us_equity", "qty": quantity}])
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["skipped_concentration"] == 1 and counts["submitted"] == 0
    assert counts["blocked_exposure"] == 0
    assert client.orders == []


# --------------------------------------------------------------------------- #
# opening delay + alignment gate (shared with the options book)
# --------------------------------------------------------------------------- #


def test_entry_delayed_in_first_15_minutes(engine):
    _seed(engine, "early")
    client = _FakeClient()
    counts = execute_pending_equities(
        engine, client, _price, now=NOW_AT_OPEN, technicals_fn=_no_tech
    )
    assert counts["entry_delayed"] == 1 and counts["submitted"] == 0
    assert client.orders == []
    # transient: still fetchable on the next cycle
    assert {i["idea_id"] for i in fetch_executable_ideas(engine, EquityExecConfig())} == {"early"}


def test_entry_delay_override_for_high_confidence(engine):
    _seed(engine, "hot", conf=0.85)
    counts = execute_pending_equities(
        engine, _FakeClient(fill=95.0), _price, now=NOW_AT_OPEN, technicals_fn=_no_tech
    )
    assert counts["submitted"] == 1 and counts["entry_delayed"] == 0


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


def test_misaligned_long_skipped(engine):
    # Going long into a confirmed downtrend at the lows -> alignment -1.0.
    _seed(engine, "mis")
    client = _FakeClient()
    counts = execute_pending_equities(
        engine, client, _price, now=NOW, technicals_fn=lambda t: _ctx("downtrend", "at_lows")
    )
    assert counts["misaligned"] == 1 and counts["submitted"] == 0
    assert client.orders == []


def test_aligned_long_submitted(engine):
    _seed(engine, "al")
    counts = execute_pending_equities(
        engine,
        _FakeClient(fill=95.0),
        _price,
        now=NOW,
        technicals_fn=lambda t: _ctx("uptrend", "at_highs"),
    )
    assert counts["submitted"] == 1


# --------------------------------------------------------------------------- #
# crash safety
# --------------------------------------------------------------------------- #


def test_duplicate_client_order_id_recovers_not_retrades(engine):
    """Crash-after-fill-before-record: the resubmit hits Alpaca's uniqueness
    check; record the idea as already-executed, never trade again."""
    _seed(engine, "dup")

    class _DupClient(_FakeClient):
        def submit_equity_order(self, symbol, qty, side="buy", client_order_id=None, **kw):
            raise RuntimeError("422 client_order_id must be unique: order already exists")

    counts = execute_pending_equities(engine, _DupClient(), _price, now=NOW, technicals_fn=_no_tech)
    assert counts.get("recovered") == 1
    assert counts["submitted"] == 0 and counts["error"] == 0
    row = _row(engine, "dup")
    assert row["status"] == "submitted" and "recovered" in row["detail"]
    assert fetch_executable_ideas(engine, EquityExecConfig()) == []


def test_unfilled_order_falls_back_to_the_submit_price(engine):
    """A market order usually returns before the fill. Recording NO basis
    would leave the exit manager unable to compute P&L, so the submit-time
    price stands in until the position's avg_entry_price is available."""
    _seed(engine, "pending")
    client = _FakeClient(fill=None)
    counts = execute_pending_equities(engine, client, _price, now=NOW, technicals_fn=_no_tech)
    assert counts["submitted"] == 1
    assert _row(engine, "pending")["entry_price"] == pytest.approx(95.0)


def test_trade_ideas_row_is_not_marked(engine):
    """CL-ncbq: the options executor may express the SAME idea — the two books
    are a deliberate A/B, so neither may consume the idea."""
    _seed(engine, "ab")
    execute_pending_equities(
        engine, _FakeClient(fill=95.0), _price, now=NOW, technicals_fn=_no_tech
    )
    with engine.connect() as c:
        assert (
            c.execute(text("SELECT status FROM trade_ideas WHERE idea_id='ab'")).scalar()
            == "pending"
        )
