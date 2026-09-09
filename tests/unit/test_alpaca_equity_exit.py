"""Tests for the Alpaca EQUITY (shares) exit manager (CL-ncbq).

Sqlite alpaca_equity_orders (mig 019) + trade_ideas + geo_events; fake Alpaca
client with canned positions/quotes. Covers every exit rule in priority order,
the SIGNED short-side P&L, the idea's own advisory stop/target overriding the
config defaults, the vanished-position finalizations, fill confirmation, and
the market-hours gate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.execution.alpaca_equity_exit import (
    EquityExitConfig,
    ExitReason,
    evaluate_equity_exit,
    manage_equity_exits,
    signed_pnl_pct,
)

NOW = datetime(2026, 7, 22, 15, 0, tzinfo=UTC)


class _FakeClient:
    def __init__(
        self,
        positions: list[dict[str, Any]] | None = None,
        market_open: bool = True,
        quote: tuple[float | None, float | None] | None = None,
        order_status: str = "accepted",
        fill: float | None = None,
    ) -> None:
        self._positions = positions or []
        self._market_open = market_open
        self._quote = quote
        self._order_status = order_status
        self._fill = fill
        self.orders: list[Any] = []
        self.cids: list[str | None] = []

    def is_market_open(self, **kw: Any) -> bool:
        return self._market_open

    def list_equity_positions(self, **kw: Any) -> list[dict[str, Any]]:
        return self._positions

    def get_stock_quote(self, symbol: str, **kw: Any) -> tuple[float | None, float | None]:
        return self._quote if self._quote is not None else (None, None)

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
        return {"id": f"ord-{len(self.orders)}", "status": "accepted"}

    def get_order(self, order_id: str, **kw: Any) -> dict[str, Any] | None:
        return {
            "id": order_id,
            "status": self._order_status,
            "filled_avg_price": self._fill,
        }


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'eq.db'}")
    with eng.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE trade_ideas (
                idea_id TEXT PRIMARY KEY, geo_event_id INTEGER, ticker TEXT,
                action TEXT, confidence FLOAT, time_stop_days INTEGER,
                stop_loss_pct FLOAT, target_prices TEXT,
                price_at_signal FLOAT, notes TEXT,
                status TEXT DEFAULT 'pending', status_updated_at TEXT,
                created_at TEXT)
        """)
        )
        conn.execute(text("CREATE TABLE geo_events (id INTEGER PRIMARY KEY, status TEXT)"))
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
    side="buy",
    qty=10,
    entry_price=100.0,
    submitted_at="2026-07-20",
    time_stop_days=10,
    idea_status="pending",
    event_status="TRADED",
    exit_status=None,
    exit_reason=None,
    exit_order_id=None,
    stop_loss_pct=None,
    target_prices=None,
    price_at_signal=None,
):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO geo_events (status) VALUES (:es)"), {"es": event_status})
        gid = conn.execute(text("SELECT max(id) FROM geo_events")).scalar()
        conn.execute(
            text("""
            INSERT INTO trade_ideas (idea_id, geo_event_id, ticker, action,
                confidence, time_stop_days, stop_loss_pct, target_prices,
                price_at_signal, status, status_updated_at, created_at)
            VALUES (:i,:g,:t,'buy_calls',0.6,:ts,:sl,:tp,:pas,:st,:sa,:sa)
        """),
            {
                "i": idea_id,
                "g": gid,
                "t": ticker,
                "ts": time_stop_days,
                "sl": stop_loss_pct,
                "tp": json.dumps(target_prices) if target_prices else None,
                "pas": price_at_signal,
                "st": idea_status,
                "sa": submitted_at,
            },
        )
        conn.execute(
            text("""
            INSERT INTO alpaca_equity_orders (idea_id, ticker, side, qty,
                notional_est, entry_price, alpaca_order_id, status, detail,
                submitted_at, exit_status, exit_reason, exit_order_id)
            VALUES (:i,:t,:sd,:q,:n,:ep,'ord-x','submitted',NULL,:sa,
                    :xs,:xr,:xo)
        """),
            {
                "i": idea_id,
                "t": ticker,
                "sd": side,
                "q": qty,
                "n": qty * entry_price,
                "ep": entry_price,
                "sa": submitted_at,
                "xs": exit_status,
                "xr": exit_reason,
                "xo": exit_order_id,
            },
        )


def _pos(symbol="RTX", qty="10", avg="100.0", cur="101.0"):
    return {
        "symbol": symbol,
        "asset_class": "us_equity",
        "qty": qty,
        "avg_entry_price": avg,
        "current_price": cur,
    }


@pytest.mark.parametrize(
    "payload", [None, {}, [{"symbol": "RTX", "asset_class": "us_equity", "qty": "NaN"}]]
)
def test_invalid_broker_snapshot_cannot_close_position(engine: Any, payload: object) -> None:
    from src.execution.alpaca_equity import AlpacaEquityClient

    _seed(engine, "unknown")
    requests: list[tuple[str, str]] = []

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        params: dict[str, str] | None,
        body: dict[str, Any] | None,
    ) -> object:
        requests.append((method, url))
        return {"is_open": True} if url.endswith("/clock") else payload

    client = AlpacaEquityClient("K", "S", request_fn=request)
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["error"] == 1
    assert _row(engine, "unknown")["exit_status"] is None
    assert all(method == "GET" for method, _ in requests)


def _row(engine, idea_id):
    with engine.connect() as c:
        return dict(
            c.execute(text("SELECT * FROM alpaca_equity_orders WHERE idea_id=:i"), {"i": idea_id})
            .one()
            ._mapping
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("held", [False, True])
def test_shared_symbol_never_consumes_another_rows_position(engine, reverse, held):
    rows = [("old", "submitted"), ("new", None)]
    for idea_id, exit_status in reversed(rows) if reverse else rows:
        _seed(engine, idea_id, exit_status=exit_status, submitted_at="2026-07-01")
    before = {idea_id: _row(engine, idea_id) for idea_id, _ in rows}
    client = _FakeClient(positions=[_pos()] if held else [])
    counts = manage_equity_exits(engine, client, now=NOW)
    # One broker net position is not two independently owned allocations. No
    # first-row aggregate sale and no second-row invented external closure.
    assert client.orders == []
    assert {idea_id: _row(engine, idea_id) for idea_id, _ in rows} == before
    assert counts["reconciliation_required"] == 2


# --------------------------------------------------------------------------- #
# signed P&L — the short book must never be judged by long-side signs
# --------------------------------------------------------------------------- #


def test_signed_pnl_is_side_aware():
    assert signed_pnl_pct("buy", 100.0, 110.0) == pytest.approx(0.10)
    assert signed_pnl_pct("buy", 100.0, 90.0) == pytest.approx(-0.10)
    # A short that FELL is a win.
    assert signed_pnl_pct("sell_short", 100.0, 90.0) == pytest.approx(0.10)
    assert signed_pnl_pct("sell_short", 100.0, 110.0) == pytest.approx(-0.10)
    # Never fabricated.
    assert signed_pnl_pct("buy", None, 110.0) is None
    assert signed_pnl_pct("buy", 100.0, None) is None
    assert signed_pnl_pct("buy", 0.0, 110.0) is None


# --------------------------------------------------------------------------- #
# exit rules (priority order)
# --------------------------------------------------------------------------- #


def test_stop_loss_long(engine):
    _seed(engine, "loss")  # entry 100
    client = _FakeClient([_pos(cur="94.0")])  # -6% <= -5% default
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert client.orders == [("RTX", 10, "sell")]
    row = _row(engine, "loss")
    assert row["exit_reason"] == "stop_loss"
    assert row["pnl_pct"] == pytest.approx(-0.06)


def test_stop_loss_short_is_signed(engine):
    """A short is stopped out when the share RISES. The long-side sign would
    read +6% here and hold a losing short forever."""
    _seed(engine, "shortloss", side="sell_short")
    client = _FakeClient([_pos(cur="106.0")])  # +6% on the share = -6% for us
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert client.orders == [("RTX", 10, "buy")]  # buy to COVER
    row = _row(engine, "shortloss")
    assert row["exit_reason"] == "stop_loss"
    assert row["pnl_pct"] == pytest.approx(-0.06)


def test_short_profit_target_when_share_falls(engine):
    _seed(engine, "shortwin", side="sell_short")
    client = _FakeClient([_pos(cur="88.0")])  # -12% on the share = +12% for us
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    row = _row(engine, "shortwin")
    assert row["exit_reason"] == "profit_target"
    assert row["pnl_pct"] == pytest.approx(0.12)


def test_profit_target_uses_the_ideas_own_advisory_target(engine):
    """The impact agent already reasoned about how far THIS thesis runs
    (target_prices vs price_at_signal = +5%); the flat 10% default would sit
    through the exit the desk actually called."""
    _seed(engine, "adv", target_prices=[105.0], price_at_signal=100.0)
    client = _FakeClient([_pos(cur="106.0")])  # +6%: past the idea's 5%
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "adv")["exit_reason"] == "profit_target"


def test_config_target_applies_when_the_idea_has_none(engine):
    _seed(engine, "flat")  # no advisory levels
    client = _FakeClient([_pos(cur="106.0")])  # +6% < 10% default
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0


def test_stop_uses_the_ideas_own_advisory_stop(engine):
    # The idea says an 8% adverse move kills it; -6% must NOT stop out.
    _seed(engine, "wide", stop_loss_pct=0.08)
    counts = manage_equity_exits(engine, _FakeClient([_pos(cur="94.0")]), now=NOW)
    assert counts["held"] == 1
    # -9% does.
    _seed(engine, "wide2", ticker="XOM", stop_loss_pct=0.08)
    counts = manage_equity_exits(engine, _FakeClient([_pos(symbol="XOM", cur="91.0")]), now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "wide2")["exit_reason"] == "stop_loss"


def test_absurd_advisory_pct_falls_back_to_config(engine):
    # stop_loss_pct=0 would exit instantly; a sane default must win.
    _seed(engine, "zero", stop_loss_pct=0.0)
    counts = manage_equity_exits(engine, _FakeClient([_pos(cur="101.0")]), now=NOW)
    assert counts["held"] == 1


def test_time_stop_closes(engine):
    _seed(engine, "old", submitted_at="2026-07-01", time_stop_days=10)
    client = _FakeClient([_pos(cur="100.5")])  # flat — only the clock triggers
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "old")["exit_reason"] == "time_stop"


def test_idea_auto_expired_is_a_time_stop(engine):
    _seed(engine, "exp", idea_status="expired")
    counts = manage_equity_exits(engine, _FakeClient([_pos(cur="100.5")]), now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "exp")["exit_reason"] == "time_stop"


def test_thesis_invalidated_beats_profit(engine):
    _seed(engine, "dead", event_status="DISMISSED")
    manage_equity_exits(engine, _FakeClient([_pos(cur="130.0")]), now=NOW)
    assert _row(engine, "dead")["exit_reason"] == "thesis_invalidated"


def test_idea_cancelled_invalidates(engine):
    _seed(engine, "cxl", idea_status="cancelled")
    manage_equity_exits(engine, _FakeClient([_pos(cur="100.5")]), now=NOW)
    assert _row(engine, "cxl")["exit_reason"] == "thesis_invalidated"


def test_event_expired_is_not_thesis_invalidation(engine):
    """geo_event EXPIRED is the ~2h intraday FX gate lapsing — routine — NOT a
    verdict on a multi-day equity thesis."""
    _seed(engine, "gate", event_status="EXPIRED")
    counts = manage_equity_exits(engine, _FakeClient([_pos(cur="101.0")]), now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0


def test_healthy_position_held(engine):
    _seed(engine, "hold")
    client = _FakeClient([_pos(cur="101.0")])  # +1%
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0
    assert client.orders == []
    assert _row(engine, "hold")["exit_status"] is None


def test_default_time_stop_when_the_idea_carries_none():
    """An idea with no time_stop_days still has a deadline — the config
    default. (Rule 5, stale, is deliberately unreachable while rule 2 can
    evaluate; it is the net for rows whose entry timestamp was unreadable on
    earlier cycles and later healed.)"""
    row = {
        "side": "buy",
        "entry_price": 100.0,
        "submitted_at": "2026-07-01",  # 21d held
        "time_stop_days": None,
        "idea_status": "pending",
        "event_status": "TRADED",
    }
    reason, detail = evaluate_equity_exit(row, _pos(cur="100.5"), EquityExitConfig(), NOW)
    assert reason == ExitReason.TIME_STOP
    assert "time stop 10d" in detail


def test_priority_time_stop_over_stop_loss():
    row = {
        "side": "buy",
        "entry_price": 100.0,
        "submitted_at": "2026-07-01",
        "time_stop_days": 10,
        "idea_status": "pending",
        "event_status": "TRADED",
    }
    reason, _ = evaluate_equity_exit(row, _pos(cur="80.0"), EquityExitConfig(), NOW)
    assert reason == ExitReason.TIME_STOP  # not stop_loss


def test_missing_price_data_skips_only_the_pnl_rules():
    row = {
        "side": "buy",
        "entry_price": 100.0,
        "submitted_at": "2026-07-20",
        "time_stop_days": 10,
        "idea_status": "pending",
        "event_status": "TRADED",
    }
    # No mark anywhere: P&L rules can't run, date rules find nothing -> hold.
    assert evaluate_equity_exit(row, _pos(cur=None), EquityExitConfig(), NOW) is None


def test_quote_mid_is_preferred_over_the_position_mark():
    row = {
        "side": "buy",
        "entry_price": 100.0,
        "submitted_at": "2026-07-20",
        "time_stop_days": 10,
        "idea_status": "pending",
        "event_status": "TRADED",
    }
    # Stale position mark says flat; the live two-sided quote says -6%.
    decision = evaluate_equity_exit(
        row, _pos(cur="100.0"), EquityExitConfig(), NOW, quote=(93.99, 94.01)
    )
    assert decision is not None and decision[0] == ExitReason.STOP_LOSS


# --------------------------------------------------------------------------- #
# lifecycle / confirmation
# --------------------------------------------------------------------------- #


def test_exit_confirm_writes_pnl_pct_from_the_real_fill(engine):
    """The share fill IS the honest basis (CL-ncbq) — on confirmation the
    estimate is REPLACED by the realized number, not COALESCEd away."""
    _seed(engine, "conf", exit_status="submitted", exit_reason="stop_loss", exit_order_id="ord-9")
    client = _FakeClient([_pos(cur="94.0")], order_status="filled", fill=93.50)
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["closed_confirmed"] == 1
    assert client.orders == []  # never resubmitted
    row = _row(engine, "conf")
    assert row["exit_status"] == "closed"
    assert row["exit_price"] == pytest.approx(93.50)
    assert row["pnl_pct"] == pytest.approx(-0.065)


def test_exit_pending_when_the_order_has_not_filled(engine):
    _seed(engine, "pend", exit_status="submitted", exit_reason="stop_loss", exit_order_id="ord-9")
    client = _FakeClient([_pos(cur="94.0")], order_status="new")
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["exit_pending"] == 1
    assert client.orders == []


def test_vanished_after_exit_submitted_confirms(engine):
    _seed(engine, "done", exit_status="submitted", exit_reason="stop_loss")
    counts = manage_equity_exits(engine, _FakeClient([]), now=NOW)
    assert counts["closed_confirmed"] == 1
    row = _row(engine, "done")
    assert row["exit_status"] == "closed"
    assert row["exit_reason"] == "stop_loss"  # submit-time reason kept


def test_vanished_without_our_order_is_external(engine):
    _seed(engine, "gone")
    counts = manage_equity_exits(engine, _FakeClient([]), now=NOW)
    assert counts["closed_external"] == 1
    assert _row(engine, "gone")["exit_reason"] == "closed_external"


def test_unmatched_position_never_managed(engine):
    # The account is shared with the options book and manual trades.
    client = _FakeClient([_pos(symbol="MYST")])
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["unmatched"] == 1
    assert client.orders == []


def test_exit_uses_the_namespaced_client_order_id(engine):
    _seed(engine, "cid", submitted_at="2026-07-01")  # time stop
    client = _FakeClient([_pos()])
    manage_equity_exits(engine, client, now=NOW)
    assert client.cids == ["curlit-eq-exit-cid"]


def test_duplicate_exit_cid_recovers(engine):
    _seed(engine, "dup", submitted_at="2026-07-01")

    class _DupClient(_FakeClient):
        def submit_equity_order(self, symbol, qty, side="buy", client_order_id=None, **kw):
            raise RuntimeError("422 client_order_id must be unique: order already exists")

    counts = manage_equity_exits(engine, _DupClient([_pos()]), now=NOW)
    assert counts["recovered"] == 1 and counts["error"] == 0
    row = _row(engine, "dup")
    assert row["exit_status"] == "submitted"
    assert row["exit_order_id"] == "recovered"


def test_market_closed_noop(engine):
    _seed(engine, "closed", submitted_at="2026-07-01")
    client = _FakeClient([_pos()], market_open=False)
    counts = manage_equity_exits(engine, client, now=NOW)
    assert counts["market_closed"] == 1
    assert client.orders == []
    assert _row(engine, "closed")["exit_status"] is None


def test_trade_ideas_row_is_not_closed_by_the_exit(engine):
    """CL-ncbq: the options book may hold the same idea — closing it here
    would corrupt that book's view and end the A/B early."""
    _seed(engine, "ab", submitted_at="2026-07-01")
    manage_equity_exits(engine, _FakeClient([_pos()]), now=NOW)
    with engine.connect() as c:
        assert (
            c.execute(text("SELECT status FROM trade_ideas WHERE idea_id='ab'")).scalar()
            == "pending"
        )
