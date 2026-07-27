"""Tests for the Alpaca options exit manager (CL-3rho).

Sqlite alpaca_option_orders (mig 014 + 016) + trade_ideas + geo_events;
fake Alpaca client with canned positions. Covers every exit rule in
priority order, the vanished-position finalizations, the unmatched-position
non-management, sell dedup recovery, and the market-hours gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.execution.alpaca_options_exit import (
    ExitReason,
    OptionsExitConfig,
    evaluate_exit,
    manage_option_exits,
    occ_expiry,
)

NOW = datetime(2026, 7, 22, 15, 0, tzinfo=UTC)
OCC = "RTX260821C00105000"  # expires 2026-08-21 → 30 DTE from NOW
OCC_NEAR = "RTX260724C00105000"  # expires 2026-07-24 → 2 DTE from NOW


class _FakeClient:
    def __init__(
        self, positions: list[dict[str, Any]] | None = None, market_open: bool = True
    ) -> None:
        self._positions = positions or []
        self._market_open = market_open
        self.orders: list[Any] = []
        self.cids: list[str | None] = []

    def is_market_open(self) -> bool:
        return self._market_open

    def list_option_positions(self) -> list[dict[str, Any]]:
        return self._positions

    def submit_option_order(
        self, occ: str, qty: int, side: str = "buy", client_order_id: str | None = None
    ) -> dict[str, Any]:
        self.orders.append((occ, qty, side))
        self.cids.append(client_order_id)
        return {"id": f"ord-{len(self.orders)}", "status": "accepted"}


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'a.db'}")
    with eng.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE trade_ideas (
                idea_id TEXT PRIMARY KEY, geo_event_id INTEGER, ticker TEXT,
                action TEXT, confidence FLOAT, time_stop_days INTEGER,
                notes TEXT, status TEXT DEFAULT 'pending',
                status_updated_at TEXT, created_at TEXT)
        """)
        )
        conn.execute(text("CREATE TABLE geo_events (id INTEGER PRIMARY KEY, status TEXT)"))
        for mig in ("014_alpaca_option_orders.sql", "016_alpaca_option_exits.sql"):
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
    occ=OCC,
    ticker="RTX",
    submitted_at="2026-07-20",
    time_stop_days=10,
    idea_status="pending",
    event_status="TRADED",
    exit_status=None,
    exit_reason=None,
):
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO geo_events (status) VALUES (:es)
        """),
            {"es": event_status},
        )
        gid = conn.execute(text("SELECT max(id) FROM geo_events")).scalar()
        conn.execute(
            text("""
            INSERT INTO trade_ideas (idea_id, geo_event_id, ticker, action,
                confidence, time_stop_days, status, status_updated_at,
                created_at)
            VALUES (:i,:g,:t,'buy_calls',0.6,:ts,:st,:sa,:sa)
        """),
            {
                "i": idea_id,
                "g": gid,
                "t": ticker,
                "ts": time_stop_days,
                "st": idea_status,
                "sa": submitted_at,
            },
        )
        conn.execute(
            text("""
            INSERT INTO alpaca_option_orders (idea_id, ticker, occ_symbol,
                opt_type, qty, premium_est, alpaca_order_id, status, detail,
                submitted_at, exit_status, exit_reason)
            VALUES (:i,:t,:o,'call',1,200.0,'ord-x','submitted',NULL,:sa,
                    :xs,:xr)
        """),
            {
                "i": idea_id,
                "t": ticker,
                "o": occ,
                "sa": submitted_at,
                "xs": exit_status,
                "xr": exit_reason,
            },
        )


def _pos(occ=OCC, qty="1", avg="2.0", cur="2.1", plpc=None):
    return {
        "symbol": occ,
        "asset_class": "us_option",
        "qty": qty,
        "avg_entry_price": avg,
        "current_price": cur,
        "unrealized_plpc": plpc,
    }


def _row(engine, idea_id):
    with engine.connect() as c:
        return dict(
            c.execute(text("SELECT * FROM alpaca_option_orders WHERE idea_id=:i"), {"i": idea_id})
            .one()
            ._mapping
        )


def _idea_status(engine, idea_id):
    with engine.connect() as c:
        return c.execute(
            text("SELECT status FROM trade_ideas WHERE idea_id=:i"), {"i": idea_id}
        ).scalar()


# --------------------------------------------------------------------------- #
# occ parsing
# --------------------------------------------------------------------------- #


def test_occ_expiry_parses():
    from datetime import date

    assert occ_expiry("RTX260821C00105000") == date(2026, 8, 21)
    assert occ_expiry("ULCC260821P00005000") == date(2026, 8, 21)
    assert occ_expiry("not-an-occ") is None
    assert occ_expiry(None) is None


# --------------------------------------------------------------------------- #
# exit rules (priority order)
# --------------------------------------------------------------------------- #


def test_profit_target_closes(engine):
    _seed(engine, "win")
    client = _FakeClient([_pos(avg="2.0", cur="4.0")])  # +100%
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert client.orders == [(OCC, 1, "sell")]
    row = _row(engine, "win")
    assert row["exit_status"] == "submitted"
    assert row["exit_reason"] == "profit_target"
    assert row["pnl_pct"] == pytest.approx(1.0)
    assert _idea_status(engine, "win") == "closed"


def test_stop_loss_closes(engine):
    _seed(engine, "loss")
    client = _FakeClient([_pos(avg="2.0", cur="1.0")])  # -50%
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "loss")["exit_reason"] == "stop_loss"


def test_time_stop_closes(engine):
    _seed(engine, "old", submitted_at="2026-07-01", time_stop_days=10)
    client = _FakeClient([_pos()])  # flat pnl — only the clock triggers
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "old")["exit_reason"] == "time_stop"


def test_thesis_invalidated_beats_profit(engine):
    # Event DISMISSED and the position is +100%: thesis wins the priority.
    _seed(engine, "dead", event_status="DISMISSED")
    client = _FakeClient([_pos(avg="2.0", cur="4.0")])
    manage_option_exits(engine, client, now=NOW)
    assert _row(engine, "dead")["exit_reason"] == "thesis_invalidated"


def test_event_expired_is_not_thesis_invalidation(engine):
    """geo_event EXPIRED is the ~2h intraday FX gate lapsing — routine for
    virtually every event — NOT an options-thesis verdict. A healthy
    position on an EXPIRED event must be HELD (its deadline is the idea's
    own time stop)."""
    _seed(engine, "gate", event_status="EXPIRED")  # 2d held, 30 DTE, +5%
    client = _FakeClient([_pos(avg="2.0", cur="2.1")])
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0
    assert client.orders == []


def test_expiry_protect_fires_without_quotes(engine):
    # 2 DTE and NO price data: the date-based rule still closes it.
    _seed(engine, "near", occ=OCC_NEAR)
    client = _FakeClient([_pos(occ=OCC_NEAR, avg="0", cur="0")])
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    row = _row(engine, "near")
    assert row["exit_reason"] == "expiry_protect"
    assert row["pnl_pct"] is None  # never fabricated
    assert row["exit_premium"] is None


def test_healthy_position_held(engine):
    _seed(engine, "hold", submitted_at="2026-07-20")  # 2d held, 30 DTE
    client = _FakeClient([_pos(avg="2.0", cur="2.1")])  # +5%
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0
    assert client.orders == []
    assert _row(engine, "hold")["exit_status"] is None


ENTRY_TODAY = "2026-07-22T13:31:00+00:00"  # 9:31 ET, same NY date as NOW


def test_entry_day_grace_holds_through_stop_level(engine):
    # -50% on the entry day is (mostly) opening spread — grace holds it.
    _seed(engine, "grace", submitted_at=ENTRY_TODAY)
    client = _FakeClient([_pos(avg="2.0", cur="1.0")])  # -50%
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0
    assert client.orders == []


def test_entry_day_extreme_valve_still_fires(engine):
    # -65% breaches the -60% valve on entry day, AFTER the 15-min settle
    # window (ENTRY_TODAY = 9:31 ET, ~89 min before NOW).
    _seed(engine, "valve", submitted_at=ENTRY_TODAY)
    client = _FakeClient([_pos(avg="2.0", cur="0.7")])  # -65%
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    row = _row(engine, "valve")
    assert row["exit_reason"] == "stop_loss"
    assert "entry day" in _detail_of_last_eval(engine, "valve")


def test_extreme_valve_suppressed_during_settle_window(engine):
    # CL-h02l: -65% just 5 min after entry is spread noise on a cheap
    # contract — the settle window must HOLD it, not sell on the valve.
    _seed(engine, "fresh", submitted_at="2026-07-22T14:55:00+00:00")  # 5min
    client = _FakeClient([_pos(avg="2.0", cur="0.7")])  # -65%
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0
    assert client.orders == []


def _detail_of_last_eval(engine, idea_id):
    # detail lives only in the log; assert via evaluate_exit directly.
    from src.execution.alpaca_options_exit import OptionsExitConfig, evaluate_exit

    row = _row(engine, idea_id)
    row["idea_status"], row["event_status"] = "pending", "TRADED"
    reason, detail = evaluate_exit(row, _pos(avg="2.0", cur="0.7"), OptionsExitConfig(), NOW)
    return detail


def test_day_two_normal_stop_applies(engine):
    # Same -50% one day later: normal stop fires.
    _seed(engine, "d2", submitted_at="2026-07-21T13:31:00+00:00")
    client = _FakeClient([_pos(avg="2.0", cur="1.0")])
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "d2")["exit_reason"] == "stop_loss"


def test_near_expiry_winner_left_to_run(engine):
    # 2 DTE but +30% >= 25% min profit: expiry protect stands aside.
    _seed(engine, "winner", occ=OCC_NEAR)
    client = _FakeClient([_pos(occ=OCC_NEAR, avg="2.0", cur="2.6")])  # +30%
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["held"] == 1 and counts["exit_submitted"] == 0


def test_final_day_closes_regardless_of_profit(engine):
    # 1 DTE: the backstop closes even a +30% winner — never ride expiry.
    occ_final = "RTX260723C00105000"  # expires tomorrow
    _seed(engine, "final", occ=occ_final)
    client = _FakeClient([_pos(occ=occ_final, avg="2.0", cur="2.6")])
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert _row(engine, "final")["exit_reason"] == "expiry_protect"


def test_evaluate_priority_time_stop_over_stop_loss():
    row = {
        "occ_symbol": OCC,
        "submitted_at": "2026-07-01",
        "time_stop_days": 10,
        "idea_status": "pending",
        "event_status": "TRADED",
    }
    reason, _ = evaluate_exit(row, _pos(avg="2.0", cur="1.0"), OptionsExitConfig(), NOW)
    assert reason == ExitReason.TIME_STOP  # not stop_loss


# --------------------------------------------------------------------------- #
# lifecycle / vanished positions
# --------------------------------------------------------------------------- #


def test_exit_pending_not_resubmitted(engine):
    _seed(engine, "pend", exit_status="submitted", exit_reason="stop_loss")
    client = _FakeClient([_pos()])  # sell not yet filled — still a position
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_pending"] == 1
    assert client.orders == []


def test_vanished_after_exit_submitted_confirms(engine):
    _seed(engine, "done", exit_status="submitted", exit_reason="stop_loss")
    counts = manage_option_exits(engine, _FakeClient([]), now=NOW)
    assert counts["closed_confirmed"] == 1
    row = _row(engine, "done")
    assert row["exit_status"] == "closed"
    assert row["exit_reason"] == "stop_loss"  # submit-time reason kept


def test_vanished_past_expiry_marked_worthless(engine):
    _seed(engine, "worthless", occ="RTX260717C00105000")  # expired 07-17
    counts = manage_option_exits(engine, _FakeClient([]), now=NOW)
    assert counts["expired_worthless"] == 1
    row = _row(engine, "worthless")
    assert row["exit_status"] == "closed"
    assert row["pnl_pct"] == pytest.approx(-1.0)
    assert _idea_status(engine, "worthless") == "closed"


def test_vanished_unexpired_marked_external(engine):
    _seed(engine, "gone")  # 30 DTE but Alpaca has no position
    counts = manage_option_exits(engine, _FakeClient([]), now=NOW)
    assert counts["closed_external"] == 1
    assert _row(engine, "gone")["exit_reason"] == "closed_external"


def test_unmatched_position_never_managed(engine):
    # Alpaca holds something we have no row for: flag, don't touch.
    client = _FakeClient([_pos(occ="MYST260821C00050000")])
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["unmatched"] == 1
    assert client.orders == []


# --------------------------------------------------------------------------- #
# dedup + gates
# --------------------------------------------------------------------------- #


def test_sell_uses_exit_client_order_id(engine):
    _seed(engine, "cid", submitted_at="2026-07-01")  # time stop
    client = _FakeClient([_pos()])
    manage_option_exits(engine, client, now=NOW)
    assert client.cids == ["curlit-exit-cid"]


def test_duplicate_exit_cid_recovers(engine):
    _seed(engine, "dup", submitted_at="2026-07-01")

    class _DupClient(_FakeClient):
        def submit_option_order(self, occ, qty, side="buy", client_order_id=None):
            raise RuntimeError("422 client_order_id must be unique: order already exists")

    counts = manage_option_exits(engine, _DupClient([_pos()]), now=NOW)
    assert counts["recovered"] == 1 and counts["error"] == 0
    row = _row(engine, "dup")
    assert row["exit_status"] == "submitted"
    assert row["exit_order_id"] == "recovered"


def test_market_closed_noop(engine):
    _seed(engine, "closed", submitted_at="2026-07-01")
    client = _FakeClient([_pos()], market_open=False)
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["market_closed"] == 1
    assert client.orders == []
    assert _row(engine, "closed")["exit_status"] is None


# --------------------------------------------------------------------- #
# No-bid / unsellable contracts (CL-hptt)
# --------------------------------------------------------------------- #


class _BidClient(_FakeClient):
    """FakeClient that also answers bid probes."""

    def __init__(self, positions, bid, **kw):  # type: ignore[no-untyped-def]
        super().__init__(positions, **kw)
        self._bid = bid
        self.bid_calls: list[str] = []

    def get_option_bid(self, occ: str):  # type: ignore[no-untyped-def]
        self.bid_calls.append(occ)
        return self._bid


def test_no_bid_contract_is_not_submitted(engine, monkeypatch):  # type: ignore[no-untyped-def]
    """CL-hptt: a deep-OTM contract can carry a live ask and NO bid. A market
    sell into an empty book is 403-rejected, and the old code just logged
    'error managing' and retried every 5-min cycle forever."""
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.execution.alpaca_options_exit.notify_operator",
        lambda t, m, *a, **k: sent.append((t, m)),
    )
    _seed(engine, "loss")
    client = _BidClient([_pos(avg="2.0", cur="1.0")], bid=0.0)  # -50%, no bid

    counts = manage_option_exits(engine, client, now=NOW)

    assert counts["unsellable"] == 1
    assert counts["exit_submitted"] == 0
    assert client.orders == []  # nothing submitted into an empty book
    assert _row(engine, "loss")["exit_status"] == "unsellable"
    assert len(sent) == 1 and "no bid" in sent[0][0].lower()


def test_unsellable_alerts_once_not_every_cycle(engine, monkeypatch):  # type: ignore[no-untyped-def]
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.execution.alpaca_options_exit.notify_operator",
        lambda t, m, *a, **k: sent.append((t, m)),
    )
    _seed(engine, "loss")
    client = _BidClient([_pos(avg="2.0", cur="1.0")], bid=0.0)

    for _ in range(3):  # three consecutive cycles
        manage_option_exits(engine, client, now=NOW)

    assert len(sent) == 1, "must not re-page the operator every cycle"
    assert client.orders == []


def test_unsellable_position_retries_when_the_bid_returns(engine, monkeypatch):  # type: ignore[no-untyped-def]
    """The row must STAY in the working set. Marking it unsellable and then
    excluding it from the query would silently ABANDON an open position."""
    monkeypatch.setattr("src.execution.alpaca_options_exit.notify_operator", lambda *a, **k: None)
    _seed(engine, "loss")
    pos = [_pos(avg="2.0", cur="1.0")]

    manage_option_exits(engine, _BidClient(pos, bid=0.0), now=NOW)
    assert _row(engine, "loss")["exit_status"] == "unsellable"

    # Bid comes back → the position is picked up again and actually sold.
    client = _BidClient(pos, bid=0.85)
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
    assert client.orders and client.orders[0][2] == "sell"
    assert _row(engine, "loss")["exit_status"] == "submitted"


def test_missing_bid_probe_fails_open(engine):  # type: ignore[no-untyped-def]
    # A client with no get_option_bid (or a probe that errors) must NOT block
    # the exit — exits always flow.
    _seed(engine, "loss")
    client = _FakeClient([_pos(avg="2.0", cur="1.0")])  # no get_option_bid
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1


def test_unavailable_quote_fails_open(engine):  # type: ignore[no-untyped-def]
    # bid=None means "quote unavailable", which is transient — submit anyway.
    _seed(engine, "loss")
    client = _BidClient([_pos(avg="2.0", cur="1.0")], bid=None)
    counts = manage_option_exits(engine, client, now=NOW)
    assert counts["exit_submitted"] == 1
