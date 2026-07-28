"""END-TO-END money-path integration tests (CL-0hr9).

Every stage of the two money paths already has unit coverage, but nothing
walked a single row the WHOLE way. These tests do — on ONE shared sqlite
engine per path, with fakes only (no network, no LLM, no DB server):

  A. FX event lifecycle — a NEW ``geo_events`` row is assessed by the real
     :class:`EventImpactAgent` (mock LLM), then picked up by the real
     :class:`EventDrivenStrategy` (real :class:`EventConfluence`) and driven
     NEW → ASSESSED → CONFIRMED → TRADED, emitting a real entry OrderIntent,
     demoting the cross-theme leg (CL-9nvq), alerting the operator, and
     booking the leg in the event book (submit → broker-confirmed fill).
     Negative path: the same flow with a below-bar confidence never confirms.

  B. Options loop — a niche/red-team ``buy_calls`` idea linked to a
     ``geo_events`` row is bought by ``execute_pending_options`` (entry mid +
     spread recorded), taken profit by ``manage_option_exits``, and finalized
     ``closed`` on the next cycle once the sell fills, so the row tells the
     whole story (entry premium → entry mid → exit reason → pnl_pct).
     Negative path: a wide-spread contract is never bought (CL-d44a).

Fixture/fake patterns are deliberately mirrored from the per-stage modules
(``test_event_impact_agent.py``, ``test_event_driven.py``,
``test_alpaca_options_executor.py``, ``test_alpaca_options_exit.py``) rather
than imported, so those files stay free to evolve independently.

Determinism: the options path uses fixed datetimes end to end. The FX path
must anchor ``seen_at`` to ``datetime.now(UTC)`` because the confluence
window is measured against the wall clock inside ``generate_intents`` (same
approach as ``test_event_driven.insert_event``); the OFFSET is fixed, there
are no sleeps, and the stale-fact ceiling is switched off so the shipped
playbook's ``last_reviewed`` date cannot age the test into failure.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.events.impact_agent import EventImpactAgent
from src.execution.alpaca_options_executor import (
    OptionsExecConfig,
    execute_pending_options,
    fetch_executable_ideas,
)
from src.execution.alpaca_options_exit import manage_option_exits
from src.strategies.event_driven import EventDrivenConfig, EventDrivenStrategy

# --------------------------------------------------------------------------- #
# shared sqlite plumbing
# --------------------------------------------------------------------------- #


def _shim(sql: str) -> str:
    """Postgres DDL → sqlite (pattern from test_event_impact_agent.py /
    test_alpaca_options_executor.py)."""
    return (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("JSONB", "TEXT")
        .replace("BIGSERIAL", "INTEGER")
        .replace("NUMERIC", "FLOAT")
        .replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
    )


def _apply_migration(engine: Engine, filename: str) -> None:
    from migrations.run import _strip_sql_comments

    sql = _shim(_strip_sql_comments(Path("migrations", filename).read_text()))
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))


@pytest.fixture(autouse=True)
def _triage_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fast triage tier (CL-cunh) is opt-in; the operator's .env may have
    leaked EVENT_TRIAGE_ENABLED into os.environ earlier in the run."""
    monkeypatch.delenv("EVENT_TRIAGE_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def sent_alerts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, int]]:
    """Capture notify_operator; also guarantees no network attempt. Alert
    dispatch lives in the events layer (CL-ikz2), so the patch target is the
    notifier module (pattern from test_event_driven.py)."""
    calls: list[tuple[str, str, int]] = []

    def _capture(title: str, message: str, priority: int = 0) -> None:
        calls.append((title, message, priority))

    monkeypatch.setattr("src.events.event_notifier.notify_operator", _capture)
    return calls


# =========================================================================== #
# A. FX event lifecycle: NEW -> ASSESSED -> CONFIRMED -> TRADED
# =========================================================================== #

#: In-theme (energy_chokepoint lists BCO_USD) vs cross-theme (USD_JPY is on
#: the impact agent's cross-theme whitelist but NOT in this playbook).
IN_THEME = "BCO_USD"
CROSS_THEME = "USD_JPY"

#: Intraday reference prices at seen_at; the live tick below is +1.01% on
#: both legs, well over the quarter-sigma bar (0.25 x 1% daily vol).
REFS = {IN_THEME: 79.20, CROSS_THEME: 148.50}
TICKS = {
    IN_THEME: {"bid": 79.99, "ask": 80.01},
    CROSS_THEME: {"bid": 149.99, "ask": 150.01},
}
EQUITY = 100_000.0
#: Inside the 30-120min confirmation window.
SEEN_MINUTES_AGO = 60.0


class _MockLLMClient:
    """LLMClient stand-in returning canned assessment text (pattern from
    test_event_impact_agent.MockLLMClient)."""

    def __init__(self, text_out: str) -> None:
        self.text_out = text_out
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": messages, "model": model})
        return SimpleNamespace(
            text=self.text_out,
            model=model,
            provider="mock",
            input_tokens=10,
            output_tokens=10,
            usd_cost=0.0,
            elapsed_sec=0.01,
        )


class _MultiSymbolProvider:
    """DataProvider stand-in (pattern from test_event_driven.FakeProvider,
    extended to a PER-SYMBOL intraday reference so the two legs carry
    realistic price levels). get_realized_vol returns the annualized value
    the real provider returns; Gate B de-annualizes it."""

    def __init__(self, refs: dict[str, float], daily_vol: float = 0.01) -> None:
        self.refs = refs
        self.daily_vol = daily_vol

    def get_intraday_value(
        self,
        symbol: str,
        as_of: datetime,
        max_staleness_minutes: int = 15,
    ) -> float | None:
        return self.refs.get(symbol)

    def get_latest_value(self, series_id: str, as_of: datetime) -> float | None:
        return self.refs.get(series_id)

    def get_realized_vol(
        self,
        pair: str,
        window: int = 20,
        as_of: datetime | None = None,
    ) -> float | None:
        return self.daily_vol * math.sqrt(252.0)


class _FakeBroker:
    """Equity for sizing plus a MUTABLE position book, so a submitted entry
    can be confirmed as filled on a later tick (patterns from
    test_event_driven.FakeBroker / _BrokerWithPositions)."""

    def __init__(self, equity: float = EQUITY) -> None:
        self.equity = equity
        self.positions: dict[str, float] = {}

    def get_account(self) -> Any:
        return SimpleNamespace(balance=self.equity, equity=self.equity, margin_used=0.0)

    def get_positions(self) -> list[Any]:
        return [
            SimpleNamespace(symbol=sym, quantity=qty, avg_price=1.0)
            for sym, qty in self.positions.items()
        ]


def _assessment_payload(confidence: float = 0.88) -> dict[str, Any]:
    """A high-urgency Hormuz assessment naming one IN-theme tradable, one
    CROSS-theme tradable, and one equity watch."""
    return {
        "core_event": "Iran announces closure of the Strait of Hormuz",
        "direction": "bullish",
        "urgency": 9,
        "horizon": "hours",
        "confidence": confidence,
        "affected": [
            {
                "instrument": IN_THEME,
                "kind": "oanda",
                "direction": "long",
                "reason": "supply risk premium on Brent",
            },
            {
                "instrument": CROSS_THEME,
                "kind": "fx",
                "direction": "long",
                "reason": "dollar bid on the energy shock (cross-theme)",
            },
            {
                "instrument": "FRO",
                "kind": "equity_watch",
                "direction": "watch",
                "reason": "tanker rates",
            },
        ],
        "rationale": "A chokepoint closure is the canonical oil supply shock.",
    }


@pytest.fixture
def geo_engine(tmp_path: Path) -> Engine:
    """sqlite geo_events built from the REAL producer migration 005."""
    engine = create_engine(f"sqlite:///{tmp_path / 'geo.db'}")
    _apply_migration(engine, "005_geo_events.sql")
    return engine


def _insert_new_event(
    engine: Engine,
    *,
    external_id: str = "gdelt:hormuz-1",
    headline: str = "Iran moves to close the Strait of Hormuz, tankers turn back",
    theme: str = "energy_chokepoint",
    minutes_ago: float = SEEN_MINUTES_AGO,
) -> int:
    seen = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO geo_events (seen_at, source, external_id, headline, "
                "url, theme, status, status_updated_at) "
                "VALUES (:seen, 'gdelt', :eid, :hl, 'https://news.test/hormuz', "
                ":theme, 'NEW', :seen)"
            ),
            {"seen": seen, "eid": external_id, "hl": headline, "theme": theme},
        )
        return int(conn.execute(text("SELECT max(id) FROM geo_events")).scalar_one())


def _row(engine: Engine, event_id: int) -> dict[str, Any]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status, assessment FROM geo_events WHERE id = :i"),
            {"i": event_id},
        ).one()
    return {
        "status": row[0],
        "assessment": json.loads(row[1]) if row[1] else None,
    }


def _make_strategy(tmp_path: Path, engine: Engine) -> EventDrivenStrategy:
    config = EventDrivenConfig(
        event_book_state_path=str(tmp_path / "event_book_state.json"),
        # CL-ylak has its own dedicated tests; disable the ceiling here so
        # this e2e cannot start failing the day the shipped playbook's
        # last_reviewed date drifts past 90 days.
        stale_review_days=0,
    )
    return EventDrivenStrategy(
        config,
        data_provider=_MultiSymbolProvider(REFS),
        db_engine=engine,
    )


def _record_transitions(strategy: EventDrivenStrategy) -> list[tuple[Any, str, str, bool]]:
    """Tap the guarded status transitions so the INTERMEDIATE CONFIRMED step
    is observable — the DB only ever shows the final status."""
    seen: list[tuple[Any, str, str, bool]] = []
    original = strategy.confluence.transition

    def _tap(event_id: Any, old: str, new: str) -> bool:
        ok = original(event_id, old, new)
        seen.append((event_id, old, new, ok))
        return ok

    strategy.confluence.transition = _tap  # type: ignore[method-assign]
    return seen


def _record_confluence_results(strategy: EventDrivenStrategy) -> list[Any]:
    """Tap the per-event ConfluenceResult so the test can see WHICH legs
    individually passed Gate B (the strategy only exposes the outcome)."""
    seen: list[Any] = []
    original = strategy.confluence.evaluate_and_transition

    def _tap(event: dict[str, Any], **kwargs: Any) -> Any:
        result = original(event, **kwargs)
        seen.append(result)
        return result

    strategy.confluence.evaluate_and_transition = _tap  # type: ignore[method-assign]
    return seen


def _generate(strategy: EventDrivenStrategy, broker: _FakeBroker) -> list[Any]:
    """One engine tick against the live tick prices (the engine drives
    generate_intents from its asyncio loop; tests use asyncio.run, as in
    test_event_driven.run)."""
    return asyncio.run(strategy.generate_intents(TICKS, broker))


class TestFxEventLifecycleE2E:
    """One geo_event row, one sqlite engine, the whole FX money path."""

    def test_new_event_becomes_a_traded_position(
        self,
        tmp_path: Path,
        geo_engine: Engine,
        sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        # -- 1. the producer ingests a NEW row -------------------------- #
        event_id = _insert_new_event(geo_engine)
        assert _row(geo_engine, event_id)["status"] == "NEW"

        # -- 2. the impact agent assesses it (mock LLM, real DB write) --- #
        client = _MockLLMClient(json.dumps(_assessment_payload()))
        results = EventImpactAgent(geo_engine, client=client).assess_new_events()  # type: ignore[arg-type]

        assert [r.status for r in results] == ["ASSESSED"]
        stored = _row(geo_engine, event_id)
        assert stored["status"] == "ASSESSED"
        assessment = stored["assessment"]
        assert assessment["urgency"] == 9
        assert assessment["confidence"] == pytest.approx(0.88)
        # Both tradables survived normalisation (USD_JPY is vetted by the
        # cross-theme whitelist), the equity is watch-only.
        by_instrument = {a["instrument"]: a for a in assessment["affected"]}
        assert by_instrument[IN_THEME]["direction"] == "long"
        assert by_instrument[CROSS_THEME]["direction"] == "long"
        assert by_instrument["FRO"]["direction"] == "watch"
        # The prompt carried this event's playbook context.
        assert "energy_chokepoint" in client.calls[0]["messages"][1].content

        # -- 3. the strategy confirms + trades it ------------------------ #
        strategy = _make_strategy(tmp_path, geo_engine)
        transitions = _record_transitions(strategy)
        gate_results = _record_confluence_results(strategy)
        broker = _FakeBroker()
        intents = _generate(strategy, broker)

        # ASSESSED -> CONFIRMED -> TRADED, both guarded writes won.
        assert transitions == [
            (event_id, "ASSESSED", "CONFIRMED", True),
            (event_id, "CONFIRMED", "TRADED", True),
        ]
        assert _row(geo_engine, event_id)["status"] == "TRADED"

        # Exactly ONE entry leg: the in-theme instrument.
        assert len(intents) == 1
        intent = intents[0]
        assert intent.symbol == IN_THEME
        assert intent.strategy_id == strategy.id
        assert intent.urgency == "urgent"
        # equity * event_risk_pct / stop_distance, long at the ask.
        entry_price = TICKS[IN_THEME]["ask"]
        expected = EQUITY * 0.005 / (entry_price * 0.01)
        assert intent.target_position == pytest.approx(expected)
        assert intent.target_position > 0

        # The cross-theme leg INDIVIDUALLY passed Gate B and there was a free
        # concurrency slot (default max is 2 legs), so theme-primary scoping
        # (CL-9nvq) is the ONLY thing keeping it out of the book.
        assert len(gate_results) == 1
        assert {c.instrument for c in gate_results[0].checks if c.confirmed} == {
            IN_THEME,
            CROSS_THEME,
        }
        assert CROSS_THEME not in {i.symbol for i in intents}

        # -- the operator was alerted, with the demotion spelled out ----- #
        confirmed = [c for c in sent_alerts if c[0] == "Event confirmed"]
        assert len(confirmed) == 1
        _title, message, priority = confirmed[0]
        assert priority == 1
        assert "Iran moves to close the Strait of Hormuz" in message
        assert f"Trade: {IN_THEME} long" in message
        assert f"Skipped: {CROSS_THEME} (cross_theme)" in message
        assert "Watch: FRO" in message
        assert "Urgency: 9/10" in message

        # -- 4. the event book recorded the entry ------------------------ #
        booked = strategy.open_positions
        assert set(booked) == {IN_THEME}
        leg = booked[IN_THEME]
        assert leg.event_id == event_id
        assert leg.entry_price == pytest.approx(entry_price)
        assert leg.stop_price == pytest.approx(entry_price * 0.99)
        assert "Hormuz" in leg.headline
        # Submitted, not yet filled: the book carries the INTENDED size in
        # pending_entries and books nothing reconciler-facing (CL-hqyj).
        pending = strategy.book.pending_entries[IN_THEME]
        assert pending.position.quantity == pytest.approx(expected)
        assert leg.quantity == 0.0
        state = json.loads((tmp_path / "event_book_state.json").read_text())
        assert state["closed_trades"] == 0

        # Next tick: the broker shows the fill -> the leg is promoted at the
        # OBSERVED size and no new intent is emitted (the row is TRADED).
        broker.positions[IN_THEME] = expected
        assert _generate(strategy, broker) == []
        assert strategy.book.pending_entries == {}
        assert strategy.book.open_positions[IN_THEME].quantity == pytest.approx(expected)
        assert _row(geo_engine, event_id)["status"] == "TRADED"
        # Still one alert — a traded row is never re-confirmed or re-alerted.
        assert len([c for c in sent_alerts if c[0] == "Event confirmed"]) == 1

    def test_low_confidence_assessment_never_confirms(
        self,
        tmp_path: Path,
        geo_engine: Engine,
        sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        """Negative path: the SAME event, market moving the SAME way, but the
        assessment's confidence is below min_confidence — Gate A holds, so the
        row rides out the window as ASSESSED and nothing is traded."""
        event_id = _insert_new_event(geo_engine, external_id="gdelt:hormuz-hedged")
        client = _MockLLMClient(json.dumps(_assessment_payload(confidence=0.60)))
        results = EventImpactAgent(geo_engine, client=client).assess_new_events()  # type: ignore[arg-type]

        # A hedged assessment is still a VALID one: the row is ASSESSED.
        assert [r.status for r in results] == ["ASSESSED"]
        assert _row(geo_engine, event_id)["assessment"]["confidence"] == pytest.approx(0.60)

        strategy = _make_strategy(tmp_path, geo_engine)
        transitions = _record_transitions(strategy)
        intents = _generate(strategy, _FakeBroker())

        assert intents == []
        assert transitions == []  # no status write at all
        assert _row(geo_engine, event_id)["status"] == "ASSESSED"
        assert strategy.open_positions == {}
        assert [c for c in sent_alerts if c[0] == "Event confirmed"] == []


# =========================================================================== #
# B. Options loop: pending idea -> bought -> profit exit -> closed
# =========================================================================== #

ENTRY_NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)  # 12:00 ET — no open delay
EXIT_NOW = datetime(2026, 7, 22, 15, 0, tzinfo=UTC)  # next session, 11:00 ET
FILL_NOW = datetime(2026, 7, 22, 15, 5, tzinfo=UTC)  # the cycle after the sell
OCC = "FRO260821C00030000"  # FRO 30 calls, expiring 2026-08-21
CONTRACT = {"symbol": OCC, "strike_price": "30", "expiration_date": "2026-08-21"}
IDEA_ID = "idea-hormuz-fro"


def _no_technicals(_ticker: str) -> None:
    """Unit tests never touch live yfinance — the gate fails open."""
    return None


def _underlying_price(_ticker: str) -> float:
    return 28.0


class _FakeAlpaca:
    """One options client for BOTH halves of the loop, with a mutable quote
    and position book so a single instance can play the whole session
    (patterns from test_alpaca_options_executor._FakeClient/_QuoteClient and
    test_alpaca_options_exit._FakeClient)."""

    def __init__(
        self,
        quote: tuple[float | None, float | None],
        contract: dict[str, Any] | None = None,
        market_open: bool = True,
    ) -> None:
        self.quote = quote
        self.contract = contract if contract is not None else CONTRACT
        self.market_open = market_open
        self.positions: list[dict[str, Any]] = []
        self.orders: list[tuple[str, int, str]] = []
        self.client_order_ids: list[str | None] = []

    # -- market data --------------------------------------------------- #
    def is_market_open(self) -> bool:
        return self.market_open

    def find_contracts(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [self.contract] if self.contract else []

    def get_option_quote(self, occ: str) -> tuple[float | None, float | None]:
        return self.quote

    def get_option_ask(self, occ: str) -> float | None:
        return self.quote[1]

    # -- account ------------------------------------------------------- #
    def list_option_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)

    def submit_option_order(
        self,
        occ: str,
        qty: int,
        side: str = "buy",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        self.orders.append((occ, qty, side))
        self.client_order_ids.append(client_order_id)
        return {"id": f"ord-{len(self.orders)}", "status": "accepted"}


def _option_position(*, avg: str, cur: str, qty: str = "1") -> dict[str, Any]:
    return {
        "symbol": OCC,
        "asset_class": "us_option",
        "qty": qty,
        "avg_entry_price": avg,
        "current_price": cur,
        "unrealized_plpc": None,
    }


@pytest.fixture
def options_engine(tmp_path: Path) -> Engine:
    """sqlite with the REAL option-order migrations (014 entry + 016 exit +
    018 entry mid), a geo_events table from migration 005, and the
    trade_ideas columns both halves of the loop read."""
    engine = create_engine(f"sqlite:///{tmp_path / 'options.db'}")
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE trade_ideas (
                idea_id TEXT PRIMARY KEY, geo_event_id INTEGER, ticker TEXT,
                action TEXT, confidence FLOAT, preferred_instrument TEXT,
                time_stop_days INTEGER, notes TEXT,
                status TEXT DEFAULT 'pending', status_updated_at TEXT,
                created_at TEXT)
        """)
        )
    _apply_migration(engine, "005_geo_events.sql")
    for migration in (
        "014_alpaca_option_orders.sql",
        "016_alpaca_option_exits.sql",
        "018_option_entry_mid.sql",
    ):
        _apply_migration(engine, migration)
    return engine


def _seed_event_and_idea(
    engine: Engine,
    *,
    idea_id: str = IDEA_ID,
    ticker: str = "FRO",
    confidence: float = 0.66,
    time_stop_days: int = 21,
) -> int:
    """A TRADED geo_event plus the pending niche/red-team buy_calls idea it
    produced — the shape the executor's policy filter demands."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO geo_events (seen_at, source, external_id, headline, "
                "url, theme, status, status_updated_at) "
                "VALUES ('2026-07-21T13:00:00+00:00', 'gdelt', :eid, "
                "'Iran moves to close the Strait of Hormuz', '', "
                "'energy_chokepoint', 'TRADED', '2026-07-21T14:00:00+00:00')"
            ),
            {"eid": f"gdelt:{idea_id}"},
        )
        event_id = int(conn.execute(text("SELECT max(id) FROM geo_events")).scalar_one())
        conn.execute(
            text("""
            INSERT INTO trade_ideas (idea_id, geo_event_id, ticker, action,
                confidence, preferred_instrument, time_stop_days, notes,
                status, status_updated_at, created_at)
            VALUES (:i, :g, :t, 'buy_calls', :c, '~5% OTM, 4 weeks', :ts, :n,
                'pending', '2026-07-21', '2026-07-21')
        """),
            {
                "i": idea_id,
                "g": event_id,
                "t": ticker,
                "c": confidence,
                "ts": time_stop_days,
                # The executor's policy filter demands both markers.
                "n": (
                    "[niche 3hop asym0.7] tanker day-rates torque | "
                    "survived red-team; top risk: swift reopening"
                ),
            },
        )
    return event_id


def _order_row(engine: Engine, idea_id: str = IDEA_ID) -> dict[str, Any]:
    with engine.connect() as conn:
        return dict(
            conn.execute(
                text("SELECT * FROM alpaca_option_orders WHERE idea_id = :i"),
                {"i": idea_id},
            )
            .one()
            ._mapping
        )


def _idea_status(engine: Engine, idea_id: str = IDEA_ID) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT status FROM trade_ideas WHERE idea_id = :i"),
            {"i": idea_id},
        ).scalar()


class TestOptionsLoopE2E:
    """One idea, one sqlite engine, one Alpaca stand-in: buy → manage → close."""

    def test_idea_is_bought_taken_profit_and_finalized(
        self,
        options_engine: Engine,
    ) -> None:
        event_id = _seed_event_and_idea(options_engine)
        client = _FakeAlpaca(quote=(1.90, 2.00))  # 5% spread — buyable

        # -- entry ------------------------------------------------------- #
        counts = execute_pending_options(
            options_engine,
            client,  # type: ignore[arg-type]
            _underlying_price,
            now=ENTRY_NOW,
            technicals_fn=_no_technicals,
        )
        assert counts["submitted"] == 1
        assert counts["skipped_spread"] == 0
        assert client.orders == [(OCC, 1, "buy")]
        assert client.client_order_ids == [f"curlit-{IDEA_ID}"]

        row = _order_row(options_engine)
        assert row["status"] == "submitted"
        assert row["occ_symbol"] == OCC
        assert row["opt_type"] == "call"
        assert row["qty"] == 1
        assert row["premium_est"] == pytest.approx(200.0)  # ask 2.00 x 100
        assert row["entry_mid"] == pytest.approx(1.95)
        assert row["entry_spread_pct"] == pytest.approx(0.05)
        assert row["exit_status"] is None
        # The idea is now spoken for — a second cycle must not re-buy it.
        assert fetch_executable_ideas(options_engine, OptionsExecConfig()) == []

        # -- the position exists and the thesis works -------------------- #
        client.positions = [_option_position(avg="2.00", cur="3.70")]  # +85%
        client.quote = (3.65, 3.75)  # mid 3.70, tight

        counts = manage_option_exits(options_engine, client, now=EXIT_NOW)  # type: ignore[arg-type]
        assert counts["exit_submitted"] == 1
        assert counts["held"] == 0
        assert counts["unmatched"] == 0
        assert client.orders[-1] == (OCC, 1, "sell")
        assert client.client_order_ids[-1] == f"curlit-exit-{IDEA_ID}"

        row = _order_row(options_engine)
        assert row["exit_status"] == "submitted"
        assert row["exit_reason"] == "profit_target"
        assert row["pnl_pct"] == pytest.approx(0.85)
        assert row["exit_premium"] == pytest.approx(370.0)
        # Mirrored onto the idea as soon as the sell is placed.
        assert _idea_status(options_engine) == "closed"

        # -- the sell fills: the position vanishes from Alpaca ----------- #
        client.positions = []
        counts = manage_option_exits(options_engine, client, now=FILL_NOW)  # type: ignore[arg-type]
        assert counts["closed_confirmed"] == 1
        assert counts["exit_submitted"] == 0
        assert len(client.orders) == 2  # no double-sell

        # -- the scorecard: the row tells the whole story ---------------- #
        row = _order_row(options_engine)
        assert row["idea_id"] == IDEA_ID
        assert row["ticker"] == "FRO"
        assert row["premium_est"] == pytest.approx(200.0)  # what we paid
        assert row["entry_mid"] == pytest.approx(1.95)  # honest entry mark
        assert row["entry_spread_pct"] == pytest.approx(0.05)
        assert row["exit_status"] == "closed"
        assert row["exit_reason"] == "profit_target"  # submit-time reason kept
        assert row["exit_premium"] == pytest.approx(370.0)
        assert row["pnl_pct"] == pytest.approx(0.85)
        assert row["pnl_pct"] > 0
        assert row["exited_at"] is not None
        assert _idea_status(options_engine) == "closed"
        # The originating event row is untouched by the options loop.
        with options_engine.connect() as conn:
            assert (
                conn.execute(
                    text("SELECT status FROM geo_events WHERE id = :i"),
                    {"i": event_id},
                ).scalar()
                == "TRADED"
            )

        # A further cycle is a no-op: the finalized row leaves the working
        # set and Alpaca holds nothing we cannot explain.
        counts = manage_option_exits(options_engine, client, now=FILL_NOW)  # type: ignore[arg-type]
        assert counts["exit_submitted"] == 0
        assert counts["closed_confirmed"] == 0
        assert counts["unmatched"] == 0
        assert counts["error"] == 0
        assert len(client.orders) == 2

    def test_wide_spread_contract_is_never_bought(
        self,
        options_engine: Engine,
    ) -> None:
        """Negative path (CL-d44a): 0.13/0.42 is a 69% spread — buying the ask
        and marking the bid is an instant -69%. Nothing is bought, NOTHING is
        recorded (the skip is transient), and the exit loop has no phantom."""
        _seed_event_and_idea(options_engine)
        client = _FakeAlpaca(quote=(0.13, 0.42))

        counts = execute_pending_options(
            options_engine,
            client,  # type: ignore[arg-type]
            _underlying_price,
            now=ENTRY_NOW,
            technicals_fn=_no_technicals,
        )
        assert counts["skipped_spread"] == 1
        assert counts["submitted"] == 0
        assert client.orders == []
        with options_engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM alpaca_option_orders")).scalar_one() == 0
        # Transient: the idea is still pending and retries next cycle.
        assert _idea_status(options_engine) == "pending"
        assert {
            i["idea_id"] for i in fetch_executable_ideas(options_engine, OptionsExecConfig())
        } == {IDEA_ID}

        # And the exit manager has nothing to manage — no row, no position.
        exit_counts = manage_option_exits(options_engine, client, now=EXIT_NOW)  # type: ignore[arg-type]
        assert exit_counts["exit_submitted"] == 0
        assert exit_counts["unmatched"] == 0
        assert exit_counts["error"] == 0

        # When the market tightens on a later cycle, the SAME idea buys.
        client.quote = (1.90, 2.00)
        counts = execute_pending_options(
            options_engine,
            client,  # type: ignore[arg-type]
            _underlying_price,
            now=ENTRY_NOW,
            technicals_fn=_no_technicals,
        )
        assert counts["submitted"] == 1
        assert _order_row(options_engine)["entry_mid"] == pytest.approx(1.95)
