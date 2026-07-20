"""Unit tests — event confluence + EventDrivenStrategy (CL-mnhw).

No live DB / network: geo_events rows live in an in-memory sqlite
fixture (the REAL table is owned by the producer's migration 005 — the
fixture only mirrors the shared schema), prices/vol come from a fake
provider, and the broker is a stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from src.execution.oms import OrderIntent
from src.strategies.event_driven import (
    EventDrivenConfig,
    EventDrivenStrategy,
    EventPosition,
)

# =============================================================================
# Fixtures / helpers
# =============================================================================


def make_db() -> Any:
    """In-memory sqlite engine with a geo_events table mirroring the
    producer's 005_geo_events.sql schema (timestamps as ISO TEXT)."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE geo_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seen_at TEXT NOT NULL,
                source TEXT NOT NULL,
                external_id TEXT UNIQUE NOT NULL,
                headline TEXT NOT NULL,
                url TEXT,
                theme TEXT,
                assessment TEXT,
                status TEXT NOT NULL DEFAULT 'NEW',
                status_updated_at TEXT NOT NULL
            )
        """))
    return engine


_EVENT_SEQ = {"n": 0}


def insert_event(
    db: Any,
    *,
    minutes_ago: float = 60,
    urgency: int = 8,
    confidence: float = 0.9,
    affected: list[dict[str, Any]] | None = None,
    status: str = "ASSESSED",
    headline: str = "Explosion reported at major Norwegian gas terminal",
    direction: str = "bullish",
) -> int:
    _EVENT_SEQ["n"] += 1
    assessment = {
        "core_event": "gas supply disruption",
        "direction": direction,
        "urgency": urgency,
        "horizon": "hours",
        "confidence": confidence,
        "affected": affected if affected is not None else [
            {"instrument": "USD_CAD", "kind": "oanda",
             "direction": "long", "reason": "test"},
        ],
        "rationale": "test fixture",
    }
    seen = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    with db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO geo_events (seen_at, source, external_id, headline, "
                "url, theme, assessment, status, status_updated_at) "
                "VALUES (:seen, 'gdelt', :ext, :headline, '', 'energy', "
                ":assessment, :status, :seen)"
            ),
            {
                "seen": seen,
                "ext": f"test-{_EVENT_SEQ['n']}",
                "headline": headline,
                "assessment": json.dumps(assessment),
                "status": status,
            },
        )
        event_id = conn.execute(text("SELECT max(id) FROM geo_events")).scalar()
    return int(event_id)


def get_status(db: Any, event_id: int) -> str:
    with db.connect() as conn:
        return conn.execute(
            text("SELECT status FROM geo_events WHERE id = :id"), {"id": event_id},
        ).scalar()


class FakeProvider:
    """DataProvider stand-in.

    get_latest_value always returns ``p0`` (the reference price at
    seen_at) — tests supply the CURRENT price via the live ``prices``
    tick dict, so the two legs stay unambiguous. get_realized_vol
    returns ``daily_vol`` re-annualized, matching the real provider's
    annualized contract.
    """

    def __init__(self, p0: float = 1.0, daily_vol: float = 0.01) -> None:
        self.p0 = p0
        self.daily_vol = daily_vol

    def get_latest_value(self, series_id: str, as_of: datetime) -> float | None:
        return self.p0

    def get_realized_vol(
        self, pair: str, window: int = 20, as_of: datetime | None = None,
    ) -> float | None:
        return self.daily_vol * math.sqrt(252.0)


class FakeBroker:
    def __init__(self, equity: float = 100_000.0) -> None:
        self._equity = equity

    def get_account(self) -> Any:
        return SimpleNamespace(
            balance=self._equity, equity=self._equity, margin_used=0.0,
        )


def make_strategy(
    tmp_path: Any,
    db: Any = None,
    provider: Any = None,
    **overrides: Any,
) -> EventDrivenStrategy:
    overrides.setdefault("event_book_state_path", str(tmp_path / "event_book_state.json"))
    cfg = EventDrivenConfig(**overrides)
    return EventDrivenStrategy(cfg, data_provider=provider, db_engine=db)


def run(strategy: EventDrivenStrategy, prices: dict[str, Any] | None = None,
        broker: Any = None) -> list[OrderIntent]:
    return asyncio.run(
        strategy.generate_intents(prices or {}, broker or FakeBroker()),
    )


def tick(price: float) -> dict[str, float]:
    return {"bid": price, "ask": price}


@pytest.fixture(autouse=True)
def sent_alerts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, int]]:
    """Capture notify_operator calls; also guarantees no network attempt."""
    calls: list[tuple[str, str, int]] = []

    def _capture(title: str, message: str, priority: int = 0) -> None:
        calls.append((title, message, priority))

    monkeypatch.setattr("src.strategies.event_driven.notify_operator", _capture)
    return calls


# A confirming setup: p0=0.99, current tick 1.0 → move +1.01% vs a
# quarter-sigma threshold of 0.25% (daily vol 1%).
CONFIRM_PRICES = {"USD_CAD": tick(1.0)}


def confirming_provider() -> FakeProvider:
    return FakeProvider(p0=0.99, daily_vol=0.01)


# =============================================================================
# Gate A — quality thresholds
# =============================================================================


class TestQualityGate:
    def test_low_urgency_blocks_confirmation(self, tmp_path: Any) -> None:
        db = make_db()
        eid = insert_event(db, urgency=6, confidence=0.9)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert run(strat, CONFIRM_PRICES) == []
        # Quality failures ride out the window (→ EXPIRED later), they
        # are not confirmed and not dismissed early.
        assert get_status(db, eid) == "ASSESSED"

    def test_low_confidence_blocks_confirmation(self, tmp_path: Any) -> None:
        db = make_db()
        eid = insert_event(db, urgency=9, confidence=0.70)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert run(strat, CONFIRM_PRICES) == []
        assert get_status(db, eid) == "ASSESSED"

    def test_thresholds_are_inclusive(self, tmp_path: Any) -> None:
        db = make_db()
        eid = insert_event(db, urgency=7, confidence=0.75)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        intents = run(strat, CONFIRM_PRICES)
        assert len(intents) == 1
        assert get_status(db, eid) == "TRADED"


# =============================================================================
# Gate B — direction confirmation math
# =============================================================================


class TestDirectionConfirmation:
    def test_long_needs_positive_move(self, tmp_path: Any) -> None:
        db = make_db()
        eid = insert_event(db)  # long USD_CAD
        # Price FELL since seen_at: p0=1.0 → 0.99.
        strat = make_strategy(tmp_path, db=db, provider=FakeProvider(p0=1.0, daily_vol=0.01))
        assert run(strat, {"USD_CAD": tick(0.99)}) == []
        assert get_status(db, eid) == "ASSESSED"

    def test_short_needs_negative_move(self, tmp_path: Any) -> None:
        db = make_db()
        affected = [{"instrument": "USD_CAD", "kind": "fx",
                     "direction": "short", "reason": "test"}]
        eid = insert_event(db, affected=affected, direction="bearish")
        strat = make_strategy(tmp_path, db=db, provider=FakeProvider(p0=1.0, daily_vol=0.01))
        intents = run(strat, {"USD_CAD": tick(0.99)})
        assert len(intents) == 1
        assert intents[0].target_position < 0
        assert get_status(db, eid) == "TRADED"

    def test_short_rejects_positive_move(self, tmp_path: Any) -> None:
        db = make_db()
        affected = [{"instrument": "USD_CAD", "kind": "fx",
                     "direction": "short", "reason": "test"}]
        eid = insert_event(db, affected=affected, direction="bearish")
        strat = make_strategy(tmp_path, db=db, provider=FakeProvider(p0=0.99, daily_vol=0.01))
        assert run(strat, {"USD_CAD": tick(1.0)}) == []
        assert get_status(db, eid) == "ASSESSED"

    def test_quarter_sigma_scaling(self, tmp_path: Any) -> None:
        """Threshold = confirm_move_frac x daily vol: with 0.8% daily vol
        the bar is 0.2%; +0.15% must fail and +0.25% must pass."""
        provider = FakeProvider(p0=1.0, daily_vol=0.008)

        db_small = make_db()
        eid_small = insert_event(db_small)
        strat = make_strategy(tmp_path, db=db_small, provider=provider)
        assert run(strat, {"USD_CAD": tick(1.0015)}) == []
        assert get_status(db_small, eid_small) == "ASSESSED"

        db_big = make_db()
        eid_big = insert_event(db_big)
        strat2 = make_strategy(tmp_path, db=db_big, provider=provider)
        intents = run(strat2, {"USD_CAD": tick(1.0025)})
        assert len(intents) == 1
        assert get_status(db_big, eid_big) == "TRADED"


# =============================================================================
# Window semantics
# =============================================================================


class TestWindow:
    def test_before_window_stays_pending(self, tmp_path: Any) -> None:
        db = make_db()
        eid = insert_event(db, minutes_ago=10)  # < 30min min-window
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert run(strat, CONFIRM_PRICES) == []
        assert get_status(db, eid) == "ASSESSED"

    def test_past_window_expires(self, tmp_path: Any) -> None:
        db = make_db()
        eid = insert_event(db, minutes_ago=200)  # > 120min max-window
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert run(strat, CONFIRM_PRICES) == []
        assert get_status(db, eid) == "EXPIRED"

    def test_expired_high_urgency_alert_capped_at_one_per_run(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        insert_event(db, minutes_ago=300, urgency=9)
        insert_event(db, minutes_ago=310, urgency=10)
        insert_event(db, minutes_ago=320, urgency=5)  # below alert bar
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        run(strat, CONFIRM_PRICES)
        expired = [c for c in sent_alerts if c[0] == "Event expired unconfirmed"]
        assert len(expired) == 1
        assert "No trade taken" in expired[0][1]


# =============================================================================
# Sizing / stop math
# =============================================================================


class TestSizing:
    def test_long_size_and_stop(self, tmp_path: Any) -> None:
        db = make_db()
        insert_event(db)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        intents = run(strat, CONFIRM_PRICES, FakeBroker(equity=100_000))
        assert len(intents) == 1
        # equity * risk / stop_distance = 100000 * 0.005 / (1.0 * 0.01)
        assert intents[0].target_position == pytest.approx(50_000.0)
        assert intents[0].symbol == "USD_CAD"
        assert intents[0].urgency == "high"
        pos = strat.open_positions["USD_CAD"]
        assert pos.entry_price == pytest.approx(1.0)
        assert pos.stop_price == pytest.approx(0.99)

    def test_short_size_is_negative_with_stop_above(self, tmp_path: Any) -> None:
        db = make_db()
        affected = [{"instrument": "USD_CAD", "kind": "oanda",
                     "direction": "short", "reason": "test"}]
        insert_event(db, affected=affected, direction="bearish")
        strat = make_strategy(tmp_path, db=db, provider=FakeProvider(p0=1.02, daily_vol=0.01))
        intents = run(strat, {"USD_CAD": tick(1.0)}, FakeBroker(equity=100_000))
        assert len(intents) == 1
        assert intents[0].target_position == pytest.approx(-50_000.0)
        assert strat.open_positions["USD_CAD"].stop_price == pytest.approx(1.01)


# =============================================================================
# Exits — hard TIME STOP and hard stop
# =============================================================================


def seed_position(
    strat: EventDrivenStrategy,
    symbol: str = "USD_CAD",
    hours_ago: float = 5.0,
    entry_price: float = 0.99,
    quantity: float = 50_000.0,
    direction: int = 1,
    stop_price: float = 0.9801,
) -> None:
    strat.open_positions[symbol] = EventPosition(
        symbol=symbol, event_id=1,
        entry_ts=datetime.now(UTC) - timedelta(hours=hours_ago),
        entry_price=entry_price, quantity=quantity, direction=direction,
        stop_price=stop_price, headline="seeded",
    )


class TestExits:
    def test_time_stop_emits_exit_after_max_holding(self, tmp_path: Any) -> None:
        db = make_db()  # empty table — exits must not need events
        strat = make_strategy(tmp_path, db=db)
        seed_position(strat, hours_ago=5.0)  # > 4h default
        intents = run(strat, {"USD_CAD": tick(1.0)})
        assert len(intents) == 1
        assert intents[0].target_position == 0
        assert strat.open_positions == {}
        # Realized P&L booked: (1.0 - 0.99) * 50000 = 500, persisted.
        state = json.loads((tmp_path / "event_book_state.json").read_text())
        assert state["realized_pnl"] == pytest.approx(500.0)
        assert state["closed_trades"] == 1

    def test_time_stop_fires_even_without_price(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=6.0)
        intents = run(strat, {})  # no tick, no provider
        assert len(intents) == 1
        assert intents[0].target_position == 0

    def test_no_exit_before_time_stop(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=1.0)
        assert run(strat, {"USD_CAD": tick(1.0)}) == []
        assert "USD_CAD" in strat.open_positions

    def test_hard_stop_exit(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=1.0, entry_price=1.0, stop_price=0.99)
        intents = run(strat, {"USD_CAD": tick(0.985)})
        assert len(intents) == 1
        assert intents[0].target_position == 0
        assert strat.open_positions == {}


# =============================================================================
# Caps
# =============================================================================


class TestCaps:
    def test_max_concurrent_blocks_new_entries(self, tmp_path: Any) -> None:
        db = make_db()
        eid = insert_event(db)
        strat = make_strategy(
            tmp_path, db=db, provider=confirming_provider(),
            max_concurrent_event_positions=2,
        )
        seed_position(strat, symbol="XAU_USD", hours_ago=0.5)
        seed_position(strat, symbol="BCO_USD", hours_ago=0.5)
        intents = run(strat, CONFIRM_PRICES)
        assert intents == []
        # Confirmed (and alerted) but NOT traded — row stays CONFIRMED.
        assert get_status(db, eid) == "CONFIRMED"

    def test_event_book_loss_cap_blocks_entries_and_logs_critical(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        state_path = tmp_path / "event_book_state.json"
        state_path.write_text(json.dumps({
            "version": 1, "realized_pnl": -2500.0,
            "closed_trades": 3, "open_positions": {},
        }))
        db = make_db()
        eid = insert_event(db)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        with caplog.at_level(logging.CRITICAL, logger="src.strategies.event_driven"):
            intents = run(strat, CONFIRM_PRICES, FakeBroker(equity=100_000))
        assert intents == []
        assert get_status(db, eid) == "CONFIRMED"  # confirmed, never traded
        crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(crit) == 1
        assert "LOSS CAP" in crit[0].getMessage()

    def test_book_under_cap_allows_entries(self, tmp_path: Any) -> None:
        state_path = tmp_path / "event_book_state.json"
        state_path.write_text(json.dumps({
            "version": 1, "realized_pnl": -1000.0,  # under 2% of 100k
            "closed_trades": 1, "open_positions": {},
        }))
        db = make_db()
        insert_event(db)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert len(run(strat, CONFIRM_PRICES, FakeBroker(equity=100_000))) == 1


# =============================================================================
# Missing table / missing DB — engine boot must never break
# =============================================================================


class TestMissingTable:
    def test_noop_and_logs_once(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        db = sa.create_engine("sqlite://")  # no geo_events table
        strat = make_strategy(tmp_path, db=db)
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_driven"):
            assert run(strat, {}) == []
            assert run(strat, {}) == []
            assert run(strat, {}) == []
        warnings = [r for r in caplog.records if "geo_events" in r.getMessage()]
        assert len(warnings) == 1

    def test_no_db_handle_is_noop(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=None)
        assert run(strat, {}) == []

    def test_recovers_when_table_appears(self, tmp_path: Any) -> None:
        db = sa.create_engine("sqlite://")
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert run(strat, CONFIRM_PRICES) == []
        with db.begin() as conn:
            conn.execute(text("""
                CREATE TABLE geo_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seen_at TEXT, source TEXT, external_id TEXT,
                    headline TEXT, url TEXT, theme TEXT,
                    assessment TEXT, status TEXT, status_updated_at TEXT)
            """))
        insert_event(db)
        assert len(run(strat, CONFIRM_PRICES)) == 1


# =============================================================================
# Instrument mapping
# =============================================================================


class TestInstrumentMapping:
    def test_unknown_instrument_skipped_with_warning(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        db = make_db()
        affected = [{"instrument": "KC_COFFEE", "kind": "fx",
                     "direction": "long", "reason": "frost"}]
        eid = insert_event(db, affected=affected)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_driven"):
            intents = run(strat, {"KC_COFFEE": tick(1.0)})
        assert intents == []  # never traded blind, never crashed
        assert get_status(db, eid) == "CONFIRMED"
        assert any("KC_COFFEE" in r.getMessage() for r in caplog.records)

    def test_alias_maps_to_broker_instrument(self, tmp_path: Any) -> None:
        db = make_db()
        affected = [{"instrument": "USDCAD", "kind": "fx",
                     "direction": "long", "reason": "test"}]
        insert_event(db, affected=affected)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        intents = run(strat, CONFIRM_PRICES)
        assert len(intents) == 1
        assert intents[0].symbol == "USD_CAD"

    def test_equity_watch_never_gates_or_trades(self, tmp_path: Any) -> None:
        """An event whose only affected entries are equity_watch can
        never confirm (nothing tradable gates) — alert-only by design."""
        db = make_db()
        affected = [{"instrument": "NVDA", "kind": "equity_watch",
                     "direction": "watch", "reason": "test"}]
        eid = insert_event(db, affected=affected)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert run(strat, {"NVDA": tick(1.0)}) == []
        assert get_status(db, eid) == "ASSESSED"


# =============================================================================
# Alerts
# =============================================================================


class TestAlerts:
    def test_confirmed_alert_content(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        affected = [
            {"instrument": "USD_CAD", "kind": "oanda",
             "direction": "long", "reason": "test"},
            {"instrument": "NVDA", "kind": "equity_watch",
             "direction": "watch", "reason": "test"},
        ]
        insert_event(db, affected=affected, headline="Major pipeline explosion")
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        run(strat, CONFIRM_PRICES)
        confirmed = [c for c in sent_alerts if c[0] == "Event confirmed"]
        assert len(confirmed) == 1
        _, message, priority = confirmed[0]
        assert priority == 1
        assert "Major pipeline explosion" in message
        assert "Trade: USD_CAD long" in message
        assert "Risk: 0.50% of equity per trade" in message
        assert "Stop: 1.00% from entry" in message
        assert "Time stop: 4h" in message
        assert "Urgency: 8/10" in message
        assert "Confidence: 0.90" in message
        assert "Watch: NVDA" in message


# =============================================================================
# build_strategies routing smoke
# =============================================================================


class TestBuildStrategies:
    def test_event_id_routes_to_event_strategy(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import src.runtime.run_engine as run_engine

        monkeypatch.setattr(
            run_engine, "_build_db_engine", lambda: sa.create_engine("sqlite://"),
        )
        config = {"strategies": [{
            "id": "event_driven",
            "enabled": True,
            "config": {
                "event_risk_pct": 0.007,
                "event_book_state_path": str(tmp_path / "book.json"),
            },
        }]}
        strategies = run_engine.build_strategies(config, broker=None, oms=None)
        assert len(strategies) == 1
        assert isinstance(strategies[0], EventDrivenStrategy)
        assert strategies[0].config.event_risk_pct == 0.007
        assert strategies[0].db is not None

    def test_enabled_false_skips_strategy(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import src.runtime.run_engine as run_engine

        monkeypatch.setattr(
            run_engine, "_build_db_engine", lambda: sa.create_engine("sqlite://"),
        )
        config = {"strategies": [
            {"id": "event_driven", "enabled": False,
             "config": {"event_book_state_path": str(tmp_path / "book.json")}},
            {"id": "cb_sentiment_shift"},
        ]}
        strategies = run_engine.build_strategies(config, broker=None, oms=None)
        assert len(strategies) == 1
        assert not isinstance(strategies[0], EventDrivenStrategy)


# =============================================================================
# State persistence
# =============================================================================


class TestStatePersistence:
    def test_open_positions_survive_restart(self, tmp_path: Any) -> None:
        db = make_db()
        insert_event(db)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        run(strat, CONFIRM_PRICES)
        assert "USD_CAD" in strat.open_positions

        # "Restart": a fresh instance reloads the persisted position, so
        # the hard time stop still fires after an engine bounce.
        strat2 = make_strategy(tmp_path, db=db, provider=confirming_provider())
        assert "USD_CAD" in strat2.open_positions
        assert strat2.open_positions["USD_CAD"].entry_price == pytest.approx(1.0)

    def test_corrupt_state_file_does_not_break_boot(self, tmp_path: Any) -> None:
        state_path = tmp_path / "event_book_state.json"
        state_path.write_text("{not json !!!")
        strat = make_strategy(tmp_path, db=make_db())
        assert strat.open_positions == {}
        assert run(strat, {}) == []
        assert (tmp_path / "event_book_state.json.corrupt").exists()
