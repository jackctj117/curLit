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
# Haven concentration cap (CL-5mkf) — real enforcement on gold/silver legs
# =============================================================================


class TestHavenCap:
    """The machine cap on combined open notional in HAVEN_INSTRUMENTS
    (XAU_USD, XAG_USD). Injectable equity (FakeBroker) + tracked
    positions (seed_position); no DB/network needed for the sizing
    exercise itself — we drive _haven_capped_size directly and via the
    full confirmed-entry path."""

    def test_default_is_twenty_pct(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        assert strat.config.haven_max_pct == pytest.approx(0.20)

    def test_reduces_size_when_over_cap(self, tmp_path: Any) -> None:
        # Equity 100k, cap 20% = 20k haven notional. 12k already open in
        # gold; a proposed 15k silver leg would total 27k → trimmed to
        # exactly the 8k headroom (÷ price 1.0 → 8000 units).
        strat = make_strategy(tmp_path, db=make_db(), haven_max_pct=0.20)
        strat.open_positions["XAU_USD"] = EventPosition(
            symbol="XAU_USD", event_id=1, entry_ts=datetime.now(UTC),
            entry_price=1.0, quantity=12_000.0, direction=1,
            stop_price=0.99, headline="seeded gold",
        )
        capped = strat._haven_capped_size(
            "XAG_USD", size=15_000.0, entry_price=1.0, equity=100_000.0,
        )
        assert capped == pytest.approx(8_000.0)  # fills remaining headroom

    def test_skips_when_already_at_cap(self, tmp_path: Any) -> None:
        # 20k gold already open == the full 20k cap → a new haven leg is
        # skipped (size 0), sign preserved regardless of direction.
        strat = make_strategy(tmp_path, db=make_db(), haven_max_pct=0.20)
        strat.open_positions["XAU_USD"] = EventPosition(
            symbol="XAU_USD", event_id=1, entry_ts=datetime.now(UTC),
            entry_price=1.0, quantity=20_000.0, direction=1,
            stop_price=0.99, headline="seeded gold",
        )
        assert strat._haven_capped_size(
            "XAG_USD", size=-5_000.0, entry_price=1.0, equity=100_000.0,
        ) == 0.0

    def test_under_cap_size_unchanged(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db(), haven_max_pct=0.20)
        # Nothing open; a 5k gold leg is well under the 20k cap → intact.
        assert strat._haven_capped_size(
            "XAU_USD", size=5_000.0, entry_price=1.0, equity=100_000.0,
        ) == pytest.approx(5_000.0)

    def test_non_haven_untouched(self, tmp_path: Any) -> None:
        # A non-haven symbol never gets capped — even with gold at the cap.
        strat = make_strategy(tmp_path, db=make_db(), haven_max_pct=0.20)
        strat.open_positions["XAU_USD"] = EventPosition(
            symbol="XAU_USD", event_id=1, entry_ts=datetime.now(UTC),
            entry_price=1.0, quantity=30_000.0, direction=1,
            stop_price=0.99, headline="seeded gold",
        )
        assert strat._haven_capped_size(
            "BCO_USD", size=99_000.0, entry_price=1.0, equity=100_000.0,
        ) == pytest.approx(99_000.0)

    def test_confirmed_haven_entry_skipped_at_cap_logs_warning(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Full confirmed-entry path: silver already at the cap → the new
        # GOLD entry is skipped (no intent), row stays CONFIRMED, WARNING
        # names the haven exposure. (Different haven symbol so it isn't
        # short-circuited as already_open.)
        db = make_db()
        affected = [{"instrument": "XAU_USD", "kind": "oanda",
                     "direction": "long", "reason": "risk-off"}]
        eid = insert_event(db, affected=affected)
        strat = make_strategy(
            tmp_path, db=db, provider=FakeProvider(p0=0.99, daily_vol=0.01),
            haven_max_pct=0.20, max_concurrent_event_positions=5,
        )
        # 20k silver already open (keyed by symbol) = the 20k cap @ 100k.
        strat.open_positions["XAG_USD"] = EventPosition(
            symbol="XAG_USD", event_id=99, entry_ts=datetime.now(UTC),
            entry_price=1.0, quantity=20_000.0, direction=1,
            stop_price=0.90, headline="seeded silver",
        )
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_driven"):
            intents = run(strat, {"XAU_USD": tick(1.0)}, FakeBroker(equity=100_000))
        assert intents == []
        assert get_status(db, eid) == "CONFIRMED"  # confirmed, never traded
        assert any(
            "Haven concentration cap" in r.getMessage() for r in caplog.records
        )

    def test_confirmed_haven_entry_reduced_under_cap(
        self, tmp_path: Any,
    ) -> None:
        # Silver partially open (10k of a 20k cap); a fresh confirmed GOLD
        # entry is trimmed to the 10k headroom rather than skipped.
        db = make_db()
        affected = [{"instrument": "XAU_USD", "kind": "oanda",
                     "direction": "long", "reason": "risk-off"}]
        insert_event(db, affected=affected)
        # event_stop_pct default 0.01, entry 1.0 → stop_distance 0.01;
        # unconstrained size = 100k * 0.005 / 0.01 = 50k notional (way
        # over the 10k headroom) → trimmed to 10k units at price 1.0.
        strat = make_strategy(
            tmp_path, db=db, provider=FakeProvider(p0=0.99, daily_vol=0.01),
            haven_max_pct=0.20, max_concurrent_event_positions=5,
        )
        strat.open_positions["XAG_USD"] = EventPosition(
            symbol="XAG_USD", event_id=99, entry_ts=datetime.now(UTC),
            entry_price=1.0, quantity=10_000.0, direction=1,
            stop_price=0.90, headline="seeded silver",
        )
        intents = run(strat, {"XAU_USD": tick(1.0)}, FakeBroker(equity=100_000))
        assert len(intents) == 1
        # Trimmed to the remaining 10k headroom (÷ entry 1.0 = 10k units).
        assert intents[0].target_position == pytest.approx(10_000.0)


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


# =============================================================================
# Alert enrichment (CL-mgcp): price, reason, age, advisory ideas
# =============================================================================


def _add_advisory(db: Any, eid: int, ideas: list[dict[str, Any]]) -> None:
    """Splice trade_ideas into a stored assessment (the fixture's
    insert_event predates the advisory keys)."""
    with db.begin() as conn:
        raw = conn.execute(
            text("SELECT assessment FROM geo_events WHERE id = :id"), {"id": eid},
        ).scalar()
        assessment = json.loads(raw)
        assessment["trade_ideas"] = ideas
        conn.execute(
            text("UPDATE geo_events SET assessment = :a WHERE id = :id"),
            {"a": json.dumps(assessment), "id": eid},
        )


_IDEAS = [
    {"ticker": "TSM", "action": "buy_puts", "direction": "bearish",
     "confidence": 0.7, "rationale": "advanced-node concentration",
     "time_horizon": "short", "holding_period_days": "2-6",
     "time_stop_days": 5},
    {"ticker": "RTX", "action": "long", "direction": "bullish",
     "confidence": 0.5, "rationale": "defense demand",
     "time_horizon": "medium", "holding_period_days": "10-20",
     "time_stop_days": 20},
]


class TestConfirmedAlertEnrichment:
    def _confirm(
        self, tmp_path: Any, db: Any, minutes_ago: float = 60,
    ) -> None:
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        run(strat, CONFIRM_PRICES)

    def test_trade_line_has_entry_price_and_reason(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        insert_event(db)
        self._confirm(tmp_path, db)
        message = next(m for t, m, _ in sent_alerts if t == "Event confirmed")
        assert "Trade: USD_CAD long" in message
        assert "@ 1" in message  # ask tick 1.0 formatted %g
        assert "Why: test" in message

    def test_age_line_since_first_seen(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        insert_event(db, minutes_ago=60)
        self._confirm(tmp_path, db)
        message = next(m for t, m, _ in sent_alerts if t == "Event confirmed")
        assert "Age: 1h since first seen" in message

    def test_advisory_ideas_block_clearly_separated(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        eid = insert_event(db)
        _add_advisory(db, eid, _IDEAS)
        self._confirm(tmp_path, db)
        message = next(m for t, m, _ in sent_alerts if t == "Event confirmed")
        assert "Operator ideas (not machine-traded):" in message
        assert "- TSM buy_puts short stop5d — advanced-node concentration" in message
        assert "- RTX long medium stop20d — defense demand" in message
        # Advisory block comes AFTER the machine-trade facts.
        assert message.index("Trade: USD_CAD") < message.index("Operator ideas")

    def test_no_ideas_no_advisory_block(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        insert_event(db)
        self._confirm(tmp_path, db)
        message = next(m for t, m, _ in sent_alerts if t == "Event confirmed")
        assert "Operator ideas" not in message

    def test_top_idea_shows_grounded_card_when_price_resolves(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        # CL-jiqq: the top idea's GROUNDED card numbers surface in the
        # confirmed alert when a live price resolves. USD_CAD is in the
        # engine's price feed (CONFIRM_PRICES = tick 1.0), so its dollar
        # stop/target/R:R render under the idea line.
        db = make_db()
        eid = insert_event(db)
        _add_advisory(db, eid, [{
            "ticker": "USD_CAD", "action": "long", "direction": "bullish",
            "confidence": 0.9, "rationale": "chokepoint reopening",
            "time_horizon": "short", "holding_period_days": "2-6",
            "time_stop_days": 5, "stop_loss_pct": 0.05,
            "target_pct": [0.08, 0.15],
            "entry_trigger": "on confirmed reopening",
            "invalidation": "renewed blockade",
        }])
        self._confirm(tmp_path, db)
        message = next(m for t, m, _ in sent_alerts if t == "Event confirmed")
        # long → stop below 1.0, targets above; card grounds them.
        assert "stop $0.95" in message
        assert "tgt $1.08/$1.15" in message
        assert "R:R" in message
        assert "entry on confirmed reopening" in message
        assert "invalid if renewed blockade" in message

    def test_top_idea_card_lines_no_price_no_dollar_levels(
        self, tmp_path: Any,
    ) -> None:
        # Directly exercise the card helper: no price → no fabricated
        # dollar stop/target/strike; only price-free facts (the DTE
        # window for an option) may show. A bare STOCK idea shows nothing.
        strat = make_strategy(tmp_path, db=make_db(), provider=None)
        opt = {"ticker": "TSM", "action": "buy_puts", "direction": "bearish",
               "time_horizon": "short", "confidence": 0.7}
        opt_lines = strat._top_idea_card_lines(opt, prices={}, now=None)
        assert opt_lines == ["  1-3 weeks to expiry"]  # DTE is price-free
        assert not any("$" in ln for ln in opt_lines)  # no fake dollars

        stock = {"ticker": "TSM", "action": "short", "direction": "bearish",
                 "time_horizon": "short", "confidence": 0.7}
        assert strat._top_idea_card_lines(stock, prices={}, now=None) == []


class _CrossAssetProvider:
    """Provider that confirms Gate B on USD_CAD AND moves the theme's
    cross-asset instruments so the corroboration line renders (CL-6mzn).

    ``get_latest_value`` returns a per-instrument reference value at
    seen_at and a moved value at 'now' (distinguished by as_of), so the
    cross-asset layer sees real % moves; USD_CAD is priced flat so the
    live tick (CONFIRM_PRICES) drives Gate B.
    """

    def __init__(self, ref: float, cur: float, seen_cutoff: datetime) -> None:
        self.seen_cutoff = seen_cutoff
        # BCO/WTI up (agree "up"); USD_CAD down at the CROSS-ASSET leg but
        # Gate B uses the live tick 1.0 vs its own ref 0.99 → still long-OK.
        self._legs = {
            "BCO_USD": (80.0, 84.0),
            "WTICO_USD": (76.0, 79.0),
            "USD_CAD": (0.99, 0.99),  # cross-asset flat; Gate B via live tick
            "USD_NOK": (10.5, 10.4),
        }

    def get_latest_value(self, instrument: str, as_of: datetime) -> float | None:
        leg = self._legs.get(instrument)
        if leg is None:
            return 0.99  # generic ref for any Gate-B instrument
        p0, p1 = leg
        return p1 if as_of > self.seen_cutoff else p0

    def get_realized_vol(self, pair: str, window: int = 20,
                         as_of: datetime | None = None) -> float | None:
        return 0.01 * math.sqrt(252.0)


class TestCrossAssetLine:
    """The cross-asset corroboration line on the confirmed-event alert
    (CL-6mzn) — display-only, never a gate."""

    def _insert_energy(self, db: Any, minutes_ago: float = 60) -> int:
        # An energy_chokepoint event (a configured cross-asset theme),
        # tradable long USD_CAD so Gate B can confirm via the live tick.
        _EVENT_SEQ["n"] += 1
        assessment = {
            "core_event": "hormuz threat", "direction": "bullish",
            "urgency": 8, "horizon": "hours", "confidence": 0.9,
            "affected": [{"instrument": "USD_CAD", "kind": "fx",
                          "direction": "long", "reason": "petro-fx"}],
            "rationale": "chokepoint",
        }
        seen = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
        with db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO geo_events (seen_at, source, external_id, "
                    "headline, url, theme, assessment, status, status_updated_at) "
                    "VALUES (:seen, 'gdelt', :ext, 'Hormuz closure threat', '', "
                    "'energy_chokepoint', :a, 'ASSESSED', :seen)"
                ),
                {"seen": seen, "ext": f"xa-{_EVENT_SEQ['n']}",
                 "a": json.dumps(assessment)},
            )
            return int(conn.execute(text("SELECT max(id) FROM geo_events")).scalar())

    def test_confirmed_alert_shows_cross_asset_confirms(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        self._insert_energy(db)
        cutoff = datetime.now(UTC) - timedelta(minutes=30)
        strat = make_strategy(
            tmp_path, db=db, provider=_CrossAssetProvider(0.99, 1.0, cutoff),
        )
        run(strat, CONFIRM_PRICES)  # USD_CAD live tick 1.0 confirms Gate B
        message = next(m for t, m, _ in sent_alerts if t == "Event confirmed")
        assert "Cross-asset:" in message
        assert "BCO_USD" in message
        assert "✓" in message
        assert "confirms" in message

    def test_unconfigured_theme_omits_cross_asset(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        # The default insert_event uses theme='energy' (NOT a configured
        # cross-asset theme) → unknown → no cross-asset line.
        db = make_db()
        insert_event(db)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        run(strat, CONFIRM_PRICES)
        message = next(m for t, m, _ in sent_alerts if t == "Event confirmed")
        assert "Cross-asset:" not in message


class TestExpiredAlertEnrichment:
    def test_age_and_top_idea(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        eid = insert_event(db, minutes_ago=300, urgency=9)
        _add_advisory(db, eid, _IDEAS)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        run(strat, CONFIRM_PRICES)
        message = next(
            m for t, m, _ in sent_alerts if t == "Event expired unconfirmed"
        )
        assert "Age: 5h since first seen" in message
        # Highest-confidence idea wins the single Top idea line.
        assert "Top idea: TSM buy_puts (stop 5d) — advanced-node concentration" in message
        assert "RTX" not in message
        assert "No trade taken" in message

    def test_no_ideas_no_top_idea_line(
        self, tmp_path: Any, sent_alerts: list[tuple[str, str, int]],
    ) -> None:
        db = make_db()
        insert_event(db, minutes_ago=300, urgency=9)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        run(strat, CONFIRM_PRICES)
        message = next(
            m for t, m, _ in sent_alerts if t == "Event expired unconfirmed"
        )
        assert "Top idea" not in message
