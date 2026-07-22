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

from src.execution.oms import OrderIntent, Urgency
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
    """Capture notify_operator calls; also guarantees no network attempt.

    Alert dispatch lives in the events layer since CL-ikz2 — the patch
    target is the notifier module, not the strategy."""
    calls: list[tuple[str, str, int]] = []

    def _capture(title: str, message: str, priority: int = 0) -> None:
        calls.append((title, message, priority))

    monkeypatch.setattr("src.events.event_notifier.notify_operator", _capture)
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
    # The base event_risk_pct sizing here produces a 50k-notional leg on a
    # 100k account (50% notional — its RISK at the 1% stop is only 0.5% of
    # equity). That is over the default 25% per-instrument cap (CL-wbmw),
    # so these tests raise the cap out of the way to exercise the base
    # sizing FORMULA in isolation; the cap's own trimming has its own
    # class (TestConcentrationCap).

    def test_long_size_and_stop(self, tmp_path: Any) -> None:
        db = make_db()
        insert_event(db)
        strat = make_strategy(
            tmp_path, db=db, provider=confirming_provider(),
            per_instrument_max_pct=1.0,
        )
        intents = run(strat, CONFIRM_PRICES, FakeBroker(equity=100_000))
        assert len(intents) == 1
        # equity * risk / stop_distance = 100000 * 0.005 / (1.0 * 0.01)
        assert intents[0].target_position == pytest.approx(50_000.0)
        assert intents[0].symbol == "USD_CAD"
        # Canonical urgency vocabulary (CL-ikz2) — the coordinator's rank
        # map knows "urgent"; the legacy "high" never escalated.
        assert intents[0].urgency == "urgent"
        pos = strat.open_positions["USD_CAD"]
        assert pos.entry_price == pytest.approx(1.0)
        assert pos.stop_price == pytest.approx(0.99)

    def test_short_size_is_negative_with_stop_above(self, tmp_path: Any) -> None:
        db = make_db()
        affected = [{"instrument": "USD_CAD", "kind": "oanda",
                     "direction": "short", "reason": "test"}]
        insert_event(db, affected=affected, direction="bearish")
        strat = make_strategy(
            tmp_path, db=db, provider=FakeProvider(p0=1.02, daily_vol=0.01),
            per_instrument_max_pct=1.0,
        )
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
    # Direct book assignment: strategy.open_positions is a merged VIEW
    # (open + pending exits) since CL-8cw1 — mutations target the book.
    strat.book.open_positions[symbol] = EventPosition(
        symbol=symbol, event_id=1,
        entry_ts=datetime.now(UTC) - timedelta(hours=hours_ago),
        entry_price=entry_price, quantity=quantity, direction=direction,
        stop_price=stop_price, headline="seeded",
    )


class TestExits:
    def test_time_stop_finalizes_on_broker_flat_confirmation(
        self, tmp_path: Any,
    ) -> None:
        """Two-phase (CL-8cw1): the trigger tick emits the exit intent and
        PARKS the leg in pending_exits (nothing finalized); the next tick's
        broker-flat snapshot books the realized P&L exactly as before."""
        db = make_db()  # empty table — exits must not need events
        strat = make_strategy(tmp_path, db=db)
        seed_position(strat, hours_ago=5.0)  # > 4h default
        broker = _BrokerWithPositions({"USDCAD"})  # broker holds the leg
        intents = run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert len(intents) == 1
        assert intents[0].target_position == 0
        # Triggered, not finalized: pending, still reconciler-visible.
        assert strat.book.pending_exits["USD_CAD"].reason == "time_stop"
        assert strat.book.open_positions == {}
        assert "USD_CAD" in strat.open_positions  # merged view
        assert strat.book.realized_pnl == 0.0
        # Broker now flat → confirm finalizes and persists:
        # (1.0 - 0.99) * 50000 = 500 from the TRIGGER price.
        broker._held = set()
        assert run(strat, {"USD_CAD": tick(1.0)}, broker) == []
        assert strat.open_positions == {}
        assert strat.book.pending_exits == {}
        state = json.loads((tmp_path / "event_book_state.json").read_text())
        assert state["realized_pnl"] == pytest.approx(500.0)
        assert state["closed_trades"] == 1

    def test_time_stop_fires_even_without_price(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=6.0)
        intents = run(strat, {})  # no tick, no provider
        assert len(intents) == 1
        assert intents[0].target_position == 0
        # No price at trigger → pnl records as 0 when the exit confirms.
        assert strat.book.pending_exits["USD_CAD"].trigger_price is None
        # Broker unreadable at trigger (FakeBroker has no get_positions)
        # → no quantity capture either: confirm on broker-flat only, and
        # book P&L as before (CL-9dhg — never guess).
        assert strat.book.pending_exits["USD_CAD"].trigger_broker_qty is None

    def test_no_exit_before_time_stop(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=1.0)
        assert run(strat, {"USD_CAD": tick(1.0)}) == []
        assert "USD_CAD" in strat.open_positions
        assert strat.book.pending_exits == {}

    def test_hard_stop_exit(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=1.0, entry_price=1.0, stop_price=0.99)
        broker = _BrokerWithPositions({"USDCAD"})
        intents = run(strat, {"USD_CAD": tick(0.985)}, broker)
        assert len(intents) == 1
        assert intents[0].target_position == 0
        assert strat.book.pending_exits["USD_CAD"].reason == "hard_stop"
        broker._held = set()  # exit filled
        run(strat, {"USD_CAD": tick(0.985)}, broker)
        assert strat.open_positions == {}
        # Loss booked from the trigger price: (0.985 - 1.0) * 50000.
        assert strat.book.realized_pnl == pytest.approx(-750.0)


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
# Concentration caps (CL-wbmw generalizes CL-5mkf) — real enforcement
# =============================================================================


def _seed(strat: EventDrivenStrategy, symbol: str, units: float,
          price: float = 1.0, stop_price: float | None = None) -> None:
    """Seed a tracked open position (notional = |units| * price). Default
    stop is a far-away 0.5 so the full-path tests' exit check (against the
    FakeProvider's p0=0.99 fallback price) doesn't close the seed before
    the new entry is sized."""
    strat.book.open_positions[symbol] = EventPosition(
        symbol=symbol, event_id=1, entry_ts=datetime.now(UTC),
        entry_price=price, quantity=units, direction=1 if units >= 0 else -1,
        stop_price=stop_price if stop_price is not None else 0.5,
        headline=f"seeded {symbol}",
    )


class TestConcentrationCap:
    """The generalized machine cap: a per-instrument cap on EVERY event
    instrument (per_instrument_max_pct) plus the tighter haven-cluster cap
    (haven_max_pct) on gold+silver. Injectable equity (FakeBroker) +
    tracked positions; we drive _concentration_capped_size directly and
    via the full confirmed-entry path."""

    def test_defaults(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, db=make_db())
        # Ceilings ABOVE one base leg (~50% notional) so the first leg
        # passes intact and the caps bind on accumulation (CL-wbmw retune).
        assert strat.config.per_instrument_max_pct == pytest.approx(0.55)
        assert strat.config.haven_max_pct == pytest.approx(0.60)

    # ---- per-instrument cap on a NON-haven single name (e.g. BCO_USD) ----

    def test_per_instrument_trims_non_haven(self, tmp_path: Any) -> None:
        # Equity 100k, per-instrument cap 25% = 25k. 20k BCO_USD already
        # open; a proposed 15k leg would total 35k → trimmed to the 5k
        # headroom (÷ price 1.0 → 5000 units).
        strat = make_strategy(tmp_path, db=make_db(), per_instrument_max_pct=0.25)
        _seed(strat, "BCO_USD", 20_000.0)
        capped = strat._concentration_capped_size(
            "BCO_USD", size=15_000.0, entry_price=1.0, equity=100_000.0,
        )
        assert capped == pytest.approx(5_000.0)

    def test_per_instrument_skips_when_exhausted(self, tmp_path: Any) -> None:
        # 25k BCO_USD open == the full 25k cap → a new leg is skipped
        # (size 0), sign preserved regardless of direction.
        strat = make_strategy(tmp_path, db=make_db(), per_instrument_max_pct=0.25)
        _seed(strat, "BCO_USD", 25_000.0)
        assert strat._concentration_capped_size(
            "BCO_USD", size=-9_000.0, entry_price=1.0, equity=100_000.0,
        ) == 0.0

    def test_under_cap_unchanged(self, tmp_path: Any) -> None:
        # Nothing open; a 5k BCO leg is well under the 25k cap → intact.
        # The cap is a ceiling, not a flat limiter — under-cap passes as-is.
        strat = make_strategy(tmp_path, db=make_db(), per_instrument_max_pct=0.25)
        assert strat._concentration_capped_size(
            "BCO_USD", size=5_000.0, entry_price=1.0, equity=100_000.0,
        ) == pytest.approx(5_000.0)

    def test_other_instrument_open_does_not_count(self, tmp_path: Any) -> None:
        # BCO's per-instrument headroom ignores an unrelated open name.
        strat = make_strategy(tmp_path, db=make_db(), per_instrument_max_pct=0.25)
        _seed(strat, "WTICO_USD", 24_000.0)  # near-cap in a DIFFERENT name
        assert strat._concentration_capped_size(
            "BCO_USD", size=20_000.0, entry_price=1.0, equity=100_000.0,
        ) == pytest.approx(20_000.0)  # BCO itself is empty → full leg

    # ---- haven-cluster cap: the tighter of the two binds ----

    def test_haven_cluster_binds_over_per_instrument(self, tmp_path: Any) -> None:
        # A pure haven-cluster case: 12k SILVER already open. GOLD itself is
        # empty (per-instrument headroom 25k) but the haven cluster headroom
        # is only 20k - 12k = 8k → the cluster cap (smaller) binds. A 15k
        # gold leg is trimmed to 8k, NOT 25k.
        strat = make_strategy(
            tmp_path, db=make_db(),
            per_instrument_max_pct=0.25, haven_max_pct=0.20,
        )
        _seed(strat, "XAG_USD", 12_000.0)
        capped = strat._concentration_capped_size(
            "XAU_USD", size=15_000.0, entry_price=1.0, equity=100_000.0,
        )
        assert capped == pytest.approx(8_000.0)  # haven-cluster headroom

    def test_per_instrument_binds_on_haven(self, tmp_path: Any) -> None:
        # A haven where the per-instrument cap is the tighter one: nothing
        # else haven-open, but 23k GOLD already open. Per-instrument
        # headroom 25k - 23k = 2k; haven-cluster headroom 20k - 23k = -3k...
        # cluster would SKIP. To isolate per-instrument-binds we lift the
        # haven cap above the per-name exposure so the per-name cap wins.
        strat = make_strategy(
            tmp_path, db=make_db(),
            per_instrument_max_pct=0.25, haven_max_pct=0.40,
        )
        _seed(strat, "XAU_USD", 23_000.0)
        capped = strat._concentration_capped_size(
            "XAU_USD", size=10_000.0, entry_price=1.0, equity=100_000.0,
        )
        # per-instrument headroom 25k - 23k = 2k binds (cluster headroom
        # 40k - 23k = 17k is looser).
        assert capped == pytest.approx(2_000.0)

    def test_haven_skips_when_cluster_exhausted(self, tmp_path: Any) -> None:
        # 20k silver open == the full 20k haven cluster cap → a fresh GOLD
        # leg is skipped even though GOLD's own per-instrument headroom
        # (25k) is wide open. Either cap exhausted → skip.
        strat = make_strategy(
            tmp_path, db=make_db(),
            per_instrument_max_pct=0.25, haven_max_pct=0.20,
        )
        _seed(strat, "XAG_USD", 20_000.0)
        assert strat._concentration_capped_size(
            "XAU_USD", size=15_000.0, entry_price=1.0, equity=100_000.0,
        ) == 0.0

    # ---- full confirmed-entry path ----

    def test_per_instrument_warning_names_the_cap(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # 22k BCO_USD open (per-instrument cap 25k @ 100k); a 50k proposed
        # leg is trimmed to the 3k headroom and the WARNING names the
        # per-instrument cap (the general-instrument path, not haven).
        strat = make_strategy(
            tmp_path, db=make_db(), per_instrument_max_pct=0.25,
        )
        _seed(strat, "BCO_USD", 22_000.0)
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_driven"):
            capped = strat._concentration_capped_size(
                "BCO_USD", size=50_000.0, entry_price=1.0, equity=100_000.0,
            )
        assert capped == pytest.approx(3_000.0)
        assert any(
            "Concentration cap [per-instrument]" in r.getMessage()
            for r in caplog.records
        )

    def test_confirmed_haven_entry_skipped_at_cluster_cap_logs_warning(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Full path: silver already at the haven cluster cap → the new GOLD
        # entry is skipped (no intent), row stays CONFIRMED, WARNING names
        # the haven-cluster cap.
        db = make_db()
        affected = [{"instrument": "XAU_USD", "kind": "oanda",
                     "direction": "long", "reason": "risk-off"}]
        eid = insert_event(db, affected=affected)
        strat = make_strategy(
            tmp_path, db=db, provider=FakeProvider(p0=0.99, daily_vol=0.01),
            haven_max_pct=0.20, max_concurrent_event_positions=5,
        )
        # 20k silver open (keyed by symbol) = the 20k haven cap @ 100k.
        _seed(strat, "XAG_USD", 20_000.0)
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_driven"):
            intents = run(strat, {"XAU_USD": tick(1.0)}, FakeBroker(equity=100_000))
        assert intents == []
        assert get_status(db, eid) == "CONFIRMED"  # confirmed, never traded
        assert any(
            "Concentration cap [haven-cluster" in r.getMessage()
            for r in caplog.records
        )

    def test_confirmed_haven_entry_reduced_under_cluster_cap(
        self, tmp_path: Any,
    ) -> None:
        # Silver partially open (10k of a 20k cluster cap); a fresh confirmed
        # GOLD entry is trimmed to the 10k cluster headroom (the tighter of
        # per-instrument 25k vs cluster 10k) rather than skipped.
        db = make_db()
        affected = [{"instrument": "XAU_USD", "kind": "oanda",
                     "direction": "long", "reason": "risk-off"}]
        insert_event(db, affected=affected)
        strat = make_strategy(
            tmp_path, db=db, provider=FakeProvider(p0=0.99, daily_vol=0.01),
            haven_max_pct=0.20, per_instrument_max_pct=0.25,
            max_concurrent_event_positions=5,
        )
        _seed(strat, "XAG_USD", 10_000.0)
        intents = run(strat, {"XAU_USD": tick(1.0)}, FakeBroker(equity=100_000))
        assert len(intents) == 1
        # Trimmed to the 10k cluster headroom (÷ entry 1.0 = 10k units).
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

    def test_legacy_state_file_without_pending_key_loads_empty(
        self, tmp_path: Any,
    ) -> None:
        # The LIVE data/event_book_state.json predates pending_exits
        # (CL-8cw1) — the missing key must default to no pending exits,
        # with every legacy field loaded untouched.
        state_path = tmp_path / "event_book_state.json"
        state_path.write_text(json.dumps({
            "version": 1, "realized_pnl": -250.0, "closed_trades": 2,
            "open_positions": {
                "USD_CAD": {
                    "event_id": 7,
                    "entry_ts": datetime.now(UTC).isoformat(),
                    "entry_price": 1.0, "quantity": 1000.0,
                    "direction": 1, "stop_price": 0.99, "headline": "legacy",
                },
            },
        }))
        strat = make_strategy(tmp_path, db=make_db())
        assert strat.book.pending_exits == {}
        assert strat.book.realized_pnl == pytest.approx(-250.0)
        assert strat.book.closed_trades == 2
        assert "USD_CAD" in strat.book.open_positions


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
        opt_lines = strat.notifier._top_idea_card_lines(opt, prices={}, now=None)
        assert opt_lines == ["  1-3 weeks to expiry"]  # DTE is price-free
        assert not any("$" in ln for ln in opt_lines)  # no fake dollars

        stock = {"ticker": "TSM", "action": "short", "direction": "bearish",
                 "time_horizon": "short", "confidence": 0.7}
        assert strat.notifier._top_idea_card_lines(stock, prices={}, now=None) == []


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


# =============================================================================
# Phantom-position reconciliation (CL-v9g4)
# =============================================================================


class _BrokerWithPositions:
    def __init__(self, held: set[str]) -> None:
        self._held = held

    def get_account(self) -> Any:
        return SimpleNamespace(balance=100_000.0, equity=100_000.0, margin_used=0.0)

    def get_positions(self) -> list[Any]:
        return [SimpleNamespace(symbol=s, quantity=1.0, avg_price=1.0)
                for s in self._held]


class _BrokerRaises:
    def get_account(self) -> Any:
        return SimpleNamespace(balance=100_000.0, equity=100_000.0, margin_used=0.0)

    def get_positions(self) -> list[Any]:
        raise RuntimeError("broker positions unavailable")


def _pos(symbol: str, minutes_ago: float) -> EventPosition:
    return EventPosition(
        symbol=symbol, event_id=1,
        entry_ts=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        entry_price=1.0, quantity=100.0, direction=1, stop_price=0.99,
        headline="x",
    )


class TestPhantomReconciliation:
    def test_norm_symbol_matches_across_underscore(self) -> None:
        assert EventDrivenStrategy._norm_symbol("USD_NOK") == "USDNOK"
        assert EventDrivenStrategy._norm_symbol("usdnok") == "USDNOK"

    def test_prunes_old_phantom_not_held(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, position_reconcile_grace_sec=120)
        strat.open_positions = {"BCO_USD": _pos("BCO_USD", minutes_ago=10)}
        strat._reconcile_positions(_BrokerWithPositions(set()), datetime.now(UTC))
        assert "BCO_USD" not in strat.open_positions  # phantom pruned

    def test_keeps_held_position_across_format(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path)
        strat.open_positions = {"USD_NOK": _pos("USD_NOK", minutes_ago=10)}
        # Broker reports it OANDA-underscore-stripped (USDNOK); norm must match.
        strat._reconcile_positions(
            _BrokerWithPositions({"USDNOK"}), datetime.now(UTC))
        assert "USD_NOK" in strat.open_positions

    def test_grace_window_protects_fresh_position(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path, position_reconcile_grace_sec=120)
        # Recorded 30s ago (< 120s grace) — a real fill may not show broker-side
        # yet, so it must NOT be pruned even though the broker doesn't hold it.
        strat.open_positions = {"USD_CAD": _pos("USD_CAD", minutes_ago=0.5)}
        strat._reconcile_positions(_BrokerWithPositions(set()), datetime.now(UTC))
        assert "USD_CAD" in strat.open_positions

    def test_fail_safe_prunes_nothing_on_broker_error(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path)
        strat.open_positions = {"BCO_USD": _pos("BCO_USD", minutes_ago=10)}
        strat._reconcile_positions(_BrokerRaises(), datetime.now(UTC))
        assert "BCO_USD" in strat.open_positions  # broker error -> keep state

    def test_reconcile_frees_the_cap_for_real_legs(self, tmp_path: Any) -> None:
        # Two old phantoms fill the cap=2; after reconciliation (broker holds
        # none) both are pruned, freeing the slots.
        strat = make_strategy(
            tmp_path, max_concurrent_event_positions=2,
            position_reconcile_grace_sec=120)
        strat.open_positions = {
            "BCO_USD": _pos("BCO_USD", minutes_ago=10),
            "NATGAS_USD": _pos("NATGAS_USD", minutes_ago=10),
        }
        assert len(strat.open_positions) >= strat.config.max_concurrent_event_positions
        strat._reconcile_positions(_BrokerWithPositions(set()), datetime.now(UTC))
        assert len(strat.open_positions) == 0  # cap freed


# =============================================================================
# Pending-exit lifecycle (CL-8cw1, exit half of CL-hqyj)
# =============================================================================


class TestPendingExitLifecycle:
    """Exits finalize on broker CONFIRMATION, not intent emission — a
    rejected exit self-heals by re-emitting every tick while the book
    keeps showing the leg to the reconciler."""

    def _trigger(self, tmp_path: Any, broker: Any) -> EventDrivenStrategy:
        """Seed a leg past the time stop and run one tick: the exit
        intent emits and the leg parks in pending_exits."""
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=5.0)  # > 4h default time stop
        intents = run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert [i.target_position for i in intents] == [0]
        assert "USD_CAD" in strat.book.pending_exits
        return strat

    def test_rejected_exit_reemits_and_stays_reconciler_visible(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Broker STILL holds the leg next tick (exit rejected): the same
        # target-0 intent re-emits, the book is NOT flat, and the leg
        # stays in the reconciler's strategy.open_positions view — so it
        # can never be classified orphaned_broker and double-flattened.
        broker = _BrokerWithPositions({"USDCAD"})
        strat = self._trigger(tmp_path, broker)
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_book"):
            intents = run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert [i.target_position for i in intents] == [0]
        assert intents[0].urgency == Urgency.URGENT.value
        book_view = strat.open_positions
        assert isinstance(book_view, dict)          # reconciler contract
        assert "USD_CAD" in book_view               # reconciler-visible
        assert strat.book.realized_pnl == 0.0       # nothing booked yet
        assert strat.book.closed_trades == 0
        assert any(
            "exit for USD_CAD not confirmed, re-emitting" in r.getMessage()
            for r in caplog.records
        )

    def test_first_emission_logs_no_reemit_warning(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        broker = _BrokerWithPositions({"USDCAD"})
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_book"):
            self._trigger(tmp_path, broker)
        assert not any("re-emitting" in r.getMessage() for r in caplog.records)

    def test_confirm_finalizes_exactly_once(self, tmp_path: Any) -> None:
        broker = _BrokerWithPositions({"USDCAD"})
        strat = self._trigger(tmp_path, broker)
        # Broker flat → finalize from the TRIGGER price (tick 1.0):
        # (1.0 - 0.99) * 50000 = 500 — even though the confirm tick's
        # price is wildly different.
        broker._held = set()
        assert run(strat, {"USD_CAD": tick(5.0)}, broker) == []
        assert strat.book.realized_pnl == pytest.approx(500.0)
        assert strat.book.closed_trades == 1
        assert strat.open_positions == {}
        # Idempotent: a second flat snapshot finalizes nothing more.
        assert strat.book.confirm_exits(broker.get_positions()) == []
        assert strat.book.realized_pnl == pytest.approx(500.0)
        assert strat.book.closed_trades == 1

    def test_broker_still_holding_never_confirms(self, tmp_path: Any) -> None:
        broker = _BrokerWithPositions({"USDCAD"})
        strat = self._trigger(tmp_path, broker)
        assert strat.book.confirm_exits(broker.get_positions()) == []
        assert "USD_CAD" in strat.book.pending_exits
        assert strat.book.realized_pnl == 0.0

    def test_no_reentry_while_pending(self, tmp_path: Any) -> None:
        # A newly-CONFIRMED event on the pending symbol must NOT re-enter:
        # the slot is still occupied by real broker risk.
        db = make_db()
        eid = insert_event(db)  # long USD_CAD, confirming setup
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        seed_position(strat, hours_ago=5.0)  # time-stops this tick
        broker = _BrokerWithPositions({"USDCAD"})
        intents = run(strat, CONFIRM_PRICES, broker)
        # Only the exit intent — no entry intent for the pending symbol.
        assert [i.target_position for i in intents] == [0]
        assert "USD_CAD" in strat.book.pending_exits
        # Confirmed (and alerted) but NOT traded — row stays CONFIRMED.
        assert get_status(db, eid) == "CONFIRMED"

    def test_pending_exit_survives_restart(self, tmp_path: Any) -> None:
        broker = _BrokerWithPositions({"USDCAD"})
        strat = self._trigger(tmp_path, broker)
        del strat
        # "Restart": a fresh instance reloads the pending exit with its
        # trigger capture intact, keeps showing it to the reconciler, and
        # still finalizes once the broker confirms flat.
        strat2 = make_strategy(tmp_path, db=make_db())
        entry = strat2.book.pending_exits["USD_CAD"]
        assert entry.reason == "time_stop"
        assert entry.trigger_price == pytest.approx(1.0)
        assert entry.triggered_ts.tzinfo is not None
        assert entry.position.entry_price == pytest.approx(0.99)
        assert "USD_CAD" in strat2.open_positions
        broker._held = set()
        run(strat2, {}, broker)
        assert strat2.book.pending_exits == {}
        assert strat2.book.realized_pnl == pytest.approx(500.0)
        assert strat2.book.closed_trades == 1


# =============================================================================
# Trigger-time broker capture: co-held residual confirmation, phantom
# finalization, snapshot-once, overlap fail-loud (CL-9dhg)
# =============================================================================


class _BrokerWithNetQty:
    """Broker stub reporting account-wide NET quantity per symbol
    (CL-9dhg) — mutate ``net`` between ticks to move the account."""

    def __init__(self, net: dict[str, float]) -> None:
        self.net = dict(net)

    def get_account(self) -> Any:
        return SimpleNamespace(balance=100_000.0, equity=100_000.0, margin_used=0.0)

    def get_positions(self) -> list[Any]:
        return [SimpleNamespace(symbol=s, quantity=q, avg_price=1.0)
                for s, q in self.net.items()]


class _RecordingSnapshotStore:
    def __init__(self) -> None:
        self.snapshots: list[Any] = []

    def store(self, snapshot: Any) -> str:
        self.snapshots.append(snapshot)
        return "snap-id"


def _legacy_pending_state(tmp_path: Any) -> None:
    """Write a pre-CL-9dhg state file: a pending exit WITHOUT the
    trigger_broker_qty key (the persisted live format at rollout)."""
    now = datetime.now(UTC)
    (tmp_path / "event_book_state.json").write_text(json.dumps({
        "version": 1, "realized_pnl": 0.0, "closed_trades": 0,
        "open_positions": {},
        "pending_exits": {
            "USD_CAD": {
                "position": {
                    "event_id": 1,
                    "entry_ts": (now - timedelta(hours=5)).isoformat(),
                    "entry_price": 0.99, "quantity": 50_000.0,
                    "direction": 1, "stop_price": 0.9801,
                    "headline": "legacy",
                },
                "reason": "time_stop",
                "triggered_ts": now.isoformat(),
                "trigger_price": 1.0,
                "emit_count": 1,
            },
        },
    }))


class TestTriggerBrokerCapture:
    """CL-9dhg findings 1 + 2: broker positions are account-wide NET, so
    a sibling strategy co-holding the instrument means "flat" never
    happens — the pending exit must confirm from the co-holder RESIDUAL.
    And a never-filled leg (trigger capture 0) must finalize as PHANTOM
    with no fabricated realized P&L."""

    def test_coheld_symbol_confirms_on_residual(self, tmp_path: Any) -> None:
        # Our leg is 50k; a co-holder owns another 30k → account net 80k
        # at trigger. When the net drops to exactly the 30k residual, OUR
        # share is out — confirm and book P&L even though the account is
        # never flat.
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=5.0)  # quantity 50k, entry 0.99
        broker = _BrokerWithNetQty({"USDCAD": 80_000.0})
        intents = run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert [i.target_position for i in intents] == [0]
        entry = strat.book.pending_exits["USD_CAD"]
        assert entry.trigger_broker_qty == pytest.approx(80_000.0)
        # The capture persists (a restart must not forget it).
        state = json.loads((tmp_path / "event_book_state.json").read_text())
        assert state["pending_exits"]["USD_CAD"]["trigger_broker_qty"] == (
            pytest.approx(80_000.0)
        )
        # Co-holder residual remains → confirmed, P&L from trigger price.
        broker.net = {"USDCAD": 30_000.0}
        assert run(strat, {"USD_CAD": tick(1.0)}, broker) == []
        assert strat.book.pending_exits == {}
        assert strat.book.realized_pnl == pytest.approx(500.0)
        assert strat.book.closed_trades == 1

    def test_partial_fill_outside_tolerance_stays_pending(
        self, tmp_path: Any,
    ) -> None:
        # Net 80k at trigger; only 20k of our 50k exit filled → net 60k,
        # 30k away from the expected 30k residual (tolerance is
        # max(1, 1% of 50k) = 500) → NOT confirmed, keeps re-emitting.
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=5.0)
        broker = _BrokerWithNetQty({"USDCAD": 80_000.0})
        run(strat, {"USD_CAD": tick(1.0)}, broker)
        broker.net = {"USDCAD": 60_000.0}
        intents = run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert [i.target_position for i in intents] == [0]  # re-emitted
        assert "USD_CAD" in strat.book.pending_exits
        assert strat.book.realized_pnl == 0.0
        assert strat.book.closed_trades == 0

    def test_phantom_leg_finalizes_without_pnl(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A rejected entry inside the reconcile grace: the broker NEVER
        # held the leg (capture 0), its stop crosses, it parks pending —
        # and must finalize as PHANTOM: no realized P&L, no
        # closed_trades, no ExitRecord; a fabricated loss would poison
        # reflective_review and the loss-cap freeze.
        strat = make_strategy(tmp_path, db=make_db())
        # Fresh (36s < 120s grace → not pruned), hard stop crosses.
        seed_position(strat, hours_ago=0.01, entry_price=1.0, stop_price=0.99)
        broker = _BrokerWithNetQty({})  # broker demonstrably holds nothing
        intents = run(strat, {"USD_CAD": tick(0.985)}, broker)
        assert [i.target_position for i in intents] == [0]
        entry = strat.book.pending_exits["USD_CAD"]
        assert entry.trigger_broker_qty == pytest.approx(0.0)
        with caplog.at_level(logging.WARNING, logger="src.strategies.event_book"):
            confirmed = strat.book.confirm_exits(broker.get_positions())
        assert confirmed == []  # phantom yields NO ExitRecord
        assert strat.book.pending_exits == {}
        assert strat.book.realized_pnl == 0.0  # no fabricated loss
        assert strat.book.closed_trades == 0
        assert any("PHANTOM" in r.getMessage() for r in caplog.records)
        state = json.loads((tmp_path / "event_book_state.json").read_text())
        assert state["pending_exits"] == {}
        assert state["realized_pnl"] == 0.0
        assert state["closed_trades"] == 0

    def test_legacy_pending_confirms_only_on_broker_flat(
        self, tmp_path: Any,
    ) -> None:
        # A persisted pre-CL-9dhg pending entry has NO trigger capture →
        # loads as None: the residual rule must never apply (a 30k net
        # would match "residual" if a capture of 80k existed, but with
        # None we cannot know), only broker-flat confirms — and P&L
        # books exactly as before.
        _legacy_pending_state(tmp_path)
        strat = make_strategy(tmp_path, db=make_db())
        assert strat.book.pending_exits["USD_CAD"].trigger_broker_qty is None
        # Broker not flat → stays pending regardless of the quantity.
        broker = _BrokerWithNetQty({"USDCAD": 30_000.0})
        assert strat.book.confirm_exits(broker.get_positions()) == []
        assert "USD_CAD" in strat.book.pending_exits
        # Broker flat → confirms, books (1.0 - 0.99) * 50000 = 500.
        broker.net = {}
        records = strat.book.confirm_exits(broker.get_positions())
        assert [r.symbol for r in records] == ["USD_CAD"]
        assert strat.book.realized_pnl == pytest.approx(500.0)
        assert strat.book.closed_trades == 1

    def test_exit_snapshot_recorded_only_on_first_emission(
        self, tmp_path: Any,
    ) -> None:
        # CL-9dhg finding 10: re-emissions derive every value from the
        # trigger-time capture — one FeatureSnapshot per tick per pending
        # exit is pure spam. Only emission 1 records; re-emitted intents
        # still flow, just without a snapshot payload.
        store = _RecordingSnapshotStore()
        cfg = EventDrivenConfig(
            event_book_state_path=str(tmp_path / "event_book_state.json"),
        )
        strat = EventDrivenStrategy(cfg, db_engine=make_db(), snapshot_store=store)
        seed_position(strat, hours_ago=5.0)
        broker = _BrokerWithNetQty({"USDCAD": 50_000.0})  # exit never fills
        first = run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert first[0].metadata  # trigger emission carries the snapshot ref
        for _ in range(2):
            intents = run(strat, {"USD_CAD": tick(1.0)}, broker)
            assert [i.target_position for i in intents] == [0]
            assert intents[0].metadata == {}  # re-emission: no snapshot
        assert strat.book.pending_exits["USD_CAD"].emit_count == 3
        exit_snaps = [
            s for s in store.snapshots if s.values.get("trigger") == "exit"
        ]
        assert len(exit_snaps) == 1

    def test_symbol_in_both_books_refuses_to_load(self, tmp_path: Any) -> None:
        # CL-9dhg finding 11: a symbol in BOTH open_positions and
        # pending_exits is corrupt state — silently preferring either
        # book loses realized P&L or double-tracks broker risk. Repo
        # rule: refuse to start.
        now = datetime.now(UTC)
        pos_payload = {
            "event_id": 1, "entry_ts": now.isoformat(), "entry_price": 1.0,
            "quantity": 1000.0, "direction": 1, "stop_price": 0.99,
            "headline": "dup",
        }
        (tmp_path / "event_book_state.json").write_text(json.dumps({
            "version": 1, "realized_pnl": 0.0, "closed_trades": 0,
            "open_positions": {"USD_CAD": pos_payload},
            "pending_exits": {
                "USD_CAD": {
                    "position": pos_payload, "reason": "hard_stop",
                    "triggered_ts": now.isoformat(), "trigger_price": 0.99,
                    "emit_count": 1,
                },
            },
        }))
        with pytest.raises(
            ValueError, match="appear in more than one of open_positions",
        ):
            make_strategy(tmp_path, db=make_db())


# =============================================================================
# Entry confirmed-fill lifecycle (ENTRY half of CL-hqyj) — the book records
# the ACTUAL broker fill, never the intended size, so a partial/over/rejected
# fill can never trip the portfolio reconciler's reconciliation_failure kill
# switch (book -308 vs broker -214 was the P1 live halt).
# =============================================================================


class TestEntryLifecycle:
    """A newly-submitted entry parks in ``pending_entries`` at INTENDED
    size (for slot/cap accounting only) and PROMOTES into open_positions at
    the OBSERVED broker fill — the co-held-safe delta from a submit-time
    baseline — or REJECTS (booking nothing) with no fill after grace.
    Pending entries value the RECONCILER-facing view at confirmed_qty (0
    until filled), so the intended phantom is never exposed."""

    def _confirmed_entry_strat(
        self, tmp_path: Any, broker: Any,
    ) -> tuple[EventDrivenStrategy, int]:
        """Confirm one long-USD_CAD event and run a tick against ``broker``;
        the leg lands in pending_entries at its intended 50k size."""
        db = make_db()
        eid = insert_event(db)  # long USD_CAD, confirming setup
        strat = make_strategy(
            tmp_path, db=db, provider=confirming_provider(),
            per_instrument_max_pct=1.0,
        )
        intents = run(strat, CONFIRM_PRICES, broker)
        assert len(intents) == 1
        assert intents[0].target_position == pytest.approx(50_000.0)
        assert "USD_CAD" in strat.book.pending_entries
        # NOT yet in open_positions — awaiting the broker fill.
        assert strat.book.open_positions == {}
        return strat, eid

    def test_partial_fill_books_actual_not_intended(self, tmp_path: Any) -> None:
        # THE root-cause fix: intended 50k long, broker fills only 30k. The
        # book must record the ACTUAL 30k, never the intended 50k — the
        # -308-vs-214 divergence that tripped reconciliation_failure.
        broker = _BrokerWithNetQty({})  # flat at submit → baseline 0
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        assert strat.book.pending_entries["USD_CAD"].entry_broker_qty == (
            pytest.approx(0.0)
        )
        # Broker fills 30k of the 50k order.
        broker.net = {"USDCAD": 30_000.0}
        run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert strat.book.pending_entries == {}
        booked = strat.book.open_positions["USD_CAD"]
        assert booked.quantity == pytest.approx(30_000.0)  # ACTUAL, not 50k
        assert booked.entry_price == pytest.approx(1.0)  # preserved
        assert booked.stop_price == pytest.approx(0.99)
        # Reconciler-facing size now equals the broker fill.
        assert strat.open_positions["USD_CAD"].quantity == pytest.approx(30_000.0)

    def test_full_fill_promotes_at_broker_delta(self, tmp_path: Any) -> None:
        broker = _BrokerWithNetQty({})
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        broker.net = {"USDCAD": 50_000.0}  # full fill
        promoted = strat.book.confirm_entries(broker.get_positions(),
                                              datetime.now(UTC))
        assert [p.quantity for p in promoted] == [pytest.approx(50_000.0)]
        assert strat.book.open_positions["USD_CAD"].quantity == (
            pytest.approx(50_000.0))
        assert strat.book.pending_entries == {}

    def test_rejected_entry_leaves_zero_residue(self, tmp_path: Any) -> None:
        # No fill ever appears; after grace the pending entry is REJECTED
        # and NOTHING is booked — no open leg, no pending, no P&L.
        broker = _BrokerWithNetQty({})  # broker flat, stays flat
        strat, _ = self._confirmed_entry_strat(
            tmp_path, broker,
        )
        # Age the submit past the grace window, then confirm.
        entry = strat.book.pending_entries["USD_CAD"]
        entry.submitted_ts = datetime.now(UTC) - timedelta(seconds=200)
        promoted = strat.book.confirm_entries(broker.get_positions(),
                                              datetime.now(UTC))
        assert promoted == []
        assert strat.book.pending_entries == {}
        assert strat.book.open_positions == {}
        assert strat.open_positions == {}  # zero residue, reconciler sees none
        assert strat.book.realized_pnl == 0.0
        assert strat.book.closed_trades == 0

    def test_within_grace_no_fill_stays_pending(self, tmp_path: Any) -> None:
        broker = _BrokerWithNetQty({})
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        # Fresh submit, broker still flat → NOT rejected yet, still pending.
        promoted = strat.book.confirm_entries(broker.get_positions(),
                                              datetime.now(UTC))
        assert promoted == []
        assert "USD_CAD" in strat.book.pending_entries
        assert strat.book.open_positions == {}

    def test_coheld_entry_books_only_our_delta(self, tmp_path: Any) -> None:
        # A sibling strategy (rate_diff) already holds 40k USD_CAD → the
        # account net is 40k at submit. Our 50k order must book from the
        # DELTA, not the account net: when net rises to 90k our fill is
        # 90k - 40k = 50k, and we book 50k, NOT the 90k account total.
        broker = _BrokerWithNetQty({"USDCAD": 40_000.0})  # co-holder baseline
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        assert strat.book.pending_entries["USD_CAD"].entry_broker_qty == (
            pytest.approx(40_000.0))
        broker.net = {"USDCAD": 90_000.0}  # our 50k fills on top
        run(strat, {"USD_CAD": tick(1.0)}, broker)
        booked = strat.book.open_positions["USD_CAD"]
        assert booked.quantity == pytest.approx(50_000.0)  # our delta, not 90k

    def test_pending_entry_occupies_a_slot(self, tmp_path: Any) -> None:
        # A submitted-but-unfilled entry must count against
        # max_concurrent_event_positions so the strategy can't over-submit
        # while the fill is in flight. cap=1: after the first entry parks
        # pending, a second confirmed event on a DIFFERENT symbol is
        # slot-blocked.
        db = make_db()
        insert_event(db)  # long USD_CAD
        affected2 = [{"instrument": "BCO_USD", "kind": "oanda",
                      "direction": "long", "reason": "second"}]
        eid2 = insert_event(db, affected=affected2)
        strat = make_strategy(
            tmp_path, db=db, provider=confirming_provider(),
            max_concurrent_event_positions=1, per_instrument_max_pct=1.0,
        )
        broker = _BrokerWithNetQty({})  # never fills → first stays pending
        prices = {"USD_CAD": tick(1.0), "BCO_USD": tick(1.0)}
        intents = run(strat, prices, broker)
        # Only ONE entry intent — the pending leg occupies the single slot.
        assert len(intents) == 1
        assert "USD_CAD" in strat.book.pending_entries
        # The second symbol was slot-blocked, its row stays CONFIRMED.
        assert get_status(db, eid2) == "CONFIRMED"

    def test_pending_entry_counts_toward_concentration_cap(
        self, tmp_path: Any,
    ) -> None:
        # A pending entry occupies its INTENDED notional against the
        # per-instrument cap (accounting view), so a second leg in the same
        # name can't blow the cap while the first is unfilled.
        strat = make_strategy(
            tmp_path, db=make_db(), per_instrument_max_pct=0.25,
        )
        # Park a pending 20k BCO_USD entry (intended).
        strat.book.record_entry(
            EventPosition(
                symbol="BCO_USD", event_id=1, entry_ts=datetime.now(UTC),
                entry_price=1.0, quantity=20_000.0, direction=1,
                stop_price=0.99, headline="pending",
            ),
            [], datetime.now(UTC),
        )
        # 25k cap @ 100k, 20k pending → 5k headroom for a second leg.
        capped = strat._concentration_capped_size(
            "BCO_USD", size=15_000.0, entry_price=1.0, equity=100_000.0,
        )
        assert capped == pytest.approx(5_000.0)

    def test_pending_entry_hidden_from_reconciler_until_filled(
        self, tmp_path: Any,
    ) -> None:
        # The reconciler-facing open_positions view values a pending entry
        # at confirmed_qty (0), NOT its intended 50k — so the phantom that
        # tripped reconciliation_failure is never exposed. After the fill it
        # converges to the broker size.
        broker = _BrokerWithNetQty({})
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        # Reconciler view: symbol present but quantity 0 (no phantom size).
        assert strat.open_positions["USD_CAD"].quantity == pytest.approx(0.0)
        # Accounting view still counts the intended magnitude for slots/caps.
        assert strat.book.accounting_positions()["USD_CAD"].quantity == (
            pytest.approx(50_000.0))
        broker.net = {"USDCAD": 50_000.0}
        strat.book.confirm_entries(broker.get_positions(), datetime.now(UTC))
        assert strat.open_positions["USD_CAD"].quantity == pytest.approx(50_000.0)

    def test_promoted_entry_is_eligible_for_time_stop(
        self, tmp_path: Any,
    ) -> None:
        # A promoted leg gets full stop/time-stop evaluation. A pending
        # entry does NOT (no exposure yet) — only after promotion.
        broker = _BrokerWithNetQty({})
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        # Backdate the pending entry so it would time-stop IF it were open.
        strat.book.pending_entries["USD_CAD"].position.entry_ts = (
            datetime.now(UTC) - timedelta(hours=5)
        )
        # Fill it: promotion carries the backdated entry_ts, so the SAME
        # tick's check_exits time-stops it.
        broker.net = {"USDCAD": 50_000.0}
        intents = run(strat, {"USD_CAD": tick(1.0)}, broker)
        # Promoted then immediately time-stopped → an exit intent this tick.
        assert [i.target_position for i in intents] == [0]
        assert strat.book.pending_entries == {}
        assert "USD_CAD" in strat.book.pending_exits

    def test_no_reentry_while_entry_pending(self, tmp_path: Any) -> None:
        # A newly-CONFIRMED event on a symbol with an entry already pending
        # must NOT submit a second order (would double the position).
        db = make_db()
        insert_event(db)  # long USD_CAD
        strat = make_strategy(
            tmp_path, db=db, provider=confirming_provider(),
            per_instrument_max_pct=1.0,
        )
        broker = _BrokerWithNetQty({})  # never fills
        run(strat, CONFIRM_PRICES, broker)
        assert "USD_CAD" in strat.book.pending_entries
        # A SECOND confirming event for the same symbol on the next tick.
        eid2 = insert_event(db)
        intents = run(strat, CONFIRM_PRICES, broker)
        assert intents == []  # no second submit
        assert get_status(db, eid2) == "CONFIRMED"  # confirmed, not traded

    def test_state_round_trip_with_pending_entries(self, tmp_path: Any) -> None:
        broker = _BrokerWithNetQty({"USDCAD": 40_000.0})
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        del strat
        # "Restart": the pending entry reloads with its baseline intact and
        # still promotes at OUR delta once the broker shows the fill.
        strat2 = make_strategy(tmp_path, db=make_db())
        entry = strat2.book.pending_entries["USD_CAD"]
        assert entry.entry_broker_qty == pytest.approx(40_000.0)
        assert entry.position.quantity == pytest.approx(50_000.0)  # intended
        assert entry.submitted_ts.tzinfo is not None
        broker.net = {"USDCAD": 90_000.0}
        strat2.book.confirm_entries(broker.get_positions(), datetime.now(UTC))
        assert strat2.book.open_positions["USD_CAD"].quantity == (
            pytest.approx(50_000.0))  # our 50k delta, not the 90k net

    def test_legacy_state_without_pending_entries_key_loads_empty(
        self, tmp_path: Any,
    ) -> None:
        # The LIVE data/event_book_state.json predates pending_entries — the
        # missing key MUST default to empty, every legacy field untouched
        # (backward compat is REQUIRED: the strategy is live with state).
        state_path = tmp_path / "event_book_state.json"
        state_path.write_text(json.dumps({
            "version": 1, "realized_pnl": -250.0, "closed_trades": 2,
            "open_positions": {
                "USD_CAD": {
                    "event_id": 7,
                    "entry_ts": datetime.now(UTC).isoformat(),
                    "entry_price": 1.0, "quantity": 1000.0,
                    "direction": 1, "stop_price": 0.99, "headline": "legacy",
                },
            },
            "pending_exits": {},
            # NO pending_entries key.
        }))
        strat = make_strategy(tmp_path, db=make_db())
        assert strat.book.pending_entries == {}
        assert strat.book.realized_pnl == pytest.approx(-250.0)
        assert strat.book.closed_trades == 2
        assert strat.book.open_positions["USD_CAD"].quantity == pytest.approx(1000.0)

    def test_symbol_in_pending_entry_and_open_refuses_to_load(
        self, tmp_path: Any,
    ) -> None:
        # Overlap fail-loud extends to pending_entries: a symbol in BOTH
        # pending_entries and open_positions is corrupt state → refuse.
        now = datetime.now(UTC)
        pos_payload = {
            "event_id": 1, "entry_ts": now.isoformat(), "entry_price": 1.0,
            "quantity": 1000.0, "direction": 1, "stop_price": 0.99,
            "headline": "dup",
        }
        (tmp_path / "event_book_state.json").write_text(json.dumps({
            "version": 1, "realized_pnl": 0.0, "closed_trades": 0,
            "open_positions": {"USD_CAD": pos_payload},
            "pending_exits": {},
            "pending_entries": {
                "USD_CAD": {
                    "position": pos_payload,
                    "submitted_ts": now.isoformat(),
                    "entry_broker_qty": 0.0, "confirmed_qty": 0.0,
                },
            },
        }))
        with pytest.raises(
            ValueError, match="appear in more than one of open_positions",
        ):
            make_strategy(tmp_path, db=make_db())

    def test_submit_confirm_window_never_flags_reconciliation(
        self, tmp_path: Any,
    ) -> None:
        # STEP 5 (option b): confirm_entries runs at the START of every
        # generate_intents tick from the freshest snapshot, so a normal fill
        # promotes within ONE strategy tick and the reconciler-facing size
        # equals the broker size on every alignment check thereafter.
        #
        # WORST CASE modeled here: the 300s alignment timer fires in the
        # window AFTER the broker fills but BEFORE the next strategy tick
        # promotes — the one tick where the book (confirmed_qty=0) lags the
        # broker (50k). That is a single orphaned_broker mismatch → streak
        # goes to 1. The next strategy tick promotes → MATCHED → streak
        # resets to 0. Since _ALIGNMENT_MISMATCH_STREAK_TO_FLAG=2, the
        # reconciliation_failure kill switch NEVER fires across the window.
        from src.runtime.live_engine import _ALIGNMENT_MISMATCH_STREAK_TO_FLAG

        broker = _BrokerWithNetQty({})
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        recon = _entry_reconciler(broker, strat)

        # Model the live-engine streak fold (live_engine._record_alignment_report).
        streak = 0

        def fold(report: Any) -> int:
            nonlocal streak
            if report is None:
                return streak
            streak = streak + 1 if report.has_mismatches else 0
            return streak

        # --- Check #1: submitted, broker not yet filled. Book 0, broker 0 →
        # both flat → MATCHED. Streak stays 0.
        fold(recon.check_alignment())
        assert streak == 0

        # --- Broker FILLS (order accepted) but confirm_entries has NOT run.
        broker.net = {"USDCAD": 50_000.0}
        # --- Check #2 lands in the window: book 0 vs broker 50k → mismatch.
        fold(recon.check_alignment())
        assert streak == 1  # one mismatch — NOT yet at the flag threshold
        assert streak < _ALIGNMENT_MISMATCH_STREAK_TO_FLAG

        # --- Next strategy tick: confirm_entries promotes to the ACTUAL fill.
        run(strat, {"USD_CAD": tick(1.0)}, broker)
        assert strat.book.open_positions["USD_CAD"].quantity == (
            pytest.approx(50_000.0))

        # --- Check #3: book 50k vs broker 50k → MATCHED → streak RESETS.
        fold(recon.check_alignment())
        assert streak == 0  # never reached 2 → reconciliation_failure never fires

    def test_stale_genuine_mismatch_still_flags(self, tmp_path: Any) -> None:
        # The freshness handling must NOT mask a GENUINE, persistent
        # mismatch: a promoted leg whose broker size diverges and STAYS
        # diverged (not a submit-window artifact) accumulates the streak and
        # WOULD flag. Proves step 5(b) doesn't blanket-suppress mismatches.
        from src.runtime.live_engine import _ALIGNMENT_MISMATCH_STREAK_TO_FLAG

        broker = _BrokerWithNetQty({})
        strat, _ = self._confirmed_entry_strat(tmp_path, broker)
        # Fill and promote to 50k.
        broker.net = {"USDCAD": 50_000.0}
        run(strat, {"USD_CAD": tick(1.0)}, broker)
        # Now the broker size DRIFTS to 30k and stays there (a real, ongoing
        # divergence — e.g. a partial external close) while the book holds
        # 50k. Every alignment check is a size_mismatch → the streak climbs.
        broker.net = {"USDCAD": 30_000.0}
        recon = _entry_reconciler(broker, strat)
        streak = 0
        for _ in range(_ALIGNMENT_MISMATCH_STREAK_TO_FLAG):
            report = recon.check_alignment()
            assert report is not None and report.has_mismatches
            streak = streak + 1 if report.has_mismatches else 0
        assert streak >= _ALIGNMENT_MISMATCH_STREAK_TO_FLAG  # genuine → flags


def _entry_reconciler(broker: Any, strat: EventDrivenStrategy) -> Any:
    """A PositionReconciler wired to the event strategy's book for the
    window/alignment test (step 5)."""
    from src.portfolio.reconciler import PositionReconciler

    return PositionReconciler(
        broker,  # type: ignore[arg-type]
        SimpleNamespace(submit_intent=lambda *a, **k: None),  # type: ignore[arg-type]
        SimpleNamespace(get_current_position=lambda sid: None),  # type: ignore[arg-type]
        strategies=[strat],
    )


# =============================================================================
# Cross-asset entry gate (CL-6mzn, gating half)
# =============================================================================


def _ca_row() -> dict[str, Any]:
    return {"id": 9, "headline": "Hormuz closure threat"}


def _ca_assessment() -> dict[str, Any]:
    return {"urgency": 8, "confidence": 0.8, "affected": [
        {"instrument": "USD_CAD", "kind": "fx", "direction": "short",
         "reason": "oil currency"},
    ]}


_CA_PRICES = {"USD_CAD": {"bid": 1.3999, "ask": 1.4001}}


class TestCrossAssetGate:
    def _enter(self, tmp_path: Any, ca: Any, **cfg: Any):
        strat = make_strategy(tmp_path, cross_asset_gate_enabled=True, **cfg)
        return strat._enter_confirmed(
            _ca_row(), _ca_assessment(), _CA_PRICES, 100_000.0,
            datetime.now(UTC), cross_asset=ca,
        )

    def test_contradictory_read_vetoes_entries(self, tmp_path: Any) -> None:
        intents, entered, skipped = self._enter(
            tmp_path, SimpleNamespace(confirmed=False))
        assert intents == [] and entered == []
        assert skipped == [("USD_CAD", "cross_asset_veto")]

    def test_confirming_read_allows_entries(self, tmp_path: Any) -> None:
        intents, entered, skipped = self._enter(
            tmp_path, SimpleNamespace(confirmed=True))
        assert len(intents) == 1 and len(entered) == 1

    def test_missing_read_allows_by_default(self, tmp_path: Any) -> None:
        # confirmed=None (no data) and cross_asset=None (layer off) both pass
        # under the default fail-open-on-missing posture. Distinct state
        # paths — each iteration must start with an empty book.
        for n, ca in enumerate((SimpleNamespace(confirmed=None), None)):
            strat = make_strategy(
                tmp_path, cross_asset_gate_enabled=True,
                event_book_state_path=str(tmp_path / f"book_{n}.json"),
            )
            i, _e, _s = strat._enter_confirmed(
                _ca_row(), _ca_assessment(), _CA_PRICES, 100_000.0,
                datetime.now(UTC), cross_asset=ca,
            )
            assert len(i) == 1, f"blocked unexpectedly for {ca!r}"

    def test_missing_read_blocks_in_strict_mode(self, tmp_path: Any) -> None:
        intents, entered, skipped = self._enter(
            tmp_path, None, cross_asset_block_on_missing=True)
        assert intents == []
        assert skipped == [("USD_CAD", "cross_asset_no_data")]

    def test_gate_disabled_ignores_contradiction(self, tmp_path: Any) -> None:
        strat = make_strategy(tmp_path)  # gate off (default)
        assert strat.config.cross_asset_gate_enabled is False
        intents, entered, _ = strat._enter_confirmed(
            _ca_row(), _ca_assessment(), _CA_PRICES, 100_000.0,
            datetime.now(UTC), cross_asset=SimpleNamespace(confirmed=False),
        )
        assert len(intents) == 1  # contradiction ignored when gate off


# =============================================================================
# Urgency vocabulary (CL-ikz2) — event intents speak the coordinator's enum
# =============================================================================


class TestUrgencyVocabulary:
    def test_event_exit_outranks_normal_intent_in_coordinator(
        self, tmp_path: Any,
    ) -> None:
        """Regression (review §6.1.2 / §9 item 1): event exits emitted
        urgency="high", a value the coordinator's rank map did not know —
        rank 0, below "normal" — so an event exit netted against another
        strategy's intent could NEVER escalate the aggregate. Exits now
        emit the canonical Urgency.URGENT and must win aggregation."""
        from src.execution.paper_broker import PaperBroker
        from src.portfolio.coordinator import PortfolioCoordinator

        # A real event exit intent off the time stop.
        strat = make_strategy(tmp_path, db=make_db())
        seed_position(strat, hours_ago=5.0)  # > 4h default time stop
        exit_intent = run(strat, {"USD_CAD": tick(1.0)})[0]
        assert exit_intent.target_position == 0
        assert exit_intent.urgency == Urgency.URGENT.value

        coord = PortfolioCoordinator(
            strategies=[SimpleNamespace(id="event_driven"),
                        SimpleNamespace(id="other")],
            oms=SimpleNamespace(),  # aggregation never touches the OMS
            broker=PaperBroker(),
            state=SimpleNamespace(
                record_reallocation=lambda *a, **k: None,
                record_portfolio_order=lambda *a, **k: None,
            ),
        )
        normal = OrderIntent(
            strategy_id="other", symbol=exit_intent.symbol,
            target_position=100.0, urgency=Urgency.NORMAL.value,
        )
        # Aggregation keys by CANONICAL symbol since the CL-8cw1 P0 fix
        # (USD_CAD → USDCAD) — one aggregate row per economic pair.
        agg = coord._aggregate_by_symbol([normal, exit_intent])
        canon = EventDrivenStrategy._norm_symbol(exit_intent.symbol)
        assert agg[canon]["urgency"] == "urgent"

    def test_event_entry_uses_canonical_urgent(self, tmp_path: Any) -> None:
        db = make_db()
        insert_event(db)
        strat = make_strategy(tmp_path, db=db, provider=confirming_provider())
        intents = run(strat, CONFIRM_PRICES)
        assert [i.urgency for i in intents] == [Urgency.URGENT.value]
