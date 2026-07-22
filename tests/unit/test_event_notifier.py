"""Unit tests — EventNotifier (CL-ikz2).

The operator-alert formatting extracted from EventDrivenStrategy per the
2026-07-21 structural review (§6.1.1, §9 item 2). Rendering CONTENT is
covered end-to-end through the strategy in test_event_driven.py
(TestAlerts / enrichment classes); here we cover the notifier as a unit:
construction, dispatch, the price-resolver seam, the idea-ledger stamp,
and the strategy → notifier delegation (injected recorder).

No live DB / network: sqlite in-memory engines and a patched
notify_operator only.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from src.events.event_notifier import EventNotifier
from src.strategies.event_driven import EventDrivenConfig, EventDrivenStrategy

# =============================================================================
# Fixtures / helpers
# =============================================================================


def make_notifier(db: Any = None, **overrides: Any) -> EventNotifier:
    kwargs: dict[str, Any] = {
        "event_risk_pct": 0.005,
        "event_stop_pct": 0.01,
        "event_max_holding_hours": 4.0,
        "confirm_window_max_minutes": 120,
        "db_engine": db,
    }
    kwargs.update(overrides)
    return EventNotifier(**kwargs)


@pytest.fixture()
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, int]]:
    calls: list[tuple[str, str, int]] = []

    def _capture(title: str, message: str, priority: int = 0) -> None:
        calls.append((title, message, priority))

    monkeypatch.setattr("src.events.event_notifier.notify_operator", _capture)
    return calls


def _row(headline: str = "Pipeline explosion", minutes_ago: float = 60) -> dict[str, Any]:
    seen = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    return {"id": 7, "headline": headline, "seen_at": seen}


# =============================================================================
# Dispatch + body
# =============================================================================


class TestAlertConfirmed:
    def test_body_and_priority(self, sent: list[tuple[str, str, int]]) -> None:
        notifier = make_notifier()
        notifier.alert_confirmed(
            _row(), {"urgency": 8, "confidence": 0.9},
            entered=[("USD_CAD", "long", "50000", 1.0, "petro-fx")],
            skipped=[("XAU_USD", "concentration_cap")],
            prices={"XAU_USD": {"bid": 2400.0, "ask": 2400.0}},
        )
        assert len(sent) == 1
        title, message, priority = sent[0]
        assert title == "Event confirmed"
        assert priority == 1
        assert "Headline: Pipeline explosion" in message
        assert "Age: 1h since first seen" in message
        assert "Trade: USD_CAD long (50000 units) @ 1" in message
        assert "  Why: petro-fx" in message
        # Skipped line resolves the mid from the tick dict (no resolver).
        assert "Skipped: XAU_USD (concentration_cap) @ 2400" in message
        assert "Risk: 0.50% of equity per trade" in message
        assert "Stop: 1.00% from entry" in message
        assert "Time stop: 4h" in message
        assert "Urgency: 8/10" in message
        assert "Confidence: 0.90" in message

    def test_injected_price_resolver_wins(
        self, sent: list[tuple[str, str, int]],
    ) -> None:
        notifier = make_notifier(price_resolver=lambda sym, prices, now: 1.25)
        notifier.alert_confirmed(
            _row(), {"urgency": 8, "confidence": 0.9},
            entered=[], skipped=[("USD_CAD", "no_price")],
        )
        assert "Skipped: USD_CAD (no_price) @ 1.25" in sent[0][1]

    def test_dispatch_failure_swallowed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*a: Any, **k: Any) -> None:
            raise RuntimeError("telegram down")

        monkeypatch.setattr("src.events.event_notifier.notify_operator", _boom)
        make_notifier().alert_confirmed(
            _row(), {"urgency": 8, "confidence": 0.9}, entered=[], skipped=[],
        )  # must not raise — alerting is never allowed to break the strategy


class TestAlertExpired:
    def test_body_and_priority(self, sent: list[tuple[str, str, int]]) -> None:
        notifier = make_notifier(confirm_window_max_minutes=120)
        notifier.alert_expired(_row(minutes_ago=300), urgency=9, confidence=0.9)
        title, message, priority = sent[0]
        assert title == "Event expired unconfirmed"
        assert priority == 0
        assert "Age: 5h since first seen" in message
        assert "Urgency: 9/10" in message
        assert "No market confirmation within 120min" in message
        assert "No trade taken" in message


# =============================================================================
# Idea-ledger cross-asset stamp (events-domain DB write)
# =============================================================================


def _ideas_db() -> Any:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE trade_ideas ("
            "id INTEGER PRIMARY KEY, geo_event_id INTEGER, notes TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO trade_ideas (geo_event_id, notes) VALUES (7, NULL)"
        ))
    return engine


class _Move:
    def __init__(self, instrument: str, pct: float, agrees: bool) -> None:
        self.instrument = instrument
        self.actual_move_pct = pct
        self.agrees = agrees


class _CAResult:
    def __init__(self, confirmed: Any, details: list[Any]) -> None:
        self.confirmed = confirmed
        self.details = details


class TestStampCrossAssetOnIdeas:
    def test_stamps_notes_once(self) -> None:
        db = _ideas_db()
        notifier = make_notifier(db=db)
        result = _CAResult(True, [_Move("BCO_USD", 1.8, True)])
        notifier.stamp_cross_asset_on_ideas(7, result)
        with db.connect() as conn:
            notes = conn.execute(
                text("SELECT notes FROM trade_ideas WHERE geo_event_id = 7"),
            ).scalar()
        assert notes == "cross-asset: confirms 1/1 (BCO_USD +1.8%)"

        # Idempotent-ish: a second stamp must not double-append.
        notifier.stamp_cross_asset_on_ideas(7, result)
        with db.connect() as conn:
            notes2 = conn.execute(
                text("SELECT notes FROM trade_ideas WHERE geo_event_id = 7"),
            ).scalar()
        assert notes2 == notes

    def test_no_db_or_unknown_result_is_noop(self) -> None:
        make_notifier(db=None).stamp_cross_asset_on_ideas(
            7, _CAResult(True, [_Move("BCO_USD", 1.8, True)]),
        )  # no db handle — silently skipped
        db = _ideas_db()
        make_notifier(db=db).stamp_cross_asset_on_ideas(7, _CAResult(None, []))
        with db.connect() as conn:
            notes = conn.execute(text("SELECT notes FROM trade_ideas")).scalar()
        assert notes is None  # unknown read never stamps

    def test_db_error_swallowed(self) -> None:
        db = sa.create_engine("sqlite://")  # no trade_ideas table at all
        make_notifier(db=db).stamp_cross_asset_on_ideas(
            7, _CAResult(True, [_Move("BCO_USD", 1.8, True)]),
        )  # must not raise — annotation only


# =============================================================================
# Strategy → notifier delegation (injected recorder)
# =============================================================================


class _RecordingNotifier:
    """Duck-typed EventNotifier stand-in proving the injection seam."""

    def __init__(self) -> None:
        self.confirmed: list[Any] = []
        self.expired: list[Any] = []
        self.stamped: list[Any] = []

    def alert_confirmed(self, row: Any, assessment: Any, entered: Any,
                        skipped: Any, **kwargs: Any) -> None:
        self.confirmed.append((row, entered, skipped))

    def alert_expired(self, row: Any, urgency: int, confidence: float) -> None:
        self.expired.append((row, urgency))

    def stamp_cross_asset_on_ideas(self, geo_event_id: Any, result: Any) -> None:
        self.stamped.append(geo_event_id)


class TestStrategyDelegation:
    def test_confirmed_flow_calls_injected_notifier(self, tmp_path: Any) -> None:
        from tests.unit.test_event_driven import (
            CONFIRM_PRICES,
            FakeBroker,
            confirming_provider,
            insert_event,
            make_db,
        )

        db = make_db()
        eid = insert_event(db)
        recorder = _RecordingNotifier()
        strat = EventDrivenStrategy(
            EventDrivenConfig(
                event_book_state_path=str(tmp_path / "book.json"),
            ),
            data_provider=confirming_provider(),
            db_engine=db,
            notifier=recorder,  # type: ignore[arg-type]
        )
        intents = asyncio.run(strat.generate_intents(CONFIRM_PRICES, FakeBroker()))
        assert len(intents) == 1
        assert len(recorder.confirmed) == 1
        _row_arg, entered, _skipped = recorder.confirmed[0]
        assert entered[0][0] == "USD_CAD"
        assert recorder.stamped == [eid]
        assert recorder.expired == []

    def test_default_notifier_is_wired_from_config(self, tmp_path: Any) -> None:
        strat = EventDrivenStrategy(
            EventDrivenConfig(
                event_risk_pct=0.007,
                event_book_state_path=str(tmp_path / "book.json"),
            ),
        )
        assert isinstance(strat.notifier, EventNotifier)
        assert strat.notifier._event_risk_pct == pytest.approx(0.007)
        # Price resolution stays defined in ONE place — the strategy's.
        assert strat.notifier._price_resolver == strat._current_price
