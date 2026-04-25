"""Unit tests — data.economic_calendar (CL-ahdu)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

from src.data.economic_calendar import (
    BlackoutAction,
    BlackoutEvaluator,
    BlackoutPolicy,
    EconomicCalendar,
    EconomicEvent,
    SeverityTier,
    load_calendar_from_yaml,
)

T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _evt(
    hours_from_t0: float,
    event_type: str = "NFP",
    tier: SeverityTier = SeverityTier.TIER_1,
    currency: str | None = None,
) -> EconomicEvent:
    return EconomicEvent(
        ts=T0 + timedelta(hours=hours_from_t0),
        event_type=event_type,
        severity_tier=tier,
        description=event_type,
        currency=currency,
    )


# =============================================================================
# Event validation
# =============================================================================


class TestEvent:
    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(AssertionError):
            EconomicEvent(
                ts=datetime(2026, 5, 1, 12, 0),  # no tz
                event_type="NFP",
                severity_tier=SeverityTier.TIER_1,
            )

    def test_tz_aware_accepted(self) -> None:
        e = EconomicEvent(
            ts=T0, event_type="NFP", severity_tier=SeverityTier.TIER_1,
        )
        assert e.event_type == "NFP"


# =============================================================================
# Calendar
# =============================================================================


class TestCalendar:
    def test_upcoming_filters_by_horizon(self) -> None:
        cal = EconomicCalendar([
            _evt(hours_from_t0=2),    # in horizon
            _evt(hours_from_t0=72),   # outside 48h horizon
            _evt(hours_from_t0=-1),   # in past
        ])
        upcoming = cal.upcoming(now=T0, horizon_hours=48.0)
        assert len(upcoming) == 1

    def test_upcoming_filters_by_tier(self) -> None:
        cal = EconomicCalendar([
            _evt(hours_from_t0=1, tier=SeverityTier.TIER_1),
            _evt(hours_from_t0=2, tier=SeverityTier.TIER_2),
        ])
        only_t1 = cal.upcoming(
            now=T0, horizon_hours=48.0, tiers=[SeverityTier.TIER_1],
        )
        assert len(only_t1) == 1
        assert only_t1[0].severity_tier == SeverityTier.TIER_1

    def test_upcoming_filters_by_currency(self) -> None:
        cal = EconomicCalendar([
            _evt(hours_from_t0=1, currency="USD"),
            _evt(hours_from_t0=2, currency="EUR"),
            _evt(hours_from_t0=3, currency=None),  # global
        ])
        usd_only = cal.upcoming(now=T0, horizon_hours=48.0, currency="USD")
        # USD event AND global event (currency=None) match.
        assert len(usd_only) == 2

    def test_events_sorted_by_ts(self) -> None:
        cal = EconomicCalendar()
        cal.add(_evt(hours_from_t0=10))
        cal.add(_evt(hours_from_t0=2))
        cal.add(_evt(hours_from_t0=5))
        ts_order = [e.ts for e in cal.events]
        assert ts_order == sorted(ts_order)


# =============================================================================
# Policy validation
# =============================================================================


class TestBlackoutPolicy:
    def test_negative_window_rejected(self) -> None:
        with pytest.raises(AssertionError):
            BlackoutPolicy(tier1_size_down_hours=-1.0)

    def test_zero_disables_level(self) -> None:
        # Tier 3 disabled by default — windows_for returns 0s.
        policy = BlackoutPolicy()
        windows = policy.windows_for(SeverityTier.TIER_3)
        assert windows == (0.0, 0.0, 0.0)

    def test_windows_for_tier1_defaults(self) -> None:
        policy = BlackoutPolicy()
        size_down, pause, exit_flat = policy.windows_for(SeverityTier.TIER_1)
        assert size_down == 24
        assert pause == 1.0
        assert exit_flat == 30


# =============================================================================
# Evaluator
# =============================================================================


class TestEvaluator:
    def test_full_size_when_no_upcoming(self) -> None:
        cal = EconomicCalendar()
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        assert decision.action == BlackoutAction.FULL_SIZE

    def test_full_size_when_event_outside_window(self) -> None:
        # Event 30 hours out (beyond default 24h tier-1 window).
        cal = EconomicCalendar([_evt(hours_from_t0=30)])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        assert decision.action == BlackoutAction.FULL_SIZE

    def test_size_down_within_24h_window(self) -> None:
        # Event 12h out — inside 24h size-down but outside 1h pause.
        cal = EconomicCalendar([_evt(hours_from_t0=12)])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        assert decision.action == BlackoutAction.SIZE_DOWN_50PCT
        assert decision.triggering_event is not None
        assert decision.minutes_until_event is not None
        assert decision.minutes_until_event == pytest.approx(720.0)

    def test_pause_within_1h_window(self) -> None:
        # 30 min out — inside 1h pause but outside 30min exit-flat.
        cal = EconomicCalendar([_evt(hours_from_t0=0.5)])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        # 30 min = 30 min, so technically AT the exit_flat boundary — verify this is exit_flat.
        # Default exit_flat_minutes=30 so minutes_until=30 should hit exit_flat first.
        assert decision.action == BlackoutAction.EXIT_FLAT

    def test_pause_strictly_inside(self) -> None:
        # 45 min out — inside 1h pause but outside 30min exit-flat.
        cal = EconomicCalendar([_evt(hours_from_t0=0.75)])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        assert decision.action == BlackoutAction.PAUSE_NEW_ENTRIES

    def test_exit_flat_within_30min_window(self) -> None:
        # 15 min out — inside 30min exit-flat (most restrictive).
        cal = EconomicCalendar([_evt(hours_from_t0=0.25)])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        assert decision.action == BlackoutAction.EXIT_FLAT

    def test_most_restrictive_wins_across_events(self) -> None:
        # One event 12h out (size-down), another 15min out (exit-flat) —
        # exit-flat must win.
        cal = EconomicCalendar([
            _evt(hours_from_t0=12),
            _evt(hours_from_t0=0.25),
        ])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        assert decision.action == BlackoutAction.EXIT_FLAT

    def test_tier_3_does_not_trigger_by_default(self) -> None:
        cal = EconomicCalendar([
            _evt(hours_from_t0=0.1, tier=SeverityTier.TIER_3),
        ])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0)
        # All tier-3 windows are 0 by default.
        assert decision.action == BlackoutAction.FULL_SIZE

    def test_currency_filter_excludes_unrelated_events(self) -> None:
        # JPY event in 12h, but we ask for USD currency only.
        cal = EconomicCalendar([
            _evt(hours_from_t0=12, currency="JPY"),
        ])
        ev = BlackoutEvaluator(cal)
        decision = ev.evaluate(now=T0, currency="USD")
        # JPY event filtered out — no upcoming for USD-relevant trades.
        assert decision.action == BlackoutAction.FULL_SIZE


# =============================================================================
# YAML loader
# =============================================================================


class TestLoadFromYaml:
    def test_load_sample_calendar(self, tmp_path: Path) -> None:
        yaml_text = dedent("""
            events:
              - ts: 2026-05-02T12:30:00Z
                event_type: NFP
                severity_tier: 1
                description: US NFP
                currency: USD
              - ts: 2026-05-07T18:00:00Z
                event_type: FOMC
                severity_tier: 1
                currency: USD
        """)
        p = tmp_path / "cal.yaml"
        p.write_text(yaml_text)
        cal = load_calendar_from_yaml(p)
        assert len(cal) == 2
        assert cal.events[0].event_type == "NFP"
        assert cal.events[0].currency == "USD"
        assert cal.events[1].event_type == "FOMC"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_calendar_from_yaml(tmp_path / "missing.yaml")

    def test_malformed_missing_events_key(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("foo: bar\n")
        with pytest.raises(ValueError, match="events"):
            load_calendar_from_yaml(p)

    def test_shipped_sample_calendar_parses(self) -> None:
        # Sanity: the file we ship under configs/ parses correctly.
        cal = load_calendar_from_yaml(Path("configs/economic_calendar.yaml"))
        assert len(cal) > 0


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_decision_to_dict(self) -> None:
        cal = EconomicCalendar([_evt(hours_from_t0=12)])
        ev = BlackoutEvaluator(cal)
        d = ev.evaluate(now=T0).to_dict()
        for key in ("action", "triggering_event", "minutes_until_event", "reason"):
            assert key in d

    def test_event_to_dict(self) -> None:
        e = _evt(hours_from_t0=12, currency="USD")
        d = e.to_dict()
        assert d["event_type"] == "NFP"
        assert d["currency"] == "USD"
