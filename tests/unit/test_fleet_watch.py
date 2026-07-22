"""Tests for the fleet watchdog decisions (CL-fmqp)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.monitoring.fleet_watch import (
    decide_fleet_alerts,
    decide_x_staleness_alert,
    parse_status,
)

NOW = datetime(2026, 7, 22, 19, 30, tzinfo=UTC)

STATUS = """  ✓ engine (pid 65695)
  ✗ x_monitor NOT RUNNING
  ✓ health_watch (pid 999)
"""


def test_parse_status():
    assert parse_status(STATUS) == {
        "engine": True, "x_monitor": False, "health_watch": True,
    }


def test_down_pages_once_then_silent():
    current = {"engine": True, "x_monitor": False}
    alerts, down = decide_fleet_alerts(current, {}, NOW)
    assert len(alerts) == 1 and "DOWN: x_monitor" in alerts[0]
    # Next cycle, still down: no re-page.
    alerts2, down2 = decide_fleet_alerts(current, down, NOW)
    assert alerts2 == [] and "x_monitor" in down2


def test_recovery_notice_with_duration():
    down = {"x_monitor": (NOW - timedelta(minutes=34)).isoformat()}
    alerts, new_down = decide_fleet_alerts(
        {"engine": True, "x_monitor": True}, down, NOW,
    )
    assert len(alerts) == 1
    assert "RECOVERED: x_monitor" in alerts[0] and "34min" in alerts[0]
    assert new_down == {}


def test_watchdog_never_reports_itself():
    alerts, _ = decide_fleet_alerts({"health_watch": False}, {}, NOW)
    assert alerts == []


def test_vanished_daemon_stays_tracked():
    down = {"x_monitor": NOW.isoformat()}
    _, new_down = decide_fleet_alerts({"engine": True}, down, NOW)
    assert "x_monitor" in new_down  # absent from status ≠ recovered


def test_x_staleness_onset_recovery_and_dedup():
    fresh = NOW - timedelta(minutes=30)
    old = NOW - timedelta(hours=5)
    # onset pages
    msg, stale = decide_x_staleness_alert(old, False, NOW)
    assert stale and msg and "STALE" in msg
    # still stale: silent
    msg2, stale2 = decide_x_staleness_alert(old, True, NOW)
    assert stale2 and msg2 is None
    # recovery notices
    msg3, stale3 = decide_x_staleness_alert(fresh, True, NOW)
    assert not stale3 and msg3 and "recovered" in msg3
    # healthy + was healthy: silent
    msg4, stale4 = decide_x_staleness_alert(fresh, False, NOW)
    assert not stale4 and msg4 is None


def test_x_staleness_empty_feed_is_stale():
    msg, stale = decide_x_staleness_alert(None, False, NOW)
    assert stale and msg is not None and "never" in msg
