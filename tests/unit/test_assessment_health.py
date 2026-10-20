"""CL-x40b: availability failures must not masquerade as quiet research."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.events.impact_agent import AssessmentResult, EventImpactAgent
from src.monitoring.assessment_health import (
    AssessmentHealth,
    decide_assessment_alert,
    read_assessment_health,
    record_assessment_health,
)
from src.research.llm.claude_code import ClaudeAuthenticationError

NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)


def result(status: str, reason: str | None = None) -> AssessmentResult:
    return AssessmentResult(1, "fixture", None, status, {}, failure_reason=reason)


def test_auth_pages_once_and_only_later_assessment_recovers() -> None:
    failed = AssessmentHealth(NOW, last_auth_failure_at=NOW)
    message, state = decide_assessment_alert(failed, NOW, True, {}, NOW)
    assert message is not None and "Claude CLI login expired" in message
    assert decide_assessment_alert(failed, NOW, True, state, NOW)[0] is None
    later = NOW + timedelta(minutes=5)
    # Empty results and missing health evidence aren't successful recovery.
    assert decide_assessment_alert(None, None, True, state, later) == (None, state)
    assert decide_assessment_alert(failed, None, True, state, later) == (None, state)
    healthy = AssessmentHealth(later, last_assessed_at=later, last_auth_failure_at=NOW)
    assert decide_assessment_alert(healthy, None, False, state, later) == (None, state)
    message, state = decide_assessment_alert(healthy, None, True, state, later)
    assert message is not None and "recovered" in message
    assert state == {}
    assert decide_assessment_alert(healthy, None, True, state, later) == (None, {})


@given(st.integers(min_value=0, max_value=365 * 24))
def test_quiet_queue_never_pages_even_with_old_success(hours: int) -> None:
    old = NOW - timedelta(hours=hours)
    health = AssessmentHealth(old, last_assessed_at=old)
    assert decide_assessment_alert(health, None, True, {}, NOW) == (None, {})


def test_stall_requires_old_backlog_and_old_success() -> None:
    old = NOW - timedelta(hours=3)
    health = AssessmentHealth(old, last_assessed_at=old)
    assert decide_assessment_alert(health, NOW, True, {}, NOW) == (None, {})
    assert decide_assessment_alert(health, old, False, {}, NOW) == (None, {})
    message, state = decide_assessment_alert(health, old, True, {}, NOW)
    assert message is not None and "stalled" in message
    assert decide_assessment_alert(health, old, True, state, NOW) == (None, state)
    recent = AssessmentHealth(NOW, last_assessed_at=NOW)
    assert decide_assessment_alert(recent, old, True, {}, NOW) == (None, {})
    # Missing status file doesn't hide a known old backlog.
    assert decide_assessment_alert(None, old, True, {}, NOW)[0] is not None


def test_health_file_preserves_auth_through_quiet_batches() -> None:
    record_assessment_health([result("NEW", "subscription_auth_unavailable")], now=NOW)
    later = NOW + timedelta(minutes=5)
    record_assessment_health([result("DISMISSED")], now=later)
    health = read_assessment_health()
    assert health == AssessmentHealth(later, None, NOW)
    record_assessment_health([result("ASSESSED")], now=later)
    assert read_assessment_health() == AssessmentHealth(later, later, NOW)


def test_corrupt_file_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"updated_at": "not a timestamp"}')
    monkeypatch.setenv("ASSESSMENT_HEALTH_STATE", str(path))
    assert read_assessment_health() is None


def test_auth_failure_keeps_event_queued_and_is_machine_readable() -> None:
    client = Mock()
    client.complete.side_effect = ClaudeAuthenticationError("login unavailable")
    agent = EventImpactAgent(engine=Mock(), client=client)
    outcome = agent.assess_row({"id": 1, "headline": "fixture", "theme": None})
    assert outcome.status == "NEW"
    assert outcome.failure_reason == "subscription_auth_unavailable"
    assert "queued" in outcome.summary_line()
    assert "DISMISSED" not in outcome.summary_line()


def test_watchdog_wires_real_status_and_transition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import health_watch as watch

    monkeypatch.setattr(watch, "STATE_PATH", tmp_path / "watch.json")
    monkeypatch.setattr(watch, "_fleet_status_output", lambda: "  ✓ engine (pid 1)\n")
    monkeypatch.setattr(watch, "_engine_system_status", lambda: {"oms_wired": True, "oms_halted": False})
    monkeypatch.setattr(watch, "_newest_x_event", lambda _: NOW)
    monkeypatch.setattr(watch, "_assessment_backlog", lambda _: (True, NOW))
    engine = Mock()
    monkeypatch.setattr("sqlalchemy.create_engine", lambda _: engine)
    monkeypatch.setattr("src.data.db_env.build_db_url", lambda: "sqlite://")
    notify = Mock(return_value=SimpleNamespace(any_succeeded=True))
    monkeypatch.setattr("src.research.notifications.notify_operator", notify)
    record_assessment_health([result("NEW", "subscription_auth_unavailable")], now=NOW)
    assert watch.run_once(now=NOW) == 1
    assert watch.run_once(now=NOW) == 0
    later = NOW + timedelta(minutes=5)
    record_assessment_health([result("ASSESSED")], now=later)
    assert watch.run_once(now=later) == 1
    assert watch.run_once(now=later) == 0
    assert notify.call_count == 2
    assert "Claude CLI login expired" in notify.call_args_list[0].args[1]
    assert "recovered" in notify.call_args_list[1].args[1]
    assert engine.dispose.call_count == 4
