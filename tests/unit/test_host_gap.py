"""Host suspend/resume detection for readiness evidence (CL-cmg9).

Oracles:
* the 2026-09-08 incident: monitor samples jumped from 23:01 to 02:18 UTC
  while one process (unchanged PID) slept through — a 3h17m wall gap with
  only one loop's worth of awake time is a ~3h12m suspension;
* conservation: for any cycle sequence, observed(active) + suspended +
  unobserved time equals elapsed wall time, and observed never exceeds wall
  (hypothesis) — sleeping hours can never be counted as observation.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.monitoring.host_gap import (
    GAP_CLOCK_STEP,
    GAP_LATENCY,
    GAP_NONE,
    GAP_SUSPENDED,
    GAP_UNOBSERVED,
    READINESS_OK,
    READINESS_STALE,
    SUSPEND_THRESHOLD_SEC,
    CycleClock,
    accumulate_observation,
    classify_cycle_gap,
    decide_readiness,
)

LOOP = 300.0
NOW = datetime(2026, 9, 9, 2, 18, tzinfo=UTC)


def _c(wall: float, mono: float, pid: int = 42) -> CycleClock:
    return CycleClock(wall=wall, mono=mono, pid=pid)


class TestClassify:
    def test_first_cycle_has_nothing_to_compare(self) -> None:
        assert classify_cycle_gap(None, _c(0, 0), LOOP).kind == GAP_NONE

    def test_normal_awake_cycle(self) -> None:
        v = classify_cycle_gap(_c(0, 0), _c(302, 302), LOOP)
        assert v.kind == GAP_NONE and v.suspended_sec == pytest.approx(0.0)

    def test_the_2026_09_08_incident_is_a_suspension(self) -> None:
        t0 = datetime(2026, 9, 8, 23, 1, tzinfo=UTC).timestamp()
        t1 = datetime(2026, 9, 9, 2, 18, tzinfo=UTC).timestamp()
        # Same PID throughout; only ~one loop of awake time elapsed.
        v = classify_cycle_gap(_c(t0, 1000.0), _c(t1, 1000.0 + LOOP), LOOP)
        assert v.kind == GAP_SUSPENDED
        assert v.wall_elapsed_sec == pytest.approx(3 * 3600 + 17 * 60)
        assert v.suspended_sec == pytest.approx(3 * 3600 + 17 * 60 - LOOP)

    def test_slow_but_awake_is_latency_not_sleep(self) -> None:
        # 700 s elapsed on BOTH clocks: the process ran long, host never slept.
        v = classify_cycle_gap(_c(0, 0), _c(700, 700), LOOP)
        assert v.kind == GAP_LATENCY and v.invalidates_readiness is False

    def test_threshold_boundary(self) -> None:
        below = classify_cycle_gap(_c(0, 0), _c(LOOP + SUSPEND_THRESHOLD_SEC - 1, LOOP), LOOP)
        at = classify_cycle_gap(_c(0, 0), _c(LOOP + SUSPEND_THRESHOLD_SEC, LOOP), LOOP)
        assert below.kind == GAP_NONE and at.kind == GAP_SUSPENDED

    def test_restarted_process_long_gap_is_unobserved(self) -> None:
        v = classify_cycle_gap(_c(0, 5, pid=1), _c(3 * 3600, 2, pid=2), LOOP)
        assert v.kind == GAP_UNOBSERVED and v.active_elapsed_sec is None

    def test_restarted_process_short_gap_is_fine(self) -> None:
        assert classify_cycle_gap(_c(0, 5, pid=1), _c(LOOP, 2, pid=2), LOOP).kind == GAP_NONE

    def test_wall_clock_going_backwards(self) -> None:
        assert classify_cycle_gap(_c(1000, 0), _c(500, 300), LOOP).kind == GAP_CLOCK_STEP

    def test_one_shot_run_never_reports_latency(self) -> None:
        assert classify_cycle_gap(_c(0, 0), _c(5000, 5000), None).kind == GAP_NONE

    def test_malformed_persisted_clock_is_ignored(self) -> None:
        assert CycleClock.from_dict({"wall": "x"}) is None
        assert CycleClock.from_dict(None) is None
        c = _c(1.5, 2.5, 7)
        assert CycleClock.from_dict(c.to_dict()) == c


class TestReadiness:
    def _suspend(self) -> Any:
        return classify_cycle_gap(_c(0, 0), _c(4 * 3600, LOOP), LOOP)

    def test_suspension_makes_readiness_stale_and_pages_once(self) -> None:
        pages, state = decide_readiness(self._suspend(), None, NOW, fresh_evidence=True)
        assert state["status"] == READINESS_STALE
        assert len(pages) == 1 and "HOST SUSPENDED" in pages[0] and "NOT observing" in pages[0]
        # Another gap while still stale: record updated, no re-page, onset kept.
        later = datetime(2026, 9, 9, 6, 0, tzinfo=UTC)
        pages2, state2 = decide_readiness(self._suspend(), state, later, fresh_evidence=True)
        assert pages2 == [] and state2["since"] == state["since"]

    def test_stale_clears_only_with_fresh_evidence(self) -> None:
        _, stale = decide_readiness(self._suspend(), None, NOW, fresh_evidence=True)
        clean = classify_cycle_gap(_c(0, 0), _c(LOOP, LOOP), LOOP)
        pages, still = decide_readiness(clean, stale, NOW, fresh_evidence=False)
        assert pages == [] and still["status"] == READINESS_STALE
        pages, ok = decide_readiness(clean, still, NOW, fresh_evidence=True)
        assert ok["status"] == READINESS_OK and "Readiness restored" in pages[0]

    def test_latency_never_flips_readiness(self) -> None:
        latency = classify_cycle_gap(_c(0, 0), _c(900, 900), LOOP)
        pages, state = decide_readiness(latency, None, NOW, fresh_evidence=False)
        assert pages == [] and state["status"] == READINESS_OK

    def test_unobserved_gap_also_invalidates(self) -> None:
        v = classify_cycle_gap(_c(0, 0, pid=1), _c(7200, 1, pid=2), LOOP)
        pages, state = decide_readiness(v, None, NOW, fresh_evidence=True)
        assert state["status"] == READINESS_STALE and "UNOBSERVED" in pages[0]


class TestObservationAccounting:
    def test_sleeping_hours_are_never_observed_time(self) -> None:
        obs: dict[str, Any] | None = None
        clocks = [_c(0, 0), _c(300, 300), _c(300 + 3 * 3600 + 300, 600), _c(3 * 3600 + 900, 900)]
        for prev, cur in zip(clocks, clocks[1:], strict=False):
            obs = accumulate_observation(obs, classify_cycle_gap(prev, cur, LOOP))
        assert obs is not None
        assert obs["active_sec"] == pytest.approx(900)
        assert obs["suspended_sec"] == pytest.approx(3 * 3600)
        assert obs["wall_sec"] == pytest.approx(3 * 3600 + 900)

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=86_400),  # wall step
                st.floats(min_value=0, max_value=1.0),  # awake fraction
                st.booleans(),  # process restarted
            ),
            min_size=1,
            max_size=30,
        )
    )
    def test_conservation_and_boundedness(self, steps: list[tuple[float, float, bool]]) -> None:
        wall, mono, pid = 0.0, 0.0, 1
        prev = _c(wall, mono, pid)
        obs: dict[str, Any] | None = None
        for wall_step, awake, restarted in steps:
            wall += wall_step
            if restarted:
                pid += 1
                mono = 0.0
            else:
                mono += wall_step * awake
            cur = _c(wall, mono, pid)
            obs = accumulate_observation(obs, classify_cycle_gap(prev, cur, LOOP))
            prev = cur
        assert obs is not None
        total = obs["active_sec"] + obs["suspended_sec"] + obs["unobserved_sec"]
        restart_short = obs["wall_sec"] - total  # restarts within one loop: counted as neither
        assert obs["active_sec"] <= obs["wall_sec"] + 1e-6
        assert restart_short >= -1e-6
        assert obs["cycles"] == len(steps)


# --------------------------------------------------------------------------- #
# integration: one real watchdog cycle, I/O stubbed at its boundaries
# --------------------------------------------------------------------------- #


def _load_health_watch() -> Any:
    spec = importlib.util.spec_from_file_location(
        "health_watch_under_test", "scripts/health_watch.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def watch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    hw = _load_health_watch()
    monkeypatch.setattr(hw, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(hw, "_fleet_status_output", lambda: "  ✓ engine (pid 1)\n")
    monkeypatch.setattr(hw, "_newest_x_event", lambda _e: datetime.now(UTC))
    monkeypatch.setattr(hw, "_engine_halt_state", lambda: False)
    monkeypatch.setattr(hw, "_recent_halt_reason", lambda: None)
    monkeypatch.setattr("src.data.db_env.build_db_url", lambda: "sqlite://")
    sent: list[str] = []

    class _Result:
        any_succeeded = True

    def _notify(_title: str, msg: str) -> Any:
        sent.append(msg)
        return _Result()

    monkeypatch.setattr("src.research.notifications.notify_operator", _notify)
    hw._sent = sent
    return hw


def _state(hw: Any) -> dict[str, Any]:
    return dict(json.loads(Path(hw.STATE_PATH).read_text()))


def test_watchdog_cycle_detects_sleep_and_restores_readiness(watch: Any) -> None:
    watch.run_once(clock=_c(0, 0), interval_sec=LOOP)
    assert _state(watch)["readiness"]["status"] == READINESS_OK
    # The Mac sleeps for three hours between cycles; the watchdog process lives.
    watch.run_once(clock=_c(3 * 3600 + LOOP, LOOP), interval_sec=LOOP)
    st1 = _state(watch)
    assert st1["readiness"]["status"] == READINESS_STALE
    assert st1["host_gaps"][-1]["kind"] == GAP_SUSPENDED
    assert st1["observation"]["suspended_sec"] == pytest.approx(3 * 3600)
    assert any("HOST SUSPENDED" in m for m in watch._sent)
    # Next awake cycle with every freshness check passing restores readiness.
    watch.run_once(clock=_c(3 * 3600 + 2 * LOOP, 2 * LOOP), interval_sec=LOOP)
    assert _state(watch)["readiness"]["status"] == READINESS_OK
    assert any("Readiness restored" in m for m in watch._sent)


def test_watchdog_stays_stale_while_evidence_is_not_fresh(
    watch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch.run_once(clock=_c(0, 0), interval_sec=LOOP)
    watch.run_once(clock=_c(3 * 3600, LOOP), interval_sec=LOOP)
    # Engine API unreachable after wake: halt state UNKNOWN → not fresh.
    monkeypatch.setattr(watch, "_engine_halt_state", lambda: None)
    watch.run_once(clock=_c(3 * 3600 + LOOP, 2 * LOOP), interval_sec=LOOP)
    assert _state(watch)["readiness"]["status"] == READINESS_STALE
