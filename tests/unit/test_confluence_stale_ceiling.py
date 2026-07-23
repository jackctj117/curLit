"""Stale-fact confidence ceiling (CL-ylak).

A CONFIRMED-candidate whose matched theme's playbook facts were last
reviewed more than ``stale_review_days`` ago has its LLM ``confidence``
capped at ``stale_confidence_ceiling`` BEFORE Gate A — so aging
ownership/control facts can no longer clear ``min_confidence`` on the
model's conviction alone. Undated themes, fresh themes, already-hedged
assessments, and a disabled ceiling are all left untouched.

The events here sit inside the window but BEFORE ``confirm_window_min``,
so ``evaluate_and_transition`` returns right after Gate A — no Gate-B
price plumbing needed, and ``db_engine`` is never touched (no transition).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from src.events.confluence import ConfluenceConfig, EventConfluence
from src.events.playbooks import Playbook, PlaybookInstrument

_SEEN = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
_NOW = _SEEN + timedelta(minutes=10)  # inside window, before confirm_window_min


def _pb(key: str, last_reviewed: date | None) -> Playbook:
    return Playbook(
        key=key,
        name=key,
        description="",
        watch_terms=("x",),
        instruments=(PlaybookInstrument("XAU_USD", "oanda", "long", "r"),),
        last_reviewed=last_reviewed,
    )


def _event(theme: str, confidence: float, urgency: int = 9) -> dict[str, Any]:
    return {
        "id": 1,
        "seen_at": _SEEN.isoformat(),
        "theme": theme,
        "assessment": {
            "urgency": urgency,
            "confidence": confidence,
            "affected": [
                {"instrument": "XAU_USD", "kind": "oanda", "direction": "long", "reason": "r"},
            ],
        },
    }


def _conf(playbooks: dict[str, Playbook] | None, **cfg: Any) -> EventConfluence:
    base = {"min_confidence": 0.75, "stale_review_days": 90, "stale_confidence_ceiling": 0.58}
    base.update(cfg)
    return EventConfluence(ConfluenceConfig(**base), playbooks=playbooks)


def test_stale_theme_caps_confidence_and_fails_gate_a() -> None:
    # last_reviewed ~200 days before _NOW → past the 90d window.
    conf = _conf({"t": _pb("t", date(2026, 1, 1))})
    res = conf.evaluate_and_transition(_event("t", 0.9), now=_NOW)
    assert res.stale_capped is True
    assert res.confidence == 0.58
    assert res.quality_passed is False  # 0.58 < min_confidence 0.75


def test_fresh_theme_not_capped_and_passes_gate_a() -> None:
    conf = _conf({"t": _pb("t", date(2026, 7, 19))})  # 1 day old
    res = conf.evaluate_and_transition(_event("t", 0.9), now=_NOW)
    assert res.stale_capped is False
    assert res.confidence == 0.9
    assert res.quality_passed is True


def test_undated_theme_never_capped() -> None:
    conf = _conf({"t": _pb("t", None)})
    res = conf.evaluate_and_transition(_event("t", 0.9), now=_NOW)
    assert res.stale_capped is False
    assert res.confidence == 0.9


def test_already_hedged_confidence_not_touched() -> None:
    # Below the ceiling already — a stale theme must never RAISE confidence.
    conf = _conf({"t": _pb("t", date(2026, 1, 1))})
    res = conf.evaluate_and_transition(_event("t", 0.5), now=_NOW)
    assert res.stale_capped is False
    assert res.confidence == 0.5


def test_disabled_when_review_days_zero() -> None:
    conf = _conf({"t": _pb("t", date(2020, 1, 1))}, stale_review_days=0)
    res = conf.evaluate_and_transition(_event("t", 0.9), now=_NOW)
    assert res.stale_capped is False
    assert res.confidence == 0.9


def test_dormant_when_no_playbooks_injected() -> None:
    # Default construction (no playbooks) leaves the ceiling inert — this is
    # what keeps every pre-CL-ylak confluence test unchanged.
    conf = _conf(None)
    res = conf.evaluate_and_transition(_event("t", 0.9), now=_NOW)
    assert res.stale_capped is False
    assert res.confidence == 0.9


def test_real_playbooks_cap_once_aged_past_window() -> None:
    # End-to-end with the REAL YAML dates (2026-07-20): an event seen >90d
    # later gets capped, proving the config dates flow through the loader.
    from src.events.playbooks import load_playbooks

    pbs = load_playbooks("configs/event_playbooks.yaml")
    seen = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)  # 104 days after 2026-07-20
    now = seen + timedelta(minutes=10)
    conf = EventConfluence(
        ConfluenceConfig(min_confidence=0.75, stale_review_days=90, stale_confidence_ceiling=0.58),
        playbooks=pbs,
    )
    ev = {
        "id": 2,
        "seen_at": seen.isoformat(),
        "theme": "energy_chokepoint",
        "assessment": {
            "urgency": 9,
            "confidence": 0.9,
            "affected": [
                {"instrument": "XAU_USD", "kind": "oanda", "direction": "long", "reason": "r"},
            ],
        },
    }
    res = conf.evaluate_and_transition(ev, now=now)
    assert res.stale_capped is True
    assert res.confidence == 0.58


# --------------------------------------------------------------------------- #
# Generic-theme machine bar (CL-gn6k)
# --------------------------------------------------------------------------- #


def _tiered_pb(key: str, tier: str) -> Playbook:
    # Fresh review date so the stale ceiling never interferes with tier tests.
    return Playbook(
        key=key,
        name=key,
        description="",
        watch_terms=("x",),
        instruments=(PlaybookInstrument("XAU_USD", "oanda", "long", "r"),),
        tier=tier,
        last_reviewed=date(2026, 7, 19),
    )


def _gen_conf(**cfg: Any) -> EventConfluence:
    base = {
        "min_confidence": 0.75,
        "generic_min_confidence": 0.82,
        "generic_min_confirmed_instruments": 2,
    }
    base.update(cfg)
    return EventConfluence(
        ConfluenceConfig(**base),
        playbooks={"gen": _tiered_pb("gen", "generic"), "spec": _tiered_pb("spec", "specific")},
    )


def test_generic_theme_needs_higher_confidence_at_gate_a() -> None:
    conf = _gen_conf()
    # 0.80 clears the base 0.75 but NOT the generic 0.82 bar → Gate A fails.
    res = conf.evaluate_and_transition(_event("gen", 0.80), now=_NOW)
    assert res.generic is True
    assert res.quality_passed is False


def test_specific_theme_unaffected_by_generic_bar() -> None:
    conf = _gen_conf()
    # Same 0.80 confidence on a SPECIFIC theme clears the base 0.75 bar.
    res = conf.evaluate_and_transition(_event("spec", 0.80), now=_NOW)
    assert res.generic is False
    assert res.quality_passed is True


def test_generic_theme_clears_gate_a_above_bar() -> None:
    conf = _gen_conf()
    res = conf.evaluate_and_transition(_event("gen", 0.85), now=_NOW)
    assert res.generic is True
    assert res.quality_passed is True


def test_generic_bar_disabled_when_thresholds_at_base() -> None:
    # generic_min_confidence <= base → the max() floor is inert.
    conf = _gen_conf(generic_min_confidence=0.75, generic_min_confirmed_instruments=1)
    res = conf.evaluate_and_transition(_event("gen", 0.80), now=_NOW)
    assert res.generic is True  # still flagged...
    assert res.quality_passed is True  # ...but gated like a specific theme


class _OneLegConfirmProvider:
    """Intraday shows a +2% move on the single leg (confirms Gate B); the
    daily close is flat. Mirrors test_intraday_pricer._FakeProvider."""

    def get_intraday_value(self, symbol: str, as_of: datetime, max_staleness_minutes: Any = None):
        ref = _SEEN + timedelta(minutes=30)
        return 100.0 if as_of <= ref else 102.0

    def get_latest_value(self, symbol: str, as_of: datetime) -> float:
        return 100.0

    def get_realized_vol(self, symbol: str, window: int, as_of: datetime) -> float:
        return 0.16  # annualized → daily ≈ 1%, threshold ≈ 0.25%


def _gate_b_conf(tier: str) -> EventConfluence:
    return EventConfluence(
        ConfluenceConfig(
            min_confidence=0.75,
            intraday_max_staleness_minutes=15,
            generic_min_confidence=0.82,
            generic_min_confirmed_instruments=2,
        ),
        data_provider=_OneLegConfirmProvider(),
        playbooks={"t": _tiered_pb("t", tier)},
    )


def test_generic_needs_two_confirmed_so_one_leg_stays_pending() -> None:
    # now inside the 30-120 window so Gate B actually runs.
    now = _SEEN + timedelta(minutes=60)
    # SPECIFIC theme: 1 confirmed leg >= 1 → confirms.
    res_spec = _gate_b_conf("specific").evaluate_and_transition(_event("t", 0.9), now=now)
    assert res_spec.outcome == "confirmed"
    # GENERIC theme with the SAME single confirming leg: needs 2 → stays
    # pending (rides out to the operator's expired alert, never machine-trades).
    res_gen = _gate_b_conf("generic").evaluate_and_transition(_event("t", 0.9), now=now)
    assert res_gen.generic is True
    assert res_gen.outcome == "pending"
