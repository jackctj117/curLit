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
