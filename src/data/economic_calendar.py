"""Economic-release calendar + blackout policy (CL-ahdu).

NFP, FOMC, ECB / BoE / BoJ rate decisions drive the largest FX vol spikes
(50+ bps slippage on poorly-timed entries). Not encoding the calendar is
free risk reduction we're leaving on the table.

This module provides:

    EconomicEvent      — one scheduled release: ts, event_type, severity_tier,
                         description, optional currency.
    EconomicCalendar   — append-only store of events; query upcoming.
    BlackoutPolicy     — per-tier sizing rules (size-down within 24h of
                         tier-1, pause new entries within 1h, exit-flat
                         within 30min).
    BlackoutEvaluator  — given current ts + calendar + policy, returns a
                         BlackoutAction the risk sizer reads.

Calendar ingestion is via load_calendar_from_yaml(path) for hand-curated
events. A vendor-feed (e.g. FRED release schedule, ForexFactory) ingester
is a separate piece — this module is the *consumer* side.

Wire into the regime sizer: when BlackoutEvaluator.evaluate(now) returns
EXIT_FLAT, halt all new entries and flatten existing positions. SIZE_DOWN_50PCT
caps new positions at half normal size. PAUSE_NEW_ENTRIES blocks new opens
but holds existing positions.

Reference: CL-ahdu.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Default blackout windows for tier-1 events. Calibrated to FX vol-spike
# behavior — vol typically rises 4–6 hours before the event, peaks at
# release, decays over the following hour.
_DEFAULT_TIER1_SIZE_DOWN_HOURS: int = 24
_DEFAULT_TIER1_PAUSE_HOURS: float = 1.0
_DEFAULT_TIER1_EXIT_FLAT_MINUTES: int = 30
_DEFAULT_TIER2_SIZE_DOWN_HOURS: int = 4   # tier-2 has a milder window
_DEFAULT_TIER2_PAUSE_HOURS: float = 0.0   # no pause for tier 2 by default


class SeverityTier(Enum):
    TIER_1 = 1  # NFP, FOMC, ECB/BoE/BoJ rate decisions, CPI
    TIER_2 = 2  # GDP, retail sales, ISM, ADP
    TIER_3 = 3  # minor releases


class BlackoutAction(Enum):
    FULL_SIZE = "full_size"
    SIZE_DOWN_50PCT = "size_down_50pct"
    PAUSE_NEW_ENTRIES = "pause_new_entries"
    EXIT_FLAT = "exit_flat"


# Numeric encoding for downstream sizers / metrics — higher = more restrictive.
_ACTION_RANK: dict[BlackoutAction, int] = {
    BlackoutAction.FULL_SIZE: 0,
    BlackoutAction.SIZE_DOWN_50PCT: 1,
    BlackoutAction.PAUSE_NEW_ENTRIES: 2,
    BlackoutAction.EXIT_FLAT: 3,
}


@dataclass(frozen=True)
class EconomicEvent:
    """One scheduled economic release."""

    ts: datetime
    event_type: str
    severity_tier: SeverityTier
    description: str = ""
    currency: str | None = None  # primary currency affected (None = global)

    def __post_init__(self) -> None:
        assert self.ts.tzinfo is not None, (
            f"event ts must be tz-aware, got naive datetime {self.ts}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "event_type": self.event_type,
            "severity_tier": self.severity_tier.value,
            "description": self.description,
            "currency": self.currency,
        }


@dataclass
class BlackoutPolicy:
    """Per-tier sizing rules.

    For each tier, three thresholds cascade as the event approaches:
        size_down_hours_before > 0  → SIZE_DOWN_50PCT inside this window
        pause_hours_before > 0      → PAUSE_NEW_ENTRIES inside (size_down → pause transition)
        exit_flat_minutes_before > 0 → EXIT_FLAT inside (most restrictive)
    Set any value to 0 to disable that level for the tier.
    """

    tier1_size_down_hours: float = _DEFAULT_TIER1_SIZE_DOWN_HOURS
    tier1_pause_hours: float = _DEFAULT_TIER1_PAUSE_HOURS
    tier1_exit_flat_minutes: float = _DEFAULT_TIER1_EXIT_FLAT_MINUTES
    tier2_size_down_hours: float = _DEFAULT_TIER2_SIZE_DOWN_HOURS
    tier2_pause_hours: float = _DEFAULT_TIER2_PAUSE_HOURS
    tier2_exit_flat_minutes: float = 0.0
    # Tier-3 events don't trigger blackout by default.
    tier3_size_down_hours: float = 0.0
    tier3_pause_hours: float = 0.0
    tier3_exit_flat_minutes: float = 0.0

    def __post_init__(self) -> None:
        for f in (
            self.tier1_size_down_hours, self.tier1_pause_hours,
            self.tier1_exit_flat_minutes, self.tier2_size_down_hours,
            self.tier2_pause_hours, self.tier2_exit_flat_minutes,
            self.tier3_size_down_hours, self.tier3_pause_hours,
            self.tier3_exit_flat_minutes,
        ):
            assert f >= 0, f"blackout window must be non-negative, got {f}"

    def windows_for(self, tier: SeverityTier) -> tuple[float, float, float]:
        """Return (size_down_hours, pause_hours, exit_flat_minutes) for the tier."""
        if tier == SeverityTier.TIER_1:
            return (
                self.tier1_size_down_hours,
                self.tier1_pause_hours,
                self.tier1_exit_flat_minutes,
            )
        if tier == SeverityTier.TIER_2:
            return (
                self.tier2_size_down_hours,
                self.tier2_pause_hours,
                self.tier2_exit_flat_minutes,
            )
        return (
            self.tier3_size_down_hours,
            self.tier3_pause_hours,
            self.tier3_exit_flat_minutes,
        )


@dataclass
class BlackoutDecision:
    """Output of BlackoutEvaluator.evaluate."""

    action: BlackoutAction
    triggering_event: EconomicEvent | None
    minutes_until_event: float | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "triggering_event": (
                self.triggering_event.to_dict() if self.triggering_event else None
            ),
            "minutes_until_event": self.minutes_until_event,
            "reason": self.reason,
        }


# =============================================================================
# Calendar
# =============================================================================


class EconomicCalendar:
    """Append-only store of economic events with upcoming-events query.

    Construct empty and call add(); or load via load_calendar_from_yaml.
    Internally sorted by ts so upcoming queries are O(n) over the upcoming
    horizon (which is small for typical use — tens of events per month).
    """

    def __init__(self, events: list[EconomicEvent] | None = None) -> None:
        self._events: list[EconomicEvent] = []
        if events:
            for e in events:
                self.add(e)

    def add(self, event: EconomicEvent) -> None:
        """Append an event and resort by ts."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.ts)

    def upcoming(
        self,
        now: datetime,
        horizon_hours: float = 48.0,
        tiers: list[SeverityTier] | None = None,
        currency: str | None = None,
    ) -> list[EconomicEvent]:
        """Events occurring in (now, now + horizon_hours] window.

        Filters by tiers (any-of) and currency (None or matching event currency).
        """
        end = now + timedelta(hours=horizon_hours)
        out: list[EconomicEvent] = []
        for e in self._events:
            if e.ts <= now or e.ts > end:
                continue
            if tiers is not None and e.severity_tier not in tiers:
                continue
            if currency is not None and e.currency is not None and e.currency != currency:
                continue
            out.append(e)
        return out

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> list[EconomicEvent]:
        return list(self._events)


# =============================================================================
# YAML loader
# =============================================================================


def load_calendar_from_yaml(path: Path | str) -> EconomicCalendar:
    """Load an EconomicCalendar from a YAML file.

    YAML schema:
        events:
          - ts: 2026-05-01T12:30:00Z
            event_type: NFP
            severity_tier: 1
            description: US Non-Farm Payrolls
            currency: USD
    """
    p = Path(path)
    if not p.exists():
        msg = f"economic calendar not found at {p}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict) or "events" not in raw:
        msg = f"calendar at {p} missing 'events' top-level key"
        raise ValueError(msg)

    events: list[EconomicEvent] = []
    for entry in raw.get("events", []):
        raw_ts = entry["ts"]
        # PyYAML auto-parses ISO 8601 to datetime; we only need to parse if
        # the value came in as a string (e.g. from JSON or hand-edited).
        if isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        events.append(
            EconomicEvent(
                ts=ts,
                event_type=str(entry["event_type"]),
                severity_tier=SeverityTier(int(entry["severity_tier"])),
                description=str(entry.get("description", "")),
                currency=entry.get("currency"),
            )
        )
    logger.info("Loaded %d economic events from %s", len(events), p)
    return EconomicCalendar(events)


# =============================================================================
# Evaluator
# =============================================================================


class BlackoutEvaluator:
    """Decides the most-restrictive BlackoutAction at a given time.

    Among all events occurring within their tier's size-down window of
    `now`, the one with the most-restrictive action wins. Returns
    FULL_SIZE when no event window applies.
    """

    def __init__(
        self,
        calendar: EconomicCalendar,
        policy: BlackoutPolicy | None = None,
    ) -> None:
        self.calendar = calendar
        self.policy = policy or BlackoutPolicy()

    def evaluate(
        self,
        now: datetime,
        currency: str | None = None,
    ) -> BlackoutDecision:
        """Return the most-restrictive applicable action across upcoming events."""
        # Look 48h ahead — that's wider than any tier-1 size-down window.
        upcoming = self.calendar.upcoming(
            now, horizon_hours=48.0, currency=currency,
        )
        worst: BlackoutAction = BlackoutAction.FULL_SIZE
        worst_event: EconomicEvent | None = None
        worst_minutes: float | None = None

        for event in upcoming:
            size_down_hours, pause_hours, exit_flat_minutes = self.policy.windows_for(
                event.severity_tier,
            )
            minutes_until = (event.ts - now).total_seconds() / 60.0

            action = BlackoutAction.FULL_SIZE
            if exit_flat_minutes > 0 and minutes_until <= exit_flat_minutes:
                action = BlackoutAction.EXIT_FLAT
            elif pause_hours > 0 and minutes_until <= pause_hours * 60:
                action = BlackoutAction.PAUSE_NEW_ENTRIES
            elif size_down_hours > 0 and minutes_until <= size_down_hours * 60:
                action = BlackoutAction.SIZE_DOWN_50PCT

            if _ACTION_RANK[action] > _ACTION_RANK[worst]:
                worst = action
                worst_event = event
                worst_minutes = minutes_until

        if worst == BlackoutAction.FULL_SIZE:
            return BlackoutDecision(
                action=worst,
                triggering_event=None,
                minutes_until_event=None,
                reason="no upcoming event within blackout windows",
            )

        assert worst_event is not None
        reason = (
            f"{worst.value} for {worst_event.event_type} "
            f"({worst_event.severity_tier.value}) in "
            f"{worst_minutes:.0f} min"
        ) if worst_minutes is not None else worst.value
        return BlackoutDecision(
            action=worst,
            triggering_event=worst_event,
            minutes_until_event=worst_minutes,
            reason=reason,
        )


__all__ = [
    "BlackoutAction",
    "BlackoutDecision",
    "BlackoutEvaluator",
    "BlackoutPolicy",
    "EconomicCalendar",
    "EconomicEvent",
    "SeverityTier",
    "load_calendar_from_yaml",
]
