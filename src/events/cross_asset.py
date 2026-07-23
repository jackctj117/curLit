"""Cross-asset confirmation layer for event trade ideas (CL-6mzn).

Operator rule (mandatory filter): *never look at the asset in isolation —
if the related commodity/sector isn't moving, the equity reaction is
likely to fade.* Before acting on an event, check that the theme's
related commodity/asset is CORROBORATING the move.

For a theme (``energy_chokepoint``, ``taiwan_semiconductor``,
``russia_ukraine``, the Africa-minerals cluster, ...) the config
``configs/cross_asset_checks.yaml`` names a handful of OANDA-tradable
corroborating instruments and the direction of the move that would
CONFIRM the escalation/positive-thesis reading (see that file's header
for the risk-off sign conventions — the easy-to-get-wrong part). This
module measures each instrument's actual % move since the event's
``seen_at`` (latest value vs the close at ``seen_at``, the same read
:class:`src.events.confluence.EventConfluence` uses) and votes:

  * an instrument AGREES when the sign of its move matches
    ``expected_direction``;
  * ``confirmed`` is True when the (weighted) fraction of agreeing
    instruments-with-data is >= ``threshold`` (default 0.5) AND at
    least one instrument had data;
  * an instrument with NO price data is EXCLUDED from the vote — it is
    not a fail. Zero instruments with data → ``confirmed = None``
    (unknown): the layer never blocks on missing data.

This is a DISPLAY / ANNOTATION layer, not a hard gate. Rendering helpers
live in :mod:`src.events.digest` / the event-driven strategy; this module
is pure-ish (its only side channel is the injected ``data_provider``) and
fully unit-tested with a fake provider.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CHECKS_PATH = Path("configs/cross_asset_checks.yaml")

#: Default fraction of (weighted) instruments-with-data that must agree
#: for the cross-asset read to CONFIRM.
DEFAULT_THRESHOLD = 0.5

VALID_DIRECTIONS = frozenset({"up", "down"})

#: OANDA-style symbol shape — the same "clearly reachable" gate the
#: playbook loader uses for tradables (BCO_USD, USD_NOK, SPX500_USD ...).
_INSTRUMENT_RE = re.compile(r"^[A-Z0-9_]+$")


@dataclass(frozen=True)
class CrossAssetCheck:
    """One corroborating instrument for a theme."""

    instrument: str
    expected_direction: str  # "up" | "down"
    weight: float = 1.0


@dataclass
class CrossAssetConfig:
    """Validated view of ``configs/cross_asset_checks.yaml``."""

    themes: dict[str, tuple[CrossAssetCheck, ...]] = field(default_factory=dict)

    def checks_for(self, theme: str | None) -> tuple[CrossAssetCheck, ...]:
        """Corroborating instruments for ``theme`` (empty tuple when the
        theme has no cross-asset entry — the layer degrades to unknown)."""
        return self.themes.get(str(theme or ""), ())


def load_cross_asset_config(
    path: Path | str = DEFAULT_CHECKS_PATH,
) -> CrossAssetConfig:
    """Load + validate the cross-asset checks config. Fail-loud on
    structural problems (bad direction, non-OANDA-shaped instrument,
    empty instrument list) — a silently-empty theme would make the
    corroboration a no-op that quietly never confirms."""
    path = Path(path)
    if not path.exists():
        msg = f"cross-asset checks config not found: {path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(path.read_text()) or {}
    themes = raw.get("themes")
    if not isinstance(themes, dict) or not themes:
        msg = f"{path}: missing or empty 'themes' mapping"
        raise ValueError(msg)

    parsed: dict[str, tuple[CrossAssetCheck, ...]] = {}
    for theme, body in themes.items():
        if not isinstance(body, dict):
            msg = f"cross_asset theme {theme!r}: body must be a mapping"
            raise ValueError(msg)
        entries = body.get("instruments")
        if not isinstance(entries, list) or not entries:
            msg = f"cross_asset theme {theme!r}: instruments must be a non-empty list"
            raise ValueError(msg)
        checks = tuple(_validate_check(theme, e) for e in entries)
        parsed[str(theme)] = checks
    return CrossAssetConfig(themes=parsed)


def _validate_check(theme: str, entry: Any) -> CrossAssetCheck:
    if not isinstance(entry, dict):
        msg = f"cross_asset theme {theme!r}: each instrument entry must be a mapping"
        raise ValueError(msg)
    instrument = str(entry.get("instrument", "")).strip()
    direction = str(entry.get("expected_direction", "")).strip().lower()
    if not instrument:
        msg = f"cross_asset theme {theme!r}: instrument entry missing 'instrument'"
        raise ValueError(msg)
    if not _INSTRUMENT_RE.match(instrument):
        msg = (
            f"cross_asset theme {theme!r} / {instrument}: instrument must be an "
            f"OANDA-shaped symbol {_INSTRUMENT_RE.pattern}"
        )
        raise ValueError(msg)
    if direction not in VALID_DIRECTIONS:
        msg = (
            f"cross_asset theme {theme!r} / {instrument}: expected_direction "
            f"{direction!r} not in {sorted(VALID_DIRECTIONS)}"
        )
        raise ValueError(msg)
    raw_weight = entry.get("weight", 1.0)
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        msg = f"cross_asset theme {theme!r} / {instrument}: weight {raw_weight!r} is not a number"
        raise ValueError(msg) from exc
    if weight <= 0.0:
        msg = f"cross_asset theme {theme!r} / {instrument}: weight must be > 0, got {weight}"
        raise ValueError(msg)
    return CrossAssetCheck(
        instrument=instrument,
        expected_direction=direction,
        weight=weight,
    )


@dataclass
class InstrumentMove:
    """Per-instrument cross-asset vote outcome."""

    instrument: str
    expected: str  # "up" | "down"
    actual_move_pct: float | None  # % move since `since` (None = no data)
    agrees: bool | None  # None when there was no data to vote


@dataclass
class CrossAssetResult:
    """Outcome of a theme's cross-asset corroboration check.

    ``confirmed`` is a tri-state:
      * ``True``  — the (weighted) agreeing fraction cleared the threshold;
      * ``False`` — data existed but the corroboration failed (fade risk);
      * ``None``  — UNKNOWN: no instrument had usable data (never blocks).
    ``score`` is the weighted agreeing fraction in [0, 1] (0.0 when
    unknown). ``details`` carries the per-instrument reads for rendering.
    """

    confirmed: bool | None
    score: float
    details: list[InstrumentMove] = field(default_factory=list)

    @property
    def n_voting(self) -> int:
        """Instruments that actually voted (had data)."""
        return sum(1 for d in self.details if d.agrees is not None)

    @property
    def n_agree(self) -> int:
        return sum(1 for d in self.details if d.agrees is True)


def _price_at(data_provider: Any, instrument: str, as_of: datetime) -> float | None:
    """Latest value of ``instrument`` at/before ``as_of`` via the
    DataProvider (same read the confluence layer uses). Fail-soft: any
    provider error → None (excluded from the vote, not a crash)."""
    if data_provider is None:
        return None
    try:
        value = data_provider.get_latest_value(instrument, as_of)
    except Exception as exc:
        logger.debug(
            "cross_asset price lookup failed for %s @ %s: %s: %s",
            instrument,
            as_of,
            type(exc).__name__,
            exc,
        )
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _move_agrees(move_pct: float, expected: str) -> bool:
    """A move AGREES with the thesis when its sign matches the expected
    direction. A dead-flat move (0.0) does NOT corroborate — the related
    asset isn't moving, which is exactly the fade signal the operator
    warned about."""
    if expected == "up":
        return move_pct > 0.0
    return move_pct < 0.0


def cross_asset_confirmation(
    data_provider: Any,
    theme: str | None,
    since: datetime,
    checks_config: CrossAssetConfig,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    now: datetime | None = None,
) -> CrossAssetResult:
    """Measure whether ``theme``'s corroborating instruments are moving
    the way the escalation/positive-thesis reading needs.

    For each configured instrument: read the price at ``since`` (event
    ``seen_at``) and the latest price (``now``), compute the % move, and
    compare its sign to ``expected_direction``. An instrument missing
    either leg is EXCLUDED from the vote (its ``agrees`` is ``None``).

    ``confirmed`` is True when the weighted agreeing fraction over
    voting instruments is >= ``threshold`` AND at least one instrument
    voted; ``None`` when no instrument had data (unknown, never blocks);
    ``False`` otherwise (data existed, corroboration failed → fade risk).
    """
    now = now or datetime.now(tz=since.tzinfo)
    checks = checks_config.checks_for(theme)
    details: list[InstrumentMove] = []
    agree_weight = 0.0
    voting_weight = 0.0

    for check in checks:
        p0 = _price_at(data_provider, check.instrument, since)
        p1 = _price_at(data_provider, check.instrument, now)
        if p0 is None or p1 is None or p0 <= 0.0:
            details.append(
                InstrumentMove(
                    instrument=check.instrument,
                    expected=check.expected_direction,
                    actual_move_pct=None,
                    agrees=None,
                )
            )
            continue
        move_pct = (p1 - p0) / p0 * 100.0
        agrees = _move_agrees(move_pct, check.expected_direction)
        details.append(
            InstrumentMove(
                instrument=check.instrument,
                expected=check.expected_direction,
                actual_move_pct=move_pct,
                agrees=agrees,
            )
        )
        voting_weight += check.weight
        if agrees:
            agree_weight += check.weight

    if voting_weight <= 0.0:
        # No instrument had usable data — UNKNOWN. Never blocks.
        return CrossAssetResult(confirmed=None, score=0.0, details=details)

    score = agree_weight / voting_weight
    confirmed = score >= threshold
    return CrossAssetResult(confirmed=confirmed, score=score, details=details)
