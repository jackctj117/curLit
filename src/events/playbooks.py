"""Event playbook loader (CL-6iu7) — validated view of
``configs/event_playbooks.yaml``.

A playbook is a pre-researched geopolitical theme: what to watch for on
GDELT (``watch_terms``) and which reachable instruments historically
react (``instruments`` with direction hints). Both the GDELT ingester
(query construction, theme tagging) and the Event Impact Agent (LLM
context + instrument whitelist) consume this module, so validation is
fail-loud at load time:

  * kind must be oanda | fx | equity_watch | polymarket
  * tradable instruments (oanda/fx) must look like OANDA symbols
    (``^[A-Z0-9_]+$`` — BCO_USD, USD_NOK, SPX500_USD ...)
  * equity tickers are ALERT-ONLY: kind equity_watch, direction watch
    (curLit has no equity execution path)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

DEFAULT_PLAYBOOKS_PATH = Path("configs/event_playbooks.yaml")

VALID_KINDS = frozenset({"oanda", "fx", "equity_watch", "polymarket"})
TRADABLE_KINDS = frozenset({"oanda", "fx"})
VALID_DIRECTIONS = frozenset({"long", "short", "watch"})
#: Theme tiers (CL-gn6k). "specific" = a named geography/entity/mechanism
#: with real trading edge; "generic" = a broad catch-all (war_escalation,
#: natural_disaster, africa_power_shift) — good for ingest coverage, weak
#: edge, so it faces a harder machine-trading bar and loses theme-attribution
#: ties to a specific theme. Absent tier defaults to "specific".
VALID_TIERS = frozenset({"specific", "generic"})

#: OANDA-style symbol shape — the "clearly reachable" gate for tradables.
INSTRUMENT_RE = re.compile(r"^[A-Z0-9_]+$")


@dataclass(frozen=True)
class PlaybookInstrument:
    instrument: str
    kind: str  # oanda | fx | equity_watch | polymarket
    direction: str  # long | short | watch (hint, not an order)
    rationale: str


@dataclass(frozen=True)
class Playbook:
    key: str
    name: str
    description: str
    watch_terms: tuple[str, ...]
    instruments: tuple[PlaybookInstrument, ...]
    #: Theme tier (CL-gn6k): "specific" (default) or "generic". Generic
    #: catch-alls face a harder machine-trading bar and lose attribution
    #: ties to a specific theme (see VALID_TIERS).
    tier: str = "specific"
    #: When the operator last reviewed/curated this theme's FACTS (mine
    #: ownership, territorial control, supply routes). Feeds the stale-fact
    #: confidence ceiling (CL-ylak): once older than the configured window,
    #: the confluence layer caps an event's confidence so aging facts can't
    #: clear Gate A on conviction alone. None = undated → never capped.
    last_reviewed: date | None = None

    @property
    def tradable_instruments(self) -> tuple[PlaybookInstrument, ...]:
        return tuple(i for i in self.instruments if i.kind in TRADABLE_KINDS)


def _parse_review_date(theme: str, raw: object) -> date | None:
    """Parse an optional ``last_reviewed`` value. YAML already yields a
    ``date`` for an unquoted ``2026-07-20``; strings (quoted / other loaders)
    are parsed as ISO. Fail-LOUD on a malformed value — a review date the
    loader silently drops would make the staleness ceiling a no-op."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw.strip())
        except ValueError as exc:
            msg = f"playbook {theme!r}: last_reviewed {raw!r} is not ISO YYYY-MM-DD"
            raise ValueError(msg) from exc
    msg = f"playbook {theme!r}: last_reviewed must be a date, got {type(raw).__name__}"
    raise ValueError(msg)


def _validate_instrument(theme: str, entry: dict[str, object]) -> PlaybookInstrument:
    instrument = str(entry.get("instrument", "")).strip()
    kind = str(entry.get("kind", "")).strip()
    direction = str(entry.get("direction", "")).strip()
    rationale = str(entry.get("rationale", "")).strip()

    if not instrument:
        msg = f"playbook {theme!r}: instrument entry missing 'instrument'"
        raise ValueError(msg)
    if kind not in VALID_KINDS:
        msg = f"playbook {theme!r} / {instrument}: kind {kind!r} not in {sorted(VALID_KINDS)}"
        raise ValueError(msg)
    if direction not in VALID_DIRECTIONS:
        msg = (
            f"playbook {theme!r} / {instrument}: direction {direction!r} "
            f"not in {sorted(VALID_DIRECTIONS)}"
        )
        raise ValueError(msg)
    if kind in TRADABLE_KINDS and not INSTRUMENT_RE.match(instrument):
        msg = (
            f"playbook {theme!r} / {instrument}: tradable ({kind}) "
            f"instrument must match {INSTRUMENT_RE.pattern}"
        )
        raise ValueError(msg)
    if kind == "equity_watch" and direction != "watch":
        msg = (
            f"playbook {theme!r} / {instrument}: equities are alert-only — "
            f"kind equity_watch requires direction 'watch', got {direction!r}"
        )
        raise ValueError(msg)
    return PlaybookInstrument(
        instrument=instrument,
        kind=kind,
        direction=direction,
        rationale=rationale,
    )


def load_playbooks(
    path: Path | str = DEFAULT_PLAYBOOKS_PATH,
) -> dict[str, Playbook]:
    """Load + validate the playbook config. Raises on structural problems
    — a silently-empty playbook would make the whole pipeline a no-op."""
    path = Path(path)
    if not path.exists():
        msg = f"event playbooks config not found: {path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(path.read_text()) or {}
    themes = raw.get("themes")
    if not isinstance(themes, dict) or not themes:
        msg = f"{path}: missing or empty 'themes' mapping"
        raise ValueError(msg)

    playbooks: dict[str, Playbook] = {}
    for key, body in themes.items():
        if not isinstance(body, dict):
            msg = f"playbook {key!r}: body must be a mapping"
            raise ValueError(msg)
        watch_terms = tuple(str(t).strip() for t in body.get("watch_terms") or [] if str(t).strip())
        if not watch_terms:
            msg = f"playbook {key!r}: watch_terms must be a non-empty list"
            raise ValueError(msg)
        instruments = tuple(_validate_instrument(key, e) for e in body.get("instruments") or [])
        if not instruments:
            msg = f"playbook {key!r}: instruments must be a non-empty list"
            raise ValueError(msg)
        tier = str(body.get("tier", "specific")).strip().lower()
        if tier not in VALID_TIERS:
            msg = f"playbook {key!r}: tier {tier!r} not in {sorted(VALID_TIERS)}"
            raise ValueError(msg)
        playbooks[key] = Playbook(
            key=key,
            name=str(body.get("name", key)),
            description=str(body.get("description", "")).strip(),
            watch_terms=watch_terms,
            instruments=instruments,
            tier=tier,
            last_reviewed=_parse_review_date(key, body.get("last_reviewed")),
        )
    return playbooks


def all_tradable_instruments(playbooks: dict[str, Playbook]) -> set[str]:
    """Union of oanda/fx instrument names across all themes — the impact
    agent's cross-theme whitelist fallback."""
    return {
        i.instrument
        for pb in playbooks.values()
        for i in pb.instruments
        if i.kind in TRADABLE_KINDS
    }
