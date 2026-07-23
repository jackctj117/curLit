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
from pathlib import Path

import yaml

DEFAULT_PLAYBOOKS_PATH = Path("configs/event_playbooks.yaml")

VALID_KINDS = frozenset({"oanda", "fx", "equity_watch", "polymarket"})
TRADABLE_KINDS = frozenset({"oanda", "fx"})
VALID_DIRECTIONS = frozenset({"long", "short", "watch"})

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

    @property
    def tradable_instruments(self) -> tuple[PlaybookInstrument, ...]:
        return tuple(i for i in self.instruments if i.kind in TRADABLE_KINDS)


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
        playbooks[key] = Playbook(
            key=key,
            name=str(body.get("name", key)),
            description=str(body.get("description", "")).strip(),
            watch_terms=watch_terms,
            instruments=instruments,
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
