"""Typed advisory trade idea (CL-59mk) — the dataclass form of the dicts
that flow ``impact_agent → geo_events.assessment["trade_ideas"] →
idea_ledger → digest``.

Why this exists: the idea dicts ride the MONEY path (they become
``trade_ideas`` rows, which the Alpaca options executor buys off) and
until now every consumer re-derived the shape by hand — ``str(idea.get(
"ticker") or "").strip()`` in four modules, three different fallbacks for
``time_stop_days``. This module names the shape once.

THE DICT IS STILL THE CONTRACT. ``geo_events.assessment`` is persisted
JSON and ``trade_ideas`` is a table; nothing here changes either. So
:meth:`TradeIdea.to_dict` reproduces the historical dicts EXACTLY — same
keys, same insertion order, so ``json.dumps`` stays byte-identical — and
:meth:`TradeIdea.from_dict` round-trips them.

Two shapes, one type:

* the **core** 15 keys emitted by
  :func:`src.events.impact_agent._normalise_trade_ideas` (the LLM's
  advisory idea, enums validated and numbers clamped THERE — this module
  deliberately does NOT re-implement that validation);
* those plus the **niche annotations** appended by
  :meth:`src.events.niche_scoring.NicheIdea.to_trade_idea` (``niche``,
  ``company_name``, ``hop_count``, ``torque_reason``,
  ``asymmetry_score``, ``liquidity_flag``, ``exchange``,
  ``robinhood_tradeable``, ``red_team_verdict``). ``to_dict`` emits that
  block ONLY when ``niche`` is set — a plain idea keeps exactly its 15
  keys, as today.

:meth:`from_dict` is a TOLERANT, LOSSLESS READER — exactly as tolerant as
the hand-rolled reads it replaces: missing keys take the same defaults
(``""`` / ``None`` / ``[]``), unparseable numbers go ``None``, nothing
raises, and a string is carried through UNTOUCHED (no stripping — the
readers disagreed about that, so the callers that want it still ask).
It does NOT clamp, validate enums, or default ``confidence`` /
``time_stop_days`` the way ``_normalise_trade_ideas`` does — that is
producer-side business logic and it stays there. Consequence: for any
dict a producer emitted, ``TradeIdea.from_dict(d).to_dict() == d``; for a
truncated legacy dict, ``to_dict`` fills the canonical keys with honest
empty defaults rather than inventing clamped values.

Boundaries this type deliberately STOPS at (each read at least one idea
key, and each was left on dicts on purpose):

* :class:`src.events.impact_agent.Assessment` keeps
  ``trade_ideas: list[dict]``. Typing the list would make
  ``Assessment.from_dict``/``to_dict`` NORMALIZE the ideas it round-trips
  — a partial idea dict would gain keys — which changes the persisted
  ``geo_events.assessment`` payload. The dict list is the contract.
* :func:`src.events.instrument_selector.decision_for_idea` and
  :func:`src.events.trade_card.build_trade_card` keep taking dicts: both
  are also called on ``trade_ideas`` TABLE rows and on legacy rows, which
  are a different (partly overlapping) shape.
* :func:`src.execution.alpaca_options_executor.fetch_executable_ideas`
  and :func:`src.monitoring.morning_digest.build_long_ideas_digest` read
  ``trade_ideas`` ROWS (``idea_id`` + a 5-6 column projection), not
  assessment ideas. Parsing those into a ``TradeIdea`` would claim 10
  fields the query never selected and still need the dict for
  ``idea_id`` — strictly worse than the row dict.
* :meth:`src.events.event_notifier.EventNotifier._ideas_block` picks its
  top idea by object IDENTITY (``if idea is top``) over the assessment's
  own list; re-parsing would break that, and the notifier's helpers take
  dicts throughout.
* :func:`src.events.digest._advisory_entries` attaches a digest-local
  ``corroboration`` block to a shallow copy of the idea. That is a render
  annotation, never persisted, so it stays a dict key (read alongside the
  typed fields in ``_idea_line``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Actions whose thesis is "the underlying goes UP". Shared by the impact
#: agent's direction inference, the niche merge, and :meth:`TradeIdea.is_bullish`.
BULLISH_ACTIONS = frozenset({"long", "buy_calls"})

#: Actions expressed as bought options (defined risk) rather than shares.
OPTION_ACTIONS = frozenset({"buy_calls", "buy_puts"})

#: The canonical key order of a persisted trade idea, in the exact order
#: ``_normalise_trade_ideas`` has always emitted it (json.dumps stability).
CORE_KEYS: tuple[str, ...] = (
    "ticker",
    "action",
    "direction",
    "confidence",
    "rationale",
    "time_horizon",
    "holding_period_days",
    "time_stop_days",
    "stop_loss_pct",
    "target_pct",
    "entry_trigger",
    "invalidation",
    "suggested_entry",
    "preferred_instrument",
    "notes",
)

#: The additive niche block, in ``NicheIdea.to_trade_idea`` order. Present
#: only on ideas the multi-hop niche pass merged in (CL-u2ph).
NICHE_KEYS: tuple[str, ...] = (
    "niche",
    "company_name",
    "hop_count",
    "torque_reason",
    "asymmetry_score",
    "liquidity_flag",
    "exchange",
    "robinhood_tradeable",
    "red_team_verdict",
)


def _as_str(value: Any) -> str:
    """Exactly the ``str(x or "")`` idiom the readers used — ANY falsy
    value (``None``, ``0``, ``False``, ``[]``) reads as ``""``, not as
    ``"0"``.

    Deliberately does NOT strip. Readers disagree about that (the ledger
    strips ``ticker``/``action``/``preferred_instrument``/``notes`` but
    not its seven other string reads), so stripping here would silently
    pick a side AND make the round-trip lossy. Both producers already
    strip every string field, so the callers that want a stripped value
    still ask for it explicitly."""
    return str(value) if value else ""


def _as_optional_str(value: Any) -> str | None:
    """Like :func:`_as_str` but preserves ``None`` — for the niche fields
    (``exchange``, ``red_team_verdict``) whose producer emits ``None``,
    not ``""``, when there is nothing to say."""
    if value is None:
        return None
    return str(value) if value else ""


def _as_float(value: Any) -> float | None:
    """Best-effort float; ``None`` when absent or unparseable (the
    ledger's ``_float_or_none`` contract)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    """Best-effort int; ``None`` when absent or unparseable — the ledger's
    former ``_int_or_none`` contract (note ``int("5.5")`` is unparseable,
    exactly as before)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_list(value: Any) -> list[float]:
    """``target_pct`` → a list of floats. Accepts the canonical list, a
    bare scalar, or garbage (→ ``[]``); unparseable members are skipped.
    Order is PRESERVED — the sort/dedup/cap belongs to the producer's
    ``_clean_targets``, not to a reader."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[float] = []
    for item in items:
        parsed = _as_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


@dataclass
class TradeIdea:
    """One advisory trade idea — the operator's action feed, and (via the
    ledger) what the Alpaca paper options executor buys.

    Field order IS the persisted key order; see :data:`CORE_KEYS` /
    :data:`NICHE_KEYS`. Numeric fields are ``| None`` because a reader can
    legitimately meet a row that never had them: the producer path always
    fills them (clamped) before persisting.
    """

    ticker: str = ""
    action: str = ""
    direction: str = ""
    confidence: float | None = None
    rationale: str = ""
    time_horizon: str = ""
    holding_period_days: str = ""
    time_stop_days: int | None = None
    stop_loss_pct: float | None = None
    target_pct: list[float] = field(default_factory=list)
    entry_trigger: str = ""
    invalidation: str = ""
    suggested_entry: str = ""
    preferred_instrument: str = ""
    notes: str = ""

    # -- niche annotations (CL-u2ph); emitted only when ``niche`` is set.
    niche: bool = False
    company_name: str = ""
    hop_count: int | None = None
    torque_reason: str = ""
    asymmetry_score: float | None = None
    liquidity_flag: bool = False
    exchange: str | None = None
    robinhood_tradeable: bool = False
    red_team_verdict: str | None = None
    research: dict[str, Any] | None = None

    # -- derived reads ---------------------------------------------------

    @property
    def is_bullish(self) -> bool:
        """True when the idea profits from the underlying RISING.
        ``action`` is authoritative; ``direction`` only breaks the tie on
        malformed input (same precedence as the trade-card grounder)."""
        action = self.action.strip().lower()
        if action in BULLISH_ACTIONS:
            return True
        if action in ("short", "buy_puts"):
            return False
        return self.direction.strip().lower() != "bearish"

    @property
    def is_options_action(self) -> bool:
        """True for ``buy_calls`` / ``buy_puts`` — the defined-risk legs
        the options executor can act on (and the only ones whose
        ``stop_loss_pct`` reads as a fraction of PREMIUM, not of price)."""
        return self.action.strip().lower() in OPTION_ACTIONS

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The canonical persisted dict — the exact keys, values and
        insertion order the hand-built dicts have always had."""
        out: dict[str, Any] = {
            "ticker": self.ticker,
            "action": self.action,
            "direction": self.direction,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "time_horizon": self.time_horizon,
            "holding_period_days": self.holding_period_days,
            "time_stop_days": self.time_stop_days,
            "stop_loss_pct": self.stop_loss_pct,
            "target_pct": list(self.target_pct),
            "entry_trigger": self.entry_trigger,
            "invalidation": self.invalidation,
            "suggested_entry": self.suggested_entry,
            "preferred_instrument": self.preferred_instrument,
            "notes": self.notes,
        }
        if not self.niche:
            return out
        # Additive niche block — existing consumers ignore unknown keys,
        # and a NON-niche idea must not grow them (CL-u2ph shape).
        out["niche"] = True
        out["company_name"] = self.company_name
        out["hop_count"] = self.hop_count
        out["torque_reason"] = self.torque_reason
        out["asymmetry_score"] = self.asymmetry_score
        out["liquidity_flag"] = self.liquidity_flag
        out["exchange"] = self.exchange
        out["robinhood_tradeable"] = self.robinhood_tradeable
        out["red_team_verdict"] = self.red_team_verdict
        if self.research is not None:
            out["research"] = self.research
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TradeIdea:
        """Rehydrate from an assessment/ledger idea dict. Tolerant by
        design (LLM-sourced data read back out of JSON): unknown keys are
        ignored, missing keys take the reader defaults, and an unparseable
        number becomes ``None`` rather than raising."""
        return cls(
            ticker=_as_str(payload.get("ticker")),
            action=_as_str(payload.get("action")),
            direction=_as_str(payload.get("direction")),
            confidence=_as_float(payload.get("confidence")),
            rationale=_as_str(payload.get("rationale")),
            time_horizon=_as_str(payload.get("time_horizon")),
            holding_period_days=_as_str(payload.get("holding_period_days")),
            time_stop_days=_as_int(payload.get("time_stop_days")),
            stop_loss_pct=_as_float(payload.get("stop_loss_pct")),
            target_pct=_as_float_list(payload.get("target_pct")),
            entry_trigger=_as_str(payload.get("entry_trigger")),
            invalidation=_as_str(payload.get("invalidation")),
            suggested_entry=_as_str(payload.get("suggested_entry")),
            preferred_instrument=_as_str(payload.get("preferred_instrument")),
            notes=_as_str(payload.get("notes")),
            niche=bool(payload.get("niche")),
            company_name=_as_str(payload.get("company_name")),
            hop_count=_as_int(payload.get("hop_count")),
            torque_reason=_as_str(payload.get("torque_reason")),
            asymmetry_score=_as_float(payload.get("asymmetry_score")),
            liquidity_flag=bool(payload.get("liquidity_flag")),
            exchange=_as_optional_str(payload.get("exchange")),
            robinhood_tradeable=bool(payload.get("robinhood_tradeable")),
            red_team_verdict=_as_optional_str(payload.get("red_team_verdict")),
            research=payload.get("research") if isinstance(payload.get("research"), dict) else None,
        )
