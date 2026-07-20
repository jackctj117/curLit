"""Horizon-based instrument-preference normalizer (CL-mgcp).

The impact agent's advisory ``trade_ideas`` sometimes arrive with a
missing ``preferred_instrument`` (and never carry a stop-loss). This
module turns a direction + expected holding period into a concrete,
defensible instrument decision so the ledger and the operator alerts
are never blank where it matters:

  * holding <= 8 days       → defined-risk options (buy_puts /
                              buy_calls), stop = 40% of premium,
                              1-3 weeks to expiry — event moves this
                              fast can gap through any stock stop.
  * 9-25 days               → still defined-risk options by default
                              (stop = 35% of premium, 3-6 weeks to
                              expiry); with ``prefer_defined_risk=False``
                              → stock (long/short) with a 10% stop.
  * > 25 days / structural  → stock (long/short) with a 12% stop —
                              option theta over a multi-month thesis
                              costs more than the gap protection buys.

IV honesty: ``iv_rank`` is accepted but INERT unless the caller
actually supplies a value — this codebase has NO options data source
today, so there is no default, no fake proxy, no silent guess. When a
real IV source lands and passes ``iv_rank >= 75``, an options decision
flips to stock (paying top-quartile premium for defined risk is
usually the worse trade). ``iv_rank=None`` means the IV overlay simply
does not run.

The selector NEVER overrides the LLM's own ``action``. It is used by
:mod:`src.events.idea_ledger` to (a) fill gaps — missing
``preferred_instrument`` / ``stop_loss_pct`` — and (b) annotate
disagreements into the idea's ``notes``
("selector prefers buy_puts (short horizon)").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Holding-period band edges (calendar days).
SHORT_MAX_DAYS = 8
MEDIUM_MAX_DAYS = 25

#: IV-rank threshold above which options are "expensive" — only ever
#: consulted when the caller supplies a real iv_rank (no data source
#: exists yet; see module docstring).
HIGH_IV_RANK = 75.0

#: Stop-loss fractions. Options stops are a fraction OF PREMIUM PAID
#: (defined risk: the premium is the max loss; the stop just exits the
#: decay earlier). Stock stops are a fraction of entry price.
OPTION_STOP_SHORT = 0.40
OPTION_STOP_MEDIUM = 0.35
STOCK_STOP_MEDIUM = 0.10
STOCK_STOP_LONG = 0.12

_BULLISH = frozenset({"bullish", "long", "buy_calls"})
_BEARISH = frozenset({"bearish", "short", "buy_puts"})

#: Fallback expected-holding-days per LLM time_horizon, used by
#: :func:`decision_for_idea` when the idea has no parseable
#: holding_period_days and no time_stop_days.
HORIZON_DEFAULT_DAYS = {
    "immediate": 2.0,
    "short": 4.0,
    "medium": 15.0,
    "structural": 45.0,
}

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class InstrumentDecision:
    """The selector's verdict for one (direction, horizon) input.

    ``stop_loss_pct`` is a fraction of premium for options actions and
    a fraction of entry price for stock actions — the
    ``preferred_instrument`` text says which reading applies.
    ``horizon_band`` is 'short' | 'medium' | 'structural' (the band the
    holding period fell into, used for disagreement notes).
    """

    action: str  # buy_puts | buy_calls | long | short
    preferred_instrument: str
    reason: str
    stop_loss_pct: float
    horizon_band: str

    @property
    def is_options(self) -> bool:
        return self.action in ("buy_puts", "buy_calls")


def _normalise_direction(direction: str) -> str:
    d = str(direction).strip().lower()
    if d in _BULLISH:
        return "bullish"
    if d in _BEARISH:
        return "bearish"
    msg = f"unrecognised direction {direction!r} (want bullish/bearish)"
    raise ValueError(msg)


def choose_instrument(
    direction: str,
    expected_holding_days: float,
    iv_rank: float | None = None,
    prefer_defined_risk: bool = True,
) -> InstrumentDecision:
    """Pick the instrument for a directional idea held ``expected_holding_days``.

    ``direction`` accepts bullish/long/buy_calls or bearish/short/
    buy_puts (raises ``ValueError`` otherwise). Non-positive holding
    periods clamp to 1 day (shortest band) rather than raising — the
    input is LLM-derived data, not operator config.

    ``iv_rank``: see module docstring — inert when ``None``.
    """
    bearish = _normalise_direction(direction) == "bearish"
    days = max(1.0, float(expected_holding_days))

    if days <= SHORT_MAX_DAYS:
        band = "short"
        option_action = "buy_puts" if bearish else "buy_calls"
        decision = InstrumentDecision(
            action=option_action,
            preferred_instrument=(
                f"{'puts' if bearish else 'calls'}, 1-3 weeks to expiry"
            ),
            reason=(
                f"short horizon (~{days:g}d): defined-risk options — "
                f"premium caps the loss if the event gaps; stop at "
                f"{OPTION_STOP_SHORT:.0%} of premium"
            ),
            stop_loss_pct=OPTION_STOP_SHORT,
            horizon_band=band,
        )
    elif days <= MEDIUM_MAX_DAYS:
        band = "medium"
        if prefer_defined_risk:
            option_action = "buy_puts" if bearish else "buy_calls"
            decision = InstrumentDecision(
                action=option_action,
                preferred_instrument=(
                    f"{'puts' if bearish else 'calls'}, 3-6 weeks to expiry"
                ),
                reason=(
                    f"medium horizon (~{days:g}d): defined-risk options "
                    f"preferred; stop at {OPTION_STOP_MEDIUM:.0%} of premium"
                ),
                stop_loss_pct=OPTION_STOP_MEDIUM,
                horizon_band=band,
            )
        else:
            decision = InstrumentDecision(
                action="short" if bearish else "long",
                preferred_instrument="stock",
                reason=(
                    f"medium horizon (~{days:g}d), defined risk not "
                    f"required: stock with a {STOCK_STOP_MEDIUM:.0%} stop"
                ),
                stop_loss_pct=STOCK_STOP_MEDIUM,
                horizon_band=band,
            )
    else:
        band = "structural"
        decision = InstrumentDecision(
            action="short" if bearish else "long",
            preferred_instrument="stock",
            reason=(
                f"structural horizon (~{days:g}d): stock — option theta "
                f"over months costs more than the gap protection buys; "
                f"{STOCK_STOP_LONG:.0%} stop"
            ),
            stop_loss_pct=STOCK_STOP_LONG,
            horizon_band=band,
        )

    # IV overlay — ONLY when a real iv_rank was supplied (no options
    # data source exists yet; None disables the overlay entirely).
    if iv_rank is not None and decision.is_options and float(iv_rank) >= HIGH_IV_RANK:
        stock_stop = STOCK_STOP_MEDIUM if days <= MEDIUM_MAX_DAYS else STOCK_STOP_LONG
        decision = InstrumentDecision(
            action="short" if bearish else "long",
            preferred_instrument="stock",
            reason=(
                f"IV rank {float(iv_rank):.0f} >= {HIGH_IV_RANK:.0f}: options "
                f"premium is top-quartile expensive — stock with a "
                f"{stock_stop:.0%} stop instead ({decision.reason})"
            ),
            stop_loss_pct=stock_stop,
            horizon_band=band,
        )
    return decision


def parse_holding_days(holding_period_days: Any) -> float | None:
    """Best-effort parse of the LLM's free-text holding period.

    '2-7' → 4.5 (midpoint), '5' → 5.0, '1-2 weeks' → 1.5 (numbers only
    — no unit handling; the agent prompt asks for days). ``None`` when
    nothing numeric is present.
    """
    nums = [float(m) for m in _NUM_RE.findall(str(holding_period_days or ""))]
    if not nums:
        return None
    return (min(nums) + max(nums)) / 2.0


def decision_for_idea(idea: dict[str, Any]) -> InstrumentDecision | None:
    """Selector decision for one normalised trade idea, or ``None``
    when the idea has no usable direction (never raises — ideas are
    advisory LLM output, a bad one just goes un-annotated).

    Expected holding days: parsed ``holding_period_days`` midpoint,
    else ``time_stop_days``, else the horizon-band default.
    """
    direction = str(idea.get("direction") or idea.get("action") or "")
    days = parse_holding_days(idea.get("holding_period_days"))
    if days is None:
        try:
            days = float(idea["time_stop_days"])
        except (KeyError, TypeError, ValueError):
            days = HORIZON_DEFAULT_DAYS.get(
                str(idea.get("time_horizon") or "").strip().lower(), 4.0,
            )
    try:
        return choose_instrument(direction, days)
    except ValueError:
        return None
