"""Event Impact Agent (CL-6iu7) — turns a NEW ``geo_events`` headline
into a structured tradability assessment.

Flow per event:
  1. Look up the row's matched playbook (theme set by the GDELT
     ingester) for pre-researched instrument mappings + direction hints.
  2. One strict-JSON LLM call (existing ``claude-code`` driver — rides
     the operator's subscription, no API billing).
  3. Defensive parse: extract JSON from prose/fences, schema-validate,
     clamp numeric ranges, drop unreachable instruments, force equities
     to watch-only.
  4. Write ``assessment`` + status ASSESSED (or DISMISSED with a
     rationale when the output is unusable).

The agent proposes; it never trades. The consumer half (confirmation +
strategy, sibling work) decides what to do with ASSESSED rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.events.playbooks import (
    DEFAULT_PLAYBOOKS_PATH,
    INSTRUMENT_RE,
    Playbook,
    all_tradable_instruments,
    load_playbooks,
)
from src.events.triage import EventTriage
from src.events.x_ingest import source_credibility_note
from src.research.llm import Message, get_client
from src.research.llm.client import LLMClient

logger = logging.getLogger(__name__)

#: Sonnet-class is plenty for single-headline classification and keeps
#: per-event subscription load small; override via EVENT_IMPACT_MODEL.
DEFAULT_MODEL = "claude-sonnet-4-6"

VALID_EVENT_DIRECTIONS = frozenset({"bullish", "bearish", "neutral"})
VALID_HORIZONS = frozenset({"minutes", "hours", "days"})
VALID_AFFECTED_KINDS = frozenset({"oanda", "fx", "equity_watch", "polymarket"})
VALID_AFFECTED_DIRECTIONS = frozenset({"long", "short", "watch"})

# -- advisory trade ideas (CL-01zt) — operator-facing, never machine-traded
VALID_IDEA_ACTIONS = frozenset({"long", "short", "buy_calls", "buy_puts"})
VALID_IDEA_HORIZONS = frozenset({"immediate", "short", "medium", "structural"})
_BULLISH_IDEA_ACTIONS = frozenset({"long", "buy_calls"})
#: Hard time-stop defaults (days) when the LLM omits one — event edge decays.
_DEFAULT_TIME_STOP_DAYS = {
    "immediate": 3, "short": 5, "medium": 20, "structural": 60,
}
MAX_TRADE_IDEAS = 8
MAX_FADE_CANDIDATES = 5
#: Concrete-levels caps (CL-jiqq) — the LLM supplies PERCENTAGES (it has
#: no live market data); the enrichment layer grounds them in real
#: prices. A stop worth more than the position is nonsense; a profit
#: target beyond a double on a news trade is fantasy. Clamp both.
MAX_STOP_LOSS_PCT = 0.90
MAX_TARGET_PCT = 3.0
MAX_TARGETS = 2

_SYSTEM_PROMPT = """\
You are the Event Impact Agent for an FX/CFD trading system. You receive
one news headline (from GDELT, so it is already ~15+ minutes old) plus a
pre-researched playbook for its matched theme. Your job is honest
interpretation, not speed: decide whether this specific headline is a
tradable macro event, which instruments it touches, and how urgent it is.

Rules:
- Respond with ONE JSON object and NOTHING else. No markdown fences, no
  commentary.
- Only propose instruments from the playbook mapping, or other clearly
  reachable OANDA CFD/FX symbols (format like EUR_USD, BCO_USD,
  XAU_USD, SPX500_USD).
- Equity tickers are ALERT-ONLY: kind "equity_watch", direction "watch".
  Never mark an equity long or short.
- Be skeptical: opinion pieces, retrospectives, market-recap articles,
  and stale news deserve direction "neutral", low urgency, and an empty
  or watch-only affected list. Confidence above 0.7 requires a concrete,
  new, physical or policy event.
- Be CONSERVATIVE: overstating impact destroys capital. When in doubt,
  lower urgency and confidence rather than raise them.
- Second-order thinking: this headline is already 15+ minutes old, so
  the obvious first move may be priced. When the primary instrument has
  likely already moved, prefer the SECOND-ORDER effects that reprice
  more slowly (substitute producers, input costs, freight, currencies).
- Prefer liquid instruments; propose a thin proxy only when nothing
  liquid carries the exposure.
- Primary-leg preference (applies to BOTH "affected" and "trade_ideas"):
  prefer the most specific, liquid, theme-aligned instrument as the
  primary leg. DRC/copper themes -> copper (XCU_USD) or the exposed
  miners; Taiwan -> semis (TSM/SOXX) + risk-off yen; Russia energy ->
  natural gas (NATGAS_USD)/defense; energy chokepoint/Hormuz -> crude
  (BCO_USD/WTICO_USD) + tanker names; Sahel -> gold MINERS (GOLD/BTG/
  EDV.TO), not bullion. Recommend pure safe-havens — gold (XAU_USD) and
  silver (XAG_USD) — ONLY when (a) the event is a broad, systemic
  risk-off move affecting many assets at once, or (b) no higher-priority
  theme-specific liquid leg carries the exposure. Do NOT include gold as
  a reflexive default just because an event is geopolitical. When gold IS
  included, it should be secondary to the theme-specific leg, not the
  headline.
- Territorial events (coups, nationalization, resource nationalism,
  license revocation, or war in a producing region — Africa,
  Russia/Ukraine): from the playbook context, name the specific
  companies whose ASSETS sit in the affected territory and put the
  asset/territory in each "reason" (e.g. "Kamoa-Kakula copper, DRC").
  Assess: (a) which mines/projects face nationalization, license
  revocation, or disruption; (b) which companies have material
  ownership or offtake agreements in that territory; (c) second-order
  effects on global supply of the commodity (cobalt, copper, gold,
  uranium, wheat, nickel, palladium); (d) whether Western alternative
  producers likely benefit.
- China-Taiwan events: prioritize TSMC / advanced-semiconductor supply
  chain impact, escalation vs routine grey-zone probability, Western
  equipment and defense beneficiaries, and negative China-revenue
  exposure. Routine drills are usually a fade, not a trade.
- Pharma API / drug-supply events: name which companies are
  China/India-INPUT exposed generics makers (TEVA, VTRS, RDY —
  disruption-BEARISH, put candidates on an input-supply shock) vs
  diversified / China+1 beneficiaries (TMO, PFE, JNJ — relatively
  resilient). Decide whether it is a specific-company hit (an FDA
  import alert / Form 483 on ONE plant → that name only) or a broad
  input-supply shock (export controls, feedstock cutoff → the generics
  cluster + XLV/XBI). No pharma OANDA leg exists; SPX500_USD only on
  major China escalation.
- Vintage honesty: company/asset facts (yours and the playbook's) are
  training-data vintage — ownership, permits, and mine status may have
  changed. Flag that uncertainty in the "reason" where it matters.
- Be selective in "affected": list only instruments THIS specific event
  genuinely moves — do not copy the entire playbook mapping.
- "direction" is the overall risk impulse of the event itself
  (bullish/bearish for risk assets, neutral if unclear); per-instrument
  direction lives in "affected".

Advisory trade ideas (operator-facing — the system never auto-trades
equities or options; these go to the operator's alert feed):
- Generate concrete, actionable ideas in "trade_ideas" — an explicit
  long / short / buy_calls / buy_puts per idea, not just directional
  bias. Prefer defined-risk options (buy_calls/buy_puts) when
  volatility is likely to spike or the move can reverse quickly;
  recommend shorting stock only for high-conviction, longer-horizon
  ideas with easy borrow.
- Every idea gets a time_horizon ("immediate" = minutes-hours news
  reaction; "short" = 1-5 trading days; "medium" = 1-4 weeks;
  "structural" = multi-month, rare), a holding_period_days range, and
  a HARD time_stop_days (immediate/short: 3-5; medium: 15-25 max —
  event edge decays fast).
- Rules of thumb: pure news reactions get short horizons and tight
  stops. Supply disruption (mines/shipping/chips) can justify medium.
  Nationalization/regime change is medium with high uncertainty.
  Knee-jerk broad risk-off (especially China headlines) is often an
  overreaction — put the fade in "fade_candidates". NEVER recommend
  shorts/puts on a name already up/down >12-15% on this news without
  a re-acceleration catalyst.
- Decision framework: sudden military escalation -> puts on exposed
  names + long defense (2-8 days; warn about post-spike vol crush).
  Nationalization/license revocation -> short/puts on the SPECIFIC
  exposed miner (5-20 days). Grain/commodity export disruption ->
  long commodity exposure (1-4 weeks). Structural defense-spending
  shift -> long defense (weeks-months, only when confirmed).
- Leave "trade_ideas"/"fade_candidates" empty when nothing is
  genuinely actionable.

Concrete, actionable detail (CL-jiqq) — the operator risks real money
off these, so each idea must carry enough to act on. YOU HAVE NO LIVE
MARKET DATA: you see a headline, not a quote screen. So give PERCENTAGES
and LOGIC, never fabricated prices, strikes, or expiry dates — the
system converts your percentages into real dollar levels from live
prices downstream. For every idea provide:
  * "stop_loss_pct": the adverse move that kills the trade, as a
    DECIMAL FRACTION. For stock, it is the % move in the share price
    (e.g. 0.07 = a 7% adverse move). For options it is the % of
    PREMIUM you are willing to lose (e.g. 0.40). Size it to the name's
    volatility and your horizon — tight for calm large-caps, wider for
    a volatile miner.
  * "target_pct": a LIST of 1-2 profit targets, each a DECIMAL FRACTION
    of favorable move in the UNDERLYING (e.g. [0.08, 0.15] = +8% then
    +15% for a long, or the down-move for a short/puts). First target
    should be realistic for the horizon; keep them sane (a news trade
    rarely runs past a double).
  * "entry_trigger": the CONDITION to enter, in words — not a fabricated
    price. E.g. "on confirmed blockade language", "after a retest of the
    breakdown level", "only if it holds below the prior-day low",
    "immediately — this is a clean gap catalyst". If it is
    "buy now, no trigger", say so.
  * "invalidation": the observable fact that KILLS the thesis (distinct
    from the price stop) — e.g. "official denial of the strike",
    "ceasefire announced", "company confirms the mine is unaffected",
    "reclaims the breakdown level on volume".
  * Option guidance in "preferred_instrument": give a MONEYNESS BAND and
    a DTE (days-to-expiry) WINDOW, NOT a specific strike or a calendar
    expiry date (you cannot know the listed chain). E.g. "slightly OTM
    puts (~5% below spot), 2-4 weeks to expiry" or "ATM-to-1-strike-OTM
    calls, 3-6 weeks out". The operator picks the nearest listed
    strike/expiry on their broker.
Keep confidence / time_horizon / time_stop_days as before.

JSON schema (all keys required except "trade_ideas" and
"fade_candidates", which are optional advisory extras; within a trade
idea, "stop_loss_pct" / "target_pct" / "entry_trigger" / "invalidation"
are strongly preferred but the system fills sane defaults if omitted):
{
  "core_event": "<one sentence: what actually happened>",
  "direction": "bullish" | "bearish" | "neutral",
  "urgency": <int 1-10>,
  "horizon": "minutes" | "hours" | "days",
  "confidence": <float 0.0-1.0>,
  "affected": [
    {"instrument": "<symbol>",
     "kind": "oanda" | "fx" | "equity_watch" | "polymarket",
     "direction": "long" | "short" | "watch",
     "reason": "<one line>"}
  ],
  "rationale": "<2-3 sentences: why this assessment>",
  "trade_ideas": [
    {"ticker": "<symbol>",
     "action": "long" | "short" | "buy_calls" | "buy_puts",
     "direction": "bullish" | "bearish",
     "confidence": <float 0.0-1.0>,
     "rationale": "<one line, name the territorial asset if relevant>",
     "time_horizon": "immediate" | "short" | "medium" | "structural",
     "holding_period_days": "<e.g. 2-7>",
     "time_stop_days": <int>,
     "stop_loss_pct": <float 0.0-0.9, adverse move: % of share price
                       for stock / % of premium for options>,
     "target_pct": [<float>, ...],  // 1-2 favorable-move fractions
     "entry_trigger": "<condition to enter, NOT a fabricated price>",
     "invalidation": "<observable fact that kills the thesis>",
     "suggested_entry": "<entry condition / level note>",
     "preferred_instrument": "<stock, or moneyness band + DTE window —
                              NO specific strike or expiry date>",
     "notes": "<key risks: vol crush, borrow, stale-facts caveats>"}
  ],
  "fade_candidates": [
    {"ticker": "<symbol>", "action": "<e.g. fade the spike>",
     "reason": "<one line: why this is an overreaction>"}
  ]
}
"""


@dataclass
class AssessmentResult:
    """Per-event outcome, for the entrypoint's one-line summaries."""

    event_id: int
    headline: str
    theme: str | None
    status: str  # ASSESSED | DISMISSED
    assessment: dict[str, Any]

    def summary_line(self) -> str:
        if self.status != "ASSESSED":
            reason = self.assessment.get("rationale", "")
            return (
                f"event id={self.event_id} DISMISSED ({reason[:80]}) "
                f"| {self.headline[:70]}"
            )
        a = self.assessment
        affected = ",".join(
            f"{x['instrument']}:{x['direction']}" for x in a.get("affected", [])
        ) or "-"
        return (
            f"event id={self.event_id} ASSESSED theme={self.theme} "
            f"dir={a.get('direction')} urg={a.get('urgency')} "
            f"hor={a.get('horizon')} conf={a.get('confidence'):.2f} "
            f"affected=[{affected}] | {self.headline[:70]}"
        )


# ---------------------------------------------------------------------- #
# Defensive parsing
# ---------------------------------------------------------------------- #


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Pull the first balanced top-level JSON object out of LLM text.

    Tolerates markdown fences and prose around the object. Raises
    ``ValueError`` when nothing parseable is found.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence line and any trailing fence.
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    if start < 0:
        msg = "no JSON object in LLM output"
        raise ValueError(msg)
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    msg = f"JSON object failed to parse: {exc}"
                    raise ValueError(msg) from exc
                if not isinstance(obj, dict):
                    msg = "top-level JSON is not an object"
                    raise ValueError(msg)
                return obj
    msg = "unbalanced JSON object in LLM output"
    raise ValueError(msg)


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(float(value)))))


def _clamp_float(value: Any, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _optional_fraction(value: Any, lo: float, hi: float) -> float | None:
    """Parse an optional decimal fraction (stop/target %), clamped to
    ``[lo, hi]``. ``None`` when absent or unparseable — a missing level
    is honest (the enrichment layer defaults it), a fabricated one is
    not. A value <= 0 is treated as unset (no zero-width stop)."""
    if value is None:
        return None
    try:
        frac = float(value)
    except (TypeError, ValueError):
        return None
    if frac <= 0.0:
        return None
    return max(lo, min(hi, frac))


def _clean_targets(raw: Any) -> list[float]:
    """A trade idea's ``target_pct`` list → up to ``MAX_TARGETS`` valid,
    positive, clamped, ascending-deduped fractions. Accepts a bare
    scalar (one target) too. Empty when nothing parses — never raises."""
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[float] = []
    for item in items:
        frac = _optional_fraction(item, 0.0, MAX_TARGET_PCT)
        if frac is not None and frac not in out:
            out.append(frac)
    out.sort()
    return out[:MAX_TARGETS]


def _normalise_trade_ideas(raw: Any) -> list[dict[str, Any]]:
    """Advisory-only ``trade_ideas`` — validate enums, clamp numbers,
    drop malformed entries INDIVIDUALLY (a bad idea never dismisses the
    assessment; ideas are operator advisory, not machine-traded)."""
    if not isinstance(raw, list):
        return []
    ideas: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker", "")).strip()
        action = str(entry.get("action", "")).strip().lower()
        horizon = str(entry.get("time_horizon", "")).strip().lower()
        if not ticker or action not in VALID_IDEA_ACTIONS:
            logger.debug("dropping trade idea (ticker/action): %r", entry)
            continue
        if horizon not in VALID_IDEA_HORIZONS:
            logger.debug("dropping trade idea (time_horizon): %r", entry)
            continue
        direction = str(entry.get("direction", "")).strip().lower()
        if direction not in ("bullish", "bearish"):
            direction = "bullish" if action in _BULLISH_IDEA_ACTIONS else "bearish"
        try:
            confidence = _clamp_float(entry.get("confidence"), 0.0, 1.0)
        except (TypeError, ValueError):
            confidence = 0.5  # advisory default, not worth dropping over
        try:
            time_stop = _clamp_int(entry.get("time_stop_days"), 1, 120)
        except (TypeError, ValueError):
            time_stop = _DEFAULT_TIME_STOP_DAYS[horizon]
        # Concrete levels (CL-jiqq) — the LLM's PERCENTAGES, clamped to
        # sane ranges; the selector/enrichment layer fills gaps and
        # grounds them in real prices. A malformed level never drops the
        # idea (advisory), it just goes absent → default fills later.
        stop_loss_pct = _optional_fraction(
            entry.get("stop_loss_pct"), 0.0, MAX_STOP_LOSS_PCT,
        )
        target_pct = _clean_targets(entry.get("target_pct"))
        ideas.append({
            "ticker": ticker,
            "action": action,
            "direction": direction,
            "confidence": confidence,
            "rationale": str(entry.get("rationale", "")).strip(),
            "time_horizon": horizon,
            "holding_period_days": str(entry.get("holding_period_days", "")).strip(),
            "time_stop_days": time_stop,
            "stop_loss_pct": stop_loss_pct,
            "target_pct": target_pct,
            "entry_trigger": str(entry.get("entry_trigger", "")).strip(),
            "invalidation": str(entry.get("invalidation", "")).strip(),
            "suggested_entry": str(entry.get("suggested_entry", "")).strip(),
            "preferred_instrument": str(entry.get("preferred_instrument", "")).strip(),
            "notes": str(entry.get("notes", "")).strip(),
        })
        if len(ideas) >= MAX_TRADE_IDEAS:
            break
    return ideas


def _normalise_fade_candidates(raw: Any) -> list[dict[str, str]]:
    """Advisory ``fade_candidates`` — overreaction fades; same
    drop-individually posture as trade ideas."""
    if not isinstance(raw, list):
        return []
    fades: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker", "")).strip()
        if not ticker:
            continue
        fades.append({
            "ticker": ticker,
            "action": str(entry.get("action", "")).strip(),
            "reason": str(entry.get("reason", "")).strip(),
        })
        if len(fades) >= MAX_FADE_CANDIDATES:
            break
    return fades


def normalise_assessment(
    payload: dict[str, Any],
    headline: str,
    playbook: Playbook | None,
    fallback_tradables: set[str],
) -> dict[str, Any]:
    """Schema-validate + clamp a parsed LLM payload into the shared
    ``assessment`` shape. Raises ``ValueError`` on unrecoverable schema
    violations (→ row is DISMISSED); silently drops bad ``affected``
    entries and clamps out-of-range numbers.
    """
    direction = str(payload.get("direction", "")).strip().lower()
    if direction not in VALID_EVENT_DIRECTIONS:
        msg = f"invalid direction {direction!r}"
        raise ValueError(msg)

    horizon = str(payload.get("horizon", "")).strip().lower().rstrip("s") + "s"
    if horizon not in VALID_HORIZONS:
        msg = f"invalid horizon {payload.get('horizon')!r}"
        raise ValueError(msg)

    try:
        urgency = _clamp_int(payload.get("urgency"), 1, 10)
        confidence = _clamp_float(payload.get("confidence"), 0.0, 1.0)
    except (TypeError, ValueError) as exc:
        msg = f"non-numeric urgency/confidence: {exc}"
        raise ValueError(msg) from exc

    core_event = str(payload.get("core_event", "")).strip() or headline
    rationale = str(payload.get("rationale", "")).strip()

    raw_affected = payload.get("affected")
    if raw_affected is None:
        raw_affected = []
    if not isinstance(raw_affected, list):
        msg = "'affected' is not a list"
        raise ValueError(msg)

    playbook_instruments = (
        {i.instrument for i in playbook.instruments} if playbook else set()
    )
    affected: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in raw_affected:
        if not isinstance(entry, dict):
            continue
        instrument = str(entry.get("instrument", "")).strip()
        kind = str(entry.get("kind", "")).strip().lower()
        adir = str(entry.get("direction", "")).strip().lower()
        reason = str(entry.get("reason", "")).strip()
        if not instrument or kind not in VALID_AFFECTED_KINDS:
            logger.debug("dropping affected entry (bad kind): %r", entry)
            continue
        if adir not in VALID_AFFECTED_DIRECTIONS:
            adir = "watch"
        if kind == "equity_watch":
            adir = "watch"  # equities are alert-only, always
        if kind in ("oanda", "fx"):
            instrument = instrument.upper().replace("/", "_").replace("-", "_")
            if not INSTRUMENT_RE.match(instrument):
                logger.debug("dropping unreachable tradable: %r", entry)
                continue
            # Whitelist gate: playbook mapping first, any playbook's
            # tradable second. An OANDA-shaped symbol outside both is
            # demoted to watch rather than dropped — the shape says
            # reachable, but nobody pre-researched it.
            known = instrument in playbook_instruments or instrument in fallback_tradables
            if not known and adir != "watch":
                logger.debug(
                    "demoting un-vetted tradable %s to watch", instrument,
                )
                adir = "watch"
        if instrument in seen:
            continue
        seen.add(instrument)
        affected.append({
            "instrument": instrument,
            "kind": kind,
            "direction": adir,
            "reason": reason,
        })

    return {
        "core_event": core_event,
        "direction": direction,
        "urgency": urgency,
        "horizon": horizon,
        "confidence": confidence,
        "affected": affected,
        "rationale": rationale,
        # Optional advisory extras (CL-01zt) — always present as lists so
        # downstream .get() consumers see a stable shape; empty when the
        # LLM offered nothing actionable.
        "trade_ideas": _normalise_trade_ideas(payload.get("trade_ideas")),
        "fade_candidates": _normalise_fade_candidates(
            payload.get("fade_candidates"),
        ),
    }


# ---------------------------------------------------------------------- #
# The agent
# ---------------------------------------------------------------------- #


class EventImpactAgent:
    """Assess NEW ``geo_events`` rows via the existing LLM stack."""

    def __init__(
        self,
        engine: Engine,
        client: LLMClient | None = None,
        model: str = DEFAULT_MODEL,
        playbooks_path: Path | str = DEFAULT_PLAYBOOKS_PATH,
        max_tokens: int = 2000,
        triage: EventTriage | None = None,
    ) -> None:
        self.engine = engine
        self.client = client if client is not None else get_client("claude-code")
        self.model = model
        self.max_tokens = max_tokens
        self.playbooks = load_playbooks(playbooks_path)
        self._fallback_tradables = all_tradable_instruments(self.playbooks)
        # Fast triage tier (CL-cunh): a cheap Haiku batch pre-filter that lets
        # us skip the expensive full assessment on clearly-irrelevant events.
        # Shares this agent's subscription client. Disabled unless
        # EVENT_TRIAGE_ENABLED is set (opt-in), so existing callers are
        # unchanged until the operator turns it on.
        self.triage = (
            triage if triage is not None
            else EventTriage(client=self.client)
        )

    # -- prompt ---------------------------------------------------------

    def _playbook_context(self, theme: str | None) -> str:
        pb = self.playbooks.get(theme or "")
        if pb is None:
            lines = [
                "No playbook matched this headline's theme. Only propose",
                "instruments from this cross-theme whitelist (or clearly",
                "reachable OANDA symbols), and lean neutral/low-confidence:",
                ", ".join(sorted(self._fallback_tradables)),
            ]
            return "\n".join(lines)
        lines = [
            f"Matched playbook: {pb.key} — {pb.name}",
            pb.description,
            "Instrument mapping (direction = historical hint, not an order):",
        ]
        for inst in pb.instruments:
            lines.append(
                f"  - {inst.instrument} [{inst.kind}] hint={inst.direction}: "
                f"{inst.rationale}"
            )
        return "\n".join(lines)

    def _user_prompt(self, row: dict[str, Any]) -> str:
        # Provenance line (CL-esyo): rows sourced from the X watchlist
        # ("x:<handle>") carry a one-line source-credibility note so the
        # model can calibrate confidence on a fast-but-unconfirmed
        # headline account vs a noisy OSINT feed. GDELT rows get nothing.
        source_line = ""
        note = source_credibility_note(str(row.get("source") or ""))
        if note:
            source_line = f"{note}\n"
        return (
            f"HEADLINE: {row['headline']}\n"
            f"URL: {row.get('url') or 'n/a'}\n"
            f"SEEN AT (UTC): {row.get('seen_at')}\n"
            f"THEME: {row.get('theme') or 'unmatched'}\n"
            f"{source_line}\n"
            f"{self._playbook_context(row.get('theme'))}\n\n"
            "Assess this event now. Respond with the JSON object only."
        )

    # -- assessment -----------------------------------------------------

    def assess_row(self, row: dict[str, Any]) -> AssessmentResult:
        """LLM-assess one row (no DB write). Returns the outcome.

        Failure semantics (learned the hard way — a subscription
        usage-window outage once terminally DISMISSED ~175 events):
        TRANSPORT failures (the LLM call itself raised — CLI exit,
        quota window, network) leave the row NEW so the next cycle
        retries it. Only failures of a SUCCESSFUL response (JSON
        extraction / schema validation) dismiss, because retrying
        those reproduces the same bad output.
        """
        theme = row.get("theme")
        playbook = self.playbooks.get(theme or "")
        try:
            resp = self.client.complete(
                messages=[
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(role="user", content=self._user_prompt(row)),
                ],
                model=self.model,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "impact agent transport failure for event id=%s "
                "(left NEW for retry): %s", row.get("id"), str(exc)[:200],
            )
            return AssessmentResult(
                event_id=int(row["id"]),
                headline=str(row["headline"]),
                theme=theme,
                assessment={},
                status="NEW",  # no-op transition — row stays queued
            )
        try:
            payload = extract_json_object(resp.text)
            assessment = normalise_assessment(
                payload, row["headline"], playbook, self._fallback_tradables,
            )
            status = "ASSESSED"
        except Exception as exc:
            # Content failure on a successful response → DISMISSED with
            # a rationale; retrying would reproduce the same output.
            logger.warning(
                "impact agent dismissed event id=%s: %s", row.get("id"), exc,
            )
            assessment = {
                "rationale": f"impact agent failure: {exc}"[:500],
            }
            status = "DISMISSED"
        return AssessmentResult(
            event_id=int(row["id"]),
            headline=str(row["headline"]),
            theme=theme,
            status=status,
            assessment=assessment,
        )

    def assess_new_events(self, limit: int = 20) -> list[AssessmentResult]:
        """Process up to ``limit`` NEW rows (newest first — freshest
        events are the only ones with any edge left) and persist each
        outcome. Returns per-event results for summary logging.

        When the triage tier is enabled (CL-cunh), the whole batch is first
        scored in one cheap Haiku call; events that fall below the relevance
        bar are DISMISSED without paying for the full assessment. Triage fails
        OPEN — a missing verdict escalates — so this can only save cost, never
        silently drop a real event.
        """
        with self.engine.connect() as conn:
            rows = [
                dict(r._mapping)
                for r in conn.execute(text(
                    "SELECT id, seen_at, headline, url, theme, source "
                    "FROM geo_events WHERE status = 'NEW' "
                    "ORDER BY seen_at DESC LIMIT :lim",
                ), {"lim": limit})
            ]

        verdicts: dict[int, Any] = {}
        if self.triage is not None and self.triage.enabled and rows:
            verdicts = self.triage.score_batch(rows)
            skipped = sum(
                1 for r in rows
                if (v := verdicts.get(int(r["id"]))) is not None
                and not v.escalate
            )
            logger.info(
                "triage: %d scored, %d escalated, %d skipped (model=%s)",
                len(rows), len(rows) - skipped, skipped, self.triage.model,
            )

        results: list[AssessmentResult] = []
        for row in rows:
            verdict = verdicts.get(int(row["id"]))
            if verdict is not None and not verdict.escalate:
                # Triaged out — cheap DISMISS, no expensive assessment call.
                logger.debug(
                    "triaged out event id=%s (relevance=%d): %.70s",
                    row.get("id"), verdict.relevance, row.get("headline"),
                )
                result = AssessmentResult(
                    event_id=int(row["id"]),
                    headline=str(row["headline"]),
                    theme=row.get("theme"),
                    status="DISMISSED",
                    assessment={
                        "rationale": (
                            f"triaged out (relevance={verdict.relevance}): "
                            f"{verdict.reason}"
                        )[:500],
                        "triaged": True,
                        "triage_relevance": verdict.relevance,
                    },
                )
                self._persist(result)
                results.append(result)
                continue
            result = self.assess_row(row)
            if result.status != "NEW":  # transport failure = no write,
                self._persist(result)   # row stays queued for retry
            results.append(result)
        return results

    def _persist(self, result: AssessmentResult) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(
                "UPDATE geo_events "
                "SET assessment = :assessment, status = :status, "
                "    status_updated_at = :now "
                "WHERE id = :id AND status = 'NEW'",
            ), {
                "assessment": json.dumps(result.assessment),
                "status": result.status,
                "now": datetime.now(UTC),
                "id": result.event_id,
            })
