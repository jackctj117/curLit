"""Pure niche-idea scoring, parsing and verification (CL-u2ph).

The PURE half of the multi-hop niche pass, extracted from
``niche_agent.py`` per CL-ikz2 (code review 2026-07-21 §6.2.2): no LLM
calls, no HTTP, no yfinance — just dataclasses and deterministic
functions over already-fetched data, so it unit-tests without any I/O.
``niche_agent.py`` keeps the prompts / LLM cycles / market-data fetch
and re-exports these names for backward compatibility.

Contents:

* :class:`AsymmetryConfig` / :class:`NicheIdea` — the tunable weights
  and the idea record (with its ``trade_ideas`` merge shape);
* :func:`parse_niche_ideas` — defensive parse of the LLM JSON reply;
* :func:`verify_ideas` — the anti-hallucination guard against the real
  :class:`SymbolUniverse` (CL-tzug); NO unverified ticker survives;
* :func:`torque_from_reason` / :func:`asymmetry_score` /
  :func:`score_and_gate` — hop/torque/smallness scoring under the
  liquidity floor, and the surface/log gate.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.events._util import clamp_float, clamp_int
from src.events.impact_agent import extract_json_object
from src.events.research_evidence import (
    CLAIM_ROLES,
    MAX_MARKET_AGE_DAYS,
    RelationshipClaim,
    SourceDocument,
    finite_nonnegative,
    parse_claims,
    timestamp,
)
from src.events.trade_idea import BULLISH_ACTIONS, TradeIdea

logger = logging.getLogger(__name__)

#: Cap on niche ideas the LLM may return (bounds a pathological reply).
MAX_NICHE_IDEAS = 6

VALID_NICHE_ACTIONS = frozenset({"long", "short", "buy_calls", "buy_puts"})
#: Same set the typed idea model and the impact agent use (CL-59mk).
_BULLISH_NICHE_ACTIONS = BULLISH_ACTIONS

#: Qualitative torque phrasing → a 0-1 numeric prior. The LLM supplies
#: prose ("torque_reason"); we look for leverage keywords rather than
#: trusting a self-scored number it has no calibration for.
_TORQUE_KEYWORDS: dict[str, float] = {
    "single-asset": 1.0,
    "single asset": 1.0,
    "pure-play": 0.9,
    "pure play": 0.9,
    "one mine": 1.0,
    "sole": 0.9,
    "only producer": 0.9,
    "high fixed cost": 0.8,
    "operating leverage": 0.85,
    "operational leverage": 0.85,
    "financial leverage": 0.8,
    "levered": 0.8,
    "high debt": 0.75,
    "net debt": 0.7,
    "royalty": 0.7,
    "streaming": 0.7,
    "offtake": 0.7,
    "junior": 0.85,
    "microcap": 0.8,
    "micro-cap": 0.8,
    "small-cap": 0.6,
    "small cap": 0.6,
    "under-followed": 0.7,
    "underfollowed": 0.7,
    "no coverage": 0.8,
    "uncovered": 0.75,
    "non-consensus": 0.6,
    "sole supplier": 1.0,
    "bottleneck": 0.85,
    "chokepoint": 0.8,
    "convert": 0.6,
    "convertible": 0.6,
    "distress": 0.75,
}


@dataclass(frozen=True)
class AsymmetryConfig:
    """Tunable weights + floors for :func:`asymmetry_score`.

    Every field is env/CLI-populatable so the operator can retune without
    a code change. Defaults are deliberately conservative — the whole
    point is to reward genuine under-followed-ness while REFUSING to
    reward an illiquid shell that merely looks under-followed.
    """

    #: Weights on the three score components (need not sum to 1; the
    #: score is a plain weighted sum, then clamped to [0, 1]).
    hop_weight: float = 0.30
    torque_weight: float = 0.35
    smallness_weight: float = 0.35

    #: hop_count is capped here before normalising (hop 1 ≈ obvious
    #: adjacent, hop 5+ ≈ maximally overlooked); more hops = more torque
    #: prior, but with diminishing returns past the cap.
    hop_cap: int = 5

    #: LIQUIDITY FLOOR. Below this average daily dollar volume the name
    #: is flagged illiquid (a trap, not an edge) — the smallness bonus is
    #: killed and a flat penalty applied. Default ~$2M/day.
    min_avg_dollar_volume: float = 2_000_000.0

    #: Flat penalty subtracted from the score when the liquidity floor is
    #: breached (on top of zeroing the smallness bonus).
    illiquid_penalty: float = 0.40

    #: Market-cap band for the "smallness" (under-followed) bonus. A cap
    #: at/under ``small_cap_ceiling`` earns the full smallness weight; at/
    #: over ``large_cap_floor`` earns none; linear in log-space between.
    small_cap_ceiling: float = 300_000_000.0  # $300M — micro/small
    large_cap_floor: float = 20_000_000_000.0  # $20B — well-covered

    #: Only ideas scoring >= this surface; the rest are logged, not shown.
    asymmetry_threshold: float = 0.45

    #: Hard drop (not just flag) an idea whose avg $volume is below this
    #: absolute floor — too thin to advise at any score. Default $250k/day.
    drop_below_dollar_volume: float = 250_000.0


@dataclass
class NicheIdea:
    """One verified, scored niche idea. Serialises to the merged
    ``trade_ideas`` shape via :meth:`to_trade_idea`."""

    ticker: str
    company_name: str
    action: str
    direction: str
    hop_count: int
    torque_reason: str
    rationale: str
    confidence: float
    #: Set by verification.
    verified: bool = False
    corrected: bool = False
    exchange: str | None = None
    robinhood_tradeable: bool = False
    #: Set by scoring.
    asymmetry_score: float | None = None
    liquidity_flag: bool = False
    market_cap: float | None = None
    avg_dollar_volume: float | None = None
    dropped_reason: str | None = None
    #: Set by the adversarial red-team critic (CL-3v56) on survivors.
    red_team_note: str = ""
    red_team_verdict: str = ""
    #: For the honest self-scored components (debugging / logging).
    components: dict[str, float] = field(default_factory=dict)
    claims: list[RelationshipClaim] = field(default_factory=list)
    claims_parse_error: bool = False
    sources: list[SourceDocument] = field(default_factory=list)
    discovery_status: str = "not_run"
    evidence_status: str = "insufficient_evidence"
    review_status: str = "not_requested"
    review_reason: str = ""
    review_provenance: dict[str, Any] = field(default_factory=dict)
    liquidity_status: str = "unknown"
    market_observed_at: str | None = None
    market_received_at: str | None = None
    score_version: str = "unscored"
    last_close: float | None = None

    @property
    def research_eligible(self) -> bool:
        return (
            self.verified
            and self.discovery_status == "completed"
            and self.evidence_status == "source_backed"
            and self.review_status == "supported"
            and self.liquidity_status == "sufficient"
            and self.dropped_reason is None
        )

    def research_record(self) -> dict[str, Any]:
        return {
            "discovery_status": self.discovery_status,
            "review_status": self.review_status,
            "review_reason": self.review_reason,
            "review_provenance": self.review_provenance,
            "evidence_status": self.evidence_status,
            "liquidity_status": self.liquidity_status,
            "market_observed_at": self.market_observed_at,
            "market_received_at": self.market_received_at,
            "claims": [asdict(c) for c in self.claims],
            "claims_parse_error": self.claims_parse_error,
            "sources": [d.to_dict() for d in self.sources],
            "identity_verified": self.verified,
            "eligible": self.research_eligible,
            "score_version": self.score_version,
            "components": self.components,
            "avg_dollar_volume": self.avg_dollar_volume,
            "last_close": self.last_close,
            "market_cap": self.market_cap,
            "dropped_reason": self.dropped_reason,
        }

    def to_trade_idea(self) -> dict[str, Any]:
        """Merge shape for the assessment ``trade_ideas`` list. Carries
        the standard idea fields the ledger/digest expect PLUS the niche
        tags. ``time_horizon`` defaults to "short" (a niche event trade
        is tactical) so the ledger's selector/expiry logic has a value.

        Built as a :class:`src.events.trade_idea.TradeIdea` (CL-59mk) so
        the merged entry is the SAME type the impact agent emits — the
        dict it serialises to is unchanged, niche block included."""
        bullish = self.action in _BULLISH_NICHE_ACTIONS or self.direction == "bullish"
        # Fold the surviving red-team attack (CL-3v56) into the operator-visible
        # notes so the bear case rides along with the idea, not just the bull.
        notes = self.torque_reason
        if self.red_team_note:
            notes = f"{notes} | red-team ({self.review_status}): {self.red_team_note}"
        return TradeIdea(
            ticker=self.ticker,
            action=self.action,
            direction="bullish" if bullish else "bearish",
            confidence=self.confidence,
            rationale=self.rationale,
            time_horizon="short",
            holding_period_days="",
            time_stop_days=10,
            stop_loss_pct=None,
            target_pct=[],
            entry_trigger="",
            invalidation="",
            suggested_entry="",
            preferred_instrument="",
            notes=notes,
            # -- niche tags (additive; existing consumers ignore unknown keys)
            niche=True,
            company_name=self.company_name,
            hop_count=self.hop_count,
            torque_reason=self.torque_reason,
            asymmetry_score=self.asymmetry_score,
            liquidity_flag=self.liquidity_flag,
            exchange=self.exchange,
            robinhood_tradeable=self.robinhood_tradeable,
            red_team_verdict=self.red_team_verdict or None,
            research=self.research_record(),
        ).to_dict()


# ---------------------------------------------------------------------- #
# Defensive parse
# ---------------------------------------------------------------------- #


def parse_niche_ideas(raw_text: str) -> list[NicheIdea]:
    """Defensive parse of the LLM reply into :class:`NicheIdea` objects.

    Extracts the JSON object (tolerating fences/prose), then validates
    each idea INDIVIDUALLY — a malformed entry is dropped, never fatal
    (same posture as the impact agent's idea normaliser). An idea needs
    at minimum a ticker OR a company_name plus a valid action; direction
    is inferred from the action when absent/invalid.
    """
    try:
        payload = extract_json_object(raw_text)
    except ValueError:
        logger.warning("niche agent: no parseable JSON in LLM output; 0 ideas")
        return []
    raw = payload.get("niche_ideas")
    if not isinstance(raw, list):
        logger.debug("niche agent: 'niche_ideas' missing/not a list")
        return []
    ideas: list[NicheIdea] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker", "")).strip().upper()
        company = str(entry.get("company_name", "")).strip()
        action = str(entry.get("action", "")).strip().lower()
        if action not in VALID_NICHE_ACTIONS:
            logger.debug("niche agent: dropping idea (bad action): %r", entry)
            continue
        if not ticker and not company:
            logger.debug("niche agent: dropping idea (no ticker/company): %r", entry)
            continue
        direction = str(entry.get("direction", "")).strip().lower()
        if direction not in ("bullish", "bearish"):
            direction = "bullish" if action in _BULLISH_NICHE_ACTIONS else "bearish"
        claims = parse_claims(entry.get("claims"))
        ideas.append(
            NicheIdea(
                ticker=ticker,
                company_name=company,
                action=action,
                direction=direction,
                hop_count=clamp_int(entry.get("hop_count"), 1, 8, 1),
                torque_reason=str(entry.get("torque_reason", "")).strip(),
                rationale=str(entry.get("rationale", "")).strip(),
                confidence=clamp_float(entry.get("confidence"), 0.0, 1.0, 0.4),
                claims=claims,
                claims_parse_error=(
                    not isinstance(entry.get("claims"), list) or len(claims) != len(entry["claims"])
                ),
            )
        )
        if len(ideas) >= MAX_NICHE_IDEAS:
            break
    return ideas


def _idea_key(idea: NicheIdea) -> str:
    """Cross-cycle dedup key (CL-2dnf): the ticker when present, else the
    normalized company name — so the same name proposed in a later hopping
    cycle is recognised and not re-counted as a fresh discovery."""
    return idea.ticker.upper() if idea.ticker else idea.company_name.strip().lower()


# ---------------------------------------------------------------------- #
# Verification (anti-hallucination guard)
# ---------------------------------------------------------------------- #


def verify_ideas(ideas: list[NicheIdea], universe: Any) -> list[NicheIdea]:
    """Anti-hallucination guard — resolve every proposed ticker against
    the real US-listed :class:`SymbolUniverse`. Returns ONLY survivors;
    NO unverified ticker survives.

    Per idea:
      (a) ``universe.exists(ticker)`` → KEEP, tag exchange +
          robinhood_tradeable.
      (b) else ``universe.resolve_name(company_name)`` → if a confident
          match, CORRECT the ticker to the resolved symbol (logged),
          then KEEP.
      (c) else DROP (logged 'unverified niche ticker <t>/<company>
          dropped').
    """
    survivors: list[NicheIdea] = []
    for idea in ideas:
        # (a) direct hit
        if idea.ticker and universe.exists(idea.ticker):
            info = universe.get(idea.ticker) or {}
            idea.verified = True
            idea.exchange = info.get("exchange")
            idea.robinhood_tradeable = bool(universe.robinhood_tradeable(idea.ticker))
            survivors.append(idea)
            continue
        # (b) resolve by company name → correct the ticker
        matches = universe.resolve_name(idea.company_name) if idea.company_name else []
        if matches:
            best = matches[0]
            resolved = str(best.get("symbol") or "").strip()
            if resolved and universe.exists(resolved):
                logger.info(
                    "niche verify: corrected ticker %r -> %s (%s) via company name %r",
                    idea.ticker or "?",
                    resolved,
                    best.get("security_name"),
                    idea.company_name,
                )
                idea.ticker = resolved.upper()
                idea.corrected = True
                idea.verified = True
                idea.exchange = best.get("exchange")
                idea.robinhood_tradeable = bool(
                    universe.robinhood_tradeable(resolved),
                )
                survivors.append(idea)
                continue
        # (c) drop
        logger.info(
            "niche verify: unverified niche ticker %r/%r dropped",
            idea.ticker or "?",
            idea.company_name or "?",
        )
        idea.dropped_reason = "unverified"
    return survivors


# ---------------------------------------------------------------------- #
# Asymmetry + liquidity scoring
# ---------------------------------------------------------------------- #


def torque_from_reason(torque_reason: str) -> float:
    """Qualitative torque prose → a 0-1 numeric prior. Scans for leverage
    keywords (single-asset, sole supplier, junior, levered, royalty…) and
    takes the strongest signal found; empty/no-signal → a neutral 0.4."""
    text = (torque_reason or "").lower()
    best = 0.0
    for keyword, weight in _TORQUE_KEYWORDS.items():
        if keyword in text:
            best = max(best, weight)
    return best if best > 0 else 0.4


def evidence_score(
    idea: NicheIdea,
    market_data: Mapping[str, Mapping[str, Any]],
    as_of: datetime,
    cfg: AsymmetryConfig | None = None,
) -> NicheIdea:
    """Active score: source-backed coverage, never prose/hop bonuses.

    Equal role weights are a transparent completeness diagnostic, not a
    calibrated expected-return estimate. Semantic support requires the critic.
    Legacy asymmetry_score() remains available only for historical analysis.
    """
    cfg = cfg or AsymmetryConfig()
    assert as_of.tzinfo is not None, "Evidence decisions require an aware cutoff"
    idea.score_version = "evidence-coverage-v1"
    roles = {
        c.role
        for c in idea.claims
        if c.kind == "documented_fact" and c.backed(idea.sources, as_of, idea.ticker)
    }
    all_backed = (
        not idea.claims_parse_error
        and bool(idea.claims)
        and all(c.backed(idea.sources, as_of, idea.ticker) for c in idea.claims)
    )
    required = {"relationship", "exposure", "catalyst"}
    idea.evidence_status = (
        "source_backed" if required <= roles and all_backed else "insufficient_evidence"
    )
    data = market_data.get(idea.ticker) or {}
    idea.market_cap = finite_nonnegative(data.get("market_cap"))
    idea.avg_dollar_volume = finite_nonnegative(data.get("avg_dollar_volume"))
    idea.last_close = finite_nonnegative(data.get("last_close"))
    observed, received = timestamp(data.get("observed_at")), timestamp(data.get("retrieved_at"))
    idea.market_observed_at = observed.isoformat() if observed else None
    idea.market_received_at = received.isoformat() if received else None
    fresh = bool(
        observed
        and received
        and observed <= received <= as_of
        and as_of - observed <= timedelta(days=MAX_MARKET_AGE_DAYS)
    )
    idea.liquidity_status = "unknown"
    if fresh and idea.avg_dollar_volume is not None:
        idea.liquidity_status = (
            "sufficient" if idea.avg_dollar_volume >= cfg.min_avg_dollar_volume else "insufficient"
        )
    idea.liquidity_flag = idea.liquidity_status != "sufficient"
    idea.components = {role: float(role in roles) for role in CLAIM_ROLES}
    idea.asymmetry_score = len(roles) / len(CLAIM_ROLES)
    idea.dropped_reason = None
    if idea.evidence_status != "source_backed":
        idea.dropped_reason = "insufficient_evidence"
    elif idea.liquidity_status != "sufficient":
        idea.dropped_reason = "liquidity_" + idea.liquidity_status
    elif idea.asymmetry_score < cfg.asymmetry_threshold:
        idea.dropped_reason = "below_evidence_threshold"
    assert 0 <= idea.asymmetry_score <= 1
    return idea


def _smallness_bonus(market_cap: float | None, cfg: AsymmetryConfig) -> float:
    """0-1 under-followed-ness from market cap: full weight at/under the
    small-cap ceiling, none at/over the large-cap floor, linear in
    log-space between. Unknown cap → a neutral 0.5 (we don't reward an
    unknowable, but don't zero it either)."""
    if market_cap is None or market_cap <= 0:
        return 0.5
    if market_cap <= cfg.small_cap_ceiling:
        return 1.0
    if market_cap >= cfg.large_cap_floor:
        return 0.0
    lo, hi = math.log(cfg.small_cap_ceiling), math.log(cfg.large_cap_floor)
    frac = (math.log(market_cap) - lo) / (hi - lo)
    return max(0.0, min(1.0, 1.0 - frac))


def asymmetry_score(
    idea: NicheIdea,
    market_data: Mapping[str, Mapping[str, Any]] | None = None,
    cfg: AsymmetryConfig | None = None,
) -> NicheIdea:
    """Score one verified idea IN PLACE and return it. Combines:

      * hop_count (more hops = more overlooked, capped at ``hop_cap``);
      * qualitative torque (:func:`torque_from_reason`);
      * REAL smallness/under-followed-ness (market cap → smallness bonus),

    under a LIQUIDITY FLOOR: below ``min_avg_dollar_volume`` the name is
    flagged illiquid, its smallness bonus is zeroed, and a flat penalty
    is applied — an illiquid shell must NOT read as an edge. Below
    ``drop_below_dollar_volume`` the idea is hard-dropped (score 0,
    dropped_reason set). Sets ``asymmetry_score``, ``liquidity_flag``,
    ``market_cap``, ``avg_dollar_volume``, and ``components``.
    """
    cfg = cfg or AsymmetryConfig()
    data = (market_data or {}).get(idea.ticker) or {}
    market_cap = data.get("market_cap")
    adv = data.get("avg_dollar_volume")
    idea.market_cap = float(market_cap) if market_cap is not None else None
    idea.avg_dollar_volume = float(adv) if adv is not None else None

    hop_norm = min(idea.hop_count, cfg.hop_cap) / cfg.hop_cap
    idea.score_version = "legacy-hop-torque-v1"
    torque = torque_from_reason(idea.torque_reason)

    # Liquidity gating FIRST — it governs whether smallness is a bonus or
    # a trap. Unknown ADV is treated as "not proven illiquid" (no
    # penalty) but also earns no smallness benefit-of-the-doubt beyond
    # the neutral cap bonus; a known-thin ADV is penalised.
    liquidity_flag = False
    hard_drop = False
    if adv is not None:
        if adv < cfg.drop_below_dollar_volume:
            hard_drop = True
            liquidity_flag = True
        elif adv < cfg.min_avg_dollar_volume:
            liquidity_flag = True

    smallness = _smallness_bonus(idea.market_cap, cfg)
    if liquidity_flag:
        # An illiquid name looks under-followed but is a trap — kill the
        # smallness reward so thinness can never masquerade as edge.
        smallness = 0.0

    score = (
        cfg.hop_weight * hop_norm + cfg.torque_weight * torque + cfg.smallness_weight * smallness
    )
    if liquidity_flag:
        score -= cfg.illiquid_penalty
    score = max(0.0, min(1.0, score))

    if hard_drop:
        score = 0.0
        idea.dropped_reason = "illiquid (below hard $volume floor)"

    idea.asymmetry_score = round(score, 4)
    idea.liquidity_flag = liquidity_flag
    idea.components = {
        "hop_norm": round(hop_norm, 4),
        "torque": round(torque, 4),
        "smallness": round(smallness, 4),
    }
    return idea


def score_and_gate(
    ideas: list[NicheIdea],
    market_data: Mapping[str, Mapping[str, Any]] | None = None,
    cfg: AsymmetryConfig | None = None,
) -> tuple[list[NicheIdea], list[NicheIdea]]:
    """Score every idea and split into (surviving, logged). An idea
    surfaces only when it clears ``asymmetry_threshold`` AND was not
    hard-dropped for illiquidity; everything else is returned in the
    second list (LOGGED, not surfaced). Both lists are score-desc."""
    cfg = cfg or AsymmetryConfig()
    for idea in ideas:
        asymmetry_score(idea, market_data, cfg)
    surviving: list[NicheIdea] = []
    logged: list[NicheIdea] = []
    for idea in ideas:
        score = idea.asymmetry_score or 0.0
        if idea.dropped_reason is None and score >= cfg.asymmetry_threshold:
            surviving.append(idea)
        else:
            logged.append(idea)
            logger.info(
                "niche gate: %s (%s, %d hops) score=%.3f below threshold "
                "%.2f%s — logged not surfaced",
                idea.ticker,
                idea.company_name,
                idea.hop_count,
                score,
                cfg.asymmetry_threshold,
                " [illiquid]" if idea.liquidity_flag else "",
            )
    surviving.sort(key=lambda i: i.asymmetry_score or 0.0, reverse=True)
    logged.sort(key=lambda i: i.asymmetry_score or 0.0, reverse=True)
    return surviving, logged
