"""Multi-hop niche / asymmetry opportunity agent (CL-u2ph).

A SEPARATE, additive LLM pass over the existing Event Impact Agent
assessment. Given a high-urgency ASSESSED event + its playbook, it does
explicit multi-hop reasoning — directly affected → upstream suppliers /
downstream customers / competitors / financiers → niche pure-plays,
juniors, royalty/streaming cos, equipment, logistics, offtake partners
with HIGH operational/financial leverage → the highest-torque
UNDER-FOLLOWED name — and returns niche trade ideas ALONGSIDE the
obvious ones. It complements CL-5mkf (anti-reflexive-consensus): the
main assessment names the liquid trade; this pass hunts the
non-consensus second/third-order name with more upside torque.

ITERATIVE HOPPING (CL-2dnf): optionally runs several LLM cycles per event
(``NICHE_MAX_CYCLES``, default 1). Cycle 1 is the base multi-hop pass; each
further cycle feeds the discovered names back in and pushes the model DEEPER
and WIDER along explicit tree-of-thought branches (upstream / downstream /
substitutes / financial), deduping on ticker-or-company and stopping early once
a cycle surfaces nothing new — deeper discovery at a bounded extra cost.

Three hard guards keep this from being a hallucination-and-fiction
generator (LLM small-cap / junior knowledge is training-vintage and a
confident wrong ticker on an illiquid name is real money on fiction):

  1. VERIFICATION (:func:`verify_ideas`) — every proposed niche ticker is
     checked against the real US-listed :class:`SymbolUniverse` (CL-tzug,
     ~13k symbols). exists → keep; else resolve_name(company) → CORRECT
     the ticker (logged); else DROP the idea (logged). NO unverified
     ticker survives. Survivors are tagged with the verified exchange +
     robinhood_tradeable flag.

  2. ASYMMETRY + LIQUIDITY FLOOR (:func:`asymmetry_score`) — combines the
     agent's hop_count (more hops = more overlooked, capped), its
     qualitative torque, and the REAL under-followed-ness from market
     data (smaller market cap / lower avg $volume = more torque
     potential). BUT below a configurable liquidity floor the idea is
     FLAGGED illiquid and heavily penalized — under-followed and
     illiquid-shell look identical without this, and a dying illiquid
     name is a trap, not an edge. Only ideas clearing the asymmetry
     threshold surface; the rest are logged (not surfaced).

  3. HONEST FRAMING (in the prompt) — not 100x; high-asymmetry
     3-10x-if-right with DEFINED risk, cite the chain, prefer
     smaller/less-covered/non-consensus names, be conservative.

Surviving high-score ideas are merged into the event's assessment
``trade_ideas`` tagged ``niche=true`` (+ hop_count, torque_reason,
asymmetry_score, liquidity_flag) so they flow through the EXISTING
digest / ledger / retail-proxy / consolidation rendering unchanged.

The agent proposes; it never trades. Advisory equities only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from src.events.impact_agent import extract_json_object
from src.events.playbooks import Playbook
from src.research.llm import Message
from src.research.llm.client import LLMClient

logger = logging.getLogger(__name__)

#: Deeper reasoning helps the multi-hop chain; overridable via
#: NICHE_AGENT_MODEL. Sonnet-class keeps per-event subscription load
#: modest while still reasoning several hops out.
DEFAULT_MODEL = "claude-sonnet-4-6"

#: Only run this pass on genuinely high-urgency events — it burns a
#: second subscription LLM call, so we gate hard on urgency (shared
#: quota discipline). Overridable via NICHE_MIN_URGENCY.
DEFAULT_MIN_URGENCY = 7

#: Iterative-hopping depth (CL-2dnf): how many LLM cycles the agent runs
#: per event. Cycle 1 is the base multi-hop pass; each further cycle feeds
#: the names found so far back in and asks for DEEPER / WIDER connected
#: names (tree-of-thought branches A-D), stopping early when a cycle adds
#: nothing new. 1 = single pass (original behavior). Each extra cycle is
#: another subscription call, so this is OPT-IN via NICHE_MAX_CYCLES and
#: clamped to [1, 5]. Combined with the urgency gate + triage, that keeps
#: the added quota bounded.
DEFAULT_MAX_CYCLES = 1
MAX_CYCLES_CAP = 5

#: Tool-augmented hopping (CL-2czc): between cycles, ground the top few
#: discovered names in REAL SEC-filing / company data and feed it into the
#: next cycle's prompt. OPT-IN (``NICHE_TOOLS_ENABLED``) since it adds HTTP
#: latency + SEC calls; only meaningful with max_cycles > 1. The per-cycle
#: entity cap (``NICHE_TOOLS_MAX_ENTITIES``) bounds the SEC call volume +
#: prompt size.
DEFAULT_TOOLS_MAX_ENTITIES = 2


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

#: Cap on niche ideas the LLM may return (bounds a pathological reply).
MAX_NICHE_IDEAS = 6

VALID_NICHE_ACTIONS = frozenset({"long", "short", "buy_calls", "buy_puts"})
_BULLISH_NICHE_ACTIONS = frozenset({"long", "buy_calls"})

#: Qualitative torque phrasing → a 0-1 numeric prior. The LLM supplies
#: prose ("torque_reason"); we look for leverage keywords rather than
#: trusting a self-scored number it has no calibration for.
_TORQUE_KEYWORDS: dict[str, float] = {
    "single-asset": 1.0, "single asset": 1.0, "pure-play": 0.9,
    "pure play": 0.9, "one mine": 1.0, "sole": 0.9, "only producer": 0.9,
    "high fixed cost": 0.8, "operating leverage": 0.85, "operational leverage": 0.85,
    "financial leverage": 0.8, "levered": 0.8, "high debt": 0.75,
    "net debt": 0.7, "royalty": 0.7, "streaming": 0.7, "offtake": 0.7,
    "junior": 0.85, "microcap": 0.8, "micro-cap": 0.8, "small-cap": 0.6,
    "small cap": 0.6, "under-followed": 0.7, "underfollowed": 0.7,
    "no coverage": 0.8, "uncovered": 0.75, "non-consensus": 0.6,
    "sole supplier": 1.0, "bottleneck": 0.85, "chokepoint": 0.8,
    "convert": 0.6, "convertible": 0.6, "distress": 0.75,
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
    small_cap_ceiling: float = 300_000_000.0     # $300M — micro/small
    large_cap_floor: float = 20_000_000_000.0     # $20B — well-covered

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
    #: For the honest self-scored components (debugging / logging).
    components: dict[str, float] = field(default_factory=dict)

    def to_trade_idea(self) -> dict[str, Any]:
        """Merge shape for the assessment ``trade_ideas`` list. Carries
        the standard idea fields the ledger/digest expect PLUS the niche
        tags. ``time_horizon`` defaults to "short" (a niche event trade
        is tactical) so the ledger's selector/expiry logic has a value."""
        bullish = self.action in _BULLISH_NICHE_ACTIONS or self.direction == "bullish"
        return {
            "ticker": self.ticker,
            "action": self.action,
            "direction": "bullish" if bullish else "bearish",
            "confidence": self.confidence,
            "rationale": self.rationale,
            "time_horizon": "short",
            "holding_period_days": "",
            "time_stop_days": 10,
            "stop_loss_pct": None,
            "target_pct": [],
            "entry_trigger": "",
            "invalidation": "",
            "suggested_entry": "",
            "preferred_instrument": "",
            "notes": self.torque_reason,
            # -- niche tags (additive; existing consumers ignore unknown keys)
            "niche": True,
            "company_name": self.company_name,
            "hop_count": self.hop_count,
            "torque_reason": self.torque_reason,
            "asymmetry_score": self.asymmetry_score,
            "liquidity_flag": self.liquidity_flag,
            "exchange": self.exchange,
            "robinhood_tradeable": self.robinhood_tradeable,
        }


_SYSTEM_PROMPT = """\
You are the NICHE OPPORTUNITY agent for an FX/equities event desk. You
receive one event that the desk has ALREADY assessed (the obvious,
liquid trade is known) plus its themed playbook. Your job is DIFFERENT
and additive: hunt the high-torque, UNDER-FOLLOWED second- and
third-order name the obvious trade misses.

Do NOT stop at the obvious liquid names. Reason MULTI-HOP:
  hop 1: directly affected companies/assets;
  hop 2: their upstream suppliers, downstream customers, competitors,
         financiers, insurers;
  hop 3+: niche pure-plays, juniors, royalty/streaming companies,
          specialised equipment makers, logistics/freight, offtake
          partners — the names with HIGH operational or financial
          leverage to this specific chain.
Follow the chain to the HIGHEST-TORQUE under-followed name at the end.

Prefer SMALLER, LESS-COVERED, NON-CONSENSUS names with more upside
torque than the obvious large-caps — a single-asset pure-play, a junior,
a sole supplier, a levered operator — over the mega-cap everyone already
owns. The obvious name is likely already priced; the overlooked one at
the end of the chain is where the asymmetry lives.

HONEST FRAMING — this is NOT a lottery ticket. You are hunting
high-ASYMMETRY, roughly 3-10x-IF-RIGHT ideas with DEFINED risk, not
100x fantasies. Be conservative. CITE THE CHAIN explicitly in the
rationale (name each hop). Real, currently-trading, US-listed companies
ONLY where possible — do NOT invent tickers; if unsure of the exact
ticker, still give the precise company_name so it can be resolved. A
confident wrong ticker on an illiquid name is real money on fiction.

Return BOTH:
  * a small number of OBVIOUS-ADJACENT names (hop 1-2, the liquid
    second-order plays), AND
  * the NICHE names (hop 3+, the under-followed high-torque plays).

Rules:
- Respond with ONE JSON object and NOTHING else. No markdown fences, no
  commentary.
- Every idea needs a real company_name and, where you know it, a real
  US-listed ticker. Never fabricate a ticker to fill the field.
- hop_count is how many links from the OBVIOUS trade this name sits
  (1 = obvious-adjacent; 4-5 = deeply overlooked).
- torque_reason: WHY this name has more leverage than the obvious trade
  (single-asset, high fixed cost, sole supplier, royalty, junior,
  levered balance sheet, etc.) — the mechanism, in one line.
- Be conservative on confidence; a longer chain is more speculative.

JSON schema (all keys required per idea):
{
  "niche_ideas": [
    {"ticker": "<US-listed ticker if known, else best guess>",
     "company_name": "<exact company name, for ticker resolution>",
     "action": "long" | "short" | "buy_calls" | "buy_puts",
     "direction": "bullish" | "bearish",
     "hop_count": <int 1-5, links from the obvious trade>,
     "torque_reason": "<one line: the leverage mechanism>",
     "rationale": "<cite the multi-hop chain: hop1 -> hop2 -> this>",
     "confidence": <float 0.0-1.0>}
  ]
}
"""


# ---------------------------------------------------------------------- #
# Market-data provider (injectable, fail-soft)
# ---------------------------------------------------------------------- #

#: A market-data lookup: ticker -> {"market_cap": float|None,
#: "avg_dollar_volume": float|None}. Injectable so unit tests feed canned
#: values and the live path uses yfinance. Any failure yields an entry
#: with None fields (fail-soft — never a raise, never a fabricated
#: number). Missing tickers are simply absent from the returned map.
MarketDataFn = Callable[[list[str]], dict[str, dict[str, Any]]]


def yfinance_market_data(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Default provider: per-ticker market cap + average dollar volume
    via yfinance ``Ticker().info`` / recent history. Fail-soft PER
    TICKER — a lookup that raises or returns junk yields None fields, so
    a scoring caller degrades to hop/torque-only rather than dying.

    ``avg_dollar_volume`` = mean(close * volume) over the last ~30
    sessions; ``market_cap`` from ``.info`` when present, else
    ``.info['sharesOutstanding'] * last_close``.
    """
    out: dict[str, dict[str, Any]] = {}
    if not tickers:
        return out
    try:
        import yfinance as yf  # deferred — cheap import for non-live callers
    except Exception:  # pragma: no cover - import guard
        logger.warning("yfinance unavailable; niche scoring runs data-free")
        return out
    for ticker in tickers:
        entry: dict[str, Any] = {"market_cap": None, "avg_dollar_volume": None}
        try:
            tk = yf.Ticker(ticker)
            info: dict[str, Any] = {}
            try:
                info = dict(tk.info or {})
            except Exception:
                info = {}
            mcap = info.get("marketCap")
            hist = tk.history(period="1mo", interval="1d")
            if hist is not None and not hist.empty:
                dollar = (hist["Close"] * hist["Volume"]).dropna()
                if not dollar.empty:
                    entry["avg_dollar_volume"] = float(dollar.mean())
                if mcap is None:
                    shares = info.get("sharesOutstanding")
                    last = float(hist["Close"].dropna().iloc[-1]) if not hist["Close"].dropna().empty else None
                    if shares and last:
                        mcap = float(shares) * last
            if mcap is not None:
                entry["market_cap"] = float(mcap)
        except Exception:
            logger.debug("niche market data: %s unusable; scoring data-free", ticker, exc_info=True)
        out[ticker] = entry
    return out


# ---------------------------------------------------------------------- #
# Defensive parse
# ---------------------------------------------------------------------- #


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


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
        ideas.append(NicheIdea(
            ticker=ticker,
            company_name=company,
            action=action,
            direction=direction,
            hop_count=_clamp_int(entry.get("hop_count"), 1, 8, 1),
            torque_reason=str(entry.get("torque_reason", "")).strip(),
            rationale=str(entry.get("rationale", "")).strip(),
            confidence=_clamp_float(entry.get("confidence"), 0.0, 1.0, 0.4),
        ))
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
                    "niche verify: corrected ticker %r -> %s (%s) via company "
                    "name %r",
                    idea.ticker or "?", resolved, best.get("security_name"),
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
            idea.ticker or "?", idea.company_name or "?",
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


def _smallness_bonus(market_cap: float | None, cfg: AsymmetryConfig) -> float:
    """0-1 under-followed-ness from market cap: full weight at/under the
    small-cap ceiling, none at/over the large-cap floor, linear in
    log-space between. Unknown cap → a neutral 0.5 (we don't reward an
    unknowable, but don't zero it either)."""
    import math

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
        cfg.hop_weight * hop_norm
        + cfg.torque_weight * torque
        + cfg.smallness_weight * smallness
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
                idea.ticker, idea.company_name, idea.hop_count, score,
                cfg.asymmetry_threshold,
                " [illiquid]" if idea.liquidity_flag else "",
            )
    surviving.sort(key=lambda i: i.asymmetry_score or 0.0, reverse=True)
    logged.sort(key=lambda i: i.asymmetry_score or 0.0, reverse=True)
    return surviving, logged


# ---------------------------------------------------------------------- #
# The agent
# ---------------------------------------------------------------------- #


class NicheAgent:
    """Multi-hop niche opportunity pass. Given an ASSESSED event row +
    its playbook, produce VERIFIED, liquidity-gated niche ideas."""

    def __init__(
        self,
        universe: Any,
        client: LLMClient | None = None,
        model: str = DEFAULT_MODEL,
        market_data_fn: MarketDataFn | None = None,
        config: AsymmetryConfig | None = None,
        max_tokens: int = 1500,
        max_cycles: int | None = None,
        tools: Any = None,
        tools_enabled: bool | None = None,
        tools_max_entities: int | None = None,
        tool_agent: Any = None,
        tool_agent_enabled: bool | None = None,
    ) -> None:
        # ``universe`` is a SymbolUniverse (or any object exposing
        # exists/get/resolve_name/robinhood_tradeable).
        self.universe = universe
        # Deferred get_client — a data-free unit test that always injects
        # a client never touches the CLI subscription.
        if client is not None:
            self.client: LLMClient = client
        else:
            from src.research.llm import get_client  # noqa: PLC0415

            self.client = get_client("claude-code")
        self.model = model
        self.market_data_fn: MarketDataFn = market_data_fn or yfinance_market_data
        self.config = config or AsymmetryConfig()
        self.max_tokens = max_tokens
        # Iterative-hopping depth (CL-2dnf) — explicit arg wins, else env,
        # else single-pass. Clamped to [1, MAX_CYCLES_CAP].
        if max_cycles is None:
            try:
                max_cycles = int(
                    os.environ.get("NICHE_MAX_CYCLES", DEFAULT_MAX_CYCLES),
                )
            except ValueError:
                max_cycles = DEFAULT_MAX_CYCLES
        self.max_cycles = max(1, min(MAX_CYCLES_CAP, max_cycles))
        # Tool-augmented hopping (CL-2czc). Explicit tools win; else construct
        # the default ResearchTools when enabled (env opt-in). Disabled → the
        # loop skips grounding entirely (pure-reasoning hops, as before).
        if tools_enabled is None:
            tools_enabled = _env_flag("NICHE_TOOLS_ENABLED", default=False)
        self.tools_enabled = tools_enabled
        if tools is not None:
            self.tools = tools
        elif tools_enabled:
            from src.events.research_tools import ResearchTools  # noqa: PLC0415

            self.tools = ResearchTools()
        else:
            self.tools = None
        if tools_max_entities is not None:
            self.tools_max_entities = tools_max_entities
        else:
            try:
                self.tools_max_entities = int(os.environ.get(
                    "NICHE_TOOLS_MAX_ENTITIES", DEFAULT_TOOLS_MAX_ENTITIES,
                ))
            except ValueError:
                self.tools_max_entities = DEFAULT_TOOLS_MAX_ENTITIES
        # Agentic Kimi tool-loop (CL-ddzt) — an ALTERNATIVE discovery source
        # where the model itself drives the tools, on the API-billed Kimi
        # provider. Explicit agent wins; else construct one when enabled AND a
        # key is present. When set it REPLACES the claude-code cycles for
        # discovery; verification/scoring downstream are identical.
        if tool_agent_enabled is None:
            tool_agent_enabled = _env_flag("NICHE_TOOL_AGENT_ENABLED", default=False)
        if tool_agent is not None:
            self.tool_agent = tool_agent
        elif tool_agent_enabled and os.environ.get("MOONSHOT_API_KEY"):
            from src.events.kimi_tool_agent import KimiToolAgent  # noqa: PLC0415
            from src.events.research_tools import ResearchTools  # noqa: PLC0415

            self.tool_agent = KimiToolAgent(
                universe=self.universe,
                tools=self.tools or ResearchTools(),
            )
        else:
            self.tool_agent = None

    # -- prompt ---------------------------------------------------------

    def _playbook_context(self, playbook: Playbook | None) -> str:
        if playbook is None:
            return "No matched playbook — reason from the event alone."
        lines = [
            f"Matched playbook: {playbook.key} — {playbook.name}",
            playbook.description,
            "Known reachable instruments (the OBVIOUS trades — go BEYOND these):",
        ]
        for inst in playbook.instruments:
            lines.append(f"  - {inst.instrument} [{inst.kind}]: {inst.rationale}")
        return "\n".join(lines)

    def _user_prompt(
        self, event_row: Mapping[str, Any], playbook: Playbook | None,
    ) -> str:
        assessment = event_row.get("assessment") or {}
        core = ""
        obvious = ""
        if isinstance(assessment, Mapping):
            core = str(assessment.get("core_event") or "")
            obvious_names = [
                str(i.get("ticker") or i.get("instrument") or "")
                for key in ("trade_ideas", "affected")
                for i in (assessment.get(key) or [])
                if isinstance(i, Mapping)
            ]
            obvious = ", ".join(sorted({n for n in obvious_names if n})) or "n/a"
        return (
            f"EVENT: {event_row.get('headline')}\n"
            f"CORE: {core or event_row.get('headline')}\n"
            f"THEME: {event_row.get('theme') or 'unmatched'}\n"
            f"OBVIOUS NAMES ALREADY ASSESSED (go beyond these): {obvious}\n\n"
            f"{self._playbook_context(playbook)}\n\n"
            "Hunt the under-followed high-torque names now. Multi-hop. "
            "Respond with the JSON object only."
        )

    def _followup_prompt(
        self,
        event_row: Mapping[str, Any],
        playbook: Playbook | None,
        discovered: list[NicheIdea],
        grounding: list[str] | None = None,
    ) -> str:
        """Cycle-2+ prompt (CL-2dnf): feed back the names found so far and
        push the model DEEPER/WIDER along explicit tree-of-thought branches,
        stopping it from repeating what's already on the table. When
        tool-augmented (CL-2czc), ``grounding`` carries REAL SEC-filing /
        profile data on those names so the next hops build on fact."""
        lines = []
        for idea in discovered[:12]:
            tkr = f" ({idea.ticker})" if idea.ticker else ""
            lines.append(
                f"  - {idea.company_name or '?'}{tkr} [hop {idea.hop_count}]",
            )
        found = "\n".join(lines) or "  (none yet)"
        ground_block = ""
        if grounding:
            joined = "\n\n".join(grounding[:6])
            ground_block = (
                "\nREAL RESEARCH DATA on names found so far — ground your next "
                "hops in THIS (actual filings/profiles, not memory); name the "
                "specific customers / suppliers / competitors it reveals:\n"
                f"{joined}\n"
            )
        return (
            f"EVENT: {event_row.get('headline')}\n"
            f"THEME: {event_row.get('theme') or 'unmatched'}\n\n"
            f"{self._playbook_context(playbook)}\n\n"
            "Entities discovered SO FAR (do NOT repeat these — go DEEPER and "
            "WIDER, further out the chain):\n"
            f"{found}\n"
            f"{ground_block}\n"
            "Now surface ADDITIONAL under-followed, high-torque names connected "
            "to this event and to those entities, exploring EACH branch:\n"
            "  A. Upstream — deeper-tier suppliers, raw inputs, sole-source "
            "components.\n"
            "  B. Downstream — customers whose demand is destroyed or created.\n"
            "  C. Substitutes / competitors — who gains share or volume.\n"
            "  D. Financial / secondary — insurers, shippers, royalty & "
            "streaming holders, lenders/creditors, equipment lessors levered to "
            "this chain.\n"
            "Prefer names NOT already listed and further out the chain (higher "
            "hop_count). Same JSON schema. Respond with the JSON object only."
        )

    # -- run ------------------------------------------------------------

    def run(
        self, event_row: Mapping[str, Any], playbook: Playbook | None = None,
    ) -> list[NicheIdea]:
        """Full niche pass for one event: DISCOVER raw ideas → verify →
        score/gate. Returns ONLY the surviving (verified, liquidity-cleared,
        above-threshold) ideas, score-desc. Never raises.

        Discovery source: the agentic Kimi tool-loop (CL-ddzt) when a
        ``tool_agent`` is configured, else the claude-code iterative multi-hop
        cycles (CL-2dnf/CL-2czc). Verification + scoring are identical either
        way, so no unverified ticker survives regardless of source."""
        raw_accum = (
            self._discover_via_tool_agent(event_row, playbook)
            if self.tool_agent is not None
            else self._discover_via_cycles(event_row, playbook)
        )
        source = "kimi" if self.tool_agent is not None else "cycles"
        if not raw_accum:
            return []
        verified = verify_ideas(raw_accum, self.universe)
        if not verified:
            logger.info(
                "niche agent: all %d proposed ideas failed verification "
                "for event id=%s", len(raw_accum), event_row.get("id"),
            )
            return []
        market_data = {}
        try:
            market_data = self.market_data_fn([i.ticker for i in verified])
        except Exception:
            logger.warning(
                "niche agent: market-data fetch failed; scoring data-free",
                exc_info=True,
            )
        surviving, logged = score_and_gate(verified, market_data, self.config)
        logger.info(
            "niche agent: event id=%s [%s] — %d proposed, %d verified, "
            "%d surfaced, %d logged (below threshold/illiquid)",
            event_row.get("id"), source, len(raw_accum), len(verified),
            len(surviving), len(logged),
        )
        return surviving

    def _discover_via_tool_agent(
        self, event_row: Mapping[str, Any], playbook: Playbook | None,
    ) -> list[NicheIdea]:
        """Agentic Kimi tool-loop discovery (CL-ddzt): the model drives the
        SEC/ticker tools, then we parse + dedup its final JSON. Fail-soft []."""
        text = self.tool_agent.discover(event_row, playbook)
        seen: set[str] = set()
        out: list[NicheIdea] = []
        for idea in (parse_niche_ideas(text) if text else []):
            key = _idea_key(idea)
            if key not in seen:
                seen.add(key)
                out.append(idea)
        return out

    def _discover_via_cycles(
        self, event_row: Mapping[str, Any], playbook: Playbook | None,
    ) -> list[NicheIdea]:
        """Iterative claude-code multi-hop cycles (CL-2dnf) with optional
        between-cycle SEC grounding (CL-2czc) → raw deduped ideas. Fail-soft:
        a cycle-1 failure yields []; a later-cycle failure keeps earlier finds."""
        raw_accum: list[NicheIdea] = []
        seen: set[str] = set()
        grounding: list[str] = []
        cycles_run = 0
        for cycle in range(1, self.max_cycles + 1):
            cycles_run = cycle
            user = (
                self._user_prompt(event_row, playbook) if cycle == 1
                else self._followup_prompt(
                    event_row, playbook, raw_accum, grounding,
                )
            )
            try:
                resp = self.client.complete(
                    messages=[
                        Message(role="system", content=_SYSTEM_PROMPT),
                        Message(role="user", content=user),
                    ],
                    model=self.model,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                logger.warning(
                    "niche agent transport failure (cycle %d) for event id=%s: %s",
                    cycle, event_row.get("id"), str(exc)[:200],
                )
                if cycle == 1:
                    return []
                break  # keep what earlier cycles found
            parsed = parse_niche_ideas(resp.text)
            fresh = [i for i in parsed if _idea_key(i) not in seen]
            for idea in fresh:
                seen.add(_idea_key(idea))
            raw_accum.extend(fresh)
            if cycle == 1 and not parsed:
                break  # nothing to deepen
            if cycle > 1 and not fresh:
                logger.info(
                    "niche agent: cycle %d added no new names — stopping "
                    "(event id=%s)", cycle, event_row.get("id"),
                )
                break
            # Tool-augment for the NEXT cycle (CL-2czc): ground the top few
            # fresh names with a REAL ticker in SEC-filing / profile data.
            if self.tools is not None and cycle < self.max_cycles and fresh:
                candidates = [
                    i for i in fresh
                    if i.ticker and self.universe.exists(i.ticker)
                ][: self.tools_max_entities]
                if candidates:
                    try:
                        grounding.extend(self.tools.enrich(candidates, self.universe))
                    except Exception:
                        logger.warning(
                            "niche tools: enrichment failed for event id=%s; "
                            "continuing without grounding", event_row.get("id"),
                            exc_info=True,
                        )

        logger.debug(
            "niche cycles: event id=%s — %d cycle(s), %d raw ideas",
            event_row.get("id"), cycles_run, len(raw_accum),
        )
        return raw_accum

    def merge_into_assessment(
        self,
        assessment: dict[str, Any],
        niche_ideas: list[NicheIdea],
        max_total: int = 8,
    ) -> int:
        """Merge surviving niche ideas into an assessment's
        ``trade_ideas`` (tagged niche=true), deduped on (ticker, action)
        against the ideas already there — an obvious idea already present
        is not duplicated by a niche echo of it. Returns how many were
        added. Mutates ``assessment`` in place."""
        existing = assessment.setdefault("trade_ideas", [])
        if not isinstance(existing, list):
            existing = []
            assessment["trade_ideas"] = existing
        seen = {
            (str(i.get("ticker") or "").upper(), str(i.get("action") or "").lower())
            for i in existing if isinstance(i, dict)
        }
        added = 0
        for idea in niche_ideas:
            key = (idea.ticker.upper(), idea.action.lower())
            if key in seen:
                continue
            if len(existing) >= max_total:
                break
            existing.append(idea.to_trade_idea())
            seen.add(key)
            added += 1
        return added
