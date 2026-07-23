"""Polymarket prediction markets as a first-class SIGNAL (CL-r1ep).

Operator request 2026-07-20: "polymarket stuff added as notifications like
the short and puts." Prediction markets often lead the news — a rapid shift
in the YES probability of a Taiwan-blockade / Hormuz-closure / coup market is
tradeable information the moment it moves.

This module is the signal half:

* :meth:`PolymarketSignal.poll_probabilities` — for each tracked geopolitical
  market (from ``configs/polymarket_geo_markets.yaml``, discovered by
  ``scripts/discover_polymarket_markets.py --mode geo``) fetch the CURRENT YES
  probability (Gamma ``outcomePrices[0]``, CLOB ``/midpoint`` fallback) and
  append one row to ``poly_market_probs`` (migration 009). Per-market failure
  tolerant: one dead market never kills the poll.

* :meth:`detect_shifts` — compare each slug's latest observation against the
  EARLIEST observation inside a lookback window; a ``|Δ| >= threshold``
  (default 0.10 = 10 points) is a shift. Dedup: once a shift has been
  notified, its anchoring row is stamped (``notified_shift``) so the same
  shift is not re-alerted every cycle.

* :meth:`notify_shift` — a clean Telegram-HTML alert built like the event
  trade cards (``<b>Prediction market shift</b>`` / question / bold
  ``72% ↑ +14 (24h)`` / theme tag / one-line read / polymarket link).

* :meth:`latest_prob_for_theme` — most-recent YES prob per market in a theme,
  for the digest's per-event corroboration line (display only — the sibling
  owns confluence gating).

HTTP goes through an injectable shim (``HttpGetJson``) so unit tests inject
canned responses; no live network in unit tests.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml
from sqlalchemy import text

from src.research.notifications import DispatchResult, html_escape, notify_operator

logger = logging.getLogger(__name__)


GAMMA_MARKETS_URL: str = "https://gamma-api.polymarket.com/markets"
CLOB_MIDPOINT_URL: str = "https://clob.polymarket.com/midpoint"
DEFAULT_GEO_CONFIG_PATH: Path = Path("configs/polymarket_geo_markets.yaml")
DEFAULT_TIMEOUT_SEC: float = 20.0

#: Default shift threshold — 10 probability points inside the window.
DEFAULT_SHIFT_THRESHOLD: float = 0.10
#: Default lookback window for shift detection.
DEFAULT_WINDOW_HOURS: int = 24

#: Themes whose YES side reads as ESCALATION when it rises — used to pick the
#: one-line "implied read" in the alert. A closure/invasion/coup/collapse
#: market ticking UP is bad news; a de-escalation/ceasefire market rising is
#: the inverse. We infer polarity from the question text, not the theme, so a
#: "ceasefire" market in russia_ukraine reads correctly.
_ESCALATION_WORDS: tuple[str, ...] = (
    "clos",
    "closure",
    "block",
    "blockade",
    "invas",
    "invade",
    "attack",
    "strike",
    "war",
    "conflict",
    "coup",
    "seiz",
    "nationaliz",
    "collapse",
    "disrupt",
    "sanction",
    "escalat",
    "shut",
    "default",
    "nuclear",
    "declare war",
    "cross border",
)
_DEESCALATION_WORDS: tuple[str, ...] = (
    "ceasefire",
    "truce",
    "peace deal",
    "de-escalat",
    "deescalat",
    "normaliz",
    "resolved",
    "end of war",
    "end the war",
    "withdraw",
)


# Type alias for the HTTP shim (url, params) -> parsed JSON. Tests inject.
HttpGetJson = Callable[[str, dict[str, str]], Any]


def _default_http_get_json(url: str, params: dict[str, str]) -> Any:
    """Production HTTP GET returning parsed JSON. Raises on non-2xx."""
    resp = httpx.get(
        url,
        params=params,
        timeout=DEFAULT_TIMEOUT_SEC,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


@dataclass(frozen=True)
class TrackedMarket:
    """One geopolitical market we poll each cycle."""

    slug: str
    question: str
    yes_token_id: str
    theme: str


@dataclass(frozen=True)
class ProbShift:
    """A detected probability shift for one market inside the window."""

    slug: str
    question: str
    theme: str
    latest_prob: float  # current YES prob in [0, 1]
    earliest_prob: float  # window-start YES prob in [0, 1]
    delta: float  # latest - earliest (signed)
    window_hours: int
    anchor_id: Any = None  # poly_market_probs.id of the latest obs (dedup)

    @property
    def rising(self) -> bool:
        return self.delta >= 0

    @property
    def delta_points(self) -> int:
        """Signed delta in whole probability points (e.g. +14)."""
        return round(self.delta * 100)


def load_tracked_markets(
    path: Path | str = DEFAULT_GEO_CONFIG_PATH,
) -> list[TrackedMarket]:
    """Load the theme-tagged geo markets. Missing file → empty list (the
    poly step is opt-in corroboration, never a hard dependency). Entries
    missing slug/yes_token_id are skipped with a warning."""
    p = Path(path)
    if not p.exists():
        logger.info("geo markets config not found at %s; nothing to poll", p)
        return []
    raw = yaml.safe_load(p.read_text()) or {}
    markets = raw.get("markets") if isinstance(raw, dict) else None
    if not isinstance(markets, list):
        logger.warning("geo markets config %s missing 'markets' list", p)
        return []
    out: list[TrackedMarket] = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        slug = str(m.get("slug") or "").strip()
        token = str(m.get("yes_token_id") or m.get("token_id") or "").strip()
        if not slug or not token:
            logger.warning("skipping geo market missing slug/yes_token_id: %s", m)
            continue
        out.append(
            TrackedMarket(
                slug=slug,
                question=str(m.get("question") or slug),
                yes_token_id=token,
                theme=str(m.get("theme") or "other"),
            )
        )
    return out


class PolymarketSignal:
    """Polls tracked geo markets, persists probs, detects + notifies shifts."""

    def __init__(
        self,
        engine: Any,
        http_get_json: HttpGetJson | None = None,
        gamma_url: str = GAMMA_MARKETS_URL,
        midpoint_url: str = CLOB_MIDPOINT_URL,
    ) -> None:
        self.engine = engine
        self.http_get_json = http_get_json or _default_http_get_json
        self.gamma_url = gamma_url
        self.midpoint_url = midpoint_url

    # ------------------------------------------------------------------ #
    # Fetch current YES probability
    # ------------------------------------------------------------------ #

    def _prob_from_gamma(self, slug: str) -> float | None:
        """Current YES prob via Gamma ``outcomePrices[0]`` for a slug."""
        try:
            data = self.http_get_json(self.gamma_url, {"slug": slug})
        except Exception as exc:
            logger.debug(
                "gamma slug fetch failed for %s: %s: %s",
                slug,
                type(exc).__name__,
                exc,
            )
            return None
        market = None
        if isinstance(data, list) and data:
            market = data[0]
        elif isinstance(data, dict):
            market = data
        if not isinstance(market, dict):
            return None
        return _parse_outcome_yes(market.get("outcomePrices"))

    def _prob_from_midpoint(self, token_id: str) -> float | None:
        """Current YES prob via the CLOB ``/midpoint`` for a token id."""
        try:
            data = self.http_get_json(self.midpoint_url, {"token_id": token_id})
        except Exception as exc:
            logger.debug(
                "clob midpoint fetch failed for %s: %s: %s",
                token_id,
                type(exc).__name__,
                exc,
            )
            return None
        if not isinstance(data, dict):
            return None
        raw = data.get("mid")
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    def fetch_current_prob(self, market: TrackedMarket) -> float | None:
        """Current YES prob for one market: Gamma first (has the question
        context), CLOB midpoint as the fallback. None when both fail."""
        prob = self._prob_from_gamma(market.slug)
        if prob is None:
            prob = self._prob_from_midpoint(market.yes_token_id)
        if prob is None:
            return None
        # Clamp to [0, 1] — a malformed feed shouldn't poison the series.
        return max(0.0, min(1.0, prob))

    # ------------------------------------------------------------------ #
    # Poll + persist
    # ------------------------------------------------------------------ #

    def poll_probabilities(
        self,
        markets: list[TrackedMarket],
        now: datetime | None = None,
    ) -> list[tuple[TrackedMarket, float]]:
        """Fetch + persist the current YES prob for each tracked market.

        Per-market failure tolerant: a market whose prob can't be fetched
        is logged and skipped; the rest still persist. Returns the
        (market, prob) pairs that were successfully observed + written.
        """
        now = now or datetime.now(UTC)
        observed: list[tuple[TrackedMarket, float]] = []
        for market in markets:
            try:
                prob = self.fetch_current_prob(market)
            except Exception:
                logger.exception(
                    "poll failed for %s; skipping",
                    market.slug,
                )
                continue
            if prob is None:
                logger.info("no prob for %s this cycle; skipping", market.slug)
                continue
            try:
                self._persist(market, prob, now)
            except Exception:
                logger.exception(
                    "persist failed for %s; skipping",
                    market.slug,
                )
                continue
            observed.append((market, prob))
        logger.info(
            "poly: observed %d/%d markets",
            len(observed),
            len(markets),
        )
        return observed

    def _persist(
        self,
        market: TrackedMarket,
        prob: float,
        observed_at: datetime,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO poly_market_probs "
                    "(slug, question, theme, yes_prob, observed_at, source) "
                    "VALUES (:slug, :q, :theme, :prob, :ts, :src)"
                ),
                {
                    "slug": market.slug,
                    "q": market.question,
                    "theme": market.theme,
                    "prob": float(prob),
                    # ISO string binds portably (psycopg2 → timestamptz;
                    # sqlite fixtures store TEXT).
                    "ts": observed_at.isoformat(),
                    "src": "gamma",
                },
            )

    # ------------------------------------------------------------------ #
    # Shift detection
    # ------------------------------------------------------------------ #

    def detect_shifts(
        self,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        threshold: float = DEFAULT_SHIFT_THRESHOLD,
        now: datetime | None = None,
    ) -> list[ProbShift]:
        """Per slug, compare the LATEST observation against the EARLIEST
        observation inside the last ``window_hours``. A ``|Δ| >= threshold``
        is a shift. Slugs with only one in-window observation (insufficient
        history) never fire. Skips shifts already notified (dedup on the
        latest-observation row's ``notified_shift`` stamp)."""
        now = now or datetime.now(UTC)
        cutoff = (now - timedelta(hours=window_hours)).isoformat()
        shifts: list[ProbShift] = []
        with self.engine.connect() as conn:
            rows = list(
                conn.execute(
                    text(
                        "SELECT id, slug, question, theme, yes_prob, observed_at, "
                        "       notified_shift "
                        "FROM poly_market_probs "
                        "WHERE observed_at >= :cutoff "
                        "ORDER BY slug ASC, observed_at ASC"
                    ),
                    {"cutoff": cutoff},
                )
            )
        by_slug: dict[str, list[Any]] = {}
        for row in rows:
            by_slug.setdefault(row.slug, []).append(row)
        for slug, obs in by_slug.items():
            if len(obs) < 2:
                continue  # insufficient history in the window
            earliest = obs[0]
            latest = obs[-1]
            delta = float(latest.yes_prob) - float(earliest.yes_prob)
            if abs(delta) < threshold:
                continue
            if latest.notified_shift:  # already alerted this exact shift
                continue
            shifts.append(
                ProbShift(
                    slug=slug,
                    question=str(latest.question or slug),
                    theme=str(latest.theme or "other"),
                    latest_prob=float(latest.yes_prob),
                    earliest_prob=float(earliest.yes_prob),
                    delta=delta,
                    window_hours=window_hours,
                    anchor_id=latest.id,
                )
            )
        return shifts

    def _mark_notified(self, shift: ProbShift) -> None:
        """Stamp the shift's anchoring row so it isn't re-notified. The
        stamp is the signed delta-points so a LATER, larger shift on a new
        row still fires (dedup is per-observation, not per-slug-forever)."""
        if shift.anchor_id is None:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE poly_market_probs SET notified_shift = :mark WHERE id = :id"),
                {"mark": f"{shift.delta_points:+d}", "id": shift.anchor_id},
            )

    # ------------------------------------------------------------------ #
    # Notification (trade-card style)
    # ------------------------------------------------------------------ #

    def notify_shift(self, shift: ProbShift) -> DispatchResult:
        """Build + send the Telegram-HTML shift alert, then mark it
        notified so subsequent cycles don't re-alert the same shift."""
        title, message = build_shift_alert(shift)
        result = notify_operator(title, message, html=True)
        try:
            self._mark_notified(shift)
        except Exception:
            logger.exception(
                "failed to mark shift notified for %s; may re-alert",
                shift.slug,
            )
        return result

    def notify_shifts(self, shifts: list[ProbShift]) -> int:
        """Notify every shift, tolerant of individual send failures.
        Returns the count that were attempted."""
        sent = 0
        for shift in shifts:
            try:
                self.notify_shift(shift)
                sent += 1
            except Exception:
                logger.exception(
                    "shift notify failed for %s; continuing",
                    shift.slug,
                )
        return sent

    # ------------------------------------------------------------------ #
    # Confluence / digest corroboration
    # ------------------------------------------------------------------ #

    def latest_prob_for_theme(
        self,
        theme: str,
    ) -> dict[str, dict[str, Any]]:
        """Most-recent YES prob per market in ``theme`` — the digest cites
        it as corroboration ("Prediction mkt: Hormuz-closure 18% ↑"). Also
        reports a short-window direction arrow when a prior observation
        exists. Returns ``{slug: {"question","yes_prob","rising"}}``.

        Display only — this does NOT gate anything (sibling owns confluence).
        Fail-soft: any DB error returns ``{}``."""
        try:
            with self.engine.connect() as conn:
                rows = list(
                    conn.execute(
                        text(
                            "SELECT slug, question, yes_prob, observed_at "
                            "FROM poly_market_probs "
                            "WHERE theme = :theme "
                            "ORDER BY slug ASC, observed_at DESC"
                        ),
                        {"theme": theme},
                    )
                )
        except Exception:
            logger.debug(
                "latest_prob_for_theme failed for %s",
                theme,
                exc_info=True,
            )
            return {}
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            # Rows are slug ASC, observed_at DESC → first per slug is latest.
            if row.slug in out:
                # A later (older) row for the same slug gives us direction.
                prev = out[row.slug]
                if prev.get("rising") is None and prev["yes_prob"] is not None:
                    prev["rising"] = prev["yes_prob"] >= float(row.yes_prob)
                continue
            out[row.slug] = {
                "question": str(row.question or row.slug),
                "yes_prob": float(row.yes_prob),
                "rising": None,
            }
        return out


# ---------------------------------------------------------------------- #
# Pure helpers
# ---------------------------------------------------------------------- #


def _parse_outcome_yes(raw: Any) -> float | None:
    """YES prob from Gamma ``outcomePrices`` — JSON-string array
    ``"[yesProb, noProb]"`` or a native list. None if unparseable."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        prices = parsed if isinstance(parsed, list) else None
    elif isinstance(raw, list):
        prices = raw
    else:
        prices = None
    if not prices:
        return None
    try:
        return float(prices[0])
    except (ValueError, TypeError):
        return None


def _implied_read(shift: ProbShift) -> str:
    """One-line operator read of the shift's meaning. A rising YES on an
    escalation-flavored market (closure/invasion/coup/…) is escalation;
    on a de-escalation market it inverts. Falls back to a neutral phrasing
    when the question polarity is ambiguous."""
    q = shift.question.lower()
    is_escalation_q = any(w in q for w in _ESCALATION_WORDS)
    is_deescalation_q = any(w in q for w in _DEESCALATION_WORDS)
    # De-escalation words win the polarity read when both appear
    # ("ceasefire in the war" → treat YES-up as de-escalation). None =
    # ambiguous question → neutral phrasing.
    polarity_up_is_escalation: bool | None
    if is_deescalation_q:
        polarity_up_is_escalation = False
    elif is_escalation_q:
        polarity_up_is_escalation = True
    else:
        polarity_up_is_escalation = None

    if polarity_up_is_escalation is None:
        return (
            "rising YES — market pricing in higher odds"
            if shift.rising
            else "falling YES — market pricing out the outcome"
        )
    escalating = shift.rising == polarity_up_is_escalation
    if escalating:
        return "market pricing in ESCALATION — watch the theme instruments"
    return "market pricing in DE-ESCALATION / fade risk on the theme"


def build_shift_alert(shift: ProbShift) -> tuple[str, str]:
    """``(title, html_message)`` for a prob-shift Telegram alert — built
    like the event trade cards. All interpolated content is html_escaped
    (the market question is hostile input)."""
    arrow = "↑" if shift.rising else "↓"
    pct = round(shift.latest_prob * 100)
    delta = shift.delta_points  # signed int
    lines = [
        "<b>Prediction market shift</b>",
        html_escape(shift.question),
        f"<b>{pct}% {arrow} {delta:+d} ({shift.window_hours}h)</b>",
        f"<i>{html_escape(shift.theme)}</i>",
        html_escape(_implied_read(shift)),
        f"https://polymarket.com/event/{html_escape(shift.slug)}",
    ]
    title = "Prediction market shift"
    return title, "\n".join(lines)
