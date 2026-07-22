"""Operator alerting for the event-driven pipeline (CL-ikz2).

Extracted from ``src/strategies/event_driven.py`` per the 2026-07-21
structural review (§6.1.1, §9 item 2): confirmed/expired alert bodies,
advisory idea blocks, grounded trade-card lines, and the cross-asset
corroboration lines are events-domain *notification formatting*, not
trading-strategy logic. :class:`EventDrivenStrategy` constructs (or is
injected with) an :class:`EventNotifier` and delegates to it; every
``notify_operator`` call for the event pipeline lives here.

The cross-asset idea-ledger stamp (``stamp_cross_asset_on_ideas``) also
lives here — it annotates the events layer's ``trade_ideas`` rows, a DB
write that was never strategy-domain.

Rendering contract: alert bodies are plain text (short lines, one fact
per line). The Telegram-HTML variant of the cross-asset line lives in
:func:`src.events.digest.build_cross_asset_line`; the plain-text ✓/✗
form here is intentionally markup-free.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from src.events.confluence import TRADABLE_KINDS, EventConfluence, mid_price
from src.events.prices import format_age
from src.research.notifications import notify_operator

logger = logging.getLogger(__name__)

#: Resolves a current price for ``(symbol, prices_tick_dict, now)`` —
#: the strategy injects its own ``_current_price`` (tick mid with a
#: data-provider fallback) so price resolution stays defined in ONE place.
PriceResolver = Callable[[str, dict[str, Any], datetime], "float | None"]


class EventNotifier:
    """Formats and dispatches event-pipeline operator alerts.

    Constructor-injected into :class:`EventDrivenStrategy` (a real one is
    built by default); tests can inject a recorder in its place. The
    scalar knobs mirror the strategy config values the alert bodies
    display — passed individually so this module never imports from
    ``src.strategies`` (events layer must not depend on strategies).
    """

    def __init__(
        self,
        *,
        event_risk_pct: float,
        event_stop_pct: float,
        event_max_holding_hours: float,
        confirm_window_max_minutes: int,
        db_engine: Any = None,
        price_resolver: PriceResolver | None = None,
    ) -> None:
        self._event_risk_pct = event_risk_pct
        self._event_stop_pct = event_stop_pct
        self._event_max_holding_hours = event_max_holding_hours
        self._confirm_window_max_minutes = confirm_window_max_minutes
        self._db = db_engine
        self._price_resolver = price_resolver

    def _resolve_price(
        self, symbol: str, prices: dict[str, Any], now: datetime,
    ) -> float | None:
        if self._price_resolver is not None:
            return self._price_resolver(symbol, prices, now)
        return mid_price(prices.get(symbol))

    # ------------------------------------------------------------------
    # Building blocks (plain text — short lines, one fact per line)
    # ------------------------------------------------------------------

    @staticmethod
    def _watch_list(assessment: dict[str, Any]) -> list[str]:
        watch: list[str] = []
        for aff in assessment.get("affected") or []:
            if not isinstance(aff, dict):
                continue
            kind = str(aff.get("kind") or "")
            direction = str(aff.get("direction") or "")
            if kind not in TRADABLE_KINDS or direction == "watch":
                name = str(aff.get("instrument") or "")
                if name:
                    watch.append(name)
        return watch

    def _ideas_block(
        self,
        assessment: dict[str, Any],
        limit: int = 5,
        prices: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        """Advisory trade-ideas lines (CL-mgcp) — clearly separated
        from the machine trades above them: the system never trades
        equities or options; these are for the operator's own hands.
        Empty list when the assessment carries no ideas.

        The TOP idea (highest confidence) additionally shows its
        GROUNDED trade-card numbers (CL-jiqq) — dollar stop/targets/R:R
        and, for options, the suggested strike + DTE window — whenever a
        live price for that ticker resolves. Absent a price the card
        degrades to its percentages; equity idea tickers usually are not
        in the engine's FX price feed, so this often shows the %-only
        form, which is honest."""
        ideas = assessment.get("trade_ideas")
        if not isinstance(ideas, list) or not ideas:
            return []
        usable = [i for i in ideas if isinstance(i, dict) and i.get("ticker")]
        if not usable:
            return []
        top = max(usable, key=lambda i: float(i.get("confidence") or 0.0))
        lines = ["Operator ideas (not machine-traded):"]
        for idea in ideas[:limit]:
            if not isinstance(idea, dict):
                continue
            parts = [
                str(idea.get("ticker") or "?"),
                str(idea.get("action") or "?"),
            ]
            horizon = str(idea.get("time_horizon") or "").strip()
            if horizon:
                parts.append(horizon)
            time_stop = idea.get("time_stop_days")
            if time_stop is not None:
                parts.append(f"stop{time_stop}d")
            line = f"- {' '.join(parts)}"
            rationale = str(idea.get("rationale") or "").strip()
            if rationale:
                line += f" — {rationale[:60]}"
            lines.append(line)
            if idea is top:
                lines.extend(self._top_idea_card_lines(idea, prices, now))
        return lines if len(lines) > 1 else []

    def _top_idea_card_lines(
        self,
        idea: dict[str, Any],
        prices: dict[str, Any] | None,
        now: datetime | None,
    ) -> list[str]:
        """Indented grounded-card detail for the top confirmed idea
        (CL-jiqq). Returns ``[]`` when nothing concrete resolves — an
        LLM percentage with no price and no trigger isn't worth a line."""
        from src.events.trade_card import build_trade_card  # noqa: PLC0415

        ticker = str(idea.get("ticker") or "")
        current = self._resolve_price(
            ticker, prices or {}, now or datetime.now(UTC),
        )
        card = build_trade_card(dict(idea), current)
        detail: list[str] = []
        entry = str(idea.get("entry_trigger") or "").strip()
        if entry:
            detail.append(f"entry {entry[:40]}")
        stop_price = card.get("stop_price")
        if stop_price is not None:
            detail.append(f"stop ${stop_price:,.2f}")
        targets = card.get("target_prices") or []
        if targets:
            detail.append("tgt " + "/".join(f"${t:,.2f}" for t in targets))
        rr = card.get("risk_reward")
        if rr is not None:
            detail.append(f"R:R {rr}")
        if card.get("is_option"):
            strike = card.get("suggested_strike")
            if strike is not None:
                detail.append(
                    f"~{card.get('dte_window') or ''} strike ${strike:,.2f} "
                    f"(nearest listed)".strip(),
                )
            elif card.get("dte_window"):
                detail.append(f"{card.get('dte_window')} to expiry")
        inval = str(idea.get("invalidation") or "").strip()
        if inval:
            detail.append(f"invalid if {inval[:40]}")
        return [f"  {' | '.join(detail)}"] if detail else []

    @staticmethod
    def _event_age(row: dict[str, Any], now: datetime | None = None) -> str:
        """``2h`` / ``3d`` since the event was first seen, or ``""``."""
        return format_age(row.get("seen_at"), now=now)

    @staticmethod
    def _cross_asset_summary(result: Any) -> str | None:
        """One-line machine-friendly cross-asset summary for the idea
        ledger's ``notes`` (CL-6mzn) — ``cross-asset: confirms 2/3
        (BCO_USD +1.8%, USD_CAD -0.2%)`` — or None when unknown/no-data."""
        if result is None:
            return None
        confirmed = getattr(result, "confirmed", None)
        if confirmed is None:
            return None
        details = list(getattr(result, "details", []) or [])
        voting = [d for d in details if getattr(d, "agrees", None) is not None]
        if not voting:
            return None
        n_agree = sum(1 for d in voting if d.agrees)
        n_voting = len(voting)
        verb = "confirms" if confirmed else "NOT confirming (fade risk)"
        moves = ", ".join(
            f"{d.instrument} {(d.actual_move_pct or 0.0):+.1f}%" for d in voting
        )
        return f"cross-asset: {verb} {n_agree}/{n_voting} ({moves})"

    @staticmethod
    def _cross_asset_line(result: Any) -> str | None:
        """Plain-text cross-asset corroboration line for the confirmed
        alert (CL-6mzn), or None when the read is unknown/no-data.

        ``result`` is a :class:`src.events.cross_asset.CrossAssetResult`.
        The Telegram-HTML variant lives in
        :func:`src.events.digest.build_cross_asset_line`; this alert body
        is plain text, so we render ✓/✗ without markup here."""
        if result is None:
            return None
        confirmed = getattr(result, "confirmed", None)
        if confirmed is None:
            return None  # unknown / no data — omit
        details = list(getattr(result, "details", []) or [])
        voting = [d for d in details if getattr(d, "agrees", None) is not None]
        if not voting:
            return None
        n_agree = sum(1 for d in voting if d.agrees)
        n_voting = len(voting)
        parts = [
            f"{d.instrument} {(d.actual_move_pct or 0.0):+.1f}% "
            f"{'✓' if d.agrees else '✗'}"
            for d in voting
        ]
        body = " · ".join(parts)
        if confirmed:
            return f"Cross-asset: {body} · confirms ({n_agree}/{n_voting})"
        return (
            f"Cross-asset: related assets NOT confirming — fade risk "
            f"({n_agree}/{n_voting}) · {body}"
        )

    # ------------------------------------------------------------------
    # Idea-ledger stamp (events-domain DB write)
    # ------------------------------------------------------------------

    def stamp_cross_asset_on_ideas(self, geo_event_id: Any, result: Any) -> None:
        """Append the cross-asset summary to the ``notes`` of this event's
        persisted trade ideas (CL-6mzn). Additive and idempotent-ish
        (skips rows whose notes already carry a ``cross-asset:`` marker);
        never touches the schema or the machine-trade path. Best-effort —
        a DB error or missing table is swallowed (annotation only)."""
        if self._db is None or geo_event_id is None:
            return
        summary = self._cross_asset_summary(result)
        if not summary:
            return
        try:
            with self._db.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE trade_ideas "
                        "SET notes = CASE "
                        "  WHEN notes IS NULL OR notes = '' THEN :summary "
                        "  ELSE notes || ' | ' || :summary END "
                        "WHERE geo_event_id = :eid "
                        "  AND (notes IS NULL OR notes NOT LIKE '%cross-asset:%')"
                    ),
                    {"summary": summary, "eid": int(geo_event_id)},
                )
        except Exception:
            logger.debug(
                "cross-asset idea-notes stamp failed for geo_event_id=%s",
                geo_event_id, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Alert dispatch
    # ------------------------------------------------------------------

    def alert_confirmed(
        self,
        row: dict[str, Any],
        assessment: dict[str, Any],
        entered: list[tuple[str, str, str, float, str]],
        skipped: list[tuple[str, str]],
        prices: dict[str, Any] | None = None,
        now: datetime | None = None,
        cross_asset: Any = None,
    ) -> None:
        now = now or datetime.now(UTC)
        lines = [f"Headline: {str(row.get('headline') or '')[:140]}"]
        age = self._event_age(row, now)
        if age:
            lines.append(f"Age: {age} since first seen")
        cross_line = self._cross_asset_line(cross_asset)
        if cross_line:
            lines.append(cross_line)
        for symbol, dir_str, size_str, entry_price, reason in entered:
            lines.append(
                f"Trade: {symbol} {dir_str} ({size_str} units) @ {entry_price:g}",
            )
            if reason:
                lines.append(f"  Why: {reason[:100]}")
        for name, reason in skipped:
            line = f"Skipped: {name} ({reason})"
            current = self._resolve_price(name, prices or {}, now)
            if current is not None:
                line += f" @ {current:g}"
            lines.append(line)
        lines.append(f"Risk: {self._event_risk_pct * 100:.2f}% of equity per trade")
        lines.append(f"Stop: {self._event_stop_pct * 100:.2f}% from entry")
        lines.append(f"Time stop: {self._event_max_holding_hours:g}h")
        lines.append(f"Urgency: {assessment.get('urgency')}/10")
        with contextlib.suppress(TypeError, ValueError):
            lines.append(f"Confidence: {float(assessment.get('confidence') or 0.0):.2f}")
        watch = self._watch_list(assessment)
        if watch:
            lines.append("Watch: " + ", ".join(watch))
        ideas = self._ideas_block(assessment, prices=prices, now=now)
        if ideas:
            lines.append("")
            lines.extend(ideas)
        try:
            notify_operator("Event confirmed", "\n".join(lines), priority=1)
        except Exception:
            logger.exception("Confirmed-event alert dispatch failed")

    def alert_expired(self, row: dict[str, Any], urgency: int, confidence: float) -> None:
        lines = [f"Headline: {str(row.get('headline') or '')[:140]}"]
        age = self._event_age(row)
        if age:
            lines.append(f"Age: {age} since first seen")
        lines += [
            f"Urgency: {urgency}/10",
            f"Confidence: {confidence:.2f}",
            f"No market confirmation within {self._confirm_window_max_minutes}min",
            "No trade taken",
        ]
        # Top advisory idea (highest confidence) still surfaces — an
        # expired-unconfirmed event can be an operator opportunity even
        # when the machine passes (CL-mgcp).
        assessment = EventConfluence.parse_assessment(row.get("assessment")) or {}
        ideas = [
            i for i in (assessment.get("trade_ideas") or [])
            if isinstance(i, dict) and i.get("ticker")
        ]
        if ideas:
            top = max(
                ideas, key=lambda i: float(i.get("confidence") or 0.0),
            )
            line = (
                f"Top idea: {top.get('ticker')} {top.get('action') or '?'}"
            )
            if top.get("time_stop_days") is not None:
                line += f" (stop {top.get('time_stop_days')}d)"
            rationale = str(top.get("rationale") or "").strip()
            if rationale:
                line += f" — {rationale[:60]}"
            lines.append(line)
        try:
            notify_operator("Event expired unconfirmed", "\n".join(lines), priority=0)
        except Exception:
            logger.exception("Expired-event alert dispatch failed")
