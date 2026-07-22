"""Operator notifications for the research loop (CL-0hr3, CL-yta6,
CL-o2vb, CL-frn7).

Extracted from ``src.research.loop`` per the 2026-07-21 structural
review (§6.2.3): Telegram-HTML formatting is not loop logic. The loop
holds a :class:`LoopNotifier` port and asks it to announce gate events;
this module owns the phone-first message layout and the dispatch
error containment.

Contract with the loop:

  * Each ``notify_*`` method returns ``True`` when the underlying
    channel actually attempted a send (``DispatchResult.any_attempted``)
    so the loop can tally per-run notification counters.
  * Dispatch failures are contained HERE — a notifier exception is
    logged and swallowed; a notification must never kill a pipeline
    pass.

Tests keep the existing injection seam: pass a recording/silent
``notify_fn`` (``(title, message, priority) → DispatchResult``) either
straight to ``ResearchLoop(notify_fn=...)`` or wrapped in a
``LoopNotifier``. The default dispatch hits production Telegram via
``src.research.notifications.notify_operator`` (no-op when env-vars
unset) with ``html=True`` — every message built here is Telegram-HTML
with all interpolated content escaped via :func:`html_escape`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.research.agents.idea import IdeaResult
from src.research.agents.reviewer import Position
from src.research.instruments import (
    extract_brief_instruments,
    extract_candidate_instruments,
)
from src.research.notifications import (
    DispatchResult,
    html_escape,
    notify_operator,
)
from src.research.verdict import VerdictResult

logger = logging.getLogger(__name__)


# Notifier callable. Default = production Telegram via
# src.research.notifications.notify_operator; tests inject a recorder.
# Signature: (title, message, priority) → DispatchResult.
# Messages are built as Telegram-HTML (interpolations escaped via
# html_escape); the default passes html=True so Telegram renders it
# (CL-frn7).
NotifyFn = Callable[[str, str, int], DispatchResult]


def default_notify_fn(title: str, message: str, priority: int) -> DispatchResult:
    """Production dispatch: Telegram-HTML via ``notify_operator``
    (no-op when TELEGRAM_* env-vars are unset)."""
    return notify_operator(title=title, message=message, priority=priority, html=True)


# --------------------------------------------------------------------------- #
# Formatting helpers (CL-frn7)
# --------------------------------------------------------------------------- #


def _fmt_instruments(instruments: list[str]) -> str:
    """Comma-joined, HTML-escaped instrument list; explicit fallback so
    the operator sees that extraction found nothing rather than a
    silently missing line."""
    if not instruments:
        return "(unknown)"
    return html_escape(", ".join(instruments))


def _fmt_position(pos: Position | None) -> str:
    """Debate position for display. ``None`` (no parsed position)
    renders as the literal string ``None`` — same output the operator
    always saw, minus the type confusion."""
    return html_escape(pos.value if pos is not None else "None")


def _brief_thesis(brief_path: Path) -> str:
    """One-line thesis: the brief's first ``# `` heading with the
    'Hypothesis:' prefix stripped. Empty string when unavailable."""
    try:
        text = brief_path.read_text()
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            thesis = stripped[2:].strip()
            if thesis.lower().startswith("hypothesis:"):
                thesis = thesis[len("hypothesis:"):].strip()
            return thesis
    return ""


def _report_oos_metrics(report_path: str | Path) -> dict[str, Any]:
    """OOS metrics dict from a candidate report JSON; {} on any
    problem. Handles both nesting shapes the pipeline has produced
    (``backtest_metrics.oos_metrics`` and top-level ``oos_metrics``)."""
    try:
        report = json.loads(Path(report_path).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(report, dict):
        return {}
    oos = report.get("backtest_metrics", {}).get(
        "oos_metrics", {},
    ) or report.get("oos_metrics", {})
    return oos if isinstance(oos, dict) else {}


def _fmt_backtest_line(oos: dict[str, Any]) -> str:
    """Key backtest numbers on one line; empty string when none are
    available (line is then omitted from the message)."""
    parts: list[str] = []
    sharpe = oos.get("sharpe")
    if isinstance(sharpe, (int, float)):
        parts.append(f"Sharpe {sharpe:.2f}")
    max_dd = oos.get("max_drawdown")
    if isinstance(max_dd, (int, float)):
        parts.append(f"max DD {max_dd:.2%}")
    n_trades = oos.get("n_trades")
    if isinstance(n_trades, (int, float)):
        parts.append(f"{int(n_trades)} trades")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# LoopNotifier — the port ResearchLoop talks to
# --------------------------------------------------------------------------- #


class LoopNotifier:
    """Builds and dispatches the loop's operator notifications.

    Constructor-injected into :class:`src.research.loop.ResearchLoop`;
    the default instance dispatches through :func:`default_notify_fn`.
    Tests inject a recording ``notify_fn`` (the ``_silent_notifier``
    pattern) without touching this class.
    """

    def __init__(self, notify_fn: NotifyFn | None = None) -> None:
        self.notify_fn: NotifyFn = notify_fn or default_notify_fn

    # ------------------------------------------------------------------ #
    # Dispatch plumbing
    # ------------------------------------------------------------------ #

    def _dispatch(self, kind: str, title: str, message: str, priority: int) -> bool:
        """Send one notification; returns True when the channel
        attempted a send. Best-effort — notifier failures don't kill
        the loop, they just get logged."""
        try:
            disp = self.notify_fn(title, message, priority)
        except Exception as exc:
            logger.warning(
                "%s notification dispatch raised: %s: %s",
                kind, type(exc).__name__, exc,
            )
            return False
        return disp.any_attempted

    # ------------------------------------------------------------------ #
    # GATE 1 — pre-research operator approval (CL-0hr3)
    # ------------------------------------------------------------------ #

    def notify_gate1(self, result: IdeaResult, extract_hash: str) -> bool:
        """Fire pre-research approval notification.

        Phone-first Telegram-HTML layout (CL-frn7): bold header via the
        title, slug + one-line thesis, which instruments the plan would
        trade (from the brief's Data requirements), and the reply line.
        """
        title = "GATE 1 — new trading hypothesis"
        tradable: list[str] = []
        inputs: list[str] = []
        thesis = ""
        if result.hypothesis_path:
            tradable, inputs = extract_brief_instruments(result.hypothesis_path)
            thesis = _brief_thesis(Path(result.hypothesis_path))
        # Short id for the Telegram approval bot (CL-b1l6) — a prefix
        # of the extract hash; the bot resolves any unambiguous prefix.
        short_id = extract_hash[:6]
        lines = [f"<b>{html_escape(result.strategy_slug)}</b>"]
        if thesis:
            lines.append(html_escape(thesis))
        lines.append("")
        lines.append(f"<b>Trades:</b> {_fmt_instruments(tradable)}")
        if inputs:
            lines.append(f"<b>Inputs:</b> {_fmt_instruments(inputs)}")
        lines.append("")
        lines.append(f"Reply: approve {short_id} | reject {short_id}")
        return self._dispatch("GATE 1", title, "\n".join(lines), 0)

    # ------------------------------------------------------------------ #
    # ESCALATE — debate needs operator review (CL-o2vb)
    # ------------------------------------------------------------------ #

    def notify_escalate(
        self,
        slug: str,
        verdict: VerdictResult,
        candidate_report_path: Path,
        transcript_path: Path,
        candidate_report: dict[str, Any],
    ) -> bool:
        """Fire ESCALATE alert. priority=1 so the operator notices;
        dedup is the loop's job (a slug only gets debated once)."""
        # Identify which rule(s) triggered ESCALATE: missing metrics
        # are the most common cause; ambiguous-positions case shows up
        # in verdict.reason.
        problem_rules = [
            f"{e.rule_id} ({e.detail})"
            for e in verdict.rule_evaluations
            if e.missing or not e.passed
        ]
        oos = candidate_report.get("backtest_metrics", {}).get(
            "oos_metrics", {},
        ) or candidate_report.get("oos_metrics", {})
        metrics_blurb = ", ".join(
            f"{k}={v}" for k, v in oos.items()
        ) if isinstance(oos, dict) else ""
        title = f"ESCALATE: debate result needs operator review — {slug}"
        # Telegram-HTML, phone-first (CL-frn7): every interpolated
        # value escaped; short lines, blank-line separation.
        lines = [
            f"<b>{html_escape(slug)}</b>",
            f"Verdict: ESCALATE — {html_escape(verdict.reason)}",
            f"Bull: {_fmt_position(verdict.bull_position)} · "
            f"Bear: {_fmt_position(verdict.bear_position)}",
            "",
            "<b>Problem rules:</b>",
        ]
        if problem_rules:
            lines += [f"- {html_escape(r)}" for r in problem_rules]
        else:
            lines.append("(none — agent positions diverged)")
        if metrics_blurb:
            lines.append(f"OOS: {html_escape(metrics_blurb)}")
        lines += [
            "",
            f"Transcript: {html_escape(str(transcript_path))}",
            f"Candidate report: {html_escape(str(candidate_report_path))}",
            "Review the debate, then retry / archive / data-seed.",
        ]
        return self._dispatch("ESCALATE", title, "\n".join(lines), 1)

    # ------------------------------------------------------------------ #
    # GATE 2 — pre-deploy operator confirmation (CL-yta6)
    # ------------------------------------------------------------------ #

    def notify_gate2(
        self,
        slug: str,
        entry: dict[str, Any],
        code_path: str | None = None,
    ) -> bool:
        """Fire pre-deploy confirmation notification. priority=1 since
        this is a deploy decision and we want the operator to notice.

        Phone-first Telegram-HTML layout (CL-frn7): slug, which
        instruments the candidate code trades, verdict + key backtest
        numbers, reply line.
        """
        title = "GATE 2 — strategy ready to deploy"
        instruments = (
            extract_candidate_instruments(code_path) if code_path else []
        )
        reason = str(entry.get("reason") or "").strip()
        if len(reason) > 200:
            reason = reason[:197] + "..."
        verdict_line = "Verdict: PROMOTE" + (
            f" — {html_escape(reason)}" if reason else ""
        )
        backtest_line = _fmt_backtest_line(
            _report_oos_metrics(entry.get("candidate_report_path", "")),
        )
        lines = [
            f"<b>{html_escape(slug)}</b>",
            "",
            f"<b>Trades:</b> {_fmt_instruments(instruments)}",
            verdict_line,
        ]
        if backtest_line:
            lines.append(backtest_line)
        lines.append("Paper-shadow at allocation=0 — no real-money risk.")
        lines.append("")
        lines.append(f"Reply: approve {html_escape(slug)} | "
                     f"reject {html_escape(slug)}")
        return self._dispatch("GATE 2", title, "\n".join(lines), 1)
