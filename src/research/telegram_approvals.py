"""Interactive Telegram approvals for the research-loop gates (CL-b1l6).

The research loop (CL-x561) holds work at two operator gates:

  * GATE 1 — hypothesis approval: ``PENDING_OPERATOR_APPROVAL``
  * GATE 2 — deploy confirmation: ``PENDING_DEPLOY_CONFIRMATION``

Until now the only approval paths were ``scripts/research_approve.py``
and the soak dashboard. This module lets the operator reply directly
in the Telegram chat that already receives the gate notifications
(``src/research/notifications.py``):

    approve a1b2c3      # gate 1: hypothesis a1b2c3… → APPROVED
    reject vol-carry    # gate 2: slug vol-carry → DEPLOY_REJECTED
    skip a1b2c3 dupe    # gate 1: → SKIPPED with reason "dupe"
    pending             # list everything awaiting a decision
    ideas               # open trade ideas from the event pipeline
    help                # command grammar

``ideas`` (CL-mgcp) is READ-ONLY v1: it lists ``pending`` rows from
the ``trade_ideas`` ledger (migration 007) with age vs time stop and a
live price where one resolves. Ideas auto-expire in the pipeline
cycle; operator-driven transitions (taken/cancelled/closed) are
schema-reserved future work — no bot command mutates the ledger yet.

Design:

  * **Long-polling** ``getUpdates`` with the ``offset`` acknowledged
    after each batch and persisted (atomically) to
    ``data/research/telegram_offset.json`` so restarts never replay
    already-handled commands.
  * **Security** — only messages whose ``chat.id`` int-equals the
    configured ``TELEGRAM_CHAT_ID`` are processed; everything else is
    logged at WARNING and dropped. The bot token is never logged; all
    HTTP errors are scrubbed the way ``notifications.py`` scrubs them.
  * **Shared mutations** — gate transitions go through
    ``src.research.approvals`` (the same code path as the CLI
    approver), and state writes are atomic so the concurrently-running
    loop never reads a torn ``state.json``.
  * **Testability** — command parsing / target resolution / handling
    are pure functions over ``LoopState``; the HTTP layer is an
    injectable callable (mirrors the ingest ``HttpGet`` shim pattern).

Entrypoint: ``scripts/telegram_approval_bot.py``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from src.research.approvals import (
    act_gate1,
    act_gate2,
    gate1_pending,
    gate2_pending,
    save_state_atomic,
)
from src.research.loop import DEFAULT_STATE_PATH, LoopState, load_state
from src.research.notifications import _scrub_token

logger = logging.getLogger(__name__)

# Silence httpx's INFO "HTTP Request: POST <url>" line — the Telegram
# URL embeds the bot token (same rationale as notifications.py, CL-wmn4).
logging.getLogger("httpx").setLevel(logging.WARNING)


DEFAULT_OFFSET_PATH: Path = Path("data/research/telegram_offset.json")

# Long-poll window for getUpdates. Telegram caps at ~50s; the HTTP
# read timeout gets a pad on top so the server can answer at the edge.
DEFAULT_POLL_TIMEOUT_SEC: int = 50
_HTTP_TIMEOUT_PAD_SEC: float = 10.0

# Short-id length shown for GATE 1 extract hashes in notifications and
# the `pending` listing. Any longer prefix also resolves.
SHORT_ID_LEN: int = 6

# Extra sendMessage attempts after a failure (fresh connection each
# time — transient TLS handshake timeouts were observed live).
_SEND_RETRIES: int = 1

_ACTION_VERBS = frozenset({"approve", "reject", "skip"})
_KNOWN_VERBS = _ACTION_VERBS | {"pending", "help", "start", "ideas", "idea"}

HELP_TEXT = (
    "curLit research approval bot — commands:\n"
    "  approve <id> [reason]  approve a pending gate entry\n"
    "  reject <id> [reason]   decline (gate 1: SKIP, gate 2: reject deploy)\n"
    "  skip <id> [reason]     same as reject\n"
    "  pending                list entries awaiting your decision\n"
    "  ideas                  list open trade ideas (read-only)\n"
    "  idea <id>              full trade card for one idea (levels, "
    "strike, DTE)\n"
    "  help                   this message\n"
    "<id> is the short id from the gate notification or the 'pending' "
    "listing (gate 1: extract-hash prefix, gate 2: strategy slug); for "
    "'idea <id>' it is the idea's short id from the 'ideas' listing."
)


# --------------------------------------------------------------------- #
# Telegram API shim (injectable, token-scrubbed)
# --------------------------------------------------------------------- #


class TelegramApiError(RuntimeError):
    """Raised for transport / API-level failures. The message is always
    token-scrubbed before construction."""


# (method, params) → parsed JSON payload. Tests inject a fake.
ApiCall = Callable[[str, dict[str, Any]], dict[str, Any]]


def make_httpx_api(token: str) -> ApiCall:
    """Default ApiCall backed by httpx POSTs to api.telegram.org.

    Any exception is re-raised as ``TelegramApiError`` with the bot
    token scrubbed and the original exception suppressed (its repr may
    embed the request URL, which contains the token).
    """

    def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        timeout = float(params.get("timeout", 0)) + _HTTP_TIMEOUT_PAD_SEC
        try:
            resp = httpx.post(
                f"https://api.telegram.org/bot{token}/{method}",
                data=params,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
        except Exception as exc:
            sanitized = _scrub_token(str(exc), token)
            raise TelegramApiError(
                f"{method} failed: {type(exc).__name__}: {sanitized}",
            ) from None  # suppress chain — original may embed the token

    return call


# --------------------------------------------------------------------- #
# Offset persistence
# --------------------------------------------------------------------- #


def load_offset(path: Path | str = DEFAULT_OFFSET_PATH) -> int | None:
    """Read the persisted getUpdates offset; None = first run / reset."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        return int(raw["offset"])
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "offset file %s malformed (%s); treating as fresh start",
            p,
            type(exc).__name__,
        )
        return None


def save_offset(offset: int, path: Path | str = DEFAULT_OFFSET_PATH) -> None:
    """Atomically persist the next getUpdates offset (tmp + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=p.parent,
        prefix=f".{p.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"offset": offset}))
        os.replace(tmp_name, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# --------------------------------------------------------------------- #
# Command parsing (pure)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParsedCommand:
    """One operator command. ``verb`` is normalized lowercase;
    ``target`` is the (lowercased) short id for action verbs;
    ``reason`` is any free text after the id."""

    verb: str
    target: str = ""
    reason: str = ""


#: Punctuation phone keyboards append/attach to tokens ("Approve
#: a1b2c3." / "approve 'a1b2c3'"). Stripped from BOTH ends of the verb
#: and the id — never from the middle (slugs contain hyphens).
_TOKEN_TRIM_CHARS = ".,;:!?'\"()[]“”‘’"


def parse_command(text: str) -> ParsedCommand | None:
    """Parse a chat message into a command. Case-insensitive on the
    verb AND the id; a leading ``/`` is tolerated (Telegram habit);
    surrounding punctuation from mobile autocorrect is trimmed."""
    tokens = text.strip().split()
    if not tokens:
        return None
    verb = tokens[0].lower().lstrip("/").strip(_TOKEN_TRIM_CHARS)
    if verb not in _KNOWN_VERBS:
        return None
    if verb == "start":  # Telegram's default first message → help
        verb = "help"
    target = tokens[1].lower().strip(_TOKEN_TRIM_CHARS) if len(tokens) > 1 else ""
    reason = " ".join(tokens[2:])
    return ParsedCommand(verb=verb, target=target, reason=reason)


# --------------------------------------------------------------------- #
# Target resolution (pure)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateTarget:
    """A resolved pending entry. ``key`` is the state-dict key the
    shared mutation functions expect (extract hash for gate 1, slug
    for gate 2); ``short_id`` is what the operator sees/replies."""

    gate: int
    key: str
    slug: str

    @property
    def short_id(self) -> str:
        return self.key[:SHORT_ID_LEN] if self.gate == 1 else self.key


def _pending_targets(state: LoopState) -> list[GateTarget]:
    targets = [
        GateTarget(gate=1, key=h, slug=str(e.get("slug") or h)) for h, e in gate1_pending(state)
    ]
    targets += [GateTarget(gate=2, key=slug, slug=slug) for slug, e in gate2_pending(state)]
    return targets


def resolve_target(
    state: LoopState,
    token: str,
) -> tuple[GateTarget | None, str]:
    """Match a short id against pending entries.

    Gate 1 matches on extract-hash prefix or exact slug; gate 2 on
    slug prefix. Returns ``(target, "")`` on a unique match, else
    ``(None, error_reply)`` — ambiguous prefixes list the candidates,
    non-pending matches report their decided status.
    """
    token_l = token.lower()
    matches = [
        t
        for t in _pending_targets(state)
        if t.key.lower().startswith(token_l) or t.slug.lower() == token_l
    ]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        lines = [f"  {t.short_id}  {t.slug} (gate {t.gate})" for t in matches]
        return None, (
            f"Ambiguous id {token!r} — matches {len(matches)} pending "
            f"entries:\n" + "\n".join(lines) + "\nReply with a longer prefix."
        )
    decided = _decided_matches(state, token_l)
    if decided:
        return None, (
            "Already decided — no pending entry matches "
            f"{token!r}:\n" + "\n".join(f"  {d}" for d in decided)
        )
    return None, (
        f"Unknown id {token!r} — nothing pending matches. Send 'pending' to list open approvals."
    )


def _decided_matches(state: LoopState, token_l: str) -> list[str]:
    """Non-pending entries matching the token, with their statuses —
    powers the 'already decided' error reply."""
    out: list[str] = []
    for h, e in state.ideas_processed.items():
        slug = str(e.get("slug") or "")
        if h.lower().startswith(token_l) or slug.lower() == token_l:
            out.append(f"{h[:SHORT_ID_LEN]}  {slug or h} (gate 1) status={e.get('status')}")
    for slug, e in state.debates_completed.items():
        if slug.lower().startswith(token_l):
            status = e.get("deploy_status") or f"verdict={e.get('verdict')}"
            out.append(f"{slug} (gate 2) {status}")
    return out


# --------------------------------------------------------------------- #
# Command handling (pure over LoopState)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommandResult:
    """Reply text plus whether ``state`` was mutated (and therefore
    needs persisting)."""

    reply: str
    state_changed: bool = False


def _entry_trades_line(instruments: list[str]) -> str:
    """Indented ``Trades:`` line for the pending listing. An explicit
    ``(unknown)`` beats silence — the whole point of showing the
    listing is telling the operator what would be traded (CL-frn7)."""
    return "   Trades: " + (", ".join(instruments) or "(unknown)")


def render_pending(state: LoopState) -> str:
    """Phone-readable listing of open approvals: numbered, two lines
    per entry (id + slug, then which instruments it would trade).
    Plain text — bot replies echo operator input and file content, so
    staying out of parse_mode means nothing needs escaping."""
    # Local import: instruments lazily pulls in the backtest stack,
    # which the bot only needs when actually rendering this listing.
    from src.research.instruments import (
        extract_brief_instruments,
        extract_candidate_instruments,
    )

    g1 = gate1_pending(state)
    g2 = gate2_pending(state)
    if not g1 and not g2:
        return "No gate approvals pending."
    lines: list[str] = ["Pending approvals:", ""]
    n = 0
    for h, e in g1:
        n += 1
        lines.append(f"{n}. {h[:SHORT_ID_LEN]} — {e.get('slug')} (gate 1)")
        brief = e.get("hypothesis_path")
        tradable = extract_brief_instruments(brief)[0] if brief else []
        lines.append(_entry_trades_line(tradable))
    for slug, _e in g2:
        n += 1
        lines.append(f"{n}. {slug} (gate 2)")
        code_path = state.candidates_processed.get(slug, {}).get("code_path")
        instruments = extract_candidate_instruments(code_path) if code_path else []
        lines.append(_entry_trades_line(instruments))
    lines.append("")
    lines.append("Reply: approve <id> | reject <id> | skip <id>")
    return "\n".join(lines)


def _ideas_engine() -> Any:
    """Engine for the trade_ideas ledger, from the same POSTGRES_* env
    the event pipeline uses — via the shared ``build_db_url`` helper
    (CL-8lv6: DATABASE_URL override preserved; warns once per process on
    the well-known default password). Module-level so tests monkeypatch
    it."""
    from sqlalchemy import create_engine  # noqa: PLC0415

    from src.data.db_env import build_db_url  # noqa: PLC0415

    return create_engine(build_db_url())


#: Short id length shown for trade ideas in the `ideas` listing — the
#: prefix the operator replies with to `idea <id>` for the full card.
IDEA_SHORT_ID_LEN: int = 6

#: The bold advisory footer on every full idea card (CL-jiqq). These are
#: NOT machine-traded; the system has no live options/IV data, so the
#: operator MUST confirm strikes and expiries on their own broker.
IDEA_ADVISORY_FOOTER = (
    "⚠️ advisory — not machine-traded; confirm strikes/expiries on your "
    "broker; no live options/IV data"
)


def render_ideas(
    engine: Any = None,
    get_prices_fn: Any = None,
    now: Any = None,
) -> str:
    """Phone-readable listing of OPEN (pending) trade ideas (CL-mgcp):
    ``1. [a1b2c3] TSM BUY PUTS — 2d old / stop 5d — $172.40 (-1.8%)``.
    The bracketed short id is what the operator sends to ``idea <id>``
    for the full trade card (CL-jiqq).

    READ-ONLY v1 — ideas auto-expire in the pipeline cycle; there is
    no bot mutation path yet (taken/cancelled/closed are future work).
    Graceful degradation: ledger unreachable (migration 007 not
    applied, DB down) → an 'unavailable' reply, never a crash; price
    resolution failing → lines render without prices. Plain text like
    every bot reply — no parse_mode, nothing needs escaping.

    Consolidated on (ticker, action) (CL-5mkf): the ledger keeps a row
    per geo_event (audit trail), but a gold idea corroborated by six
    events reads here as ONE line with ``×6 events`` rather than six
    near-duplicate lines. The kept row is the freshest/highest-conf of
    the group, so its short id + levels are current."""
    # Local imports: the ledger/price stack is event-pipeline plumbing
    # the approval bot only needs for this listing.
    from src.events import idea_ledger  # noqa: PLC0415
    from src.events import prices as prices_mod  # noqa: PLC0415

    try:
        engine = engine if engine is not None else _ideas_engine()
        rows = idea_ledger.list_open_consolidated(engine)
    except Exception as exc:
        logger.warning(
            "ideas listing unavailable: %s: %s",
            type(exc).__name__,
            exc,
        )
        return (
            "Trade ideas unavailable (ledger unreachable — is migration "
            "007 applied and the DB up?)."
        )
    if not rows:
        return "No open trade ideas."
    fetch = get_prices_fn if get_prices_fn is not None else prices_mod.get_prices
    try:
        price_map = fetch([r["ticker"] for r in rows], engine=engine)
    except Exception:
        logger.warning("ideas listing: price fetch failed", exc_info=True)
        price_map = {}
    lines = [f"Open trade ideas ({len(rows)}):", ""]
    for n, row in enumerate(rows, 1):
        age = prices_mod.format_age(row.get("created_at"), now=now)
        short_id = str(row.get("idea_id") or "")[:IDEA_SHORT_ID_LEN]
        action = str(row.get("action") or "?").replace("_", " ").upper()
        head = f"{n}. [{short_id}] {row['ticker']} {action}"
        # ×N events (CL-5mkf): N distinct geo_events proposed this same
        # (ticker, action) → conviction, shown as one consolidated line.
        event_count = int(row.get("event_count") or 1)
        if event_count > 1:
            head += f" ×{event_count} events"
        parts = [head]
        aging = []
        if age:
            aging.append(f"{age} old")
        if row.get("time_stop_days") is not None:
            aging.append(f"stop {row['time_stop_days']}d")
        if aging:
            parts.append(" / ".join(aging))
        price = prices_mod.format_price(
            row["ticker"],
            price_map.get(row["ticker"]),
        )
        if price:
            parts.append(price)
        lines.append(" — ".join(parts))
    lines.append("")
    lines.append("Send 'idea <id>' for the full trade card.")
    lines.append("Read-only; ideas expire automatically at their time stop.")
    return "\n".join(lines)


def _fmt_dollar(value: Any) -> str:
    """``$172.40`` for a dollar level, or ``?`` when absent/unparseable."""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "?"


def render_idea_detail(
    idea_id: str,
    engine: Any = None,
    get_prices_fn: Any = None,
    now: Any = None,
) -> str:
    """The FULL trade card for one idea (CL-jiqq) — everything the
    operator needs to actually place the trade: ticker, live price,
    action + instrument, entry zone, dollar stop, dollar targets, R:R,
    suggested option strike + DTE window with the 'pick nearest listed'
    caveat, entry trigger, invalidation, rationale, age vs time stop,
    and the bold advisory footer.

    Grounded numbers are RECOMPUTED live off the current price (the
    persisted card can be stale — the price moved since the signal); the
    persisted card is the fallback when no live price resolves. Plain
    text, graceful degradation, never a crash — same posture as
    :func:`render_ideas`."""
    from src.events import idea_ledger  # noqa: PLC0415
    from src.events import prices as prices_mod  # noqa: PLC0415
    from src.events.retail_proxy import full_detail  # noqa: PLC0415
    from src.events.trade_card import build_trade_card  # noqa: PLC0415

    prefix = str(idea_id or "").strip().strip(_TOKEN_TRIM_CHARS)
    if not prefix:
        return "Usage: idea <id> — send 'ideas' for the open ideas and their ids."
    try:
        engine = engine if engine is not None else _ideas_engine()
        row = idea_ledger.get_idea(engine, prefix)
    except Exception as exc:
        logger.warning(
            "idea detail unavailable: %s: %s",
            type(exc).__name__,
            exc,
        )
        return (
            "Trade idea unavailable (ledger unreachable — is the DB up "
            "and migrations 007/008 applied?)."
        )
    if row is None:
        return f"Unknown idea id {prefix!r} — send 'ideas' to list open ideas and their ids."

    ticker = str(row.get("ticker") or "")
    fetch = get_prices_fn if get_prices_fn is not None else prices_mod.get_prices
    try:
        price_map = fetch([ticker], engine=engine)
    except Exception:
        logger.warning("idea detail: price fetch failed", exc_info=True)
        price_map = {}
    info = price_map.get(ticker) or {}
    live_price = info.get("price")
    # Recompute the card off the live price; if that misses, fall back to
    # the persisted dollar levels so the operator still sees numbers.
    card = build_trade_card(dict(row), live_price, info.get("change_pct"))

    action = str(row.get("action") or "?").replace("_", " ").upper()
    lines = [f"{ticker} — {action}"]

    # ×N events (CL-5mkf): how many distinct open geo_events proposed
    # this same (ticker, action). The card shows one consolidated idea;
    # this line names the conviction. Best-effort — a lookup failure just
    # omits the line, never breaks the card.
    with contextlib.suppress(Exception):
        raw_action = str(row.get("action") or "")
        n_events = sum(
            1
            for r in idea_ledger.list_open(engine)
            if str(r.get("ticker") or "") == ticker and str(r.get("action") or "") == raw_action
        )
        if n_events > 1:
            lines.append(f"Corroboration: ×{n_events} events proposed this")

    price_str = prices_mod.format_price(ticker, info)
    if price_str:
        lines.append(f"Price: {price_str} (live last close)")
    elif row.get("price_at_signal") is not None:
        lines.append(
            f"Price: {_fmt_dollar(row.get('price_at_signal'))} at signal (no live price now)",
        )
    else:
        lines.append("Price: no live price")

    instrument = str(row.get("preferred_instrument") or "").strip()
    if instrument:
        lines.append(f"Instrument: {instrument}")

    # Grounded levels — prefer the freshly recomputed card; fall back to
    # the persisted values (which may be from an older price).
    stop_price = card.get("stop_price")
    if stop_price is None:
        stop_price = row.get("stop_price")
    targets = card.get("target_prices") or row.get("target_prices") or []
    rr = card.get("risk_reward")
    if rr is None:
        rr = row.get("risk_reward")

    entry_zone = str(card.get("entry_zone") or "").strip()
    if entry_zone:
        lines.append(f"Entry: {entry_zone}")
    if stop_price is not None:
        # The dollar stop reflects the UNDERLYING move used by the card
        # (card['stop_loss_pct']). For an option the row's own stop_loss_pct
        # is a PREMIUM fraction, shown separately so it never reads as a
        # share-price move.
        under_pct = card.get("stop_loss_pct")
        pct_note = f" ({float(under_pct) * 100:.0f}% underlying)" if under_pct else ""
        stop_line = f"Stop: {_fmt_dollar(stop_price)}{pct_note}"
        if card.get("is_option") and row.get("stop_loss_pct"):
            stop_line += f"; exit at {float(row['stop_loss_pct']) * 100:.0f}% of premium"
        lines.append(stop_line)
    if targets:
        lines.append("Targets: " + ", ".join(_fmt_dollar(t) for t in targets))
    if rr is not None:
        lines.append(f"Risk:reward: {rr}")

    if card.get("is_option"):
        strike = card.get("suggested_strike") or row.get("suggested_strike")
        dte = str(card.get("dte_window") or row.get("dte_window") or "").strip()
        if strike is not None:
            lines.append(
                f"Suggested strike: {_fmt_dollar(strike)} — pick nearest listed strike",
            )
        if dte:
            dte_days = card.get("dte_days")
            days_note = (
                f" (choose the listed expiry nearest {dte_days} days out)" if dte_days else ""
            )
            lines.append(f"Expiry: {dte} to expiry{days_note}")

    trigger = str(row.get("entry_trigger") or "").strip()
    if trigger:
        lines.append(f"Trigger: {trigger}")
    invalidation = str(row.get("invalidation") or "").strip()
    if invalidation:
        lines.append(f"Invalidation: {invalidation}")
    rationale = str(row.get("rationale") or "").strip()
    if rationale:
        lines.append(f"Rationale: {rationale}")
    notes = str(row.get("notes") or "").strip()
    if notes:
        lines.append(f"Notes: {notes}")

    age = prices_mod.format_age(row.get("created_at"), now=now)
    time_stop = row.get("time_stop_days")
    age_parts = []
    if age:
        age_parts.append(f"{age} old")
    if time_stop is not None:
        age_parts.append(f"expires at {time_stop}d time stop")
    if age_parts:
        lines.append("Age: " + " / ".join(age_parts))

    # Robinhood execution proxy (CL-vowz): the operator can't place FX /
    # CFDs / futures on Robinhood, so map the idea's instrument to the
    # tradable ETF/stock version. The idea's ``direction`` (falling back
    # to ``action``) selects the short side (inverse ETFs / puts).
    proxy_lines = full_detail(
        ticker,
        str(row.get("direction") or row.get("action") or ""),
    )
    if proxy_lines:
        lines.append("")
        lines.extend(proxy_lines)

    lines.append("")
    lines.append(IDEA_ADVISORY_FOOTER)
    return "\n".join(lines)


def handle_text(state: LoopState, text: str) -> CommandResult:
    """Process one operator message against the loaded state. Pure
    over ``state`` — caller persists ``state`` iff ``state_changed``
    (``ideas`` reads the trade_ideas ledger, never the loop state)."""
    cmd = parse_command(text)
    if cmd is None:
        return CommandResult(
            reply=(
                "Unrecognized command. Send 'help' for the command list "
                "or 'pending' to see open approvals."
            ),
        )
    if cmd.verb == "help":
        return CommandResult(reply=HELP_TEXT)
    if cmd.verb == "pending":
        return CommandResult(reply=render_pending(state))
    if cmd.verb == "ideas":
        return CommandResult(reply=render_ideas())
    if cmd.verb == "idea":
        if not cmd.target:
            return CommandResult(
                reply=("Usage: idea <id> — send 'ideas' for the open ideas and their ids."),
            )
        return CommandResult(reply=render_idea_detail(cmd.target))
    # approve / reject / skip
    if not cmd.target:
        return CommandResult(
            reply=(f"Usage: {cmd.verb} <id> [reason] — send 'pending' for the ids awaiting you."),
        )
    target, error = resolve_target(state, cmd.target)
    if target is None:
        return CommandResult(reply=error)
    approve = cmd.verb == "approve"
    if target.gate == 1:
        result = act_gate1(state, target.key, approve=approve, reason=cmd.reason)
    else:
        result = act_gate2(state, target.key, approve=approve, reason=cmd.reason)
    if not result.ok:  # raced with another approver / the loop timeout
        return CommandResult(reply=f"⏸ Refused: {result.message}")
    return CommandResult(
        reply=_ack_reply(target, approve=approve, reason=cmd.reason),
        state_changed=True,
    )


def _ack_reply(target: GateTarget, approve: bool, reason: str) -> str:
    """One-line phone-readable acknowledgement (CL-frn7). Minimal
    emoji — a single ✅/❌ status mark, nothing else."""
    slug = target.slug
    if target.gate == 1:
        if approve:
            return f"✅ Approved {slug} — will implement next run."
        return f"❌ Skipped {slug} — {reason or 'skipped by operator'}."
    if approve:
        return f"✅ Deploy approved {slug} — paper-shadow registration next run."
    return f"❌ Deploy rejected {slug} — {reason or 'rejected by operator'}."


# --------------------------------------------------------------------- #
# Poll loop
# --------------------------------------------------------------------- #


def _chat_id_matches(actual: Any, expected: str) -> bool:
    """Strict authorization check: int-compare after str-normalize.
    Anything non-integer on either side fails closed."""
    try:
        return int(str(actual).strip()) == int(str(expected).strip())
    except (TypeError, ValueError):
        return False


@dataclass
class TelegramApprovalBot:
    """Long-polling approval bot. One ``poll_once()`` = one
    ``getUpdates`` batch: authorize, handle, reply, ack offset."""

    token: str
    chat_id: str
    state_path: Path = field(default=DEFAULT_STATE_PATH)
    offset_path: Path = field(default=DEFAULT_OFFSET_PATH)
    poll_timeout_sec: int = DEFAULT_POLL_TIMEOUT_SEC
    api_call: ApiCall | None = None

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.offset_path = Path(self.offset_path)
        if self.api_call is None:
            self.api_call = make_httpx_api(self.token)

    # -- public ------------------------------------------------------- #

    def poll_once(self, timeout_sec: int | None = None) -> int:
        """One getUpdates batch. Returns the number of updates seen
        (authorized or not). The offset is advanced past every update
        in the batch — a poison message never wedges the loop."""
        assert self.api_call is not None  # set in __post_init__
        timeout = self.poll_timeout_sec if timeout_sec is None else timeout_sec
        params: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": json.dumps(["message"]),
        }
        offset = load_offset(self.offset_path)
        if offset is not None:
            params["offset"] = offset
        payload = self.api_call("getUpdates", params)
        if not payload.get("ok", False):
            raise TelegramApiError(
                f"getUpdates returned ok=false: {payload.get('description')!r}",
            )
        updates: list[dict[str, Any]] = payload.get("result", [])
        for update in updates:
            try:
                self._process_update(update)
            except Exception as exc:
                # api_call errors are already token-scrubbed.
                logger.warning(
                    "failed to process update %s: %s: %s",
                    update.get("update_id"),
                    type(exc).__name__,
                    exc,
                )
        if updates:
            next_offset = max(int(u["update_id"]) for u in updates) + 1
            save_offset(next_offset, self.offset_path)
            logger.info(
                "processed %d update(s); offset → %d",
                len(updates),
                next_offset,
            )
        return len(updates)

    def run_forever(
        self,
        stop: threading.Event,
        error_backoff_sec: float = 5.0,
    ) -> None:
        """Poll until ``stop`` is set (SIGTERM handler sets it).
        Transient API failures back off and retry."""
        logger.info(
            "Telegram approval bot polling (long-poll %ds, state=%s)",
            self.poll_timeout_sec,
            self.state_path,
        )
        while not stop.is_set():
            try:
                self.poll_once()
            except TelegramApiError as exc:
                logger.warning(
                    "poll failed: %s — retrying in %.0fs",
                    exc,
                    error_backoff_sec,
                )
                stop.wait(error_backoff_sec)
        logger.info("Telegram approval bot stopped")

    # -- internals ---------------------------------------------------- #

    def _process_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        if not _chat_id_matches(chat.get("id"), self.chat_id):
            logger.warning(
                "ignoring update %s from unauthorized chat id=%r "
                "(only the configured TELEGRAM_CHAT_ID may issue commands)",
                update.get("update_id"),
                chat.get("id"),
            )
            return
        text = str(message.get("text") or "").strip()
        if not text:
            return
        state = load_state(self.state_path)
        result = handle_text(state, text)
        if result.state_changed:
            save_state_atomic(state, self.state_path)
            logger.info(
                "state updated via Telegram command: %r → %s",
                text,
                result.reply.splitlines()[0],
            )
        self._send_reply(result.reply)

    def _send_reply(self, reply: str) -> None:
        assert self.api_call is not None
        # Plain text, no parse_mode. Gate notifications use HTML
        # (notifications.py, CL-frn7), but bot replies echo operator
        # input and error text verbatim — plain text means none of it
        # needs escaping. One retry: live smoke
        # showed fresh-connection TLS handshakes to api.telegram.org
        # can time out transiently; a second attempt on a new
        # connection usually lands.
        for attempt in range(1 + _SEND_RETRIES):
            try:
                self.api_call(
                    "sendMessage",
                    {"chat_id": self.chat_id, "text": reply},
                )
                return
            except TelegramApiError as exc:
                if attempt >= _SEND_RETRIES:
                    raise
                logger.warning(
                    "sendMessage attempt %d failed: %s — retrying",
                    attempt + 1,
                    exc,
                )
