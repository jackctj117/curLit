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
    help                # command grammar

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
_KNOWN_VERBS = _ACTION_VERBS | {"pending", "help", "start"}

HELP_TEXT = (
    "curLit research approval bot — commands:\n"
    "  approve <id> [reason]  approve a pending gate entry\n"
    "  reject <id> [reason]   decline (gate 1: SKIP, gate 2: reject deploy)\n"
    "  skip <id> [reason]     same as reject\n"
    "  pending                list entries awaiting your decision\n"
    "  help                   this message\n"
    "<id> is the short id from the gate notification or the 'pending' "
    "listing (gate 1: extract-hash prefix, gate 2: strategy slug)."
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
            p, type(exc).__name__,
        )
        return None


def save_offset(offset: int, path: Path | str = DEFAULT_OFFSET_PATH) -> None:
    """Atomically persist the next getUpdates offset (tmp + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=p.parent, prefix=f".{p.name}.", suffix=".tmp",
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
    target = (
        tokens[1].lower().strip(_TOKEN_TRIM_CHARS) if len(tokens) > 1 else ""
    )
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
        GateTarget(gate=1, key=h, slug=str(e.get("slug") or h))
        for h, e in gate1_pending(state)
    ]
    targets += [
        GateTarget(gate=2, key=slug, slug=slug)
        for slug, e in gate2_pending(state)
    ]
    return targets


def resolve_target(
    state: LoopState, token: str,
) -> tuple[GateTarget | None, str]:
    """Match a short id against pending entries.

    Gate 1 matches on extract-hash prefix or exact slug; gate 2 on
    slug prefix. Returns ``(target, "")`` on a unique match, else
    ``(None, error_reply)`` — ambiguous prefixes list the candidates,
    non-pending matches report their decided status.
    """
    token_l = token.lower()
    matches = [
        t for t in _pending_targets(state)
        if t.key.lower().startswith(token_l) or t.slug.lower() == token_l
    ]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        lines = [
            f"  {t.short_id}  {t.slug} (gate {t.gate})" for t in matches
        ]
        return None, (
            f"Ambiguous id {token!r} — matches {len(matches)} pending "
            f"entries:\n" + "\n".join(lines)
            + "\nReply with a longer prefix."
        )
    decided = _decided_matches(state, token_l)
    if decided:
        return None, (
            "Already decided — no pending entry matches "
            f"{token!r}:\n" + "\n".join(f"  {d}" for d in decided)
        )
    return None, (
        f"Unknown id {token!r} — nothing pending matches. "
        f"Send 'pending' to list open approvals."
    )


def _decided_matches(state: LoopState, token_l: str) -> list[str]:
    """Non-pending entries matching the token, with their statuses —
    powers the 'already decided' error reply."""
    out: list[str] = []
    for h, e in state.ideas_processed.items():
        slug = str(e.get("slug") or "")
        if h.lower().startswith(token_l) or slug.lower() == token_l:
            out.append(
                f"{h[:SHORT_ID_LEN]}  {slug or h} (gate 1) "
                f"status={e.get('status')}"
            )
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
        instruments = (
            extract_candidate_instruments(code_path) if code_path else []
        )
        lines.append(_entry_trades_line(instruments))
    lines.append("")
    lines.append("Reply: approve <id> | reject <id> | skip <id>")
    return "\n".join(lines)


def handle_text(state: LoopState, text: str) -> CommandResult:
    """Process one operator message against the loaded state. Pure —
    caller persists ``state`` iff ``state_changed``."""
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
    # approve / reject / skip
    if not cmd.target:
        return CommandResult(
            reply=(
                f"Usage: {cmd.verb} <id> [reason] — send 'pending' for "
                f"the ids awaiting you."
            ),
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
        return (
            f"✅ Deploy approved {slug} — paper-shadow registration "
            f"next run."
        )
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
        timeout = (
            self.poll_timeout_sec if timeout_sec is None else timeout_sec
        )
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
                f"getUpdates returned ok=false: "
                f"{payload.get('description')!r}",
            )
        updates: list[dict[str, Any]] = payload.get("result", [])
        for update in updates:
            try:
                self._process_update(update)
            except Exception as exc:
                # api_call errors are already token-scrubbed.
                logger.warning(
                    "failed to process update %s: %s: %s",
                    update.get("update_id"), type(exc).__name__, exc,
                )
        if updates:
            next_offset = max(int(u["update_id"]) for u in updates) + 1
            save_offset(next_offset, self.offset_path)
            logger.info(
                "processed %d update(s); offset → %d",
                len(updates), next_offset,
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
            self.poll_timeout_sec, self.state_path,
        )
        while not stop.is_set():
            try:
                self.poll_once()
            except TelegramApiError as exc:
                logger.warning(
                    "poll failed: %s — retrying in %.0fs",
                    exc, error_backoff_sec,
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
                update.get("update_id"), chat.get("id"),
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
                text, result.reply.splitlines()[0],
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
                    "sendMessage", {"chat_id": self.chat_id, "text": reply},
                )
                return
            except TelegramApiError as exc:
                if attempt >= _SEND_RETRIES:
                    raise
                logger.warning(
                    "sendMessage attempt %d failed: %s — retrying",
                    attempt + 1, exc,
                )
