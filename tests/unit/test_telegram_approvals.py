"""Unit tests for the Telegram approval bot (CL-b1l6).

All HTTP is faked via the injectable ApiCall shim. State assertions
reload the temp state.json and compare against the loop's status
constants — and, for the equivalence tests, against what
scripts/research_approve.py produces on an identical copy.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.research_approve import main as approve_main

from src.research.approvals import save_state_atomic
from src.research.loop import (
    GATE1_APPROVED_STATUS,
    GATE1_PENDING_STATUS,
    GATE1_SKIPPED_STATUS,
    GATE2_APPROVED_STATUS,
    GATE2_PENDING_STATUS,
    GATE2_REJECTED_STATUS,
    LoopState,
    load_state,
)
from src.research.telegram_approvals import (
    CommandResult,
    ParsedCommand,
    TelegramApiError,
    TelegramApprovalBot,
    handle_text,
    load_offset,
    make_httpx_api,
    parse_command,
    render_pending,
    resolve_target,
    save_offset,
)

CHAT_ID = "123456789"
FOREIGN_CHAT_ID = 987654321
TOKEN = "1111111111:AAAAfake-token-must-never-leak"

HASH_A = "a1b2c3d4e5f6a7b8"  # short id a1b2c3
HASH_B = "a1b2ffee00112233"  # collides with HASH_A at prefix a1b2
HASH_C = "ff00112233445566"


def _gate1_entry(slug: str, status: str = GATE1_PENDING_STATUS) -> dict[str, Any]:
    return {
        "status": status,
        "slug": slug,
        "reason": "",
        "pending_since": "2026-07-10T00:00:00+00:00",
        "hypothesis_path": f"docs/research/hypotheses/{slug}.md",
    }


def _gate2_entry(status: str = GATE2_PENDING_STATUS) -> dict[str, Any]:
    return {
        "verdict": "PROMOTE",
        "reason": "meets all rules",
        "transcript_path": "docs/research/debates/x/transcript.md",
        "bull": "PROMOTE",
        "bear": "ABSTAIN",
        "deploy_status": status,
        "pending_since": "2026-07-11T00:00:00+00:00",
        "candidate_report_path": "reports/candidates/x.json",
    }


def _make_state(
    ideas: dict[str, dict[str, Any]] | None = None,
    debates: dict[str, dict[str, Any]] | None = None,
) -> LoopState:
    return LoopState(
        ideas_processed=dict(ideas or {}),
        debates_completed=dict(debates or {}),
    )


def _seed(path: Path, state: LoopState) -> None:
    save_state_atomic(state, path)


def _update(update_id: int, text: str, chat_id: Any = CHAT_ID) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "text": text},
    }


class FakeApi:
    """Recording ApiCall: getUpdates pops pre-loaded batches,
    sendMessage records the outgoing reply."""

    def __init__(self, batches: list[list[dict[str, Any]]] | None = None):
        self.batches = list(batches or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, dict(params)))
        if method == "getUpdates":
            batch = self.batches.pop(0) if self.batches else []
            return {"ok": True, "result": batch}
        return {"ok": True, "result": {}}

    @property
    def sent_messages(self) -> list[str]:
        return [
            str(p["text"]) for m, p in self.calls if m == "sendMessage"
        ]


def _make_bot(
    tmp_path: Path,
    batches: list[list[dict[str, Any]]] | None = None,
    state: LoopState | None = None,
) -> tuple[TelegramApprovalBot, FakeApi, Path]:
    state_path = tmp_path / "state.json"
    _seed(state_path, state or _make_state())
    api = FakeApi(batches)
    bot = TelegramApprovalBot(
        token=TOKEN,
        chat_id=CHAT_ID,
        state_path=state_path,
        offset_path=tmp_path / "telegram_offset.json",
        api_call=api,
    )
    return bot, api, state_path


# --------------------------------------------------------------------- #
# Command parsing
# --------------------------------------------------------------------- #


class TestParseCommand:
    def test_simple_approve(self) -> None:
        assert parse_command("approve a1b2c3") == ParsedCommand(
            verb="approve", target="a1b2c3",
        )

    def test_case_insensitive_and_slash(self) -> None:
        assert parse_command("  /APPROVE A1B2C3  ") == ParsedCommand(
            verb="approve", target="a1b2c3",
        )

    def test_reason_captured(self) -> None:
        cmd = parse_command("skip a1b2c3 duplicates existing strategy")
        assert cmd == ParsedCommand(
            verb="skip", target="a1b2c3",
            reason="duplicates existing strategy",
        )

    def test_bare_verbs(self) -> None:
        assert parse_command("pending") == ParsedCommand(verb="pending")
        assert parse_command("HELP") == ParsedCommand(verb="help")
        assert parse_command("/start") == ParsedCommand(verb="help")

    def test_unknown_returns_none(self) -> None:
        assert parse_command("what is going on") is None
        assert parse_command("") is None


# --------------------------------------------------------------------- #
# Chat-id security
# --------------------------------------------------------------------- #


class TestChatIdFiltering:
    def test_foreign_chat_ignored(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        bot, api, state_path = _make_bot(
            tmp_path,
            batches=[[_update(1, "approve a1b2c3", chat_id=FOREIGN_CHAT_ID)]],
            state=state,
        )
        with caplog.at_level(logging.WARNING):
            n = bot.poll_once(timeout_sec=0)
        assert n == 1
        # No state mutation, no reply to the intruder.
        reloaded = load_state(state_path)
        assert reloaded.ideas_processed[HASH_A]["status"] == GATE1_PENDING_STATUS
        assert api.sent_messages == []
        assert any("unauthorized" in r.message for r in caplog.records)
        # Offset still advances past the foreign update.
        assert load_offset(tmp_path / "telegram_offset.json") == 2

    def test_int_str_normalization(self, tmp_path: Path) -> None:
        # chat.id arrives as int from Telegram; env var is str — match.
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        bot, api, state_path = _make_bot(
            tmp_path,
            batches=[[_update(1, "approve a1b2c3", chat_id=int(CHAT_ID))]],
            state=state,
        )
        bot.poll_once(timeout_sec=0)
        reloaded = load_state(state_path)
        assert reloaded.ideas_processed[HASH_A]["status"] == GATE1_APPROVED_STATUS

    def test_missing_chat_ignored(self, tmp_path: Path) -> None:
        bot, api, _ = _make_bot(
            tmp_path, batches=[[{"update_id": 5, "message": {"text": "pending"}}]],
        )
        assert bot.poll_once(timeout_sec=0) == 1
        assert api.sent_messages == []


# --------------------------------------------------------------------- #
# State mutations (statuses must match the loop's constants)
# --------------------------------------------------------------------- #


class TestMutations:
    def test_approve_gate1(self, tmp_path: Path) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        bot, api, state_path = _make_bot(
            tmp_path, batches=[[_update(1, "approve a1b2c3")]], state=state,
        )
        bot.poll_once(timeout_sec=0)
        reloaded = load_state(state_path)
        assert reloaded.ideas_processed[HASH_A]["status"] == GATE1_APPROVED_STATUS
        # One-line phone-readable ack (CL-frn7)
        assert any(
            m == "✅ Approved alpha — will implement next run."
            for m in api.sent_messages
        )

    def test_reject_gate1_is_skip(self, tmp_path: Path) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        bot, _, state_path = _make_bot(
            tmp_path, batches=[[_update(1, "reject a1b2c3 not novel")]],
            state=state,
        )
        bot.poll_once(timeout_sec=0)
        entry = load_state(state_path).ideas_processed[HASH_A]
        assert entry["status"] == GATE1_SKIPPED_STATUS
        assert entry["reason"] == "not novel"

    def test_skip_gate1_default_reason(self, tmp_path: Path) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        bot, _, state_path = _make_bot(
            tmp_path, batches=[[_update(1, "skip a1b2c3")]], state=state,
        )
        bot.poll_once(timeout_sec=0)
        entry = load_state(state_path).ideas_processed[HASH_A]
        assert entry["status"] == GATE1_SKIPPED_STATUS
        assert entry["reason"] == "skipped by operator"

    def test_approve_gate2_by_slug(self, tmp_path: Path) -> None:
        state = _make_state(debates={"vol-carry": _gate2_entry()})
        bot, api, state_path = _make_bot(
            tmp_path, batches=[[_update(1, "approve vol-carry")]], state=state,
        )
        bot.poll_once(timeout_sec=0)
        entry = load_state(state_path).debates_completed["vol-carry"]
        assert entry["deploy_status"] == GATE2_APPROVED_STATUS
        assert any(
            m.startswith("✅ Deploy approved vol-carry")
            for m in api.sent_messages
        )

    def test_reject_gate2(self, tmp_path: Path) -> None:
        state = _make_state(debates={"vol-carry": _gate2_entry()})
        bot, _, state_path = _make_bot(
            tmp_path, batches=[[_update(1, "reject vol-carry too risky")]],
            state=state,
        )
        bot.poll_once(timeout_sec=0)
        entry = load_state(state_path).debates_completed["vol-carry"]
        assert entry["deploy_status"] == GATE2_REJECTED_STATUS
        assert entry["deploy_reason"] == "too risky"

    def test_gate2_slug_prefix(self, tmp_path: Path) -> None:
        state = _make_state(debates={"vol-carry": _gate2_entry()})
        bot, _, state_path = _make_bot(
            tmp_path, batches=[[_update(1, "approve vol-c")]], state=state,
        )
        bot.poll_once(timeout_sec=0)
        entry = load_state(state_path).debates_completed["vol-carry"]
        assert entry["deploy_status"] == GATE2_APPROVED_STATUS


class TestCliEquivalence:
    """Telegram mutations must produce byte-identical state.json to
    scripts/research_approve.py acting on the same input."""

    def _run_cli(self, argv: list[str]) -> int:
        out, err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            return approve_main(argv)
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def test_gate1_approve_matches_cli(self, tmp_path: Path) -> None:
        seed = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        tg_path = tmp_path / "tg" / "state.json"
        cli_path = tmp_path / "cli" / "state.json"
        _seed(tg_path, seed)
        _seed(cli_path, _make_state(ideas={HASH_A: _gate1_entry("alpha")}))

        bot = TelegramApprovalBot(
            token=TOKEN, chat_id=CHAT_ID, state_path=tg_path,
            offset_path=tmp_path / "off.json",
            api_call=FakeApi([[_update(1, "approve a1b2c3")]]),
        )
        bot.poll_once(timeout_sec=0)
        rc = self._run_cli([
            "--state", str(cli_path), "--slug", "alpha", "--action", "GO",
        ])
        assert rc == 0
        assert tg_path.read_text() == cli_path.read_text()

    def test_gate2_reject_matches_cli(self, tmp_path: Path) -> None:
        tg_path = tmp_path / "tg" / "state.json"
        cli_path = tmp_path / "cli" / "state.json"
        _seed(tg_path, _make_state(debates={"vol-carry": _gate2_entry()}))
        _seed(cli_path, _make_state(debates={"vol-carry": _gate2_entry()}))

        bot = TelegramApprovalBot(
            token=TOKEN, chat_id=CHAT_ID, state_path=tg_path,
            offset_path=tmp_path / "off.json",
            api_call=FakeApi([[_update(1, "reject vol-carry too risky")]]),
        )
        bot.poll_once(timeout_sec=0)
        rc = self._run_cli([
            "--state", str(cli_path), "--gate=2", "--slug", "vol-carry",
            "--action", "SKIP", "--reason", "too risky",
        ])
        assert rc == 0
        assert tg_path.read_text() == cli_path.read_text()


# --------------------------------------------------------------------- #
# Prefix resolution
# --------------------------------------------------------------------- #


class TestResolution:
    def test_unambiguous_prefix(self) -> None:
        state = _make_state(ideas={
            HASH_A: _gate1_entry("alpha"), HASH_C: _gate1_entry("gamma"),
        })
        target, err = resolve_target(state, "a1")
        assert err == ""
        assert target is not None
        assert target.gate == 1
        assert target.key == HASH_A

    def test_collision_lists_candidates(self, tmp_path: Path) -> None:
        state = _make_state(ideas={
            HASH_A: _gate1_entry("alpha"), HASH_B: _gate1_entry("beta"),
        })
        target, err = resolve_target(state, "a1b2")
        assert target is None
        assert "Ambiguous" in err
        assert HASH_A[:6] in err
        assert HASH_B[:6] in err
        # And end-to-end: state stays untouched.
        state_path = tmp_path / "state.json"
        _seed(state_path, state)
        bot = TelegramApprovalBot(
            token=TOKEN, chat_id=CHAT_ID, state_path=state_path,
            offset_path=tmp_path / "off.json",
            api_call=FakeApi([[_update(1, "approve a1b2")]]),
        )
        bot.poll_once(timeout_sec=0)
        reloaded = load_state(state_path)
        assert all(
            e["status"] == GATE1_PENDING_STATUS
            for e in reloaded.ideas_processed.values()
        )

    def test_gate1_exact_slug_match(self) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        target, err = resolve_target(state, "alpha")
        assert err == ""
        assert target is not None and target.key == HASH_A

    def test_already_decided(self) -> None:
        state = _make_state(
            ideas={HASH_A: _gate1_entry("alpha", status=GATE1_APPROVED_STATUS)},
        )
        target, err = resolve_target(state, "a1b2c3")
        assert target is None
        assert "Already decided" in err
        assert GATE1_APPROVED_STATUS in err

    def test_unknown_id(self) -> None:
        target, err = resolve_target(_make_state(), "deadbeef")
        assert target is None
        assert "Unknown id" in err
        assert "pending" in err


# --------------------------------------------------------------------- #
# handle_text: pending / help / malformed
# --------------------------------------------------------------------- #


class TestHandleText:
    def test_pending_listing(self) -> None:
        state = _make_state(
            ideas={
                HASH_A: _gate1_entry("alpha"),
                HASH_C: _gate1_entry("gamma", status="DECLINED"),
            },
            debates={"vol-carry": _gate2_entry()},
        )
        result = handle_text(state, "pending")
        assert not result.state_changed
        assert HASH_A[:6] in result.reply       # gate1 short id
        assert "alpha" in result.reply
        assert HASH_C[:6] not in result.reply   # non-pending hidden
        assert "vol-carry" in result.reply      # gate2 slug
        assert "approve <id>" in result.reply   # reply instructions

    def test_pending_empty(self) -> None:
        result = handle_text(_make_state(), "pending")
        assert "No gate approvals pending" in result.reply

    def test_help(self) -> None:
        result = handle_text(_make_state(), "help")
        assert "approve <id>" in result.reply
        assert "pending" in result.reply

    def test_malformed_missing_id(self) -> None:
        result = handle_text(_make_state(), "approve")
        assert not result.state_changed
        assert "Usage" in result.reply

    def test_unrecognized_chatter(self) -> None:
        result = handle_text(_make_state(), "hello bot how are you")
        assert not result.state_changed
        assert "help" in result.reply

    def test_render_pending_matches_handle(self) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        assert handle_text(state, "pending") == CommandResult(
            reply=render_pending(state), state_changed=False,
        )


# --------------------------------------------------------------------- #
# Phone-readable replies + instrument display (CL-frn7)
# --------------------------------------------------------------------- #


CANDIDATE_CODE = """\
class Strategy:
    symbols = ["DXY", "EURUSD"]
    execution_symbol = "EURUSD"

    def fit(self, data):
        return self

    def generate_signals(self, data):
        return None
"""

BRIEF_TEXT = """\
# Hypothesis: test thesis

## Data requirements
- `prices.{symbol}` — daily close for EURUSD and GBPUSD
- Macro input: FRED `DGS10`
"""


class TestAckReplies:
    """Ack replies are one-liners with a single status mark."""

    def test_reject_gate1_one_liner(self, tmp_path: Path) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        bot, api, _ = _make_bot(
            tmp_path, batches=[[_update(1, "reject a1b2c3 not novel")]],
            state=state,
        )
        bot.poll_once(timeout_sec=0)
        assert api.sent_messages == ["❌ Skipped alpha — not novel."]

    def test_reject_gate2_one_liner_default_reason(
        self, tmp_path: Path,
    ) -> None:
        state = _make_state(debates={"vol-carry": _gate2_entry()})
        bot, api, _ = _make_bot(
            tmp_path, batches=[[_update(1, "reject vol-carry")]], state=state,
        )
        bot.poll_once(timeout_sec=0)
        assert api.sent_messages == [
            "❌ Deploy rejected vol-carry — rejected by operator.",
        ]

    def test_refusal_marked(self) -> None:
        state = _make_state(
            ideas={HASH_A: _gate1_entry("alpha", status=GATE1_APPROVED_STATUS)},
        )
        # Force the race path: resolve by exact slug of a decided entry
        # is impossible (not pending), so drive act_gate1 directly via
        # a state where the entry flips between resolve and act.
        result = handle_text(state, "approve a1b2c3")
        assert not result.state_changed
        assert "Already decided" in result.reply or result.reply.startswith("⏸")


class TestPendingFormat:
    def test_numbered_two_lines_per_entry_with_trades(
        self, tmp_path: Path,
    ) -> None:
        brief_path = tmp_path / "brief.md"
        brief_path.write_text(BRIEF_TEXT)
        code_path = tmp_path / "cand.py"
        code_path.write_text(CANDIDATE_CODE)

        ideas = {HASH_A: _gate1_entry("alpha")}
        ideas[HASH_A]["hypothesis_path"] = str(brief_path)
        debates = {"vol-carry": _gate2_entry()}
        state = _make_state(ideas=ideas, debates=debates)
        state.candidates_processed["vol-carry"] = {
            "status": "IMPLEMENTED",
            "code_path": str(code_path),
        }

        reply = render_pending(state)
        lines = reply.splitlines()
        assert lines[0] == "Pending approvals:"
        # Gate 1 entry: id + slug then its brief's tradable instruments
        assert f"1. {HASH_A[:6]} — alpha (gate 1)" in lines
        i1 = lines.index(f"1. {HASH_A[:6]} — alpha (gate 1)")
        assert lines[i1 + 1] == "   Trades: EURUSD, GBPUSD"
        # Gate 2 entry: slug then the candidate code's instruments
        # (execution symbol first — backtest_runner's canonical read)
        assert "2. vol-carry (gate 2)" in lines
        i2 = lines.index("2. vol-carry (gate 2)")
        assert lines[i2 + 1] == "   Trades: EURUSD, DXY"
        assert lines[-1] == "Reply: approve <id> | reject <id> | skip <id>"

    def test_unknown_instruments_shown_explicitly(self) -> None:
        # Brief/code paths that don't exist → fail-safe "(unknown)",
        # never an exception inside the bot's reply path.
        state = _make_state(
            ideas={HASH_A: _gate1_entry("alpha")},
            debates={"vol-carry": _gate2_entry()},
        )
        reply = render_pending(state)
        assert reply.count("   Trades: (unknown)") == 2


# --------------------------------------------------------------------- #
# Offset persistence
# --------------------------------------------------------------------- #


class TestOffset:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "telegram_offset.json"
        assert load_offset(p) is None
        save_offset(42, p)
        assert load_offset(p) == 42
        assert json.loads(p.read_text()) == {"offset": 42}

    def test_malformed_treated_as_fresh(self, tmp_path: Path) -> None:
        p = tmp_path / "telegram_offset.json"
        p.write_text("{not json")
        assert load_offset(p) is None

    def test_poll_advances_and_reuses_offset(self, tmp_path: Path) -> None:
        bot, api, _ = _make_bot(
            tmp_path,
            batches=[
                [_update(7, "help"), _update(8, "help")],
                [],
            ],
        )
        bot.poll_once(timeout_sec=0)
        assert load_offset(tmp_path / "telegram_offset.json") == 9
        bot.poll_once(timeout_sec=0)
        get_updates_calls = [
            p for m, p in api.calls if m == "getUpdates"
        ]
        assert "offset" not in get_updates_calls[0]  # fresh start
        assert get_updates_calls[1]["offset"] == 9   # ack'd

    def test_no_updates_no_offset_write(self, tmp_path: Path) -> None:
        bot, _, _ = _make_bot(tmp_path, batches=[[]])
        assert bot.poll_once(timeout_sec=0) == 0
        assert not (tmp_path / "telegram_offset.json").exists()


# --------------------------------------------------------------------- #
# Token safety
# --------------------------------------------------------------------- #


class TestTokenSafety:
    def test_httpx_api_scrubs_token(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import httpx

        def boom(url: str, **kwargs: Any) -> Any:
            raise httpx.ConnectError(
                f"connect failed for https://api.telegram.org/bot{TOKEN}/getUpdates",
            )

        monkeypatch.setattr(httpx, "post", boom)
        api = make_httpx_api(TOKEN)
        with pytest.raises(TelegramApiError) as exc_info:
            api("getUpdates", {"timeout": 0})
        assert TOKEN not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value)
        # Chain suppressed so tracebacks can't resurface the raw URL.
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__

    def test_no_token_in_logs_or_replies(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        bot, api, _ = _make_bot(
            tmp_path,
            batches=[[
                _update(1, "approve a1b2c3"),
                _update(2, "pending", chat_id=FOREIGN_CHAT_ID),
            ]],
            state=state,
        )
        with caplog.at_level(logging.DEBUG):
            bot.poll_once(timeout_sec=0)
        for record in caplog.records:
            assert TOKEN not in record.getMessage()
        for sent in api.sent_messages:
            assert TOKEN not in sent

    def test_send_failure_logged_scrubbed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        class SendBoomApi(FakeApi):
            def __call__(
                self, method: str, params: dict[str, Any],
            ) -> dict[str, Any]:
                if method == "sendMessage":
                    raise TelegramApiError(
                        "sendMessage failed: HTTPStatusError: 400 for "
                        "url https://api.telegram.org/bot[REDACTED]/sendMessage",
                    )
                return super().__call__(method, params)

        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        state_path = tmp_path / "state.json"
        _seed(state_path, state)
        bot = TelegramApprovalBot(
            token=TOKEN, chat_id=CHAT_ID, state_path=state_path,
            offset_path=tmp_path / "off.json",
            api_call=SendBoomApi([[_update(3, "approve a1b2c3")]]),
        )
        with caplog.at_level(logging.WARNING):
            n = bot.poll_once(timeout_sec=0)
        assert n == 1
        # Mutation still landed; reply failure is logged, offset advances.
        assert (
            load_state(state_path).ideas_processed[HASH_A]["status"]
            == GATE1_APPROVED_STATUS
        )
        assert load_offset(tmp_path / "off.json") == 4
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "failed to process update" in joined
        assert TOKEN not in joined

    def test_send_retries_once_then_succeeds(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        class FlakySendApi(FakeApi):
            def __init__(self, batches: list[list[dict[str, Any]]]):
                super().__init__(batches)
                self.send_attempts = 0

            def __call__(
                self, method: str, params: dict[str, Any],
            ) -> dict[str, Any]:
                if method == "sendMessage":
                    self.send_attempts += 1
                    if self.send_attempts == 1:
                        raise TelegramApiError(
                            "sendMessage failed: ConnectTimeout: "
                            "handshake timed out",
                        )
                return super().__call__(method, params)

        state = _make_state(ideas={HASH_A: _gate1_entry("alpha")})
        state_path = tmp_path / "state.json"
        _seed(state_path, state)
        api = FlakySendApi([[_update(3, "approve a1b2c3")]])
        bot = TelegramApprovalBot(
            token=TOKEN, chat_id=CHAT_ID, state_path=state_path,
            offset_path=tmp_path / "off.json", api_call=api,
        )
        with caplog.at_level(logging.WARNING):
            bot.poll_once(timeout_sec=0)
        assert api.send_attempts == 2
        assert any("✅ Approved" in m for m in api.sent_messages)
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "retrying" in joined
        assert TOKEN not in joined


class TestMobileKeyboardMangling:
    """Phone keyboards capitalize, smart-quote, and append punctuation —
    every form must parse identically to the clean lowercase command
    (operator report 2026-07-20: replies must never be case-sensitive)."""

    def test_verb_case_variants(self) -> None:
        for text in ("PENDING", "Pending", "pEnDiNg", "/Pending"):
            cmd = parse_command(text)
            assert cmd is not None and cmd.verb == "pending", text

    def test_action_verb_and_id_uppercased(self) -> None:
        for text in ("Approve ABC123", "APPROVE abc123", "approve AbC123"):
            cmd = parse_command(text)
            assert cmd is not None, text
            assert cmd.verb == "approve" and cmd.target == "abc123", text

    def test_trailing_punctuation_trimmed(self) -> None:
        for text in ("Approve abc123.", "approve abc123,", "approve abc123!",
                     "Approve \u201cabc123\u201d", "approve 'abc123'"):
            cmd = parse_command(text)
            assert cmd is not None, text
            assert cmd.target == "abc123", text

    def test_hyphenated_slug_interior_untouched(self) -> None:
        cmd = parse_command("Reject My-Slug-Name.")
        assert cmd is not None
        assert cmd.target == "my-slug-name"

    def test_uppercase_id_resolves_against_lowercase_hash(self) -> None:
        h = "a1b2c3d4e5f6" + "0" * 52
        state = _make_state(ideas={h: _gate1_entry("some-slug")})
        cmd = parse_command("Approve A1B2C3.")
        assert cmd is not None
        target, err = resolve_target(state, cmd.target)
        assert err == "" and target is not None
        assert target.key.startswith("a1b2c3")


# --------------------------------------------------------------------- #
# 'ideas' command (CL-mgcp) — read-only trade-idea listing
# --------------------------------------------------------------------- #


def _ledger_engine() -> Any:
    """sqlite engine with the REAL migration-007 + 008 schema (types
    shimmed), mirroring tests/unit/test_idea_ledger.py. 008 adds the
    concrete trade-card level columns the ledger now writes (CL-jiqq)."""
    import sqlalchemy as sa
    from migrations.run import _strip_sql_comments
    from sqlalchemy import text as sql_text

    from tests.unit.test_idea_ledger import (
        MIGRATION,
        MIGRATION_LEVELS,
        _shim_pg_types_for_sqlite,
        _sqlite_statements,
    )

    engine = sa.create_engine("sqlite://")
    for mig in (MIGRATION, MIGRATION_LEVELS):
        sql = _shim_pg_types_for_sqlite(_strip_sql_comments(mig.read_text()))
        with engine.begin() as conn:
            for stmt in _sqlite_statements(sql):
                conn.execute(sql_text(stmt))
    return engine


class TestIdeasCommand:
    def _seed(self, engine: Any) -> None:
        from datetime import UTC, datetime, timedelta

        from src.events.idea_ledger import persist_ideas

        now = datetime.now(UTC)
        persist_ideas(engine, 1, {"trade_ideas": [{
            "ticker": "TSM", "action": "buy_puts", "direction": "bearish",
            "confidence": 0.7, "rationale": "r", "time_horizon": "short",
            "holding_period_days": "2-6", "time_stop_days": 5,
        }]}, now=now - timedelta(days=2))
        persist_ideas(engine, 2, {"trade_ideas": [{
            "ticker": "RTX", "action": "long", "direction": "bullish",
            "confidence": 0.5, "rationale": "r", "time_horizon": "medium",
            "holding_period_days": "10-20", "time_stop_days": 20,
        }]}, now=now - timedelta(hours=3))

    def test_lists_open_ideas_with_age_stop_price(self) -> None:
        from src.research.telegram_approvals import render_ideas

        engine = _ledger_engine()
        self._seed(engine)

        def fake_prices(tickers: Any, engine: Any = None) -> dict[str, Any]:
            assert sorted(tickers) == ["RTX", "TSM"]
            return {"TSM": {"price": 172.4, "change_pct": -1.8}}

        reply = render_ideas(engine=engine, get_prices_fn=fake_prices)
        assert "Open trade ideas (2):" in reply
        # Newest first; numbered; bracketed short id; imperative action;
        # age vs stop; price where resolvable (CL-jiqq).
        assert "] RTX LONG — 3h old / stop 20d" in reply
        assert "] TSM BUY PUTS — 2d old / stop 5d — $172.40 (-1.8%)" in reply
        assert reply.index("RTX") < reply.index("TSM")  # newest first
        assert "Send 'idea <id>' for the full trade card." in reply
        assert "Read-only" in reply

    def test_empty_ledger(self) -> None:
        from src.research.telegram_approvals import render_ideas

        reply = render_ideas(
            engine=_ledger_engine(), get_prices_fn=lambda *a, **k: {},
        )
        assert reply == "No open trade ideas."

    def test_table_missing_graceful(self) -> None:
        import sqlalchemy as sa

        from src.research.telegram_approvals import render_ideas

        reply = render_ideas(
            engine=sa.create_engine("sqlite://"),
            get_prices_fn=lambda *a, **k: {},
        )
        assert "Trade ideas unavailable" in reply

    def test_price_failure_renders_without_prices(self) -> None:
        from src.research.telegram_approvals import render_ideas

        engine = _ledger_engine()
        self._seed(engine)

        def boom(*_a: Any, **_k: Any) -> dict[str, Any]:
            raise RuntimeError("yahoo down")

        reply = render_ideas(engine=engine, get_prices_fn=boom)
        assert "] RTX LONG — 3h old / stop 20d" in reply
        assert "$" not in reply

    def test_handle_text_dispatches_ideas(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "src.research.telegram_approvals.render_ideas",
            lambda: "IDEAS-SENTINEL",
        )
        result = handle_text(_make_state(), "ideas")
        assert result == CommandResult(reply="IDEAS-SENTINEL", state_changed=False)

    def test_parse_command_recognises_ideas(self) -> None:
        assert parse_command("ideas") == ParsedCommand(verb="ideas")
        assert parse_command("/Ideas") == ParsedCommand(verb="ideas")

    def test_help_mentions_ideas(self) -> None:
        assert "ideas" in handle_text(_make_state(), "help").reply


class TestIdeaDetailCommand:
    """CL-jiqq — the `idea <id>` full trade card."""

    def _seed_one(self, engine: Any) -> str:
        from datetime import UTC, datetime

        from src.events.idea_ledger import make_idea_id, persist_ideas

        persist_ideas(engine, 1, {"trade_ideas": [{
            "ticker": "TSM", "action": "buy_puts", "direction": "bearish",
            "confidence": 0.7, "rationale": "advanced-node concentration",
            "time_horizon": "short", "holding_period_days": "2-6",
            "time_stop_days": 5, "stop_loss_pct": 0.40,
            "target_pct": [0.10, 0.18],
            "entry_trigger": "on confirmed blockade language",
            "invalidation": "official denial of the strike",
        }]}, prices={"TSM": {"price": 172.4, "change_pct": -1.8}},
            now=datetime(2026, 7, 20, 12, tzinfo=UTC))
        return make_idea_id(1, "TSM", "buy_puts")

    def test_full_card_found(self) -> None:
        from src.research.telegram_approvals import (
            IDEA_ADVISORY_FOOTER,
            render_idea_detail,
        )

        engine = _ledger_engine()
        idea_id = self._seed_one(engine)

        def fake_prices(tickers: Any, engine: Any = None) -> dict[str, Any]:
            return {"TSM": {"price": 172.4, "change_pct": -1.8}}

        reply = render_idea_detail(
            idea_id[:6], engine=engine, get_prices_fn=fake_prices,
        )
        assert reply.startswith("TSM — BUY PUTS")
        assert "$172.40" in reply
        assert "Stop:" in reply            # dollar stop present
        assert "Targets:" in reply
        assert "Risk:reward:" in reply
        # option guidance with the pick-nearest caveat
        assert "Suggested strike:" in reply
        assert "pick nearest listed strike" in reply
        assert "to expiry" in reply
        assert "Trigger: on confirmed blockade language" in reply
        assert "Invalidation: official denial of the strike" in reply
        assert "Rationale: advanced-node concentration" in reply
        assert "Age:" in reply
        # bold advisory footer
        assert IDEA_ADVISORY_FOOTER in reply

    def test_unknown_id(self) -> None:
        from src.research.telegram_approvals import render_idea_detail

        engine = _ledger_engine()
        self._seed_one(engine)
        reply = render_idea_detail(
            "zzzzzz", engine=engine, get_prices_fn=lambda *a, **k: {},
        )
        assert "Unknown idea id" in reply

    def test_no_live_price_falls_back_to_signal(self) -> None:
        from src.research.telegram_approvals import (
            IDEA_ADVISORY_FOOTER,
            render_idea_detail,
        )

        engine = _ledger_engine()
        idea_id = self._seed_one(engine)
        # No live price now → uses persisted price_at_signal / levels.
        reply = render_idea_detail(
            idea_id[:6], engine=engine, get_prices_fn=lambda *a, **k: {},
        )
        assert "at signal" in reply
        assert IDEA_ADVISORY_FOOTER in reply

    def test_ledger_unreachable_graceful(self) -> None:
        import sqlalchemy as sa

        from src.research.telegram_approvals import render_idea_detail

        reply = render_idea_detail(
            "abc", engine=sa.create_engine("sqlite://"),
            get_prices_fn=lambda *a, **k: {},
        )
        assert "unavailable" in reply.lower()

    def test_handle_text_dispatches_idea_detail(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake(idea_id: str) -> str:
            captured["id"] = idea_id
            return "DETAIL-SENTINEL"

        monkeypatch.setattr(
            "src.research.telegram_approvals.render_idea_detail", fake,
        )
        result = handle_text(_make_state(), "idea a1b2c3")
        assert result == CommandResult(reply="DETAIL-SENTINEL", state_changed=False)
        assert captured["id"] == "a1b2c3"

    def test_idea_without_id_gives_usage(self) -> None:
        result = handle_text(_make_state(), "idea")
        assert "Usage: idea <id>" in result.reply

    def test_parse_command_distinguishes_idea_from_ideas(self) -> None:
        assert parse_command("idea a1b2c3") == ParsedCommand(
            verb="idea", target="a1b2c3",
        )
        assert parse_command("ideas") == ParsedCommand(verb="ideas")

    def test_help_mentions_idea_detail(self) -> None:
        assert "idea <id>" in handle_text(_make_state(), "help").reply
