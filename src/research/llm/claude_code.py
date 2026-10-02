"""Claude-via-subscription driver — routes Claude calls through the
local Claude Code CLI in headless mode (``claude -p``) so they run on
the operator's Claude subscription login instead of API billing.

Why a CLI subprocess instead of the Anthropic SDK: direct Messages-API
calls can only bill to an API organization — a Claude Pro/Max
subscription cannot pay for raw API traffic. The sanctioned way to run
programmatic work on a subscription is Claude Code's headless mode,
which authenticates with the same login as interactive sessions.

Operational notes:

  * **Auth precedence trap.** An ``ANTHROPIC_API_KEY`` (or
    ``ANTHROPIC_AUTH_TOKEN``) in the environment OUTRANKS the
    subscription login inside the CLI — and ``dotenv_bootstrap`` puts
    the API key in this process's env. The driver strips both from the
    subprocess environment so calls are guaranteed to ride the
    subscription, never the API org.
  * **Quota sharing.** Subscription usage limits (5-hour / weekly
    windows) are shared with interactive Claude Code sessions. A heavy
    pipeline run can throttle the operator's own sessions.
  * **Context hygiene.** ``--system-prompt`` replaces Claude Code's
    default agent prompt with the role's persona, and the subprocess
    runs from a neutral temp cwd so project CLAUDE.md / settings are
    not loaded into every role call (~19K tokens saved per call).
Budget controls and provenance (CL-h7c1) — verified against the
installed CLI (Claude Code 2.1.287: ``claude --help`` plus the binary's
env-var handling), not assumed:

  * ``max_tokens`` IS enforced. The CLI has no ``--max-tokens`` flag but
    honours the ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` environment variable as
    the per-request output cap (thinking included); a value above the
    model's own ceiling is clamped DOWN to that ceiling, never raised.
    The driver sets it from ``max_tokens`` on every call. Per the binary,
    hitting the cap yields an API-error result ("exceeded the N output
    token maximum"); an ``is_error`` payload raises here (fail loud). Not
    yet observed live — no real calls are made in development.
  * ``temperature`` is NOT enforceable — the CLI exposes no sampling
    control. It is reported in ``LLMResponse.unenforced_params`` (and any
    other unrecognised kwarg alongside it) and logged, never dropped
    silently.
  * ``usd_cost`` is None with ``cost_provenance="subscription_unmetered"``:
    nothing is invoiced per call to the API org, but subscription usage is
    not a measured $0 either. The CLI's ``total_cost_usd`` is an
    API-EQUIVALENT ESTIMATE kept in ``nominal_usd_cost`` — never summed as
    spend.
  * Tokens come from the payload's ``usage`` block (input + cache
    creation + cache read; output). A missing / malformed field makes the
    count None (unknown), never 0.
  * ``model`` is the serving model only when ``modelUsage`` proves it: an
    entry matching the REQUESTED model (exact, or with a ``-YYYYMMDD`` /
    ``[...]`` suffix) that produced output tokens. Otherwise it is
    ``"unverified"`` — ``modelUsage`` also lists Claude Code's internal
    utility calls (e.g. Haiku), so "most output tokens" is not proof of
    who answered. The request is kept in ``requested_model``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import weakref
from typing import Any

from src.research.llm.client import (
    COST_SUBSCRIPTION_UNMETERED,
    UNVERIFIED_MODEL,
    Driver,
    LLMResponse,
    Message,
    register_driver,
)

logger = logging.getLogger(__name__)

# Generous per-call ceiling — Fable-class turns on hard prompts can run
# minutes; the pipeline's role calls are single-shot text generation.
_CALL_TIMEOUT_SEC = 900

#: Env var the claude CLI reads as its per-request output-token cap (CL-h7c1;
#: confirmed in the installed 2.1.287 binary — clamped to the model ceiling).
MAX_OUTPUT_TOKENS_ENV = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"

#: Parameters the CLI cannot enforce at all — reported, never silently dropped.
_UNENFORCEABLE_PARAMS: tuple[str, ...] = ("temperature",)

#: Cache components of ``usage`` that count toward input load.
_CACHE_INPUT_FIELDS: tuple[str, ...] = (
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

#: modelUsage key suffixes that denote the same model: a dated snapshot
#: (``claude-haiku-4-5`` → ``claude-haiku-4-5-20251001``) and/or a context
#: variant tag (``claude-fable-5[1m]``).
_DATED_SUFFIX = r"(?:-\d{8})?"
_VARIANT_TAG = re.compile(r"\[[^\]]*\]$")


def _nonneg_int(value: object) -> int | None:
    """A token count, or None when absent / malformed (bool is not a count)."""
    if type(value) is int and value >= 0:
        return value
    return None


def _usage_tokens(usage: object) -> tuple[int | None, int | None]:
    """(input incl. cache, output) from the CLI ``usage`` block.

    Any missing or malformed component makes that side None (unknown) —
    never coerced to 0 (CL-h7c1).
    """
    if not isinstance(usage, dict):
        return None, None
    in_tok = _nonneg_int(usage.get("input_tokens"))
    for key in _CACHE_INPUT_FIELDS:
        part = _nonneg_int(usage.get(key))
        in_tok = None if in_tok is None or part is None else in_tok + part
    return in_tok, _nonneg_int(usage.get("output_tokens"))


def _matches_requested(key: str, requested: str) -> bool:
    base = _VARIANT_TAG.sub("", key)
    want = _VARIANT_TAG.sub("", requested)
    return re.fullmatch(re.escape(want) + _DATED_SUFFIX, base) is not None


def _resolve_served_model(model_usage: object, requested: str) -> str:
    """The modelUsage key proven to have served ``requested``, else
    ``UNVERIFIED_MODEL``. Proof = an entry matching the request that
    produced output tokens; a different model is never promoted to
    "served" by output-token volume (the CL-h7c1 Haiku mislabel)."""
    if not isinstance(model_usage, dict):
        return UNVERIFIED_MODEL
    matches = [
        key
        for key, entry in model_usage.items()
        if isinstance(key, str)
        and _matches_requested(key, requested)
        and isinstance(entry, dict)
        and (_nonneg_int(entry.get("outputTokens")) or 0) > 0
    ]
    if len(matches) == 1:
        return matches[0]
    return UNVERIFIED_MODEL


def _nominal_cost(value: object) -> float | None:
    """The CLI's API-equivalent estimate, or None when absent/malformed."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


class ClaudeCodeDriver(Driver):
    """Runs Claude models through ``claude -p`` (headless Claude Code)
    on the operator's subscription login. No API key required."""

    name = "claude-code"
    requires_api_key = False

    def __init__(self, api_key: str = "") -> None:
        # Deliberately skip Driver.__init__'s non-empty key check —
        # auth is the CLI's stored subscription login.
        self.api_key = ""
        bin_path = shutil.which("claude")
        if not bin_path:
            msg = (
                "claude-code driver: `claude` CLI not found on PATH — "
                "install Claude Code or switch the provider back to "
                "`claude` (API billing)"
            )
            raise ValueError(msg)
        self._bin: str = bin_path
        # One WARNING per driver for the unenforceable params; every response
        # still carries them in ``unenforced_params`` (CL-h7c1).
        self._warned_unenforced = False
        # Neutral cwd so the CLI doesn't ingest this repo's CLAUDE.md /
        # settings into every role call.
        self._workdir = tempfile.mkdtemp(prefix="curlit-claude-code-")
        # Reap the workdir when this driver is collected. Long-lived daemons
        # rebuild their agents every cycle (event_pipeline: one per 900s tick,
        # now ×N under the parallel per-thread clients), and nothing else ever
        # removed these dirs — they accumulated in /tmp indefinitely.
        # weakref.finalize (not __del__) so it also runs at interpreter exit.
        self._cleanup = weakref.finalize(
            self,
            shutil.rmtree,
            self._workdir,
            ignore_errors=True,
        )

    def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        no_tools: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        # The cap is forwarded to the CLI below; a non-positive / non-int
        # value would be ignored by the CLI (falls back to its default),
        # i.e. silently uncapped — refuse it instead.
        if type(max_tokens) is not int or max_tokens <= 0:
            msg = f"claude-code: max_tokens must be a positive int, got {max_tokens!r}"
            raise ValueError(msg)
        unenforced = _UNENFORCEABLE_PARAMS + tuple(sorted(kwargs))
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        # The CLI takes one prompt string; flatten multi-turn
        # transcripts with role labels (single [system, user] calls —
        # the common case — pass through unlabelled).
        chat = [m for m in messages if m.role in ("user", "assistant")]
        if len(chat) == 1:
            prompt = chat[0].content
        else:
            prompt = "\n\n".join(f"[{m.role.upper()}]\n{m.content}" for m in chat)

        # Prompt goes over STDIN, not argv (CL-8s2a). ``claude -p`` with no
        # trailing prompt argument reads the prompt from stdin (the documented
        # pipe usage), so a bare ``-p`` here + ``input=prompt`` below is
        # equivalent to the old ``-p <prompt>`` — but the (often multi-KB)
        # prompt is no longer visible in ``ps``/process listings and can't hit
        # the OS ARG_MAX ceiling on long research prompts. Flags/behavior are
        # otherwise unchanged.
        cmd = [
            self._bin,
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--exclude-dynamic-system-prompt-sections",
        ]
        # no_tools (CL-u5cq): single-shot text→JSON callers must strip the
        # built-in toolset. With tools available, the headless session may
        # RESEARCH the prompt (web fetches on hot news events) until the
        # context window bursts and the CLI exits 1 — the 2026-08-02 mode
        # where the weekend's biggest events (Iran/Hormuz, OPEC) each blew
        # past 200k tokens and stuck NEW forever, while routine ones passed.
        if no_tools:
            cmd += ["--tools", ""]
        if system_text:
            cmd += ["--system-prompt", system_text]

        # Strip API credentials so the CLI cannot silently bill the
        # API org — an env API key outranks the subscription login.
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        }
        # CL-h7c1: enforce the requested output cap. Overrides any inherited
        # value so the caller's max_tokens — not ambient env — is the cap.
        env[MAX_OUTPUT_TOKENS_ENV] = str(max_tokens)
        if not self._warned_unenforced:
            self._warned_unenforced = True
            logger.warning(
                "claude-code: the claude CLI cannot enforce %s — requested "
                "values are reported in LLMResponse.unenforced_params, not applied",
                ", ".join(unenforced),
            )

        t0 = time.time()
        # CL-ep4q: pre-load stdin before spawn. A live PIPE can miss the
        # CLI's prompt-arrival deadline under contention. TemporaryFile is
        # private (0600), unlinked where supported, and closed on every exit.
        # Per-call files also prevent concurrent prompts overwriting each other.
        with tempfile.TemporaryFile(mode="w+b", dir=self._workdir) as prompt_file:
            prompt_file.write(prompt.encode("utf-8"))
            prompt_file.flush()
            prompt_file.seek(0)
            logger.info(
                "claude-code: starting model=%s max_output_tokens=%d (enforced via %s) "
                "temperature=%s (NOT enforceable) unenforced=%s with preloaded stdin",
                model,
                max_tokens,
                MAX_OUTPUT_TOKENS_ENV,
                temperature,
                list(unenforced),
            )
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=prompt_file,
                timeout=_CALL_TIMEOUT_SEC,
                env=env,
                cwd=self._workdir,
                check=False,
            )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            # Prefer stderr; else mine the JSON payload for the ERROR fields
            # (CL-hm2k) — the raw tail is the end of the usage block, which
            # made the Aug 4-6 1080-failure outage undiagnosable from logs.
            detail = (proc.stderr or "").strip()
            if not detail:
                raw = (proc.stdout or "").strip()
                try:
                    err = json.loads(raw)
                    detail = (
                        f"subtype={err.get('subtype')!r} "
                        f"api_error_status={err.get('api_error_status')!r} "
                        f"result={str(err.get('result'))[:300]}"
                    )
                except json.JSONDecodeError:
                    detail = raw[:500]  # head — the informative end
            msg = f"claude -p exited {proc.returncode}: {detail}"
            raise RuntimeError(msg)
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            msg = f"claude -p returned non-JSON output: {proc.stdout[:300]}"
            raise RuntimeError(msg) from exc
        if payload.get("is_error") or payload.get("subtype") != "success":
            msg = (
                f"claude -p error (subtype={payload.get('subtype')!r}): "
                f"{str(payload.get('result'))[:500]}"
            )
            raise RuntimeError(msg)

        # Count cached prefix tokens too — they're real context the call
        # consumed. Missing/malformed usage → None (unknown), never 0.
        in_tok, out_tok = _usage_tokens(payload.get("usage"))
        if in_tok is None or out_tok is None:
            logger.warning(
                "claude-code[%s]: usage not fully reported (in=%s out=%s) — "
                "recorded as UNKNOWN, not zero",
                model,
                in_tok,
                out_tok,
            )
        nominal = _nominal_cost(payload.get("total_cost_usd"))
        served = _resolve_served_model(payload.get("modelUsage"), model)
        if served == UNVERIFIED_MODEL:
            model_usage = payload.get("modelUsage")
            logger.warning(
                "claude-code: serving model UNVERIFIED for requested=%s — "
                "modelUsage keys %s do not prove it",
                model,
                sorted(model_usage) if isinstance(model_usage, dict) else None,
            )
        logger.info(
            "claude-code: done requested=%s served=%s in=%s out=%s "
            "nominal_api_equiv=%s (ESTIMATE; subscription, usd_cost unmetered) "
            "elapsed=%.2fs",
            model,
            served,
            in_tok,
            out_tok,
            "unknown" if nominal is None else f"${nominal:.4f}",
            elapsed,
        )
        return LLMResponse(
            text=str(payload.get("result", "")),
            model=served,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            # Subscription: no per-call USD is metered — unknown, not $0.
            usd_cost=None,
            elapsed_sec=elapsed,
            raw=payload,
            requested_model=model,
            cost_provenance=COST_SUBSCRIPTION_UNMETERED,
            nominal_usd_cost=nominal,
            unenforced_params=unenforced,
        )


# Self-register on import (same pattern as grok.py).
register_driver("claude-code", ClaudeCodeDriver)
