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
  * ``max_tokens`` / ``temperature`` are accepted for interface
    compatibility but not forwarded — the CLI does not expose them.
  * ``usd_cost`` is reported as 0.0 (nothing bills to the API org);
    the CLI's nominal API-equivalent cost is logged at DEBUG for
    visibility.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import weakref
from typing import Any

from src.research.llm.client import (
    Driver,
    LLMResponse,
    Message,
    register_driver,
)

logger = logging.getLogger(__name__)

# Generous per-call ceiling — Fable-class turns on hard prompts can run
# minutes; the pipeline's role calls are single-shot text generation.
_CALL_TIMEOUT_SEC = 900


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

        t0 = time.time()
        # CL-ep4q: pre-load stdin before spawn. A live PIPE can miss the
        # CLI's prompt-arrival deadline under contention. TemporaryFile is
        # private (0600), unlinked where supported, and closed on every exit.
        # Per-call files also prevent concurrent prompts overwriting each other.
        with tempfile.TemporaryFile(mode="w+b", dir=self._workdir) as prompt_file:
            prompt_file.write(prompt.encode("utf-8"))
            prompt_file.flush()
            prompt_file.seek(0)
            logger.info("claude-code: starting model=%s with preloaded stdin", model)
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

        usage = payload.get("usage") or {}
        # Count cached prefix tokens too — they're real context the
        # call consumed, and the spend summary should reflect load.
        in_tok = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )
        out_tok = int(usage.get("output_tokens", 0))
        nominal = float(payload.get("total_cost_usd") or 0.0)
        logger.debug(
            "claude-code[%s] nominal API-equivalent cost $%.4f "
            "(billed to subscription, not the API org)",
            model,
            nominal,
        )
        # modelUsage can include Claude Code's internal utility calls
        # (haiku) alongside the responder — the serving model is the
        # one that produced the output tokens.
        model_usage = payload.get("modelUsage") or {}
        served = max(
            model_usage,
            key=lambda m: model_usage[m].get("outputTokens", 0),
            default=model,
        )
        return LLMResponse(
            text=str(payload.get("result", "")),
            model=served,
            provider=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            usd_cost=0.0,  # subscription — nothing bills to the API org
            elapsed_sec=elapsed,
            raw=payload,
        )


# Self-register on import (same pattern as grok.py).
register_driver("claude-code", ClaudeCodeDriver)
