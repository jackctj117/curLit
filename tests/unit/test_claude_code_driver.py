"""Tests for the claude-code subscription driver.

All subprocess calls are mocked — no CLI, no network, no quota spend.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.research.llm.claude_code import ClaudeCodeDriver
from src.research.llm.client import Message, get_client

_OK_PAYLOAD = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "hello from fable",
    "total_cost_usd": 0.12,
    "usage": {
        "input_tokens": 100,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 25,
        "output_tokens": 7,
    },
    "modelUsage": {
        # haiku entry first on purpose — internal utility call; the
        # serving model is the one with the output tokens.
        "claude-haiku-4-5-20251001": {"outputTokens": 2},
        "claude-fable-5": {"outputTokens": 7},
    },
}


def _proc(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def _driver() -> ClaudeCodeDriver:
    with patch("shutil.which", return_value="/fake/claude"):
        return ClaudeCodeDriver()


def _capture_stdin(captured: list[str]) -> Callable[..., MagicMock]:
    """Read inside the call: production closes its private stream on return."""

    def run(*args: Any, **kwargs: Any) -> MagicMock:
        captured.append(kwargs["stdin"].read().decode("utf-8"))
        return _proc(json.dumps(_OK_PAYLOAD))

    return run


class TestConstruction:
    def test_no_api_key_required(self) -> None:
        drv = _driver()
        assert drv.api_key == ""
        assert ClaudeCodeDriver.requires_api_key is False

    def test_workdir_removed_when_driver_is_collected(self) -> None:
        """Each driver mkdtemp's a neutral cwd and nothing ever removed it.
        Long-lived daemons rebuild agents every cycle (now x N under the
        parallel per-thread clients), so these accumulated in /tmp forever."""
        import gc
        from pathlib import Path

        drv = _driver()
        workdir = Path(drv._workdir)
        assert workdir.is_dir()

        del drv
        gc.collect()
        assert not workdir.exists()

    def test_workdir_cleanup_survives_already_deleted_dir(self) -> None:
        # Cleanup must never raise if something else removed the dir first.
        import gc
        import shutil as _shutil
        from pathlib import Path

        drv = _driver()
        workdir = Path(drv._workdir)
        _shutil.rmtree(workdir)  # pre-remove
        del drv
        gc.collect()  # ignore_errors=True → no exception
        assert not workdir.exists()

    def test_missing_cli_raises(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(ValueError, match="CLI not found"),
        ):
            ClaudeCodeDriver()

    def test_get_client_skips_key_check(self) -> None:
        # No ANTHROPIC_* key plumbing — construction succeeds keyless.
        with patch("shutil.which", return_value="/fake/claude"):
            client = get_client("claude-code")
        assert client.provider == "claude-code"


class TestComplete:
    def test_real_child_reads_exact_concurrent_preloaded_prompts(self) -> None:
        """A real child reads fd 0; no Claude process, provider or quota used."""
        import subprocess
        import sys
        from concurrent.futures import ThreadPoolExecutor

        actual_run = subprocess.run
        child = (
            "import json,os,stat,sys; "
            "assert stat.S_ISREG(os.fstat(0).st_mode); "
            "print(json.dumps({'subtype':'success','result':sys.stdin.buffer.read().decode('utf-8')}))"
        )

        def local_child(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return actual_run([sys.executable, "-c", child], **kwargs)

        driver = _driver()
        prompts = [f"prompt {i}: café 🧪\n" * 10000 for i in range(8)]
        with patch("subprocess.run", side_effect=local_child), ThreadPoolExecutor(4) as pool:
            responses = list(
                pool.map(
                    lambda prompt: (
                        driver.complete([Message("user", prompt)], model="local-test").text
                    ),
                    prompts,
                )
            )
        assert responses == prompts

    def test_prompt_is_preloaded_private_file_and_closed_on_timeout(self) -> None:
        import subprocess

        prompt = "Unicode evidence café 🧪\n" * 10000
        captured: list[Any] = []

        def timeout(*args: Any, **kwargs: Any) -> None:
            assert "input" not in kwargs
            stream = kwargs["stdin"]
            captured.append(stream)
            info = os.fstat(stream.fileno())
            assert stat.S_ISREG(info.st_mode)
            assert stat.S_IMODE(info.st_mode) == 0o600
            assert stream.tell() == 0
            assert stream.read() == prompt.encode("utf-8")
            raise subprocess.TimeoutExpired(args[0], 1)

        with patch("subprocess.run", side_effect=timeout), pytest.raises(subprocess.TimeoutExpired):
            _driver().complete([Message("user", prompt)], model="test-model")
        assert len(captured) == 1 and captured[0].closed

    def test_success_parses_payload(self) -> None:
        drv = _driver()
        captured: list[str] = []
        with patch("subprocess.run", side_effect=_capture_stdin(captured)) as run:
            resp = drv.complete(
                [Message("system", "persona"), Message("user", "hi")],
                model="claude-fable-5",
            )
        assert resp.text == "hello from fable"
        assert resp.model == "claude-fable-5"
        assert resp.input_tokens == 175  # input + cache_creation + cache_read
        assert resp.output_tokens == 7
        # CL-h7c1 intentionally changed this expectation (was == 0.0): a
        # subscription call has no metered USD cost, and 0.0 read as a
        # MEASURED zero in spend totals. Unknown is None + an explicit
        # provenance; the CLI's figure is kept only as a labelled estimate.
        assert resp.usd_cost is None
        assert resp.cost_provenance == "subscription_unmetered"
        assert resp.nominal_usd_cost == pytest.approx(0.12)
        assert resp.requested_model == "claude-fable-5"
        cmd = run.call_args.args[0]
        # Prompt rides STDIN now (CL-8s2a), not argv: `-p` is bare and the
        # prompt is passed through a preloaded stdin file (CL-ep4q).
        assert cmd[:2] == ["/fake/claude", "-p"]
        assert captured == ["hi"]
        assert run.call_args.kwargs["stdin"].closed
        assert "hi" not in cmd  # never on argv (ps / ARG_MAX safe)
        assert "--system-prompt" in cmd and "persona" in cmd
        assert "--model" in cmd and "claude-fable-5" in cmd

    def test_nonzero_exit_surfaces_error_fields_not_usage_tail(self) -> None:
        # CL-hm2k: with empty stderr and a JSON payload on stdout, the raised
        # message must carry subtype / api_error_status / the HEAD of result —
        # the raw tail is the end of the usage block and diagnoses nothing.
        payload = json.dumps(
            {
                "subtype": "error_during_execution",
                "api_error_status": 429,
                "result": "Rate limited: usage window exhausted." + " pad" * 200,
                "usage": {"input_tokens": 9},
            }
        )
        drv = _driver()
        with (
            patch("subprocess.run", return_value=_proc(payload, returncode=1)),
            pytest.raises(RuntimeError) as exc,
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5")
        msg = str(exc.value)
        assert "subtype='error_during_execution'" in msg
        assert "api_error_status=429" in msg
        assert "Rate limited: usage window exhausted." in msg

    def test_nonzero_exit_prefers_stderr(self) -> None:
        drv = _driver()
        with (
            patch(
                "subprocess.run",
                return_value=_proc("{}", returncode=1, stderr="boom: real reason"),
            ),
            pytest.raises(RuntimeError, match="boom: real reason"),
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5")

    def test_no_tools_strips_the_builtin_toolset(self) -> None:
        # CL-u5cq: single-shot JSON callers pass no_tools=True; the CLI gets
        # --tools "" so the headless session cannot research the prompt into
        # a context-window overflow (the 2026-08-02 stuck-NEW mode).
        drv = _driver()
        with patch("subprocess.run", return_value=_proc(json.dumps(_OK_PAYLOAD))) as run:
            drv.complete(
                [Message("user", "hi")],
                model="claude-fable-5",
                no_tools=True,
            )
        cmd = run.call_args.args[0]
        i = cmd.index("--tools")
        assert cmd[i + 1] == ""

    def test_tools_available_by_default(self) -> None:
        drv = _driver()
        with patch("subprocess.run", return_value=_proc(json.dumps(_OK_PAYLOAD))) as run:
            drv.complete([Message("user", "hi")], model="claude-fable-5")
        assert "--tools" not in run.call_args.args[0]

    def test_api_credentials_stripped_from_subprocess_env(self) -> None:
        drv = _driver()
        with (
            patch("subprocess.run", return_value=_proc(json.dumps(_OK_PAYLOAD))) as run,
            patch.dict(
                "os.environ", {"ANTHROPIC_API_KEY": "sk-live", "ANTHROPIC_AUTH_TOKEN": "tok"}
            ),
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5")
        env = run.call_args.kwargs["env"]
        # The env key would OUTRANK the subscription login inside the
        # CLI — it must be truly absent, not empty.
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env

    def test_multi_turn_flattened_with_role_labels(self) -> None:
        drv = _driver()
        captured: list[str] = []
        with patch("subprocess.run", side_effect=_capture_stdin(captured)):
            drv.complete(
                [Message("user", "a"), Message("assistant", "b"), Message("user", "c")],
                model="claude-fable-5",
            )
        # Flattened transcript now arrives via stdin (CL-8s2a).
        prompt = captured[0]
        assert "[USER]\na" in prompt and "[ASSISTANT]\nb" in prompt

    def test_prompt_goes_to_stdin_not_argv(self) -> None:
        """CL-8s2a: the (possibly multi-KB) prompt must be passed via stdin,
        never on argv — argv is visible in `ps` and bounded by ARG_MAX."""
        drv = _driver()
        secret_long_prompt = "SENSITIVE-" + "x" * 5000
        captured: list[str] = []
        with patch(
            "subprocess.run",
            side_effect=_capture_stdin(captured),
        ) as run:
            drv.complete(
                [Message("system", "persona"), Message("user", secret_long_prompt)],
                model="claude-fable-5",
            )
        argv = run.call_args.args[0]
        # The prompt is nowhere on the command line...
        assert not any(secret_long_prompt in str(a) for a in argv)
        assert "-p" in argv and argv[argv.index("-p") + 1].startswith("--")
        # ...and is delivered as stdin instead.
        assert captured == [secret_long_prompt]

    def test_nonzero_exit_raises(self) -> None:
        drv = _driver()
        with (
            patch("subprocess.run", return_value=_proc("", 1, "boom")),
            pytest.raises(RuntimeError, match="exited 1"),
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5")

    def test_error_subtype_raises(self) -> None:
        drv = _driver()
        bad = dict(_OK_PAYLOAD, is_error=True, subtype="error_during_execution")
        with (
            patch("subprocess.run", return_value=_proc(json.dumps(bad))),
            pytest.raises(RuntimeError, match="error_during_execution"),
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5")

    def test_non_json_output_raises(self) -> None:
        drv = _driver()
        with (
            patch("subprocess.run", return_value=_proc("not json")),
            pytest.raises(RuntimeError, match="non-JSON"),
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5")


def _run_with(payload: dict[str, Any], **complete_kwargs: Any) -> tuple[Any, MagicMock]:
    """Run one complete() against a canned CLI payload; return (resp, run mock)."""
    drv = _driver()
    kwargs: dict[str, Any] = {"model": "claude-fable-5", **complete_kwargs}
    with patch("subprocess.run", return_value=_proc(json.dumps(payload))) as run:
        resp = drv.complete([Message("user", "hi")], **kwargs)
    return resp, run


class TestBudgetControls:
    """CL-h7c1: requested caps reach the CLI, or their non-enforcement is
    explicit. Oracle: the installed CLI (2.1.287) has no --max-tokens /
    --temperature flag; it reads CLAUDE_CODE_MAX_OUTPUT_TOKENS from the env.

    These run with the rollout gate ENABLED (fixture below); the gate-off
    default is covered by TestCapRolloutGate."""

    @pytest.fixture(autouse=True)
    def _enforce(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURLIT_CLAUDE_ENFORCE_OUTPUT_CAP", "1")

    def test_max_tokens_reaches_cli_as_output_cap_env(self) -> None:
        _, run = _run_with(_OK_PAYLOAD, max_tokens=1234)
        assert run.call_args.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "1234"

    def test_requested_cap_overrides_ambient_env(self) -> None:
        # An inherited operator setting must not silently replace the cap
        # the caller asked for.
        with patch.dict("os.environ", {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"}):
            _, run = _run_with(_OK_PAYLOAD, max_tokens=900)
        assert run.call_args.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "900"

    def test_default_max_tokens_is_forwarded_too(self) -> None:
        _, run = _run_with(_OK_PAYLOAD)
        assert run.call_args.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"

    @pytest.mark.parametrize("bad", [0, -5, True, 12.5])
    def test_invalid_cap_refused_before_spawn(self, bad: Any) -> None:
        # The CLI ignores a non-positive/non-int value and runs UNCAPPED;
        # refusing is the only way not to drop the cap silently.
        drv = _driver()
        with (
            patch("subprocess.run") as run,
            pytest.raises(ValueError, match="max_tokens"),
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5", max_tokens=bad)
        run.assert_not_called()

    def test_temperature_reported_unenforced_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("WARNING", logger="src.research.llm.claude_code")
        resp, run = _run_with(_OK_PAYLOAD, temperature=0.7)
        assert resp.unenforced_params == ("temperature",)
        cmd = run.call_args.args[0]
        assert not any("temperature" in str(a) for a in cmd)  # nothing pretends to apply it
        assert any("cannot enforce temperature" in r.getMessage() for r in caplog.records)

    def test_unrecognised_kwargs_reported_not_dropped(self) -> None:
        resp, _ = _run_with(_OK_PAYLOAD, top_p=0.9, stop=["x"])
        assert resp.unenforced_params == ("temperature", "stop", "top_p")


class TestUsageProvenance:
    """CL-h7c1: missing usage is UNKNOWN (None), never coerced to 0."""

    def test_missing_usage_block_is_unknown(self) -> None:
        payload = {k: v for k, v in _OK_PAYLOAD.items() if k != "usage"}
        resp, _ = _run_with(payload)
        assert resp.input_tokens is None
        assert resp.output_tokens is None
        assert resp.total_tokens is None

    def test_missing_output_tokens_is_unknown(self) -> None:
        usage = {k: v for k, v in _OK_PAYLOAD["usage"].items() if k != "output_tokens"}
        resp, _ = _run_with(dict(_OK_PAYLOAD, usage=usage))
        assert resp.input_tokens == 175
        assert resp.output_tokens is None

    def test_missing_cache_component_makes_input_unknown(self) -> None:
        usage = {k: v for k, v in _OK_PAYLOAD["usage"].items() if k != "cache_read_input_tokens"}
        resp, _ = _run_with(dict(_OK_PAYLOAD, usage=usage))
        assert resp.input_tokens is None
        assert resp.output_tokens == 7

    @pytest.mark.parametrize("bad", ["100", None, -1, 1.5, True])
    def test_malformed_count_is_unknown(self, bad: Any) -> None:
        usage = dict(_OK_PAYLOAD["usage"], input_tokens=bad)
        resp, _ = _run_with(dict(_OK_PAYLOAD, usage=usage))
        assert resp.input_tokens is None

    def test_measured_zero_output_stays_zero(self) -> None:
        # Distinctness cuts both ways: a REPORTED 0 is a measurement.
        usage = dict(_OK_PAYLOAD["usage"], output_tokens=0)
        resp, _ = _run_with(dict(_OK_PAYLOAD, usage=usage))
        assert resp.output_tokens == 0

    def test_cost_is_never_a_measured_zero(self) -> None:
        payload = {k: v for k, v in _OK_PAYLOAD.items() if k != "total_cost_usd"}
        resp, _ = _run_with(payload)
        assert resp.usd_cost is None
        assert resp.nominal_usd_cost is None  # absent estimate is unknown, not $0
        assert resp.cost_provenance == "subscription_unmetered"


class TestServingModelAttribution:
    """CL-h7c1: modelUsage lists Claude Code's internal utility calls; only an
    entry matching the REQUESTED model proves who served."""

    def test_utility_model_never_reported_as_server(self) -> None:
        # Reproduces the prior smoke: requested sonnet, modelUsage shows only
        # the Haiku utility call (6688 in / 11 out). Old code said "haiku".
        payload = dict(
            _OK_PAYLOAD,
            modelUsage={"claude-haiku-4-5-20251001": {"inputTokens": 6688, "outputTokens": 11}},
        )
        resp, _ = _run_with(payload, model="claude-sonnet-4-6")
        assert resp.model == "unverified"
        assert resp.requested_model == "claude-sonnet-4-6"

    def test_requested_model_verified_even_with_fewer_output_tokens(self) -> None:
        payload = dict(
            _OK_PAYLOAD,
            modelUsage={
                "claude-haiku-4-5-20251001": {"outputTokens": 11},
                "claude-sonnet-4-6": {"outputTokens": 5},
            },
        )
        resp, _ = _run_with(payload, model="claude-sonnet-4-6")
        assert resp.model == "claude-sonnet-4-6"

    def test_dated_snapshot_and_variant_tag_match_request(self) -> None:
        dated = dict(_OK_PAYLOAD, modelUsage={"claude-haiku-4-5-20251001": {"outputTokens": 3}})
        resp, _ = _run_with(dated, model="claude-haiku-4-5")
        assert resp.model == "claude-haiku-4-5-20251001"
        tagged = dict(_OK_PAYLOAD, modelUsage={"claude-fable-5[1m]": {"outputTokens": 3}})
        resp, _ = _run_with(tagged, model="claude-fable-5")
        assert resp.model == "claude-fable-5[1m]"

    def test_prefix_of_another_model_does_not_match(self) -> None:
        payload = dict(_OK_PAYLOAD, modelUsage={"claude-opus-4-8": {"outputTokens": 9}})
        resp, _ = _run_with(payload, model="claude-opus-4")
        assert resp.model == "unverified"

    def test_requested_entry_without_output_is_unverified(self) -> None:
        payload = dict(
            _OK_PAYLOAD,
            modelUsage={
                "claude-fable-5": {"outputTokens": 0},
                "claude-opus-4-8": {"outputTokens": 40},
            },
        )
        resp, _ = _run_with(payload)
        assert resp.model == "unverified"

    def test_missing_model_usage_is_unverified_not_the_request(self) -> None:
        payload = {k: v for k, v in _OK_PAYLOAD.items() if k != "modelUsage"}
        resp, _ = _run_with(payload)
        assert resp.model == "unverified"
        assert resp.requested_model == "claude-fable-5"


class TestCapRolloutGate:
    """CL-h7c1 round-4 review: enforcement is gated OFF by default until caller
    budgets are calibrated. Off = pre-CL-h7c1 env behavior, but the cap is
    REPORTED unenforced rather than silently dropped."""

    def test_gate_off_by_default_leaves_env_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CURLIT_CLAUDE_ENFORCE_OUTPUT_CAP", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)
        resp, run = _run_with(_OK_PAYLOAD, max_tokens=900)
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in run.call_args.kwargs["env"]
        assert "max_tokens" in resp.unenforced_params

    def test_gate_off_preserves_ambient_operator_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CURLIT_CLAUDE_ENFORCE_OUTPUT_CAP", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        _, run = _run_with(_OK_PAYLOAD, max_tokens=900)
        assert run.call_args.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"

    def test_gate_on_enforces_and_does_not_report_unenforced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CURLIT_CLAUDE_ENFORCE_OUTPUT_CAP", "1")
        resp, run = _run_with(_OK_PAYLOAD, max_tokens=900)
        assert run.call_args.kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "900"
        assert "max_tokens" not in resp.unenforced_params
