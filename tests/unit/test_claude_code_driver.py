"""Tests for the claude-code subscription driver.

All subprocess calls are mocked — no CLI, no network, no quota spend.
"""

from __future__ import annotations

import json
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
    def test_success_parses_payload(self) -> None:
        drv = _driver()
        with patch("subprocess.run", return_value=_proc(json.dumps(_OK_PAYLOAD))) as run:
            resp = drv.complete(
                [Message("system", "persona"), Message("user", "hi")],
                model="claude-fable-5",
            )
        assert resp.text == "hello from fable"
        assert resp.model == "claude-fable-5"
        assert resp.input_tokens == 175  # input + cache_creation + cache_read
        assert resp.output_tokens == 7
        assert resp.usd_cost == 0.0  # subscription — never bills the API org
        cmd = run.call_args.args[0]
        # Prompt rides STDIN now (CL-8s2a), not argv: `-p` is bare and the
        # prompt is passed as subprocess input.
        assert cmd[:2] == ["/fake/claude", "-p"]
        assert run.call_args.kwargs["input"] == "hi"
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
        with patch("subprocess.run", return_value=_proc(json.dumps(_OK_PAYLOAD))) as run:
            drv.complete(
                [Message("user", "a"), Message("assistant", "b"), Message("user", "c")],
                model="claude-fable-5",
            )
        # Flattened transcript now arrives via stdin (CL-8s2a).
        prompt = run.call_args.kwargs["input"]
        assert "[USER]\na" in prompt and "[ASSISTANT]\nb" in prompt

    def test_prompt_goes_to_stdin_not_argv(self) -> None:
        """CL-8s2a: the (possibly multi-KB) prompt must be passed via stdin,
        never on argv — argv is visible in `ps` and bounded by ARG_MAX."""
        drv = _driver()
        secret_long_prompt = "SENSITIVE-" + "x" * 5000
        with patch(
            "subprocess.run",
            return_value=_proc(json.dumps(_OK_PAYLOAD)),
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
        assert run.call_args.kwargs["input"] == secret_long_prompt

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
