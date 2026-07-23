"""Unit tests for the operator approval CLI (CL-0hr3).

Drives the script via its main(argv) entry. State file is created
under tmp_path; assertions read state.json directly to verify the
right transition happened.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from scripts.research_approve import main as approve_main

from src.research.loop import (
    LoopState,
    load_state,
    save_state,
)


def _seed_state(path: Path, entries: dict[str, dict[str, str]]) -> None:
    s = LoopState(ideas_processed=dict(entries))
    save_state(s, path)


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI and capture stdout/stderr."""
    out, err = io.StringIO(), io.StringIO()
    sys.stdout, old_out = out, sys.stdout
    sys.stderr, old_err = err, sys.stderr
    try:
        rc = approve_main(argv)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


class TestList:
    def test_list_empty_state(self, state_path: Path) -> None:
        save_state(LoopState(), state_path)
        rc, out, _ = _run(["--state", str(state_path), "--list"])
        assert rc == 0
        assert "No GATE 1 entries pending" in out

    def test_list_shows_pending_entries(self, state_path: Path) -> None:
        _seed_state(
            state_path,
            {
                "h1": {
                    "status": "PENDING_OPERATOR_APPROVAL",
                    "slug": "alpha",
                    "pending_since": "2026-04-28T00:00:00",
                    "hypothesis_path": "docs/research/hypotheses/alpha.md",
                },
                "h2": {"status": "DECLINED", "slug": "beta"},
            },
        )
        rc, out, _ = _run(["--state", str(state_path), "--list"])
        assert rc == 0
        assert "alpha" in out
        assert "beta" not in out  # only PENDING shown
        assert "1" in out  # count header


class TestApprove:
    def test_go_flips_to_approved(self, state_path: Path) -> None:
        _seed_state(
            state_path,
            {
                "h1": {
                    "status": "PENDING_OPERATOR_APPROVAL",
                    "slug": "alpha",
                    "pending_since": "2026-04-28T00:00:00",
                },
            },
        )
        rc, out, _ = _run(
            [
                "--state",
                str(state_path),
                "--slug",
                "alpha",
                "--action",
                "GO",
            ]
        )
        assert rc == 0
        assert "APPROVED" in out
        state = load_state(state_path)
        assert state.ideas_processed["h1"]["status"] == "APPROVED"

    def test_skip_flips_to_skipped_with_reason(self, state_path: Path) -> None:
        _seed_state(
            state_path,
            {
                "h1": {
                    "status": "PENDING_OPERATOR_APPROVAL",
                    "slug": "alpha",
                    "pending_since": "2026-04-28T00:00:00",
                },
            },
        )
        rc, out, _ = _run(
            [
                "--state",
                str(state_path),
                "--slug",
                "alpha",
                "--action",
                "SKIP",
                "--reason",
                "duplicate of existing strategy",
            ]
        )
        assert rc == 0
        assert "SKIPPED" in out
        state = load_state(state_path)
        entry = state.ideas_processed["h1"]
        assert entry["status"] == "SKIPPED"
        assert "duplicate" in entry["reason"]

    def test_unknown_slug_returns_error(self, state_path: Path) -> None:
        _seed_state(
            state_path,
            {
                "h1": {
                    "status": "PENDING_OPERATOR_APPROVAL",
                    "slug": "alpha",
                    "pending_since": "2026-04-28T00:00:00",
                },
            },
        )
        rc, _out, err = _run(
            [
                "--state",
                str(state_path),
                "--slug",
                "ghost",
                "--action",
                "GO",
            ]
        )
        assert rc == 2
        assert "no entry found" in err

    def test_already_approved_refuses_to_act(self, state_path: Path) -> None:
        # Status is APPROVED, not PENDING — refuse to clobber.
        _seed_state(
            state_path,
            {
                "h1": {"status": "APPROVED", "slug": "alpha"},
            },
        )
        rc, _out, err = _run(
            [
                "--state",
                str(state_path),
                "--slug",
                "alpha",
                "--action",
                "SKIP",
            ]
        )
        assert rc == 2
        assert "not 'PENDING_OPERATOR_APPROVAL'" in err

    def test_missing_action_returns_error(self, state_path: Path) -> None:
        _seed_state(
            state_path,
            {
                "h1": {"status": "PENDING_OPERATOR_APPROVAL", "slug": "alpha"},
            },
        )
        rc, _out, err = _run(
            [
                "--state",
                str(state_path),
                "--slug",
                "alpha",
            ]
        )
        assert rc == 2
        assert "required" in err


# --------------------------------------------------------------------- #
# GATE 2
# --------------------------------------------------------------------- #


def _seed_gate2(state_path: Path, debates: dict[str, dict]) -> None:
    s = LoopState(debates_completed=dict(debates))
    save_state(s, state_path)


class TestGate2List:
    def test_list_pending_only(self, state_path: Path) -> None:
        _seed_gate2(
            state_path,
            {
                "alpha": {
                    "verdict": "PROMOTE",
                    "deploy_status": "PENDING_DEPLOY_CONFIRMATION",
                    "pending_since": "2026-04-28T00:00:00",
                    "transcript_path": "docs/research/debates/alpha/transcript.md",
                    "candidate_report_path": "reports/candidates/alpha.json",
                },
                "beta": {
                    "verdict": "PROMOTE",
                    "deploy_status": "DEPLOYED",
                },
                "gamma": {
                    "verdict": "REJECT",
                },
            },
        )
        rc, out, _ = _run(["--state", str(state_path), "--gate=2", "--list"])
        assert rc == 0
        assert "alpha" in out
        assert "beta" not in out  # already deployed
        assert "gamma" not in out  # not a PROMOTE
        assert "transcript" in out

    def test_list_empty(self, state_path: Path) -> None:
        save_state(LoopState(), state_path)
        rc, out, _ = _run(["--state", str(state_path), "--gate=2", "--list"])
        assert rc == 0
        assert "No GATE 2 entries pending" in out


class TestGate2Approve:
    def test_go_flips_to_deploy_approved(self, state_path: Path) -> None:
        _seed_gate2(
            state_path,
            {
                "alpha": {
                    "verdict": "PROMOTE",
                    "deploy_status": "PENDING_DEPLOY_CONFIRMATION",
                    "pending_since": "2026-04-28T00:00:00",
                },
            },
        )
        rc, out, _ = _run(
            [
                "--state",
                str(state_path),
                "--gate=2",
                "--slug",
                "alpha",
                "--action",
                "GO",
            ]
        )
        assert rc == 0
        assert "DEPLOY_APPROVED" in out
        state = load_state(state_path)
        assert state.debates_completed["alpha"]["deploy_status"] == ("DEPLOY_APPROVED")

    def test_skip_flips_to_deploy_rejected(self, state_path: Path) -> None:
        _seed_gate2(
            state_path,
            {
                "alpha": {
                    "verdict": "PROMOTE",
                    "deploy_status": "PENDING_DEPLOY_CONFIRMATION",
                    "pending_since": "2026-04-28T00:00:00",
                },
            },
        )
        rc, out, _ = _run(
            [
                "--state",
                str(state_path),
                "--gate=2",
                "--slug",
                "alpha",
                "--action",
                "SKIP",
                "--reason",
                "duplicates existing strategy",
            ]
        )
        assert rc == 0
        assert "DEPLOY_REJECTED" in out
        state = load_state(state_path)
        assert state.debates_completed["alpha"]["deploy_status"] == ("DEPLOY_REJECTED")
        assert "duplicates" in state.debates_completed["alpha"]["deploy_reason"]

    def test_already_deployed_refuses(self, state_path: Path) -> None:
        _seed_gate2(
            state_path,
            {
                "alpha": {"verdict": "PROMOTE", "deploy_status": "DEPLOYED"},
            },
        )
        rc, _out, err = _run(
            [
                "--state",
                str(state_path),
                "--gate=2",
                "--slug",
                "alpha",
                "--action",
                "GO",
            ]
        )
        assert rc == 2
        assert "PENDING_DEPLOY_CONFIRMATION" in err

    def test_unknown_slug(self, state_path: Path) -> None:
        save_state(LoopState(), state_path)
        rc, _out, err = _run(
            [
                "--state",
                str(state_path),
                "--gate=2",
                "--slug",
                "ghost",
                "--action",
                "GO",
            ]
        )
        assert rc == 2
        assert "no debate entry" in err
