"""Operator CLI for GATE 1 / GATE 2 approvals (CL-0hr3, CL-yta6).

Usage:

    # List hypotheses awaiting GATE 1 approval
    python -m scripts.research_approve --list

    # Approve a hypothesis (loop will run implementer next pass)
    python -m scripts.research_approve --slug carry-regime --action GO

    # Skip / reject a hypothesis with a reason (loop archives it)
    python -m scripts.research_approve --slug carry-regime \
        --action SKIP --reason "duplicates existing strategy X"

The loop holds PROPOSED hypotheses at PENDING_OPERATOR_APPROVAL in
``data/research/state.json`` (CL-0hr3). This script is the CLI fallback
for the soak-dashboard panel (CL-7t8d) — both mutate the same state
file. Either approval path is valid; the dashboard is just nicer UX.

The actual gate transitions live in ``src.research.approvals``
(CL-b1l6) — shared with the Telegram approval bot
(``scripts/telegram_approval_bot.py``) so both frontends apply
identical mutations and write the state file atomically.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from src.research.approvals import (
    act_gate1,
    act_gate2,
    find_gate1_hash_by_slug,
    gate1_pending,
    gate2_pending,
    save_state_atomic,
)
from src.research.loop import (
    DEFAULT_STATE_PATH,
    load_state,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Operator approval CLI for the research loop",
    )
    p.add_argument(
        "--state", default=str(DEFAULT_STATE_PATH),
        help="Path to the loop state file",
    )
    p.add_argument(
        "--gate", type=int, choices=(1, 2), default=1,
        help="Which gate to act on: 1 = pre-research, 2 = pre-deploy",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all pending entries for the chosen gate and exit",
    )
    p.add_argument(
        "--slug", help="Strategy slug to act on (required without --list)",
    )
    p.add_argument(
        "--action", choices=("GO", "SKIP"),
        help=(
            "GATE 1: GO = approve for implementer; SKIP = archive. "
            "GATE 2: GO = run paper-shadow registrar; SKIP = reject deploy."
        ),
    )
    p.add_argument(
        "--reason", default="",
        help="Free-text reason recorded with the action",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env so credentials/paths are available without first
    # sourcing the file. Explicit env vars still win.
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    args = _build_parser().parse_args(argv)
    state = load_state(args.state)

    if args.list:
        if args.gate == 1:
            return _list_gate1(state)
        return _list_gate2(state)

    if not args.slug or not args.action:
        print(
            "ERROR: --slug and --action required (use --list to inspect)",
            file=sys.stderr,
        )
        return 2

    if args.gate == 1:
        rc = _act_gate1(state, args.slug, args.action, args.reason)
    else:
        rc = _act_gate2(state, args.slug, args.action, args.reason)
    if rc == 0:
        save_state_atomic(state, args.state)
    return rc


# --------------------------------------------------------------------- #
# GATE 1 actions
# --------------------------------------------------------------------- #


def _list_gate1(state: Any) -> int:
    pending = gate1_pending(state)
    if not pending:
        print("No GATE 1 entries pending.")
        return 0
    print(f"GATE 1 pending: {len(pending)}")
    for h, e in pending:
        print(
            f"  - slug={e.get('slug')} "
            f"extract={h[:12]}... "
            f"since={e.get('pending_since')}",
        )
        if e.get("hypothesis_path"):
            print(f"      hypothesis: {e['hypothesis_path']}")
    return 0


def _act_gate1(state: Any, slug: str, action: str, reason: str) -> int:
    target_hash = find_gate1_hash_by_slug(state, slug)
    if target_hash is None:
        print(f"ERROR: no entry found with slug={slug!r}", file=sys.stderr)
        return 2

    result = act_gate1(
        state, target_hash, approve=(action == "GO"), reason=reason,
    )
    if not result.ok:
        print(f"ERROR: {result.message}", file=sys.stderr)
        return 2
    print(result.message)
    return 0


# --------------------------------------------------------------------- #
# GATE 2 actions
# --------------------------------------------------------------------- #


def _list_gate2(state: Any) -> int:
    pending = gate2_pending(state)
    if not pending:
        print("No GATE 2 entries pending.")
        return 0
    print(f"GATE 2 pending: {len(pending)}")
    for slug, e in pending:
        print(
            f"  - slug={slug} verdict={e.get('verdict')} "
            f"since={e.get('pending_since')}",
        )
        if e.get("transcript_path"):
            print(f"      transcript: {e['transcript_path']}")
        if e.get("candidate_report_path"):
            print(f"      report: {e['candidate_report_path']}")
    return 0


def _act_gate2(state: Any, slug: str, action: str, reason: str) -> int:
    result = act_gate2(state, slug, approve=(action == "GO"), reason=reason)
    if not result.ok:
        print(f"ERROR: {result.message}", file=sys.stderr)
        return 2
    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
