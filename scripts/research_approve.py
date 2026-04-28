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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.research.loop import (
    DEFAULT_STATE_PATH,
    GATE1_APPROVED_STATUS,
    GATE1_PENDING_STATUS,
    GATE1_SKIPPED_STATUS,
    load_state,
    save_state,
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
        "--list", action="store_true",
        help="List all pending entries and exit",
    )
    p.add_argument(
        "--slug", help="Strategy slug to act on (required without --list)",
    )
    p.add_argument(
        "--action", choices=("GO", "SKIP"),
        help="GO = approve for implementer; SKIP = archive with reason",
    )
    p.add_argument(
        "--reason", default="",
        help="Free-text reason recorded with the action",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state = load_state(args.state)

    if args.list:
        pending = [
            (h, e) for h, e in state.ideas_processed.items()
            if e.get("status") == GATE1_PENDING_STATUS
        ]
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

    if not args.slug or not args.action:
        print(
            "ERROR: --slug and --action required (use --list to inspect)",
            file=sys.stderr,
        )
        return 2

    # Find the entry by slug
    target_hash: str | None = None
    for h, e in state.ideas_processed.items():
        if e.get("slug") == args.slug:
            target_hash = h
            break
    if target_hash is None:
        print(
            f"ERROR: no entry found with slug={args.slug!r}", file=sys.stderr,
        )
        return 2

    entry = state.ideas_processed[target_hash]
    if entry.get("status") != GATE1_PENDING_STATUS:
        print(
            f"ERROR: entry for slug={args.slug!r} is in status "
            f"{entry.get('status')!r}, not {GATE1_PENDING_STATUS!r}; "
            f"refusing to act",
            file=sys.stderr,
        )
        return 2

    if args.action == "GO":
        entry["status"] = GATE1_APPROVED_STATUS
        if args.reason:
            entry["reason"] = args.reason
        print(f"APPROVED: {args.slug}")
    elif args.action == "SKIP":
        entry["status"] = GATE1_SKIPPED_STATUS
        entry["reason"] = args.reason or "skipped by operator"
        print(f"SKIPPED: {args.slug} — {entry['reason']}")

    save_state(state, Path(args.state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
