"""Explicit paper-only historical repair after restored backup and stopped writers.

CL-koeg: never submits/cancels orders. Requires fresh GET/SELECT evidence and
per-idea operator approvals; corrections use locked original-row comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]  # OS process-inspection boundary.
from scripts.alpaca_recovery_report import capture, write_private
from sqlalchemy import create_engine

from src.data.db_env import build_db_url
from src.dotenv_bootstrap import load_project_env
from src.execution.alpaca_ledger_repair import apply_repairs
from src.execution.alpaca_recovery import TERMINAL, PaperEvidenceClient


def verify_writers_stopped() -> None:
    """Inspect current user's actual processes, not stale PID files/launcher status."""
    for process in psutil.process_iter(["pid", "uids", "cmdline"]):
        try:
            if process.info["uids"].real != os.getuid():
                continue
            args = process.info["cmdline"] or []
            if any(
                Path(arg).name in {"execute_options.py", "execute_equities.py"}
                or arg in {"scripts.execute_options", "scripts.execute_equities"}
                for arg in args
            ):
                raise RuntimeError(f"Alpaca writer still running: PID {process.pid}")
        except psutil.NoSuchProcess:
            continue
        # AccessDenied is deliberately not ignored: process exclusivity unknown.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--verified-backup", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restore-equity", action="append", default=[])
    parser.add_argument("--restore-option", action="append", default=[])
    parser.add_argument("--apply-paper-repair", action="store_true")
    args = parser.parse_args()
    if not args.apply_paper_repair:
        parser.error("explicit --apply-paper-repair required")
    backup = json.loads(args.verified_backup.read_text())
    archive = args.verified_backup.parent / "database.pgdump.gpg"
    if (
        backup.get("full_restore_verified") is not True
        or not archive.is_file()
        or hashlib.sha256(archive.read_bytes()).hexdigest() != backup.get("encrypted_sha256")
    ):
        parser.error("intact encrypted backup with full disposable restore proof required")
    baseline = json.loads(args.baseline.read_text())
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    verify_writers_stopped()
    load_project_env(args.env_file)
    if os.environ.get("ALPACA_PAPER", "true").lower() not in {"true", "1", "yes"}:
        parser.error("paper environment required")
    if os.environ.get("ALPACA_LEDGER_CLOSE_ONLY", "").lower() not in {"true", "1", "yes"}:
        parser.error("persistent ALPACA_LEDGER_CLOSE_ONLY=1 required before restoring management")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    client = PaperEvidenceClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
    engine = create_engine(build_db_url(), connect_args={"connect_timeout": 10})
    try:
        current = capture(client, engine, 100, include_contracts=True)
        write_private(args.output_dir / "snapshot.json", current)
        if current["internal"] != current["internal_after"]:
            raise RuntimeError("internal rows changed during capture")
        if any(order.get("status") not in TERMINAL for order in current["orders"]["records"]):
            raise RuntimeError("working or unknown broker orders block controlled repair")

        # Compare stable economic identity, not market-value/quote fluctuations.
        def inventory(positions: list[dict[str, Any]]) -> list[tuple[str, str]]:
            return sorted((p["asset_id"], p["qty"]) for p in positions)

        if inventory(current["positions_before"]) != inventory(current["positions_after"]):
            raise RuntimeError("broker inventory changed during capture")
        if inventory(client.positions()) != inventory(current["positions_after"]):
            raise RuntimeError("broker inventory changed before repair")
        verify_writers_stopped()
        result = apply_repairs(
            engine,
            baseline,
            current,
            restore_equities=set(args.restore_equity),
            restore_options=set(args.restore_option),
        )
        write_private(args.output_dir / "result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        engine.dispose()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
