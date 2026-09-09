"""Operator-only GET/SELECT evidence capture, or credential-free offline replay.

No order submission, cancellation, UPDATE or repair command exists here.
Capture requires an explicit env path and stores private evidence in a NEW
directory. Reviewing the report does not authorize applying its proposals.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.data.db_env import build_db_url
from src.dotenv_bootstrap import load_project_env
from src.execution.alpaca_recovery import (
    EvidenceError,
    PaperEvidenceClient,
    build_report,
    fingerprint,
    history_dict,
)
from src.monitoring.logging_setup import JSONFormatter, get_context_filter

logger = logging.getLogger(__name__)


def internal_snapshot(engine: Engine) -> dict[str, Any]:
    """Bounded, read-only, repeatable-read snapshot; no daemon helpers imported."""
    logger.info("Reading internal Alpaca records in read-only transaction")
    with (
        engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn,
        conn.begin(),
    ):
        conn.execute(text("SET TRANSACTION READ ONLY"))
        # Keep a stalled audit from holding a production snapshot indefinitely.
        conn.execute(text("SET LOCAL statement_timeout = '15s'"))
        return {
            "options": [
                dict(r._mapping)
                for r in conn.execute(
                    text(
                        "SELECT * FROM alpaca_option_orders WHERE status = 'submitted' ORDER BY idea_id"
                    )
                )
            ],
            "equities": [
                dict(r._mapping)
                for r in conn.execute(
                    text(
                        "SELECT * FROM alpaca_equity_orders WHERE status = 'submitted' ORDER BY idea_id"
                    )
                )
            ],
            "idea_ids": [
                r[0] for r in conn.execute(text("SELECT idea_id FROM trade_ideas ORDER BY idea_id"))
            ],
        }


def capture(
    client: PaperEvidenceClient, engine: Engine, max_pages: int, *, include_contracts: bool = False
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"started_at": datetime.now(UTC).isoformat()}
    snapshot["account_scope"] = client.account_scope()
    snapshot["positions_before"] = client.positions()
    snapshot["internal"] = internal_snapshot(engine)
    snapshot["orders"] = history_dict(client.history("orders", max_pages=max_pages))
    snapshot["activities"] = history_dict(client.history("activities", max_pages=max_pages))
    if include_contracts:
        snapshot["contracts"] = {}
        for symbol in sorted(
            {r["occ_symbol"] for r in snapshot["internal"]["options"] if r.get("occ_symbol")}
        ):
            try:
                snapshot["contracts"][symbol] = client.contract(symbol)
            except EvidenceError as exc:
                snapshot["contracts"][symbol] = {"error": str(exc)}
    snapshot["exit_lookups"] = {}
    for row in snapshot["internal"]["options"]:
        if row.get("status") != "submitted" or row.get("exit_status") != "submitted":
            continue
        client_id = f"curlit-exit-{row['idea_id']}"
        try:
            try:
                order = client.order(broker_id=row.get("exit_order_id"), client_id=client_id)
            except EvidenceError as exc:
                # Only confirmed not-found may fall back to original client ID.
                # A timeout remains unknown, not evidence of a missing order.
                if str(exc) != "http_404" or not row.get("exit_order_id"):
                    raise
                order = client.order(client_id=client_id)
            snapshot["exit_lookups"][row["idea_id"]] = {"order": order}
        except EvidenceError as exc:
            snapshot["exit_lookups"][row["idea_id"]] = {"error": str(exc)}
    snapshot["positions_after"] = client.positions()
    snapshot["internal_after"] = internal_snapshot(engine)
    snapshot["finished_at"] = datetime.now(UTC).isoformat()
    # Same canonical representation on capture and offline replay.
    result: dict[str, Any] = json.loads(json.dumps(snapshot, default=str))
    return result


def write_private(path: Path, payload: object) -> None:
    """Exclusive creation: never overwrite prior audit evidence."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(payload, output, sort_keys=True, indent=2, default=str)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, help="Offline replay; loads no credentials")
    source.add_argument(
        "--capture-paper", action="store_true", help="Explicit operator GET/SELECT capture"
    )
    parser.add_argument("--env-file", type=Path, help="Explicit operational env file, capture only")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New private evidence directory"
    )
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument(
        "--include-contracts",
        action="store_true",
        help="Capture broker contract sizes for fill ledger",
    )
    args = parser.parse_args(argv)
    if args.max_pages < 1 or (args.capture_paper and args.env_file is None):
        parser.error("capture requires --env-file; max-pages must be positive")
    if args.snapshot and args.env_file:
        parser.error("offline replay must not load an environment file")
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    handler.addFilter(get_context_filter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # Do not enable HTTP debug logging: it can contain account/order identifiers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        if args.snapshot:
            snapshot = json.loads(args.snapshot.read_text())
        else:
            if load_project_env(args.env_file) is None:
                raise ValueError("explicit environment file unavailable")
            if os.environ.get("ALPACA_PAPER", "true").lower() not in {"1", "true", "yes"}:
                raise ValueError("capture refuses a live-configured environment")
            client = PaperEvidenceClient(
                os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_API_SECRET", "")
            )
            engine = create_engine(build_db_url(), connect_args={"connect_timeout": 10})
            try:
                snapshot = capture(
                    client, engine, args.max_pages, include_contracts=args.include_contracts
                )
            finally:
                client.close()
                engine.dispose()
        write_private(args.output_dir / "snapshot.json", snapshot)
        report = build_report(snapshot)
        write_private(args.output_dir / "report.json", report)
        write_private(
            args.output_dir / "manifest.json",
            {
                "snapshot_sha256": fingerprint(snapshot),
                "report_sha256": fingerprint(report),
                "mode": "read_only",
                "repair_authorized": False,
            },
        )
        print(
            json.dumps(
                {
                    "output": str(args.output_dir),
                    "pending_options": len(report["pending_option_exits"]),
                    "unmatched_equities": len(report["unmatched_equity_holdings"]),
                    "coverage": report["coverage"],
                }
            )
        )
        return 0 if all(snapshot[k]["exhausted"] for k in ("orders", "activities")) else 2
    except Exception as exc:
        # DB exceptions can expose URLs/parameters. Keep details out of logs.
        write_private(args.output_dir / "failure.json", {"error_type": type(exc).__name__})
        logger.error(
            "Capture/report failed: %s (no operational writes performed)", type(exc).__name__
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
