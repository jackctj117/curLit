"""Operator-only SELECT export of captured research failures (CL-uofe).

No current retrieval, new company verification or invented liquidity. Exports
the first stored invocation for each explicitly selected event and retains its
original audit alongside the derived, engineering-only frozen tool collection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.alpaca_recovery_report import write_private
from sqlalchemy import create_engine, text

from src.data.db_env import build_db_url
from src.dotenv_bootstrap import load_project_env
from src.events.niche_shadow import CapturedInput


def captured_failure(audit: dict[str, Any]) -> CapturedInput:
    """Recover only measurements/identity results actually preserved in this audit."""
    report = audit["report"]
    discovery = report["discovery"]
    symbols: dict[str, dict[str, Any]] = {}
    for call in discovery["trace"]:
        if call.get("tool") == "check_ticker" and call.get("result", {}).get("exists") is True:
            result = call["result"]
            symbols[result["ticker"]] = dict(result)
    for call in discovery["trace"]:
        ticker = call.get("arguments", {}).get("ticker")
        if call.get("tool") == "get_company_profile" and ticker in symbols:
            symbols[ticker]["profile"] = call["result"]
    market = {}
    for candidate in report["candidates"]:
        ticker, research = candidate["ticker"], candidate["research"]
        market[ticker] = {
            "market_cap": research.get("market_cap"),
            "avg_dollar_volume": research.get("avg_dollar_volume"),
            "last_close": research.get("last_close"),
            "observed_at": research.get("market_observed_at"),
        }
        if research.get("market_received_at") is not None:
            market[ticker]["retrieved_at"] = research["market_received_at"]
    return CapturedInput.capture(
        {
            # DiscoveryOutcome.recorded_at_utc is construction/start time.
            # The immutable audit's persisted timestamp covers later tool results.
            "captured_at": str(audit["recorded_at"]) if audit.get("recorded_at") else None,
            "event": audit["input_snapshot"],
            "symbols": symbols,
            "market_data": market,
            "sources": discovery["sources"],
            "limitations": [
                "engineering reconstruction of archived invocation, not a forward trial",
                "only originally checked symbols/retrieved sources are available",
                "missing market observations remain unknown",
            ],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--event", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if load_project_env(args.env_file) is None:
        parser.error("explicit operational environment unavailable")
    engine = create_engine(build_db_url(), connect_args={"connect_timeout": 10})
    try:
        with engine.connect() as conn, conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text("SET LOCAL statement_timeout='15s'"))
            rows = [
                dict(
                    conn.execute(
                        text(
                                "SELECT invocation_id,event_id,recorded_at,input_snapshot,report FROM niche_research_audit "
                            "WHERE event_id=:event ORDER BY recorded_at,invocation_id LIMIT 1"
                        ),
                        {"event": event},
                    )
                    .mappings()
                    .one()
                )
                for event in args.event
            ]
        captures = [captured_failure(row) for row in rows]
        args.output_dir.mkdir(mode=0o700, exist_ok=False)
        for row, captured in zip(rows, captures, strict=True):
            write_private(args.output_dir / f"{row['event_id']}-audit.json", row)
            write_private(args.output_dir / f"{row['event_id']}-capture.json", captured.payload())
        print(f"Exported {len(rows)} archived invocations; no operational writes or provider calls")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
