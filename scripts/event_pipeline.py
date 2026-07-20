"""Current-events pipeline entrypoint (CL-6iu7).

Usage:
    .venv/bin/python scripts/event_pipeline.py --ingest --assess --once
    .venv/bin/python scripts/event_pipeline.py --ingest --loop 900
    .venv/bin/python scripts/event_pipeline.py --assess --limit 10

``--ingest`` polls GDELT (one themed query per playbook) and inserts
NEW rows into geo_events; ``--assess`` runs the Event Impact Agent over
NEW rows (capped per run). ``--loop N`` repeats every N seconds —
suitable for a systemd timer / cron / Airflow later; ``--once`` (the
default) runs a single cycle.

GDELT updates ~every 15 minutes, so looping faster than ~900s only
re-fetches the same articles (they dedup away harmlessly).

After each assess cycle a compact Telegram digest of that cycle's
urgent events (urgency >= --digest-min-urgency, default 5 or
$EVENT_DIGEST_MIN_URGENCY) is sent via src.events.digest — the
operator's "bots surfaced these tickers" feed. ``--no-digest``
disables it; quiet cycles never send anything.

Each cycle also runs the key-free relative-volume scanner (CL-i4sr)
over the playbook equity watch universe before the digest, so Watch:
tickers with an unusual spike in the last 24h render as ``FRO×3.2``.
``--no-scan`` disables it; a scan failure never kills the cycle.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("event_pipeline")


def _db_url() -> str:
    return (
        f"postgresql+psycopg2://"
        f"{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GDELT ingest + LLM impact assessment for geo_events",
    )
    p.add_argument("--ingest", action="store_true", help="Poll GDELT for new events")
    p.add_argument("--assess", action="store_true", help="Assess NEW rows via the impact agent")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one cycle (default)")
    mode.add_argument(
        "--loop", type=int, metavar="SECONDS", default=None,
        help="Repeat every N seconds until interrupted",
    )
    p.add_argument(
        "--lookback-minutes", type=int, default=60,
        help="GDELT ingest window ending now (default 60; overlap dedups away)",
    )
    p.add_argument(
        "--limit", type=int, default=20,
        help="Max NEW rows assessed per cycle (default 20)",
    )
    p.add_argument(
        "--playbooks", default="configs/event_playbooks.yaml",
        help="Path to the event playbook config",
    )
    p.add_argument(
        "--model", default=os.environ.get("EVENT_IMPACT_MODEL", ""),
        help="Impact agent model override (default: agent default)",
    )
    p.add_argument(
        "--digest", action=argparse.BooleanOptionalAction, default=True,
        help="Send a Telegram digest of urgent events after each assess "
             "cycle (default on; --no-digest to disable)",
    )
    p.add_argument(
        "--scan", action=argparse.BooleanOptionalAction, default=True,
        help="Run the relative-volume scanner over the equity watch "
             "universe each cycle, before the digest (default on; "
             "--no-scan to disable)",
    )
    p.add_argument(
        "--digest-min-urgency", type=int, metavar="N", default=None,
        help="Digest urgency threshold 1-10 (default: "
             "$EVENT_DIGEST_MIN_URGENCY or 5)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return p


def _resolve_digest_min_urgency(cli_value: int | None) -> int:
    """CLI flag wins; else $EVENT_DIGEST_MIN_URGENCY; else the module
    default. Resolved AFTER load_project_env() so .env values count."""
    from src.events.digest import DEFAULT_MIN_URGENCY  # noqa: PLC0415

    if cli_value is not None:
        return cli_value
    raw = os.environ.get("EVENT_DIGEST_MIN_URGENCY", "")
    try:
        return int(raw) if raw.strip() else DEFAULT_MIN_URGENCY
    except ValueError:
        logger.warning(
            "EVENT_DIGEST_MIN_URGENCY=%r is not an int; using %d",
            raw, DEFAULT_MIN_URGENCY,
        )
        return DEFAULT_MIN_URGENCY


def _cycle(args: argparse.Namespace) -> None:
    if args.ingest:
        from src.data.gdelt import GdeltIngester  # noqa: PLC0415

        ingester = GdeltIngester(_db_url(), playbooks_path=args.playbooks)
        end = datetime.now(UTC)
        start = end - timedelta(minutes=args.lookback_minutes)
        rows = ingester.run(start, end)
        logger.info("ingest: %d new geo_events rows", rows)

    if args.scan:
        # RVOL scan runs BEFORE assess/digest so this cycle's digest
        # can annotate Watch tickers with fresh marks. A scan failure
        # (yfinance outage, missing table) must never kill the cycle
        # — RVOL is advisory confirmation, not pipeline plumbing.
        try:
            from src.scanners.relative_volume import (  # noqa: PLC0415
                RelativeVolumeScanner,
            )

            scanner = RelativeVolumeScanner(
                _db_url(), playbooks_path=args.playbooks,
            )
            scan_rows = scanner.scan()
            logger.info(
                "scan: %d tickers, %d unusual",
                len(scan_rows), sum(1 for r in scan_rows if r.is_unusual),
            )
        except Exception:
            logger.exception("volume scan failed; continuing")

    if args.assess:
        from sqlalchemy import create_engine  # noqa: PLC0415

        from src.events.impact_agent import (  # noqa: PLC0415
            DEFAULT_MODEL,
            EventImpactAgent,
        )

        agent = EventImpactAgent(
            engine=create_engine(_db_url()),
            model=args.model or DEFAULT_MODEL,
            playbooks_path=args.playbooks,
        )
        results = agent.assess_new_events(limit=args.limit)
        for result in results:
            logger.info("%s", result.summary_line())
        assessed = sum(1 for r in results if r.status == "ASSESSED")
        logger.info(
            "assess: %d processed (%d assessed, %d dismissed)",
            len(results), assessed, len(results) - assessed,
        )

        if args.digest:
            from src.events.digest import (  # noqa: PLC0415
                fetch_volume_marks,
                send_digest,
            )

            # A digest failure must never take down the pipeline —
            # assessments are already persisted by this point.
            try:
                # fetch_volume_marks is fail-soft ({} on missing
                # table / DB blip) — the Watch line just renders
                # without ×rvol annotations.
                marks = fetch_volume_marks(create_engine(_db_url()))
                disp = send_digest(
                    results, min_urgency=args.digest_min_urgency,
                    volume_marks=marks,
                )
            except Exception:
                logger.exception("digest dispatch failed; continuing")
            else:
                if disp is not None and disp.any_attempted:
                    logger.info(
                        "digest: sent (telegram ok=%s)",
                        disp.telegram_succeeded,
                    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.ingest and not args.assess:
        _build_parser().error("nothing to do: pass --ingest and/or --assess")

    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    args.digest_min_urgency = _resolve_digest_min_urgency(args.digest_min_urgency)

    if args.loop is None:
        _cycle(args)
        return 0

    logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
    try:
        while True:
            t0 = time.time()
            try:
                _cycle(args)
            except Exception:
                # A failed cycle (GDELT hiccup, DB blip) must not kill
                # the loop — log and try again next interval.
                logger.exception("cycle failed; continuing")
            elapsed = time.time() - t0
            time.sleep(max(0.0, args.loop - elapsed))
    except KeyboardInterrupt:
        logger.info("interrupted — exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
