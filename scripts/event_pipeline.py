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

Enrichment + idea ledger (CL-mgcp): after assessments, ONE price batch
is fetched for every affected/idea/fade ticker in the cycle
(src.events.prices), each ASSESSED event's advisory ``trade_ideas``
are persisted to the ``trade_ideas`` table (migration 007, deduped on
idea_id), stale pending ideas past their time stop auto-expire, and
the digest renders prices, per-event age, and Ideas:/Fade: sections.
All of it is fail-soft — enrichment must never take down the pipeline.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger("event_pipeline")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GDELT ingest + LLM impact assessment for geo_events",
    )
    p.add_argument("--ingest", action="store_true", help="Poll GDELT for new events")
    p.add_argument("--assess", action="store_true", help="Assess NEW rows via the impact agent")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one cycle (default)")
    mode.add_argument(
        "--loop",
        type=int,
        metavar="SECONDS",
        default=None,
        help="Repeat every N seconds until interrupted",
    )
    p.add_argument(
        "--lookback-minutes",
        type=int,
        default=60,
        help="GDELT ingest window ending now (default 60; overlap dedups away)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max NEW rows assessed per cycle (default 20)",
    )
    p.add_argument(
        "--playbooks",
        default="configs/event_playbooks.yaml",
        help="Path to the event playbook config",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("EVENT_IMPACT_MODEL", ""),
        help="Impact agent model override (default: agent default)",
    )
    p.add_argument(
        "--digest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send a Telegram digest of urgent events after each assess "
        "cycle (default on; --no-digest to disable)",
    )
    p.add_argument(
        "--scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the relative-volume scanner over the equity watch "
        "universe each cycle, before the digest (default on; "
        "--no-scan to disable)",
    )
    p.add_argument(
        "--poly",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Poll tracked geopolitical Polymarket markets each cycle, "
        "persist YES probs, and Telegram-alert on rapid shifts "
        "(default on; --no-poly to disable)",
    )
    p.add_argument(
        "--niche",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the multi-hop niche/asymmetry pass (CL-u2ph) on "
        "high-urgency ASSESSED events (urgency >= $NICHE_MIN_URGENCY, "
        "default 7) — surfaces verified, liquidity-gated under-followed "
        "names, merged into trade_ideas (default on; --no-niche to "
        "disable; it burns one extra LLM call per qualifying event)",
    )
    p.add_argument(
        "--poly-config",
        default="configs/polymarket_geo_markets.yaml",
        help="Path to the theme-tagged geo markets config polled by the --poly step",
    )
    p.add_argument(
        "--digest-min-urgency",
        type=int,
        metavar="N",
        default=None,
        help="Digest urgency threshold 1-10 (default: $EVENT_DIGEST_MIN_URGENCY or 5)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return p


def _resolve_digest_min_urgency(cli_value: int | None) -> int:
    """CLI flag wins; else $EVENT_DIGEST_MIN_URGENCY; else the module
    default. Resolved AFTER load_project_env() so .env values count."""
    from src.events.digest import DAEMON_DEFAULT_MIN_URGENCY  # noqa: PLC0415

    if cli_value is not None:
        return cli_value
    raw = os.environ.get("EVENT_DIGEST_MIN_URGENCY", "")
    try:
        return int(raw) if raw.strip() else DAEMON_DEFAULT_MIN_URGENCY
    except ValueError:
        logger.warning(
            "EVENT_DIGEST_MIN_URGENCY=%r is not an int; using %d",
            raw,
            DAEMON_DEFAULT_MIN_URGENCY,
        )
        return DAEMON_DEFAULT_MIN_URGENCY


def _resolve_niche_min_urgency() -> int:
    """$NICHE_MIN_URGENCY (int, 1-10) or the niche agent's module default.
    Resolved AFTER load_project_env() so .env values count."""
    from src.events.niche_agent import DEFAULT_MIN_URGENCY  # noqa: PLC0415

    raw = os.environ.get("NICHE_MIN_URGENCY", "")
    try:
        return int(raw) if raw.strip() else DEFAULT_MIN_URGENCY
    except ValueError:
        logger.warning(
            "NICHE_MIN_URGENCY=%r is not an int; using %d",
            raw,
            DEFAULT_MIN_URGENCY,
        )
        return DEFAULT_MIN_URGENCY


#: Niche discovery fan-out width (CL-818b). Each qualifying event's niche pass
#: is independent, blocking I/O (Kimi tool loop + yfinance + red-team critic),
#: so running them concurrently collapses the cycle to ~the slowest single
#: event instead of their sum. Bounded to stay under the Moonshot/tool rate
#: limits. Token SPEND is unchanged (same calls) — this is latency-only.
_DEFAULT_NICHE_CONCURRENCY = 4
_MAX_NICHE_CONCURRENCY = 8


def _niche_max_concurrency() -> int:
    """$NICHE_MAX_CONCURRENCY (int) bounding the niche discovery fan-out;
    default 4, clamped to [1, 8]. 1 = the old serial behavior."""
    raw = os.environ.get("NICHE_MAX_CONCURRENCY", "")
    try:
        value = int(raw) if raw.strip() else _DEFAULT_NICHE_CONCURRENCY
    except ValueError:
        logger.warning(
            "NICHE_MAX_CONCURRENCY=%r is not an int; using %d",
            raw,
            _DEFAULT_NICHE_CONCURRENCY,
        )
        value = _DEFAULT_NICHE_CONCURRENCY
    return max(1, min(_MAX_NICHE_CONCURRENCY, value))


def _niche_step(engine: object, results: list, min_urgency: int) -> int:
    """CL-u2ph: multi-hop niche/asymmetry pass on high-urgency ASSESSED
    events only (bounded extra research per qualifying event).
    Surviving VERIFIED, liquidity-gated niche ideas are MERGED
    into each event's in-memory assessment ``trade_ideas`` (tagged
    niche=true) AND re-persisted to geo_events so they flow through the
    downstream enrich/persist/digest path unchanged. Returns how many
    niche ideas were surfaced across the cycle.

    Fail-soft BY DESIGN: the niche pass is additive advisory colour — any
    failure (symbol universe unavailable, LLM transport, DB blip) logs
    and returns, never kills the cycle."""
    import json as _json  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from uuid import uuid4  # noqa: PLC0415

    from sqlalchemy import text  # noqa: PLC0415

    from src.data.symbols import SymbolUniverse  # noqa: PLC0415
    from src.events.niche_agent import NicheAgent, NicheReport  # noqa: PLC0415
    from src.events.niche_audit import persist_niche_audit  # noqa: PLC0415
    from src.events.playbooks import load_playbooks  # noqa: PLC0415
    from src.events.research_evidence import DiscoveryOutcome  # noqa: PLC0415

    def _urgency(r: object) -> int:
        try:
            return int((getattr(r, "assessment", None) or {}).get("urgency", 0))
        except (TypeError, ValueError):
            return 0

    qualifying = [
        r for r in results if getattr(r, "status", "") == "ASSESSED" and _urgency(r) >= min_urgency
    ]
    if not qualifying:
        logger.info("niche: no ASSESSED events at urgency >= %d; skipped", min_urgency)
        return 0

    universe = SymbolUniverse(engine)  # type: ignore[arg-type]
    playbooks = load_playbooks("configs/event_playbooks.yaml")
    agent = NicheAgent(universe=universe)

    # Warm the SymbolUniverse in-memory cache ONCE up front, so the concurrent
    # verify_ideas reads below are pure in-memory (no N cold DB loads / no
    # first-load race across workers). Best-effort — a stub/oddball universe
    # just skips it and each worker self-loads as before.
    try:
        universe.exists("SPY")
    except Exception:
        logger.debug("symbol-universe cache warm-up skipped", exc_info=True)

    def _discover(r: Any) -> tuple[Any, NicheReport, str, dict[str, Any]]:
        """Thread body — DISCOVERY ONLY (thread-safe: verify reads the warmed
        in-memory universe; the Kimi/yfinance/critic calls are independent
        blocking I/O). Merge + DB persist stay on the caller thread. Fail-soft
        per event so one bad event never sinks the batch."""
        invocation_id = str(uuid4())
        row = {
            "id": r.event_id,
            "headline": r.headline,
            "theme": r.theme,
            "assessment": copy.deepcopy(r.assessment),
        }
        try:
            report = agent.run_report(copy.deepcopy(row), playbooks.get(r.theme or ""))
        except Exception:
            logger.exception("niche discovery failed for event id=%s; continuing", r.event_id)
            report = NicheReport(DiscoveryOutcome("unavailable", "pipeline_discovery_failure"), [])
        return r, report, invocation_id, row

    # DISCOVER concurrently (bounded), then MERGE + PERSIST serially on THIS
    # thread so the DB writes never race (CL-818b). pool.map preserves input
    # order → deterministic merges + logs. workers==1 is the old serial path.
    workers = max(1, min(_niche_max_concurrency(), len(qualifying)))
    if workers == 1:
        discovered = [_discover(r) for r in qualifying]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            discovered = list(pool.map(_discover, qualifying))

    surfaced = 0
    for r, report, invocation_id, snapshot in discovered:
        # Fail-soft PER EVENT: a transient DB blip on one event must not drop
        # the merges/writes for the events after it — they are already
        # ASSESSED, so the next cycle will NOT retry them and their niche
        # ideas would be lost for good.
        try:
            report_data = report.to_dict()
            # Independent transaction: a later event-status race or merge
            # rollback must not discard the already committed audit evidence.
            persist_niche_audit(engine, invocation_id, r.event_id, snapshot, report_data)
            enriched = copy.deepcopy(snapshot["assessment"])
            enriched["niche_research"] = {**report_data, "invocation_id": invocation_id}
            added = agent.merge_into_assessment(enriched, report.eligible)
            # Persist even empty/failed outcomes; absence is not approval.
            with engine.begin() as conn:  # type: ignore[attr-defined]
                written = conn.execute(
                    text(
                        "UPDATE geo_events SET assessment = :a "
                        "WHERE id = :id AND status = 'ASSESSED' "
                        "AND assessment = CAST(:original AS jsonb)",
                    ),
                    {
                        "a": _json.dumps(enriched), "id": r.event_id,
                        "original": _json.dumps(snapshot["assessment"]),
                    },
                )
                if written.rowcount != 1:
                    logger.warning(
                        "niche audit preserved but merge skipped: event or assessment changed "
                        "event=%s invocation=%s",
                        r.event_id, invocation_id,
                    )
                    continue
            r.assessment = enriched  # Only committed ideas may reach the later ledger step.
            surfaced += added
        except Exception:
            logger.exception(
                "niche merge/persist failed for event id=%s; continuing",
                r.event_id,
            )
    logger.info(
        "niche: %d qualifying event(s), %d niche idea(s) surfaced+merged (concurrency=%d)",
        len(qualifying),
        surfaced,
        workers,
    )
    return surfaced


def _result_tickers(result: object) -> set[str]:
    """Every ticker the enrichment could annotate for one ASSESSED
    result: affected instruments + trade-idea tickers + fade tickers."""
    tickers: set[str] = set()
    assessment = getattr(result, "assessment", None) or {}
    for entry in assessment.get("affected") or []:
        if isinstance(entry, dict) and entry.get("instrument"):
            tickers.add(str(entry["instrument"]))
    for key in ("trade_ideas", "fade_candidates"):
        for entry in assessment.get(key) or []:
            if isinstance(entry, dict) and entry.get("ticker"):
                tickers.add(str(entry["ticker"]))
    return tickers


def _enrich_and_persist(engine: object, results: list) -> tuple[dict, dict]:
    """CL-mgcp: one price batch per cycle, idea-ledger writes, idea
    auto-expiry, and the seen_at lookup for digest ages.

    Fail-soft EVERYWHERE by design — the assessments are already
    persisted by this point, so enrichment failures degrade to an
    unannotated digest, never a dead cycle. Returns (prices, seen_ats).
    """
    prices: dict = {}
    seen_ats: dict = {}
    assessed = [r for r in results if getattr(r, "status", "") == "ASSESSED"]
    try:
        from src.events.idea_ledger import expire_stale, persist_ideas  # noqa: PLC0415
        from src.events.prices import get_prices  # noqa: PLC0415

        tickers = sorted({t for r in assessed for t in _result_tickers(r)})
        if tickers:
            prices = get_prices(tickers, engine=engine)
        persisted = 0
        for r in assessed:
            persisted += persist_ideas(
                engine,
                r.event_id,
                r.assessment,
                prices=prices,
            )
        expired = expire_stale(engine)
        logger.info(
            "ideas: %d persisted, %d auto-expired (prices for %d/%d tickers)",
            persisted,
            expired,
            len(prices),
            len(tickers),
        )
    except Exception:
        logger.exception("idea ledger / price enrichment failed; continuing")
    if assessed:
        try:
            from sqlalchemy import bindparam, text  # noqa: PLC0415

            stmt = text(
                "SELECT id, seen_at FROM geo_events WHERE id IN :ids",
            ).bindparams(bindparam("ids", expanding=True))
            with engine.connect() as conn:  # type: ignore[attr-defined]
                seen_ats = {
                    int(row_id): seen_at
                    for row_id, seen_at in conn.execute(
                        stmt,
                        {"ids": [r.event_id for r in assessed]},
                    )
                }
        except Exception:
            logger.debug(
                "seen_at lookup failed; digest renders without event ages",
                exc_info=True,
            )
    return prices, seen_ats


def _poly_step(args: argparse.Namespace) -> Any:
    """Poll tracked geopolitical Polymarket markets, persist YES probs,
    detect rapid shifts, and Telegram-alert on them (CL-r1ep).

    Fail-soft BY DESIGN: prediction markets are advisory corroboration,
    not pipeline plumbing — any failure (Gamma outage, missing table,
    empty config) logs and returns, never kills the cycle. Returns the
    constructed :class:`PolymarketSignal` (or None on failure) so the
    caller can hand it to the digest for per-theme corroboration.
    """
    from sqlalchemy import create_engine  # noqa: PLC0415

    from src.events.polymarket_signal import (  # noqa: PLC0415
        PolymarketSignal,
        load_tracked_markets,
    )

    engine = create_engine(build_db_url())
    signal = PolymarketSignal(engine)
    markets = load_tracked_markets(args.poly_config)
    if not markets:
        logger.info("poly: no tracked geo markets configured; skipping poll")
        return signal
    signal.poll_probabilities(markets)
    shifts = signal.detect_shifts()
    if shifts:
        sent = signal.notify_shifts(shifts)
        logger.info("poly: %d shift(s) detected, %d alerted", len(shifts), sent)
    else:
        logger.info("poly: %d markets polled, no shifts", len(markets))
    return signal


def _cycle(args: argparse.Namespace) -> None:
    # CL-3lga: per-phase wall-clock so 48h of logs can rank the P0 work
    # (which phase actually dominates a cycle) without a profiler attached.
    _t_cycle = time.monotonic()
    _phase_ms: dict[str, float] = {}

    if args.ingest:
        from src.data.gdelt import GdeltIngester  # noqa: PLC0415

        _t = time.monotonic()
        ingester = GdeltIngester(build_db_url(), playbooks_path=args.playbooks)
        end = datetime.now(UTC)
        start = end - timedelta(minutes=args.lookback_minutes)
        rows = ingester.run(start, end)
        _phase_ms["ingest"] = (time.monotonic() - _t) * 1000.0
        logger.info("ingest: %d new geo_events rows", rows)

    if args.scan:
        # RVOL scan runs BEFORE assess/digest so this cycle's digest
        # can annotate Watch tickers with fresh marks. A scan failure
        # (yfinance outage, missing table) must never kill the cycle
        # — RVOL is advisory confirmation, not pipeline plumbing.
        _t = time.monotonic()
        try:
            from src.scanners.relative_volume import (  # noqa: PLC0415
                RelativeVolumeScanner,
            )

            scanner = RelativeVolumeScanner(
                build_db_url(),
                playbooks_path=args.playbooks,
            )
            scan_rows = scanner.scan()
            logger.info(
                "scan: %d tickers, %d unusual",
                len(scan_rows),
                sum(1 for r in scan_rows if r.is_unusual),
            )
        except Exception:
            logger.exception("volume scan failed; continuing")
        _phase_ms["scan"] = (time.monotonic() - _t) * 1000.0

    if args.assess:
        from sqlalchemy import create_engine  # noqa: PLC0415

        from src.events.impact_agent import (  # noqa: PLC0415
            DEFAULT_MODEL,
            EventImpactAgent,
        )

        _t_assess = time.monotonic()
        engine = create_engine(build_db_url())
        agent = EventImpactAgent(
            engine=engine,
            model=args.model or DEFAULT_MODEL,
            playbooks_path=args.playbooks,
        )
        # CL-esyo: X watchlist posts and GDELT headlines now share this
        # one NEW queue. assess_new_events orders by seen_at DESC, so a
        # fresh X post (recent seen_at) naturally jumps ahead of any
        # GDELT backlog — no separate path, same impact agent.
        results = agent.assess_new_events(limit=args.limit)
        for result in results:
            logger.info("%s", result.summary_line())
        assessed = sum(1 for r in results if r.status == "ASSESSED")
        logger.info(
            "assess: %d processed (%d assessed, %d dismissed)",
            len(results),
            assessed,
            len(results) - assessed,
        )

        # Multi-hop niche/asymmetry pass (CL-u2ph) BEFORE enrichment so
        # merged niche ideas flow through the same ledger/digest path.
        # High-urgency ASSESSED events only (quota discipline); fail-soft.
        if args.niche:
            _t_niche = time.monotonic()
            try:
                niche_min = getattr(args, "niche_min_urgency", None)
                if niche_min is None:
                    niche_min = _resolve_niche_min_urgency()
                _niche_step(engine, results, niche_min)
            except Exception:
                logger.exception("niche step failed; continuing")
            _phase_ms["niche"] = (time.monotonic() - _t_niche) * 1000.0

        # Idea ledger + one price batch per cycle (CL-mgcp) — fail-soft.
        prices, seen_ats = _enrich_and_persist(engine, results)

        # Prediction-market poll (CL-r1ep) runs after assess so the
        # digest can cite fresh per-theme probs as corroboration. A poly
        # failure NEVER kills the cycle — it's advisory, not plumbing.
        poly_signal = None
        if args.poly:
            try:
                poly_signal = _poly_step(args)
            except Exception:
                logger.exception("poly step failed; continuing")

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
                marks = fetch_volume_marks(engine)
                disp = send_digest(
                    results,
                    min_urgency=args.digest_min_urgency,
                    volume_marks=marks,
                    prices=prices,
                    seen_ats=seen_ats,
                    poly_signal=poly_signal,
                )
            except Exception:
                logger.exception("digest dispatch failed; continuing")
            else:
                if disp is not None and disp.any_attempted:
                    logger.info(
                        "digest: sent (telegram ok=%s)",
                        disp.telegram_succeeded,
                    )

        # assess phase spans LLM assessment + niche + enrich + digest; the
        # niche sub-phase is tracked separately above.
        _phase_ms["assess"] = (time.monotonic() - _t_assess) * 1000.0
    elif args.poly:
        # --poly without --assess: still poll + alert on shifts (the
        # notifications are the point), just no digest corroboration.
        try:
            _poly_step(args)
        except Exception:
            logger.exception("poly step failed; continuing")

    # CL-3lga: one compact phase-timing line per cycle. Ranks where wall time
    # goes (ingest vs assess vs niche) across the 48h measurement window.
    _phase_ms["total"] = (time.monotonic() - _t_cycle) * 1000.0
    logger.info(
        "cycle timing: %s",
        " ".join(f"{name}={ms:.0f}ms" for name, ms in _phase_ms.items()),
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
    args.niche_min_urgency = _resolve_niche_min_urgency()

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
