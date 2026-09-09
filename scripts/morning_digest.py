"""Morning Telegram digest daemon (CL-ydp8, supersedes CL-lpai).

Sends TWO Telegram messages per trading morning (default 09:15 ET,
MORNING_DIGEST_TIME_ET):

1. "Morning positions" — FULL book: OANDA long AND short with economic
   reading, all Alpaca option positions, closed-last-24h realized P&L,
   balances. Venue read failures render "(unavailable)" — never a
   silently flat digest.
2. "LONG ideas" — the event pipeline's current bullish shopping list
   (pending bullish trade ideas by confidence), independent of what has
   actually been bought.

Usage:
    .venv/bin/python scripts/morning_digest.py --once    # send if due
    .venv/bin/python scripts/morning_digest.py --force   # send now
    .venv/bin/python scripts/morning_digest.py --loop 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)

STATE_PATH = Path(
    os.environ.get("MORNING_DIGEST_STATE", "data/morning_digest_state.json"),
)
_OANDA_BASE = "https://api-fxpractice.oanda.com/v3/accounts"


def _load_last_sent() -> str | None:
    try:
        return str(json.loads(STATE_PATH.read_text()).get("last_sent_ny_date"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning(
            "morning digest: unreadable state %s — treating as never sent",
            STATE_PATH,
            exc_info=True,
        )
        return None


def _oanda_creds() -> tuple[str, str] | None:
    key = os.environ.get("OANDA_API_KEY", "")
    acct = os.environ.get("OANDA_ACCOUNT_ID", "")
    return (key, acct) if key and acct else None


def _fetch_oanda_positions() -> list | None:
    creds = _oanda_creds()
    if creds is None:
        return None
    try:
        from src.execution.oanda_broker import OandaBroker  # noqa: PLC0415

        return OandaBroker(creds[0], creds[1], practice=True).get_positions()
    except Exception:
        logger.warning("morning digest: OANDA positions unavailable", exc_info=True)
        return None


def _fetch_oanda_closed_24h(now: datetime) -> list[dict]:
    """Trades closed in the last 24h with realized P&L (best-effort)."""
    creds = _oanda_creds()
    if creds is None:
        return []
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(
            f"{_OANDA_BASE}/{creds[1]}/trades",
            params={"state": "CLOSED", "count": "50"},
            headers={"Authorization": f"Bearer {creds[0]}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        cut = now - timedelta(hours=24)
        out: list[dict] = []
        for t in resp.json().get("trades", []):
            close_raw = str(t.get("closeTime") or "")[:26]
            try:
                closed = datetime.fromisoformat(close_raw.replace("Z", "")).replace(tzinfo=UTC)
            except ValueError:
                continue
            if closed < cut:
                continue
            units = float(t.get("initialUnits") or 0)
            side = "short" if units < 0 else "long"
            pl = float(t.get("realizedPL") or 0.0)
            out.append(
                {
                    "venue": "OANDA",
                    "desc": f"{t.get('instrument')} {side} {abs(units):,.0f}",
                    "pl": f"{pl:+.2f}",
                }
            )
        return out
    except Exception:
        logger.warning("morning digest: OANDA closed-trades unavailable", exc_info=True)
        return []


def _fetch_oanda_balance() -> str | None:
    creds = _oanda_creds()
    if creds is None:
        return None
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(
            f"{_OANDA_BASE}/{creds[1]}/summary",
            headers={"Authorization": f"Bearer {creds[0]}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        acct = resp.json()["account"]
        return f"${float(acct['balance']):,.2f} (uPL {float(acct['unrealizedPL']):+,.2f})"
    except Exception:
        logger.warning("morning digest: OANDA balance unavailable", exc_info=True)
        return None


def _alpaca_client():  # noqa: ANN202
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        return None
    from src.execution.alpaca_options import AlpacaOptionsClient  # noqa: PLC0415

    return AlpacaOptionsClient(key, secret)


def _fetch_alpaca_positions() -> list | None:
    client = _alpaca_client()
    if client is None:
        return None
    try:
        return client.list_option_positions()
    except Exception:
        logger.warning("morning digest: Alpaca positions unavailable", exc_info=True)
        return None


def _fetch_alpaca_equity() -> str | None:
    client = _alpaca_client()
    if client is None:
        return None
    try:
        acct = client.get_account()
        return f"${float(acct['equity']):,.2f}"
    except Exception:
        logger.warning("morning digest: Alpaca account unavailable", exc_info=True)
        return None


def _fetch_alpaca_closed_24h(engine, now: datetime) -> list[dict]:  # noqa: ANN001
    from src.execution.alpaca_ledger_reporting import closed_performance  # noqa: PLC0415
    from src.monitoring.morning_digest import _describe_occ  # noqa: PLC0415

    try:
        rows = closed_performance(engine, now - timedelta(hours=24))
        out: list[dict] = []
        for r in rows:
            if r["net_realized"] is not None:
                pl_txt = f"${float(r['net_realized']):+,.0f} net"
            else:
                pl_txt = f"${float(r['gross_realized']):+,.0f} gross; costs/net unknown"
            desc = _describe_occ(str(r["symbol"])) if r["book"] == "options" else str(r["symbol"])
            out.append(
                {
                    "venue": "Alpaca",
                    "desc": f"{desc} [fill-verified close]",
                    "pl": pl_txt,
                }
            )
        return out
    except Exception:
        logger.warning("morning digest: Alpaca closed rows unavailable", exc_info=True)
        return []


def _fetch_unsellable(engine) -> list[dict]:  # noqa: ANN001
    """Option rows whose exit is currently BLOCKED for want of a bid
    (CL-p0pe / CL-hptt). These page the operator once and then stay silent,
    so the digest is what keeps a days-stuck position visible.

    Fail-soft: any read problem returns [] — a digest must still send."""
    from sqlalchemy import text  # noqa: PLC0415

    try:
        with engine.connect() as conn:
            return [
                dict(r._mapping)
                for r in conn.execute(
                    text(
                        "SELECT ticker, occ_symbol, exit_reason "
                        "FROM alpaca_option_orders "
                        "WHERE exit_status = 'unsellable' "
                        "ORDER BY ticker",
                    )
                )
            ]
    except Exception:
        logger.warning("morning digest: unsellable lookup failed", exc_info=True)
        return []


def _fetch_bullish_ideas(engine) -> list[dict]:  # noqa: ANN001
    from sqlalchemy import text  # noqa: PLC0415

    try:
        with engine.connect() as conn:
            return [
                dict(r._mapping)
                for r in conn.execute(
                    text("""
                SELECT ticker, action, confidence, preferred_instrument,
                       rationale
                FROM trade_ideas
                WHERE status = 'pending'
                  AND (direction = 'bullish'
                       OR action IN ('buy_calls', 'long'))
            """)
                )
            ]
    except Exception:
        logger.warning("morning digest: bullish ideas unavailable", exc_info=True)
        return []


def _resolve_idea_names(engine, ideas: list[dict]) -> dict[str, str]:  # noqa: ANN001
    """Ticker→company-name map for the LONG-ideas tickers (CL-ikz2) so the
    digest shows "VG (Venture Global, Inc.)". Fail-soft: any problem
    (universe unavailable, DB blip) logs and returns ``{}`` — the digest
    then renders bare tickers, exactly as before."""
    tickers = {str(i.get("ticker") or "").upper() for i in ideas}
    tickers.discard("")
    if not tickers:
        return {}
    try:
        from src.data.symbols import SymbolUniverse  # noqa: PLC0415

        return SymbolUniverse(engine).company_names(tickers)
    except Exception:
        logger.warning("morning digest: ticker-name enrichment unavailable", exc_info=True)
        return {}


def run_once(now: datetime | None = None, *, force: bool = False) -> bool:
    """Send both digests if due; True when they went out."""
    from sqlalchemy import create_engine  # noqa: PLC0415

    from src.events._util import atomic_write_json  # noqa: PLC0415
    from src.monitoring.morning_digest import (  # noqa: PLC0415
        _NY,
        build_long_ideas_digest,
        build_position_digest,
        should_send,
    )
    from src.research.notifications import notify_operator  # noqa: PLC0415

    now = now or datetime.now(UTC)
    send_time = os.environ.get(
        "MORNING_DIGEST_TIME_ET",
        os.environ.get("LONG_DIGEST_TIME_ET", "09:15"),
    )
    if not force and not should_send(now, _load_last_sent(), send_time):
        return False

    engine = create_engine(build_db_url())
    balances: dict[str, str] = {}
    if (b := _fetch_oanda_balance()) is not None:
        balances["OANDA"] = b
    if (e := _fetch_alpaca_equity()) is not None:
        balances["Alpaca"] = e

    positions_body = build_position_digest(
        _fetch_oanda_positions(),
        _fetch_alpaca_positions(),
        closed_24h=(_fetch_oanda_closed_24h(now) + _fetch_alpaca_closed_24h(engine, now)),
        balances=balances or None,
        now=now,
        unsellable=_fetch_unsellable(engine),
    )
    bullish_ideas = _fetch_bullish_ideas(engine)
    ideas_body = build_long_ideas_digest(
        bullish_ideas,
        now,
        names=_resolve_idea_names(engine, bullish_ideas),
    )

    r1 = notify_operator("☀️ Morning positions", positions_body, html=True)
    r2 = notify_operator("📈 LONG ideas (event-driven)", ideas_body, html=True)
    if not (r1.any_succeeded and r2.any_succeeded):
        # State is only saved on FULL success — a partial failure retries
        # next cycle (possible duplicate of the half that went through;
        # preferable to silently losing a morning).
        logger.warning(
            "morning digest: send failed (positions=%s ideas=%s) — will retry next cycle",
            r1.any_succeeded,
            r2.any_succeeded,
        )
        return False
    atomic_write_json(
        STATE_PATH,
        {"last_sent_ny_date": now.astimezone(_NY).date().isoformat()},
    )
    logger.info("morning digest: sent both for %s", now.astimezone(_NY).date().isoformat())
    return True


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    parser = argparse.ArgumentParser(description="Morning Telegram digests.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="send immediately, ignore time/dedup gates"
    )
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.loop:
        logger.info(
            "morning digest: looping every %ds (send at %s ET)",
            args.loop,
            os.environ.get("MORNING_DIGEST_TIME_ET", "09:15"),
        )
        try:
            while True:
                try:
                    run_once()
                except Exception:
                    logger.exception("morning digest: cycle failed — retrying")
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("morning digest: stopped")
        return 0

    sent = run_once(force=args.force)
    print("sent" if sent else "not due (use --force to send now)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
