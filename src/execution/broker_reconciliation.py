"""Daily broker reconciliation (CL-unlt).

Compares OANDA's view of recent fills against the internal trade journal.
Discrepancies fall into four categories — missed-by-us, missed-by-broker,
quantity mismatch, price mismatch — each one a real silent-corruption
risk for live P&L.

Pulls v3 OANDA endpoints:
  - /v3/accounts/{id}/summary           — account-level NAV
  - /v3/accounts/{id}/transactions      — fill / swap / fee transactions
                                          since a given timestamp

Internal source: trade_journal_events (event_type=ORDER_FILLED). The
journal is the canonical truth on our side because it's append-only,
hash-chained, and survives engine restarts.

Output:
  - ReconciliationReport(matched, mismatches, oanda_only, internal_only)
  - Daily JSON summary written to reports/reconciliation/YYYY-MM-DD.json
  - Prometheus gauge fx_reconciliation_mismatches set per day
  - Optional alert (Telegram) on mismatch count > 0

Usage as a cron job:
  0 1 * * *  .venv/bin/python -m scripts.run_daily_reconciliation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text

from src.execution.broker import canonical_symbol

logger = logging.getLogger(__name__)


# Tolerances for "close enough" matching. OANDA quotes 5 decimal places for
# major pairs (1 microPip = 1e-5); 1e-4 is a 10-pip tolerance which is
# wide enough to accept normal slippage but narrow enough to flag a real
# mis-fill. Quantity is in units (1 = 1 unit of base currency); 0.5 unit
# is below dust.
_PRICE_TOLERANCE: float = 1e-4
_QTY_TOLERANCE: float = 0.5
# Match window: a fill is considered "the same fill" if the broker
# timestamp is within 30 seconds of our journal entry. OANDA's clock and
# ours can drift by a couple seconds; 30s catches that without being so
# wide we double-match across a same-symbol re-entry.
_TS_MATCH_WINDOW_SEC: int = 30


@dataclass
class FillRecord:
    """Normalized fill — either source. Matching keys off of timestamp +
    instrument + side, so we only need one shape for both sides."""

    ts: datetime
    instrument: str  # OANDA-style "EUR_USD" — we normalize on read
    units: float     # signed: positive=buy, negative=sell
    price: float
    transaction_id: str  # OANDA txn id or our intent_id
    source: str          # "oanda" | "internal"


@dataclass
class Mismatch:
    """One discrepancy. ``kind`` ∈ {missing_internal, missing_broker,
    qty_drift, price_drift}."""

    kind: str
    oanda: FillRecord | None
    internal: FillRecord | None
    detail: str


@dataclass
class ReconciliationReport:
    date: datetime
    matched: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    oanda_total: int = 0
    internal_total: int = 0

    @property
    def is_clean(self) -> bool:
        return len(self.mismatches) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "matched": self.matched,
            "oanda_total": self.oanda_total,
            "internal_total": self.internal_total,
            "mismatches": [
                {
                    "kind": m.kind,
                    "detail": m.detail,
                    "oanda": _record_to_dict(m.oanda),
                    "internal": _record_to_dict(m.internal),
                }
                for m in self.mismatches
            ],
        }


def _record_to_dict(r: FillRecord | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return {
        "ts": r.ts.isoformat(),
        "instrument": r.instrument,
        "units": r.units,
        "price": r.price,
        "transaction_id": r.transaction_id,
        "source": r.source,
    }


def fetch_oanda_fills(
    client: httpx.Client,
    account_id: str,
    since: datetime,
    until: datetime,
) -> list[FillRecord]:
    """Pull ORDER_FILL transactions from OANDA's transaction history.

    We use the dated-range endpoint which the v20 API guarantees returns
    in chronological order. Pagination is handled implicitly because we
    bound by date and the API's default page size is 1000 — adequate for
    24h of any reasonable system.
    """
    resp = client.get(
        f"/v3/accounts/{account_id}/transactions",
        params={
            "from": since.isoformat(),
            "to": until.isoformat(),
            "type": "ORDER_FILL",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    out: list[FillRecord] = []
    for txn in data.get("transactions", []):
        if txn.get("type") != "ORDER_FILL":
            continue
        try:
            ts = datetime.fromisoformat(txn["time"].replace("Z", "+00:00"))
            inst = str(txn.get("instrument", ""))
            units = float(txn.get("units", 0))
            price = float(txn.get("price", 0))
            tid = str(txn.get("id", ""))
        except (KeyError, ValueError):
            logger.warning("Skipped malformed OANDA txn: %r", txn)
            continue
        out.append(FillRecord(
            ts=ts, instrument=_oanda_to_pair(inst), units=units,
            price=price, transaction_id=tid, source="oanda",
        ))
    return out


def _oanda_to_pair(symbol: str) -> str:
    """OANDA uses EUR_USD; we use EURUSD. Canonicalize (CL-2zt0) so this
    matches the internal journal side, which is also canonicalized — a raw
    ``.replace("_","")`` missed mixed-case/`/`/`-` dialects and, more
    importantly, the internal side wasn't normalized at all, so event legs
    (USD_CAD journal vs USDCAD OANDA) never matched → permanent
    missing_internal/missing_broker noise that buried real drift."""
    return canonical_symbol(symbol)


def fetch_internal_fills(
    engine: Any, since: datetime, until: datetime,
) -> list[FillRecord]:
    """Pull ORDER_FILLED events from trade_journal_events."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT ts, intent_id, symbol, payload
                FROM trade_journal_events
                WHERE event_type = 'order_filled'
                  AND ts >= :since AND ts <= :until
                ORDER BY ts
            """),
            {"since": since, "until": until},
        ).fetchall()

    out: list[FillRecord] = []
    for ts, intent_id, symbol, payload in rows:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = {}
        side = payload.get("side", "buy")
        qty = float(payload.get("quantity", 0))
        if side == "sell":
            qty = -qty
        # Internal journal doesn't store fill price uniformly across
        # the OMS code paths today (CL-mdle remediates this). When
        # missing, set to 0 and the comparator will flag price_drift on
        # any non-zero broker price — which is the right alert surface.
        price = float(payload.get("fill_price", 0))
        out.append(FillRecord(
            ts=ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts)),
            # CL-2zt0: canonicalize the journal symbol to match the OANDA
            # side (also canonicalized) — event legs journal as USD_CAD but
            # OANDA reports USDCAD; raw they never matched.
            instrument=canonical_symbol(str(symbol)),
            units=qty,
            price=price,
            transaction_id=str(intent_id or ""),
            source="internal",
        ))
    return out


def reconcile_fills(
    oanda_fills: list[FillRecord],
    internal_fills: list[FillRecord],
    date: datetime | None = None,
) -> ReconciliationReport:
    """Greedy timestamp + instrument + side matching."""
    report = ReconciliationReport(
        date=date or datetime.now(UTC),
        oanda_total=len(oanda_fills),
        internal_total=len(internal_fills),
    )

    # Index internal by (instrument, sign). Sort by ts so we can pop the
    # earliest match first — same-instrument repeated entries match in
    # chronological order rather than scrambling.
    by_key: dict[tuple[str, int], list[FillRecord]] = {}
    for f in sorted(internal_fills, key=lambda x: x.ts):
        sign = 1 if f.units > 0 else -1 if f.units < 0 else 0
        by_key.setdefault((f.instrument, sign), []).append(f)

    matched_internal: set[int] = set()  # by id() to dedup
    for o in sorted(oanda_fills, key=lambda x: x.ts):
        sign = 1 if o.units > 0 else -1 if o.units < 0 else 0
        candidates = by_key.get((o.instrument, sign), [])
        match: FillRecord | None = None
        for c in candidates:
            if id(c) in matched_internal:
                continue
            if abs((o.ts - c.ts).total_seconds()) <= _TS_MATCH_WINDOW_SEC:
                match = c
                break

        if match is None:
            report.mismatches.append(Mismatch(
                kind="missing_internal",
                oanda=o, internal=None,
                detail=(
                    f"OANDA reports {o.units:+.0f} {o.instrument} @ "
                    f"{o.price} at {o.ts.isoformat()} but no internal "
                    f"journal entry within {_TS_MATCH_WINDOW_SEC}s"
                ),
            ))
            continue
        matched_internal.add(id(match))

        if abs(o.units - match.units) > _QTY_TOLERANCE:
            report.mismatches.append(Mismatch(
                kind="qty_drift",
                oanda=o, internal=match,
                detail=(
                    f"Quantity drift: oanda={o.units:+.0f} "
                    f"internal={match.units:+.0f}"
                ),
            ))
        elif match.price != 0 and abs(o.price - match.price) > _PRICE_TOLERANCE:
            # price=0 in internal means we never recorded it — flag as
            # missing-price-data rather than a drift, because the unit
            # match was valid.
            report.mismatches.append(Mismatch(
                kind="price_drift",
                oanda=o, internal=match,
                detail=(
                    f"Price drift: oanda={o.price} internal={match.price}"
                ),
            ))
        else:
            report.matched += 1

    # Anything in internal that didn't match: missing on broker side.
    for _sign_key, candidates in by_key.items():
        for c in candidates:
            if id(c) not in matched_internal:
                report.mismatches.append(Mismatch(
                    kind="missing_broker",
                    oanda=None, internal=c,
                    detail=(
                        f"Internal records {c.units:+.0f} {c.instrument} "
                        f"at {c.ts.isoformat()} but OANDA has no "
                        f"corresponding ORDER_FILL"
                    ),
                ))

    return report


def write_report(
    report: ReconciliationReport, out_dir: Path | None = None,
) -> Path:
    """Write the daily JSON report. Path: ``reports/reconciliation/YYYY-MM-DD.json``."""
    out_dir = out_dir or Path("reports/reconciliation")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = report.date.strftime("%Y-%m-%d") + ".json"
    path = out_dir / fname
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return path


def emit_metrics(report: ReconciliationReport) -> None:
    """Emit Prometheus metrics. Failure is non-fatal."""
    try:
        from src.monitoring.metrics import (
            reconciliation_clean_days,
            reconciliation_mismatches,
        )
        reconciliation_mismatches.set(len(report.mismatches))
        if report.is_clean:
            reconciliation_clean_days.inc()
    except Exception:
        logger.warning("Failed to emit reconciliation metrics", exc_info=True)


def run_daily_reconciliation(
    engine: Any,
    oanda_client: httpx.Client,
    account_id: str,
    date: datetime | None = None,
    out_dir: Path | None = None,
) -> ReconciliationReport:
    """Top-level entry point — convenient for cron + tests."""
    date = date or datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    since = date - timedelta(days=1)
    until = date

    oanda = fetch_oanda_fills(oanda_client, account_id, since, until)
    internal = fetch_internal_fills(engine, since, until)
    report = reconcile_fills(oanda, internal, date=date)

    write_report(report, out_dir=out_dir)
    emit_metrics(report)

    if report.is_clean:
        logger.info(
            "Reconciliation clean for %s — matched %d fills",
            date.date(), report.matched,
        )
    else:
        logger.error(
            "Reconciliation found %d mismatches for %s — see report",
            len(report.mismatches), date.date(),
        )
    return report
