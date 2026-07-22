"""Free options-activity scanner (CL-mtum) — the confirmation pillar's
options leg, built on yfinance chain data at zero cost.

Once daily (near the close — intraday/after-hours chain data is spotty), pull
each candidate ticker's option chain across the nearest expiries and snapshot:

  * call/put VOLUME (→ put/call ratio, the positioning-direction read),
  * call/put OPEN INTEREST,
  * nearest-expiry ATM implied vol,

into ``options_activity`` (migration 015). The table becomes its own baseline:
after ~5+ snapshots, "today's total options volume is 3.2× its baseline" is
the unusual-activity signal, and :func:`activity_note` renders a compact
confirmation line for idea grounding.

HONEST SCOPE (documented everywhere): delayed aggregate positioning — shows
THAT options are active and which way volume skews, NOT sweeps or aggressor
side. That needs paid tick-level data; when/if the operator buys it (Unusual
Whales / OPRA), the SOURCE swaps and this table + consumers stay.

Chain fetch is injectable — no live yfinance in unit tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: Nearest expiries aggregated per snapshot (weeklies + front month).
DEFAULT_MAX_EXPIRIES = 4
#: Snapshots needed before a baseline/unusualness read is offered.
MIN_BASELINE_SNAPSHOTS = 5
#: Baseline window (snapshots, i.e. trading days once the daemon runs daily).
BASELINE_WINDOW = 20

#: Injectable chain fetch: ticker → ChainSummary-ish dict or None.
ChainFn = Callable[[str], dict[str, Any] | None]


@dataclass(frozen=True)
class ChainSummary:
    ticker: str
    call_volume: int
    put_volume: int
    call_oi: int
    put_oi: int
    atm_iv: float | None
    expiries_sampled: int

    @property
    def total_volume(self) -> int:
        return self.call_volume + self.put_volume

    @property
    def pc_volume_ratio(self) -> float:
        return self.put_volume / max(self.call_volume, 1)


def yfinance_chain_summary(
    ticker: str, max_expiries: int = DEFAULT_MAX_EXPIRIES,
) -> dict[str, Any] | None:
    """Aggregate the nearest expiries' chains via yfinance. Fail-soft → None.

    ATM IV comes from the nearest expiry's closest-to-spot call; yfinance
    sometimes reports degenerate IVs (~1e-5) off-hours — those become None
    rather than a lie.
    """
    try:
        import yfinance as yf  # noqa: PLC0415 — deferred
        t = yf.Ticker(ticker)
        expiries = list(t.options or [])[:max_expiries]
        if not expiries:
            return None
        cv = pv = coi = poi = 0
        atm_iv: float | None = None
        try:
            spot = float(t.fast_info["last_price"])
        except Exception:
            spot = 0.0
        for i, exp in enumerate(expiries):
            ch = t.option_chain(exp)
            calls, puts = ch.calls, ch.puts
            cv += int(calls["volume"].fillna(0).sum())
            pv += int(puts["volume"].fillna(0).sum())
            coi += int(calls["openInterest"].fillna(0).sum())
            poi += int(puts["openInterest"].fillna(0).sum())
            if i == 0 and spot > 0 and len(calls):
                atm = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
                iv = float(atm["impliedVolatility"].iloc[0])
                atm_iv = iv if iv > 0.005 else None  # degenerate off-hours IV
        return {
            "ticker": ticker, "call_volume": cv, "put_volume": pv,
            "call_oi": coi, "put_oi": poi, "atm_iv": atm_iv,
            "expiries_sampled": len(expiries),
        }
    except Exception:
        logger.debug("options activity: chain fetch failed for %s", ticker,
                     exc_info=True)
        return None


def _to_summary(raw: dict[str, Any]) -> ChainSummary | None:
    try:
        return ChainSummary(
            ticker=str(raw["ticker"]),
            call_volume=int(raw.get("call_volume") or 0),
            put_volume=int(raw.get("put_volume") or 0),
            call_oi=int(raw.get("call_oi") or 0),
            put_oi=int(raw.get("put_oi") or 0),
            atm_iv=(float(raw["atm_iv"]) if raw.get("atm_iv") is not None
                    else None),
            expiries_sampled=int(raw.get("expiries_sampled") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


_UPSERT = text("""
    INSERT INTO options_activity
        (obs_date, ticker, call_volume, put_volume, call_oi, put_oi,
         pc_volume_ratio, atm_iv, expiries_sampled, source, created_at)
    VALUES (:obs_date, :ticker, :cv, :pv, :coi, :poi, :pc, :iv, :n,
            'yfinance', :now)
    ON CONFLICT (obs_date, ticker) DO UPDATE SET
        call_volume = excluded.call_volume,
        put_volume = excluded.put_volume,
        call_oi = excluded.call_oi,
        put_oi = excluded.put_oi,
        pc_volume_ratio = excluded.pc_volume_ratio,
        atm_iv = excluded.atm_iv,
        expiries_sampled = excluded.expiries_sampled,
        created_at = excluded.created_at
""")


def snapshot_tickers(
    engine: Any,
    tickers: Sequence[str],
    chain_fn: ChainFn | None = None,
    obs_date: date | None = None,
) -> dict[str, int]:
    """One daily snapshot pass over ``tickers``. Fail-soft per ticker.
    Returns {"written", "skipped"}."""
    fetch = chain_fn or yfinance_chain_summary
    obs = obs_date or datetime.now(UTC).date()
    now = datetime.now(UTC)
    written = skipped = 0
    for raw_ticker in dict.fromkeys(tickers):  # dedup, keep order
        ticker = str(raw_ticker).strip().upper()
        if not ticker or "_" in ticker:  # OANDA ids have no equity chains
            skipped += 1
            continue
        raw = fetch(ticker)
        summary = _to_summary(raw) if raw else None
        if summary is None or summary.total_volume <= 0:
            skipped += 1
            continue
        try:
            with engine.begin() as conn:
                conn.execute(_UPSERT, {
                    "obs_date": obs, "ticker": ticker,
                    "cv": summary.call_volume, "pv": summary.put_volume,
                    "coi": summary.call_oi, "poi": summary.put_oi,
                    "pc": summary.pc_volume_ratio, "iv": summary.atm_iv,
                    "n": summary.expiries_sampled, "now": now,
                })
            written += 1
        except Exception:
            logger.warning("options activity: DB write failed for %s",
                           ticker, exc_info=True)
            skipped += 1
    logger.info("options activity: %d written, %d skipped (obs %s)",
                written, skipped, obs)
    return {"written": written, "skipped": skipped}


def volume_unusualness(
    engine: Any, ticker: str, obs_date: date | None = None,
) -> float | None:
    """Today's total volume ÷ its prior-snapshot baseline, or None when
    fewer than MIN_BASELINE_SNAPSHOTS priors exist (no fake baselines)."""
    obs = obs_date or datetime.now(UTC).date()
    try:
        with engine.connect() as conn:
            today = conn.execute(text(
                "SELECT call_volume + put_volume FROM options_activity "
                "WHERE ticker = :t AND obs_date = :d",
            ), {"t": ticker.upper(), "d": obs}).scalar()
            if today is None:
                return None
            rows = conn.execute(text(
                "SELECT call_volume + put_volume FROM options_activity "
                "WHERE ticker = :t AND obs_date < :d "
                "ORDER BY obs_date DESC LIMIT :w",
            ), {"t": ticker.upper(), "d": obs, "w": BASELINE_WINDOW}).all()
    except Exception:
        logger.debug("options activity: unusualness query failed for %s",
                     ticker, exc_info=True)
        return None
    priors = [float(r[0]) for r in rows if r[0]]
    if len(priors) < MIN_BASELINE_SNAPSHOTS:
        return None
    baseline = sum(priors) / len(priors)
    return float(today) / baseline if baseline > 0 else None


def activity_note(
    engine: Any, ticker: str, obs_date: date | None = None,
) -> str | None:
    """Compact confirmation line for grounding/notes, or None when no
    snapshot exists. E.g. 'options: P/C 0.44 (call-skewed), vol 3.1x baseline'.
    """
    obs = obs_date or datetime.now(UTC).date()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT pc_volume_ratio, call_volume, put_volume, atm_iv "
                "FROM options_activity WHERE ticker = :t "
                "AND obs_date <= :d ORDER BY obs_date DESC LIMIT 1",
            ), {"t": ticker.upper(), "d": obs}).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    pc = float(row[0]) if row[0] is not None else None
    skew = ""
    if pc is not None:
        skew = " (call-skewed)" if pc < 0.7 else (
            " (put-skewed)" if pc > 1.4 else "")
    parts = [f"options: P/C {pc:.2f}{skew}" if pc is not None else "options:"]
    unusual = volume_unusualness(engine, ticker, obs)
    if unusual is not None:
        parts.append(f"vol {unusual:.1f}x baseline")
    if row[3] is not None:
        parts.append(f"ATM IV {float(row[3]) * 100:.0f}%")
    return ", ".join(parts)
