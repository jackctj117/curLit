#!/usr/bin/env python3
"""Standalone soak-test dashboard — read-only HTML view of soak health.

Runs independently of the engine so it can be started/restarted without
disturbing the running soak. Reads:
    - logs/soak_test.jsonl    monitor samples (memory, equity, pid)
    - logs/soak_engine_*.log  engine log (recent errors)
    - Postgres tables          journal + snapshot row counts
    - psutil                  live engine PID + RSS

Usage:
    .venv/bin/python scripts/soak_dashboard.py
    open http://127.0.0.1:8201/
"""

from __future__ import annotations

import glob
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.research.dashboard_panel import (
    DecisionError,
    apply_decision,
    list_pending_approvals,
)
from src.research.loop import DEFAULT_STATE_PATH, load_state

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs"
SOAK_LOG = LOG_DIR / "soak_test.jsonl"

app = FastAPI(title="curLit Soak Dashboard", version="0.1.0")


# Lazy-initialized DB engine so dashboard polling doesn't create a fresh
# SQLAlchemy engine (with its own connection pool) per request — that
# accumulates connections until Postgres rejects with "too many clients
# already" (CL-xn6k). One engine, reused across all polls.
_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.environ.get(
            "DATABASE_URL",
            f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'fx')}:"
            f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
            f"{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/"
            f"{os.environ.get('POSTGRES_DB', 'fx')}",
        )
        # Small pool — dashboard is read-only and low-frequency. Without
        # the explicit cap a default pool of 5+10 would be wasteful.
        _engine = create_engine(url, pool_size=2, max_overflow=0, pool_pre_ping=True)
    return _engine


def _find_engine_pid() -> int | None:
    """Return PID of the currently-running engine python process.

    Filters out the parent zsh wrapper (whose cmdline contains the eval'd
    python invocation as a string) by matching on proc.name() == python.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info["name"] or "").lower()
            cmd = " ".join(proc.info["cmdline"] or [])
        except Exception:
            continue
        if "src.runtime.run_engine" in cmd and "python" in name:
            return int(proc.info["pid"])
    return None


def _engine_runtime(pid: int) -> dict[str, Any]:
    """psutil-derived runtime info for the engine process."""
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            return {
                "memory_mb": round(proc.memory_info().rss / 1e6, 1),
                "cpu_pct": round(proc.cpu_percent(interval=0.0), 1),
                "uptime_sec": int(time.time() - proc.create_time()),
                "n_threads": proc.num_threads(),
                "n_fds": (
                    proc.num_fds() if hasattr(proc, "num_fds")
                    else len(proc.open_files())
                ),
            }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _read_samples() -> list[dict[str, Any]]:
    """Read all soak samples — caps at last 500 to keep payloads bounded."""
    if not SOAK_LOG.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in SOAK_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out[-500:]


def _recent_errors(n: int = 20) -> list[str]:
    """Tail the most-recent engine log for error/critical lines."""
    logs = sorted(glob.glob(str(LOG_DIR / "soak_engine_2026*.log")))
    if not logs:
        return []
    try:
        tail = Path(logs[-1]).read_text().splitlines()[-2000:]
    except Exception:
        return []
    keywords = ("ERROR", "CRITICAL", "Traceback")
    matches = [line for line in tail if any(kw in line for kw in keywords)]
    return matches[-n:]


def _engine_api_get(path: str) -> Any:
    """Proxy GET to the engine's web API on :8200. Returns parsed JSON or
    None on any failure (engine down, secret mismatch, transport error).
    """
    secret = os.environ.get("WEB_API_SECRET", "curlit-dev")
    try:
        r = httpx.get(
            f"http://127.0.0.1:8200{path}",
            params={"secret": secret}, timeout=2.0,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _account_state() -> dict[str, Any]:
    """Equity / margin from broker via engine's web API."""
    data = _engine_api_get("/api/account")
    if data is None:
        return {"error": "engine api unreachable"}
    return data


def _positions() -> list[dict[str, Any]]:
    """Live positions from broker via engine's web API."""
    data = _engine_api_get("/api/positions")
    return data if isinstance(data, list) else []


def _strategies() -> list[dict[str, Any]]:
    """Configured strategies + their symbols."""
    data = _engine_api_get("/api/signals")
    if not data or "strategies" not in data:
        return []
    return list(data["strategies"])


def _system_status() -> dict[str, Any]:
    """Engine running flag + OMS halted flag."""
    data = _engine_api_get("/api/system")
    return data or {"error": "unreachable"}


def _recent_trades(n: int = 15) -> list[dict[str, Any]]:
    """Last N trade-journal events with the bits a human cares about.

    Each row distills payload fields most useful for "what's happening" —
    target_position / delta for INTENT_SUBMITTED, side+quantity for
    ORDER_PLACED/FILLED, rejection_class for ORDER_REJECTED, summary for
    RECONCILIATION_REPORT.
    """
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT seq, ts, event_type, intent_id, strategy_id, symbol, "
                    "  payload "
                    "FROM trade_journal_events ORDER BY seq DESC LIMIT :n"
                ),
                {"n": n},
            ).fetchall()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        seq, ts, etype, iid, sid, sym, payload = r
        # Postgres returns JSONB as dict; normalize for safety.
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        # Pull the most-relevant payload fields per event type into a
        # single short string — table cells stay readable.
        detail_bits: list[str] = []
        if etype == "intent_submitted":
            if "delta" in payload:
                detail_bits.append(f"Δ={payload['delta']:+.2f}")
            if "target_position" in payload:
                detail_bits.append(f"target={payload['target_position']:+.2f}")
        elif etype in ("order_placed", "order_filled", "order_partial_fill"):
            if "side" in payload and "quantity" in payload:
                detail_bits.append(f"{payload['side']} {payload['quantity']:.2f}")
        elif etype == "order_rejected":
            if "rejection_class" in payload:
                detail_bits.append(payload["rejection_class"])
        elif etype == "reconciliation_report":
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if summary:
                matched = summary.get("matched", 0)
                total = sum(summary.values())
                detail_bits.append(f"{matched}/{total} matched")
        out.append({
            "seq": seq,
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "event_type": etype,
            "intent_id": (iid[:8] + "…") if iid else "",
            "strategy_id": sid or "",
            "symbol": sym or "",
            "detail": " ".join(detail_bits),
        })
    return out


def _db_counts() -> dict[str, Any]:
    """Counts on trade_journal_events + feature_snapshots."""
    try:
        eng = _get_engine()
        out: dict[str, Any] = {}
        with eng.connect() as conn:
            for table in ("trade_journal_events", "feature_snapshots"):
                try:
                    out[table] = int(
                        conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
                    )
                except Exception:
                    out[table] = None
            # Most recent journal event type — proves the audit trail is moving.
            try:
                row = conn.execute(text(
                    "SELECT event_type, ts FROM trade_journal_events "
                    "ORDER BY seq DESC LIMIT 1"
                )).fetchone()
                if row is not None:
                    out["latest_event"] = {
                        "type": row[0],
                        "ts": row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1]),
                    }
            except Exception:
                pass
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _verdict(
    engine_alive: bool,
    monitor_age_sec: float | None,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Top-level health rollup: green/yellow/red with reason."""
    if not engine_alive:
        return {"status": "RED", "reason": "engine process not running"}
    if monitor_age_sec is None:
        return {"status": "YELLOW", "reason": "no soak samples yet"}
    if monitor_age_sec > 1500:  # 25 min
        return {"status": "RED", "reason": f"monitor stale {int(monitor_age_sec)}s"}
    if monitor_age_sec > 900:   # 15 min
        return {"status": "YELLOW", "reason": f"monitor stale {int(monitor_age_sec)}s"}
    if len(samples) >= 2:
        first = samples[0].get("memory_mb", 0)
        last = samples[-1].get("memory_mb", 0)
        if first > 0 and last > first * 2:
            return {"status": "RED", "reason": f"memory doubled ({first}→{last} MB)"}
        if first > 0 and last > first * 1.5:
            return {"status": "YELLOW", "reason": f"memory +{int((last/first - 1) * 100)}%"}
    return {"status": "GREEN", "reason": "all signals nominal"}


# =============================================================================
# JSON API
# =============================================================================


@app.get("/api/soak")
def soak_health() -> dict[str, Any]:
    pid = _find_engine_pid()
    runtime = _engine_runtime(pid) if pid else {}

    samples = _read_samples()
    monitor_age = (
        time.time() - SOAK_LOG.stat().st_mtime
        if SOAK_LOG.exists() else None
    )
    sample_age = None
    if samples:
        try:
            last_ts = datetime.fromisoformat(
                samples[-1]["ts"].replace("Z", "+00:00")
            )
            sample_age = (datetime.now(UTC) - last_ts).total_seconds()
        except Exception:
            pass

    db = _db_counts()
    errors = _recent_errors()

    verdict = _verdict(pid is not None, monitor_age, samples)

    # Trade-activity / account info — fetched via the engine's web API.
    # Returns empty/error fields if the engine API isn't reachable.
    account = _account_state()
    positions = _positions()
    strategies = _strategies()
    system = _system_status()
    recent_trades = _recent_trades(n=15)

    return {
        "now": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "engine": {
            "pid": pid,
            "alive": pid is not None,
            **runtime,
        },
        "monitor": {
            "log_path": str(SOAK_LOG),
            "log_age_sec": int(monitor_age) if monitor_age is not None else None,
            "sample_age_sec": int(sample_age) if sample_age is not None else None,
            "n_samples": len(samples),
        },
        "memory_history": [
            {"ts": s["ts"], "mb": s.get("memory_mb")}
            for s in samples[-100:]
        ],
        "latest_sample": samples[-1] if samples else None,
        "db": db,
        "recent_errors": errors,
        "account": account,
        "positions": positions,
        "strategies": strategies,
        "system": system,
        "recent_trades": recent_trades,
    }


# =============================================================================
# HTML page
# =============================================================================


_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>curLit soak</title>
<style>
  body { font-family: -apple-system, monospace; margin: 0; padding: 1.5em;
         background: #0e1116; color: #d4d4d4; }
  h1 { font-size: 1.2em; margin: 0 0 0.5em 0; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
  .card { background: #1a1f29; padding: 1em 1.2em; border-radius: 6px;
          border-left: 3px solid #444; }
  .card h2 { font-size: 0.85em; margin: 0 0 0.6em 0; color: #888;
             text-transform: uppercase; letter-spacing: 0.05em; }
  .v { font-size: 1.4em; font-weight: 600; }
  .small { color: #888; font-size: 0.85em; }
  .GREEN { border-left-color: #4ade80; }
  .YELLOW { border-left-color: #fbbf24; }
  .RED { border-left-color: #ef4444; }
  .GREEN .v { color: #4ade80; }
  .YELLOW .v { color: #fbbf24; }
  .RED .v { color: #ef4444; }
  pre { background: #0a0d12; padding: 0.6em; border-radius: 4px;
        font-size: 0.78em; max-height: 200px; overflow: auto; margin: 0; }
  .row { display: flex; justify-content: space-between; padding: 0.2em 0;
         border-bottom: 1px solid #2a2f3a; }
  .row:last-child { border-bottom: 0; }
  .row .k { color: #888; }
  #spark { height: 40px; margin-top: 0.5em; }
  #spark rect { fill: #4ade80; }
  .err { color: #f87171; font-size: 0.8em; padding: 0.3em 0; }
  footer { margin-top: 1em; font-size: 0.75em; color: #666; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
  th, td { text-align: left; padding: 0.3em 0.5em; border-bottom: 1px solid #2a2f3a; }
  th { color: #888; font-weight: 500; text-transform: uppercase;
       letter-spacing: 0.04em; font-size: 0.78em; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.pos { color: #4ade80; }
  td.neg { color: #f87171; }
  td.muted { color: #666; }
  .full { grid-column: 1 / -1; }
</style>
</head>
<body>
<h1>curLit soak <span id="ts" class="small"></span></h1>
<div id="root">loading…</div>
<footer>auto-refresh every 10s · <a href="/api/soak" style="color:#888">raw json</a></footer>
<script>
function fmt(n) { return n == null ? "—" : (typeof n === "number" ? n.toLocaleString() : n); }
function fmtDur(sec) {
  if (sec == null) return "—";
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
function row(k, v) { return `<div class="row"><span class="k">${k}</span><span>${v}</span></div>`; }

async function refresh() {
  let d;
  try { d = await fetch("/api/soak").then(r => r.json()); }
  catch (e) { document.getElementById("root").innerHTML =
    `<div class="card RED"><h2>Error</h2><div>${e.message}</div></div>`; return; }

  const v = d.verdict || {};
  const e = d.engine || {};
  const m = d.monitor || {};
  const db = d.db || {};
  const last = d.latest_sample || {};

  document.getElementById("ts").textContent = "· " + (d.now || "").slice(11, 19) + " UTC";

  // Memory sparkline.
  const hist = (d.memory_history || []).map(h => h.mb).filter(x => x != null);
  let spark = "";
  if (hist.length > 1) {
    const min = Math.min(...hist), max = Math.max(...hist);
    const r = Math.max(max - min, 0.01);
    const w = 100 / hist.length;
    spark = `<svg id="spark" width="100%" viewBox="0 0 100 40" preserveAspectRatio="none">` +
      hist.map((y, i) => {
        const h = ((y - min) / r) * 38 + 1;
        return `<rect x="${i * w}" y="${40 - h}" width="${w * 0.85}" height="${h}"/>`;
      }).join("") + `</svg>`;
  }

  // ----- Account / positions / strategies / trades -----
  const acct = d.account || {};
  const positions = d.positions || [];
  const strategies = d.strategies || [];
  const sys_ = d.system || {};
  const trades = d.recent_trades || [];

  const fmtMoney = (n) => n == null ? "—" : "$" + Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
  const fmtNum = (n) => n == null ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: 4 });
  const cls = (n) => n == null ? "muted" : (n > 0 ? "pos" : (n < 0 ? "neg" : "muted"));

  const positionsHtml = positions.length
    ? `<table>
         <thead><tr><th>symbol</th><th>qty</th><th>avg px</th><th>uPnL</th></tr></thead>
         <tbody>${positions.map(p => `
           <tr>
             <td>${p.symbol}</td>
             <td class="num ${cls(p.quantity)}">${fmtNum(p.quantity)}</td>
             <td class="num">${fmtNum(p.avg_price)}</td>
             <td class="num ${cls(p.pnl)}">${fmtMoney(p.pnl)}</td>
           </tr>`).join("")}
         </tbody>
       </table>`
    : `<div class="small">no open positions</div>`;

  const strategiesHtml = strategies.length
    ? `<table>
         <thead><tr><th>id</th><th>symbols</th></tr></thead>
         <tbody>${strategies.map(s => `
           <tr>
             <td>${s.id}</td>
             <td class="muted">${(s.symbols || []).join(", ")}</td>
           </tr>`).join("")}
         </tbody>
       </table>`
    : `<div class="small">engine api unreachable or no strategies</div>`;

  const tradesHtml = trades.length
    ? `<table>
         <thead><tr><th>seq</th><th>ts</th><th>event</th><th>strategy</th><th>symbol</th><th>detail</th></tr></thead>
         <tbody>${trades.map(t => `
           <tr>
             <td class="num muted">${t.seq}</td>
             <td class="muted">${(t.ts || "").slice(11, 19)}</td>
             <td>${t.event_type}</td>
             <td class="muted">${t.strategy_id || "—"}</td>
             <td>${t.symbol || "—"}</td>
             <td class="muted">${t.detail}</td>
           </tr>`).join("")}
         </tbody>
       </table>`
    : `<div class="small">no journal events yet</div>`;

  document.getElementById("root").innerHTML = `
    <div class="grid">
      <div class="card ${v.status || ''}">
        <h2>verdict</h2>
        <div class="v">${v.status || "?"}</div>
        <div class="small">${v.reason || ""}</div>
      </div>
      <div class="card ${e.alive ? 'GREEN' : 'RED'}">
        <h2>engine</h2>
        <div class="v">${e.alive ? "ALIVE" : "DEAD"}</div>
        <div class="small">PID ${fmt(e.pid)} · uptime ${fmtDur(e.uptime_sec)}</div>
        <div class="small">OMS ${sys_.oms_halted ? "HALTED" : "active"}</div>
      </div>
      <div class="card">
        <h2>account · paper</h2>
        <div class="v">${fmtMoney(acct.equity)}</div>
        <div class="small">margin used ${fmtMoney(acct.margin_used)}</div>
        ${acct.error ? `<div class="err">${acct.error}</div>` : ""}
      </div>
      <div class="card">
        <h2>memory · cpu · fds</h2>
        <div class="v">${fmt(e.memory_mb)} MB</div>
        <div class="small">cpu ${fmt(e.cpu_pct)}% · ${fmt(e.n_threads)} threads · ${fmt(e.n_fds)} fds</div>
        ${spark}
      </div>
      <div class="card ${m.sample_age_sec > 900 ? 'YELLOW' : ''}">
        <h2>monitor</h2>
        <div class="v">${fmt(m.n_samples)} samples</div>
        <div class="small">last sample ${fmtDur(m.sample_age_sec)} ago</div>
      </div>
      <div class="card">
        <h2>db rows</h2>
        ${row("trade_journal_events", fmt(db.trade_journal_events))}
        ${row("feature_snapshots", fmt(db.feature_snapshots))}
        ${db.latest_event ? row("latest", db.latest_event.type) : ""}
        ${db.error ? `<div class="err">${db.error}</div>` : ""}
      </div>
      <div class="card full">
        <h2>positions</h2>
        ${positionsHtml}
      </div>
      <div class="card full">
        <h2>strategies</h2>
        ${strategiesHtml}
      </div>
      <div class="card full">
        <h2>recent trade-journal events</h2>
        ${tradesHtml}
      </div>
      <div class="card full" id="approvals-card">
        <h2>pending approvals</h2>
        <div id="approvals-body" class="small">loading…</div>
      </div>
    </div>
    <div class="card" style="margin-top:1em">
      <h2>recent errors (engine log)</h2>
      ${d.recent_errors && d.recent_errors.length
          ? `<pre>${d.recent_errors.map(l => l.replace(/</g, "&lt;")).join("\\n")}</pre>`
          : `<div class="small">none</div>`}
    </div>
  `;
}

// ---- Approvals panel (CL-7t8d) -----------------------------------
async function refreshApprovals() {
  const body = document.getElementById("approvals-body");
  if (!body) return;
  let d;
  try { d = await fetch("/api/approvals").then(r => r.json()); }
  catch (e) {
    body.innerHTML = `<div class="err">${e.message}</div>`; return;
  }
  if (d.error) {
    body.innerHTML = `<div class="err">${d.error}</div>`; return;
  }
  const pending = d.pending || [];
  if (!pending.length) {
    body.innerHTML = `<div class="small">no entries pending operator action</div>`;
    return;
  }
  body.innerHTML = `<table>
    <thead><tr>
      <th>gate</th><th>slug</th><th>since</th><th>detail</th><th></th>
    </tr></thead>
    <tbody>${pending.map(e => `
      <tr id="row-${e.gate}-${e.slug.replace(/[^a-z0-9]/gi,'_')}">
        <td>GATE ${e.gate}</td>
        <td>${e.slug}</td>
        <td class="muted">${(e.pending_since || "").slice(11, 19)}</td>
        <td class="muted">${e.gate === 1
          ? (e.hypothesis_path || "")
          : `${e.bull || ""}/${e.bear || ""} · ${e.transcript_path || ""}`}</td>
        <td>
          <input type="text" placeholder="reason (optional)"
                 id="reason-${e.gate}-${e.slug.replace(/[^a-z0-9]/gi,'_')}"
                 style="width: 12em; background:#0a0d12; color:#d4d4d4;
                        border:1px solid #2a2f3a; padding:0.2em 0.4em;
                        font: inherit; border-radius: 3px;">
          <button onclick="decide(${e.gate}, '${e.slug.replace(/'/g,"\\'")}', 'APPROVE')"
                  style="background:#1f4d1f; color:#4ade80; border:none;
                         padding:0.3em 0.7em; cursor:pointer; border-radius: 3px;">
            APPROVE</button>
          <button onclick="decide(${e.gate}, '${e.slug.replace(/'/g,"\\'")}', 'REJECT')"
                  style="background:#4d1f1f; color:#f87171; border:none;
                         padding:0.3em 0.7em; cursor:pointer; border-radius: 3px;
                         margin-left: 0.3em;">
            REJECT</button>
        </td>
      </tr>`).join("")}
    </tbody></table>`;
}

async function decide(gate, slug, action) {
  const safeSlug = slug.replace(/[^a-z0-9]/gi,'_');
  const reasonEl = document.getElementById(`reason-${gate}-${safeSlug}`);
  const reason = reasonEl ? reasonEl.value : "";
  const secret = (new URLSearchParams(window.location.search)).get("secret") || "curlit-dev";
  try {
    const resp = await fetch(`/api/approvals/${encodeURIComponent(slug)}?secret=${encodeURIComponent(secret)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({gate, action, reason}),
    });
    if (!resp.ok) {
      const detail = await resp.text();
      alert(`error: ${resp.status} ${detail}`);
      return;
    }
  } catch (e) {
    alert(`network error: ${e.message}`); return;
  }
  await refreshApprovals();
}

refresh();
refreshApprovals();
setInterval(refresh, 10000);
setInterval(refreshApprovals, 10000);
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _HTML


# =============================================================================
# Approvals panel (CL-7t8d)
# =============================================================================
#
# Pending GATE 1 (research) + GATE 2 (deploy) entries surface here so
# the operator can approve/skip via the browser instead of running
# scripts/research_approve.py manually. Decisions write through to
# data/research/state.json (the source of truth the loop watches) and
# append a line to data/research/decisions.log for audit.
#
# Auth: same WEB_API_SECRET pattern the rest of the dashboard uses for
# state-mutating endpoints. The list endpoint is read-only and binds
# only to 127.0.0.1 (uvicorn host below) — that's the dashboard's
# existing security boundary.


_DECISIONS_LOG = Path("data/research/decisions.log")


def _check_secret(secret: str) -> None:
    """Validate the WEB_API_SECRET header. Raises HTTPException(401)
    on mismatch. Same pattern the engine API uses (see _engine_api
    above)."""
    expected = os.environ.get("WEB_API_SECRET", "curlit-dev")
    if not secret or secret != expected:
        raise HTTPException(status_code=401, detail="invalid secret")


@app.get("/api/approvals")
def api_approvals() -> dict[str, Any]:
    """Read-only listing of all pending GATE 1 + GATE 2 entries."""
    try:
        state = load_state(DEFAULT_STATE_PATH)
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "pending": []}
    pending = [e.to_json() for e in list_pending_approvals(state)]
    return {"pending": pending}


@app.post("/api/approvals/{slug}")
def api_apply_decision(
    slug: str,
    body: dict[str, Any] = Body(...),  # noqa: B008
    secret: str = Query(""),
) -> dict[str, Any]:
    """Apply an APPROVE/REJECT decision. Body keys: ``gate`` (1|2),
    ``action`` (APPROVE|REJECT), ``reason`` (optional)."""
    _check_secret(secret)
    gate = int(body.get("gate", 0))
    action = str(body.get("action", "")).upper()
    reason = str(body.get("reason", ""))
    try:
        return apply_decision(
            state_path=DEFAULT_STATE_PATH,
            gate=gate, slug=slug, action=action, reason=reason,
            decisions_log=_DECISIONS_LOG,
        )
    except DecisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    # Auto-load .env so WEB_API_SECRET / Postgres creds are available
    # without first sourcing the file. Explicit env vars still win.
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    port = int(os.environ.get("SOAK_DASHBOARD_PORT", "8201"))
    print(f"curLit soak dashboard → http://127.0.0.1:{port}/")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
