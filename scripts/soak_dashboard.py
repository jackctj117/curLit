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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs"
SOAK_LOG = LOG_DIR / "soak_test.jsonl"

app = FastAPI(title="curLit Soak Dashboard", version="0.1.0")


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


def _db_counts() -> dict[str, Any]:
    """Counts on trade_journal_events + feature_snapshots."""
    try:
        from sqlalchemy import create_engine, text
        url = os.environ.get(
            "DATABASE_URL",
            f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'fx')}:"
            f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
            f"{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/"
            f"{os.environ.get('POSTGRES_DB', 'fx')}",
        )
        eng = create_engine(url)
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
            sample_age = (datetime.now(timezone.utc) - last_ts).total_seconds()
        except Exception:
            pass

    db = _db_counts()
    errors = _recent_errors()

    verdict = _verdict(pid is not None, monitor_age, samples)

    return {
        "now": datetime.now(timezone.utc).isoformat(),
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
        <h2>latest sample</h2>
        ${row("ts", (last.ts || "—").slice(0, 19) + " UTC")}
        ${row("memory_mb", fmt(last.memory_mb))}
        ${row("equity", fmt(last.equity))}
        ${row("cpu_pct", fmt(last.cpu_pct))}
      </div>
      <div class="card">
        <h2>db rows</h2>
        ${row("trade_journal_events", fmt(db.trade_journal_events))}
        ${row("feature_snapshots", fmt(db.feature_snapshots))}
        ${db.latest_event ? row("latest", db.latest_event.type) : ""}
        ${db.error ? `<div class="err">${db.error}</div>` : ""}
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

refresh();
setInterval(refresh, 10000);
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _HTML


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    port = int(os.environ.get("SOAK_DASHBOARD_PORT", "8201"))
    print(f"curLit soak dashboard → http://127.0.0.1:{port}/")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
