#!/usr/bin/env bash
# curLit daemon fleet launcher (CL-portability) — start/stop/status/restart
# for the full native-daemon roster documented in docs/CURRENT_OPERATIONS.md §3.
#
#   ./scripts/daemons.sh start [name]     # launch everything (or one) not already running
#   ./scripts/daemons.sh stop [name]      # SIGTERM everything (or one)
#   ./scripts/daemons.sh status [name]    # one line per daemon
#   ./scripts/daemons.sh restart [name]   # stop + start + VERIFY the pid changed
#
# Idempotent: start skips anything already running (argv match). Each daemon
# logs to logs/<name>.log. Gated daemons (Alpaca/Reddit) self-disable when
# their env keys are absent — safe to run the full fleet on a fresh device.
#
# restart is the only deploy-safe way to bounce one daemon (CL-obgy): it
# waits for the old process to die, starts a new one, and FAILS (exit 1)
# unless the surviving pid differs from the old pid — a silent "already
# running (old pid)" can no longer masquerade as a successful reload. An
# unknown [name] is a hard error (exit 2), never a silent whole-fleet
# operation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

# CURLIT_RISK_PROFILE must be explicit for the engine — the soak convention
# is "aggressive"; override by exporting before calling this script.
: "${CURLIT_RISK_PROFILE:=aggressive}"
export CURLIT_RISK_PROFILE

# name | pgrep pattern | command
DAEMONS=(
  "engine|run_engine --broker oanda-practice|$PY -m src.runtime.run_engine --broker oanda-practice"
  "event_pipeline|event_pipeline.py --ingest|$PY scripts/event_pipeline.py --ingest --assess --loop 900"
  "intraday_pricer|intraday_pricer.py --loop|$PY scripts/intraday_pricer.py --loop 120"
  "refresh_rates|refresh_rates.py --loop|$PY scripts/refresh_rates.py --loop 86400"
  "x_monitor|x_monitor.py --loop|$PY scripts/x_monitor.py --loop 300"
  "execute_options|execute_options.py --loop|$PY scripts/execute_options.py --loop 300"
  # Shares A/B of the SAME ideas (CL-ncbq); self-disables (idles) unless
  # ALPACA_EQUITY_ENABLED is set, so it is safe in the roster from day one.
  "execute_equities|execute_equities.py --loop|$PY scripts/execute_equities.py --loop 300"
  "options_activity|options_activity.py --loop|$PY scripts/options_activity.py --loop 86400"
  "score_outcomes|score_outcomes.py --loop|$PY scripts/score_outcomes.py --loop 86400"
  "telegram_bot|telegram_approval_bot.py|$PY scripts/telegram_approval_bot.py"
  "morning_digest|morning_digest.py --loop|$PY scripts/morning_digest.py --loop 300"
  "truth_monitor|truth_monitor.py --loop|$PY scripts/truth_monitor.py --loop 300"
  "health_watch|health_watch.py --loop|$PY scripts/health_watch.py --loop 300"
  "weekly_event_study|weekly_event_study.py --loop|$PY scripts/weekly_event_study.py --loop 21600"
  # reddit_monitor: uncomment once Reddit API approval lands (CL-okww)
  # "reddit_monitor|reddit_monitor.py --loop|$PY scripts/reddit_monitor.py --loop 300"
)

# keep_awake (macOS only, CL-vff9 companion): caffeinate keeps the box from
# sleeping while the fleet runs — -d display, -i idle, -m disk, -s system-sleep
# (the -s assertion is AC-power-only, so it never drains the battery flat).
# Tied to the fleet: `daemons.sh stop` lets the Mac sleep again; the box must
# stay plugged in with the lid OPEN (clamshell/battery still sleeps). Skipped
# on Linux/other — `caffeinate` doesn't exist there, so on a server rely on the
# OS staying awake (no-sleep is the norm) rather than a bogus failing daemon.
# Prepended so it starts first, before the engine.
if [ "$(uname -s)" = "Darwin" ]; then
  DAEMONS=("keep_awake|caffeinate -dims|caffeinate -dims" "${DAEMONS[@]}")
fi

cmd="${1:-status}"
target="${2:-}"
fail=0
matched=0

# Process discovery and control go through a bounded, identity-verified
# helper (CL-wv3v). On 2026-09-08 each `pgrep -f` here took tens of seconds
# on macOS (it reads every process's argv) and the unbounded 1 s poll loops
# stretched a 30 s restart into minutes. The helper runs each argv scan in a
# worker it abandons at a hard timeout, tracks the old writer by pid + start
# time (no argv read; PID-reuse safe), signals only re-verified identities,
# refuses to launch when two writers match or when a scan could not confirm
# none is running, and gives restart a TOTAL deadline whose failures name the
# phase (discover / stop-wait / start / verify) and the last known writer
# state. Launch is unchanged: `nohup $launch >> logs/<name>.log 2>&1`.
# Pattern and launch command travel in the environment, never argv, so a
# helper cannot match its own pattern. Status lines are unchanged (fleet_watch
# parses them); a stalled scan prints "  ? <name> UNKNOWN", which it ignores.
PD=src/runtime/proc_discovery.py
: "${DAEMONS_RESTART_DEADLINE_S:=60}"

# All helpers read the loop variables $name / $pattern / $launch.
pd() {
  CURLIT_PROC_PATTERN="$pattern" CURLIT_PROC_LAUNCH="$launch" \
    "$PY" "$PD" "$@" --name "$name" --log "logs/${name}.log"
}

start_one() {
  pd start
}

# SIGTERM, wait for actual exit (30s grace — daemons finish their in-flight
# cycle; returning early let a following `start` see the dying process, bit
# us twice on 2026-07-22), then SIGKILL the same verified identity.
stop_one() {
  pd stop
}

# stop + start + verify the surviving identity differs from the one we began
# with, else the reload did NOT happen (CL-obgy — the 2026-07-31 near-miss: a
# wrong pid-file guess left execute_options running pre-fix code while
# everything looked green). Bounded by DAEMONS_RESTART_DEADLINE_S in total.
restart_one() {
  pd restart --deadline "$DAEMONS_RESTART_DEADLINE_S"
}

case "$cmd" in start|stop|status|restart) ;; *)
  echo "usage: $0 {start|stop|status|restart} [name]" >&2
  exit 2
  ;;
esac

for entry in "${DAEMONS[@]}"; do
  IFS='|' read -r name pattern launch <<<"$entry"
  if [ -n "$target" ] && [ "$name" != "$target" ]; then
    continue
  fi
  matched=1
  case "$cmd" in
    start)
      start_one || fail=1
      ;;
    stop)
      stop_one || true  # stop keeps its historical always-0 exit
      ;;
    restart)
      restart_one || fail=1
      ;;
    status)
      pd status || echo "  ? $name UNKNOWN — status helper failed"
      ;;
  esac
done

if [ -n "$target" ] && [ "$matched" -eq 0 ]; then
  echo "unknown daemon: $target" >&2
  echo "known daemons:" >&2
  for entry in "${DAEMONS[@]}"; do
    IFS='|' read -r name _ _ <<<"$entry"
    echo "  $name" >&2
  done
  exit 2
fi

# start/stop/status keep their historical always-0 exit (boot_recovery and
# the watchdog depend on it); restart is strict — a failed verify exits 1.
if [ "$cmd" = "restart" ]; then
  exit "$fail"
fi
