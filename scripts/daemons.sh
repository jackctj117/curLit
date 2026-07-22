#!/usr/bin/env bash
# curLit daemon fleet launcher (CL-portability) — start/stop/status for the
# full native-daemon roster documented in docs/CURRENT_OPERATIONS.md §3.
#
#   ./scripts/daemons.sh start     # launch everything not already running
#   ./scripts/daemons.sh stop      # SIGTERM everything
#   ./scripts/daemons.sh status    # one line per daemon
#
# Idempotent: start skips anything already running (pgrep match). Each daemon
# logs to logs/<name>.log. Gated daemons (Alpaca/Reddit) self-disable when
# their env keys are absent — safe to run the full fleet on a fresh device.
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
  "options_activity|options_activity.py --loop|$PY scripts/options_activity.py --loop 86400"
  "score_outcomes|score_outcomes.py --loop|$PY scripts/score_outcomes.py --loop 86400"
  "telegram_bot|telegram_approval_bot.py|$PY scripts/telegram_approval_bot.py"
  "morning_digest|morning_digest.py --loop|$PY scripts/morning_digest.py --loop 300"
  "truth_monitor|truth_monitor.py --loop|$PY scripts/truth_monitor.py --loop 300"
  # reddit_monitor: uncomment once Reddit API approval lands (CL-okww)
  # "reddit_monitor|reddit_monitor.py --loop|$PY scripts/reddit_monitor.py --loop 300"
)

cmd="${1:-status}"
for entry in "${DAEMONS[@]}"; do
  IFS='|' read -r name pattern launch <<<"$entry"
  pid="$(pgrep -f "$pattern" | head -1 || true)"
  case "$cmd" in
    start)
      if [ -n "$pid" ]; then
        echo "  ✓ $name already running (pid $pid)"
      else
        nohup $launch >> "logs/${name}.log" 2>&1 &
        disown || true
        sleep 1
        newpid="$(pgrep -f "$pattern" | head -1 || true)"
        if [ -n "$newpid" ]; then
          echo "  ▶ $name started (pid $newpid)"
        else
          echo "  ✗ $name FAILED — check logs/${name}.log"
        fi
      fi
      ;;
    stop)
      if [ -n "$pid" ]; then
        kill -TERM "$pid" && echo "  ■ $name stopped (pid $pid)"
      else
        echo "  - $name not running"
      fi
      ;;
    status)
      if [ -n "$pid" ]; then
        echo "  ✓ $name (pid $pid)"
      else
        echo "  ✗ $name NOT RUNNING"
      fi
      ;;
    *)
      echo "usage: $0 {start|stop|status}" >&2
      exit 2
      ;;
  esac
done
