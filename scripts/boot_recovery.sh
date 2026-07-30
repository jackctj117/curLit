#!/usr/bin/env bash
# Boot recovery for the curLit fleet (CL-rmvt) — run by the launchd agent
# deploy/launchd/com.curlit.fleet.plist at login/boot.
#
# Why this exists: the 2026-07-28 macOS update rebooted the box at 16:57 and
# NOTHING self-healed — the daemons are nohup processes (erased by reboot),
# and the system stayed dark 15.6h including the first hour of the US session
# (the watchdog dies with the host, so nothing could even page). This script
# restores the documented steady state: wait for Docker, ensure the live DB
# container, wait for Postgres, then the idempotent fleet start.
#
# Idempotent by construction: docker start on a running container is a no-op
# and daemons.sh start skips anything already running — safe to kickstart at
# any time.
set -u
# launchd runs with a MINIMAL environment — no Homebrew/Docker on PATH.
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")/.."
LOG="logs/boot_recovery.log"
mkdir -p logs
exec >>"$LOG" 2>&1
echo "==== boot recovery $(date '+%Y-%m-%d %H:%M:%S %Z') ===="

# 1. Docker daemon (Docker Desktop is a login item; give it time).
for i in $(seq 1 60); do
  docker info >/dev/null 2>&1 && break
  [ "$i" -eq 1 ] && open -a Docker 2>/dev/null
  sleep 6
done
if ! docker info >/dev/null 2>&1; then
  echo "docker daemon never came up — aborting (fleet NOT started)"
  exit 1
fi
echo "docker up"

# 2. Live DB container (restart=unless-stopped should do this; belt+braces).
docker start curlit-postgres-soak >/dev/null 2>&1 || true
for i in $(seq 1 30); do
  docker exec curlit-postgres-soak pg_isready -U fx >/dev/null 2>&1 && break
  sleep 4
done
docker exec curlit-postgres-soak pg_isready -U fx >/dev/null 2>&1 \
  && echo "postgres ready" || echo "WARN: postgres not ready — starting fleet anyway (daemons fail-soft)"

# 3. The fleet (idempotent).
./scripts/daemons.sh start
echo "==== boot recovery done $(date '+%Y-%m-%d %H:%M:%S %Z') ===="
