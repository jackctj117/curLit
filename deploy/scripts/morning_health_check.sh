#!/usr/bin/env bash
# CL-5yl: morning health check — runs at 06:00 UTC via cron.
# Reports any RED to stdout (cron emails it), any YELLOW to stderr only.
# Exits 0 always — this is observability, not a control loop.

set -uo pipefail

red()    { printf '\033[31m[RED]    %s\033[0m\n' "$*"; }
yellow() { printf '\033[33m[YELLOW] %s\033[0m\n' "$*" >&2; }
green()  { printf '\033[32m[OK]     %s\033[0m\n' "$*" >&2; }

ANY_RED=0

# -- Services ------------------------------------------------------------
for unit in fx-vault-agent fx-ingestion fx-live-engine fx-watchdog; do
    if systemctl is-active --quiet "$unit"; then
        green "$unit running"
    else
        red "$unit not running"
        ANY_RED=1
    fi
done

# -- Disk ----------------------------------------------------------------
USED=$(df / | awk 'NR==2 {gsub("%","",$5); print $5}')
if (( USED >= 90 )); then
    red "Root disk ${USED}% full"
    ANY_RED=1
elif (( USED >= 75 )); then
    yellow "Root disk ${USED}% full"
else
    green "Root disk ${USED}%"
fi

# -- Docker --------------------------------------------------------------
if command -v docker >/dev/null; then
    DOWN=$(docker ps --filter status=exited --format '{{.Names}}' | wc -l)
    if (( DOWN > 0 )); then
        yellow "$DOWN docker containers exited"
    fi
fi

# -- Engine API ----------------------------------------------------------
if curl -sf -o /dev/null --max-time 3 http://localhost:8099/metrics; then
    green "Engine /metrics responsive"
else
    red "Engine /metrics endpoint unreachable"
    ANY_RED=1
fi

# -- Engine errors in last 24h -------------------------------------------
ERR_COUNT=$(journalctl -u fx-live-engine --since "24h ago" -p err --no-pager 2>/dev/null | wc -l)
if (( ERR_COUNT > 50 )); then
    red "Engine logged $ERR_COUNT errors in last 24h"
    ANY_RED=1
elif (( ERR_COUNT > 5 )); then
    yellow "Engine logged $ERR_COUNT errors in last 24h"
fi

# -- Reconciliation ------------------------------------------------------
LATEST_RECON=$(ls -t /opt/curlit/reports/reconciliation/*.json 2>/dev/null | head -1)
if [[ -z "$LATEST_RECON" ]]; then
    yellow "No reconciliation reports yet"
elif [[ $(find "$LATEST_RECON" -mtime -2 2>/dev/null) ]]; then
    MISMATCHES=$(jq -r '.mismatches | length' "$LATEST_RECON" 2>/dev/null || echo "?")
    if [[ "$MISMATCHES" == "0" ]]; then
        green "Reconciliation clean ($(basename "$LATEST_RECON"))"
    else
        red "Reconciliation has $MISMATCHES mismatches in $(basename "$LATEST_RECON")"
        ANY_RED=1
    fi
else
    yellow "Reconciliation report stale: $(basename "$LATEST_RECON")"
fi

# -- Source drift gauge --------------------------------------------------
DRIFT=$(curl -sf --max-time 3 http://localhost:8099/metrics 2>/dev/null \
    | awk '/^fx_engine_source_drift / {print $2; exit}')
if [[ "${DRIFT:-0}" == "1.0" ]] || [[ "${DRIFT:-0}" == "1" ]]; then
    red "Engine source-drift gauge is 1 — running stale code, restart needed"
    ANY_RED=1
fi

if (( ANY_RED == 0 )); then
    green "Morning health check: all green"
fi
