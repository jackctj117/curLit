# LiveEngineDown

## Severity: CRITICAL

## What it means
The main trading engine is not responding.

## Immediate actions

> **Native deploy note**: on the current operator box the engine runs
> natively via nohup (see `docs/BOOT.md`), not systemd. There, check
> `/tmp/curlit-engine.log` and relaunch per BOOT.md; the
> journalctl/systemctl steps below apply to the systemd production
> deploy (`docs/STARTUP.md`).

1. Open incident dashboard
2. Check if watchdog has already restarted it: `journalctl -u fx-watchdog --since "5 min ago"`
3. Check logs: `journalctl -u fx-live-engine --since "10 min ago"`
4. Verify broker connection is still alive (check broker web portal)
5. Verify all positions are safe (broker still has them)

## Common causes
- Broker API disconnection causing engine crash
- Memory exhausted (8GB limit hit)
- Database connection failure
- Python exception in signal generation loop

## Resolution
1. If watchdog restarted successfully: monitor for 5 min, confirm normal
2. If watchdog failed: `systemctl start fx-live-engine`
3. Check memory usage before restart: `free -h`
4. After restart, verify reconciliation passes (no position mismatch)
5. Check kill switch states — resume only if clean

## Escalation
If engine fails to restart: revert to last deploy tag with `scripts/rollback.sh`
