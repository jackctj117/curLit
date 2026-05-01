# STARTUP — production server boot ritual (CL-fnu)

Step-by-step guide to bringing curLit up on the production fx-server.
This complements `docs/BOOT.md` (which is the dev/local boot guide) by
covering the systemd-managed production deploy.

The system runs as four cooperating systemd services:
- `fx-vault-agent.service` — credential server (must be first; everything else reads from it)
- `fx-ingestion.service` — Airflow scheduler (FRED, yfinance, CFTC daily ingest)
- `fx-live-engine.service` — the trading engine itself
- `fx-watchdog.service` — restarts the engine on crash; alerts on
  repeat failures

Each step lists the command and the expected output. **Do not skip
verification** — the order matters and a silent failure of vault
will cascade through the dependent services.

---

## 1. SSH in (key auth only)

```bash
ssh fx-server          # alias resolves via ~/.ssh/config
```

Expected: shell prompt as `fx-operator@fx-prod`. If password prompt
appears, your key isn't on the server — fix locally before continuing.

---

## 2. Start vault agent (passphrase prompt)

```bash
sudo systemctl start fx-vault-agent
sudo systemctl status fx-vault-agent --no-pager | head -10
```

You'll be prompted for the master vault passphrase. Type carefully — no
echo. After successful start:

Expected status:
```
● fx-vault-agent.service - curLit credential vault
   Loaded: loaded (/etc/systemd/system/fx-vault-agent.service; enabled)
   Active: active (running) since <recent>
   Main PID: <pid> (python)
```

Verify the unix socket is listening:

```bash
ls -la /run/fx-vault-agent.sock
```

Expected: `srw------- 1 fx-vault fx-vault 0 <date> /run/fx-vault-agent.sock`.

**Optional** TPM-sealed passphrase auto-unlock (trusted machine only):
configure `tpm2_unseal` in the unit file's ExecStartPre. See
`docs/SECURITY.md` for the threat model.

---

## 3. Start data ingestion (Airflow)

```bash
sudo systemctl start fx-ingestion
sudo systemctl status fx-ingestion --no-pager | head -8
```

Expected: `Active: active (running)`. Confirm the scheduler is consuming
DAGs:

```bash
journalctl -u fx-ingestion --since "1 minute ago" | grep -i "scheduler"
```

Expected: at least one line mentioning `Scheduler heartbeat`.

---

## 4. Start the engine

```bash
sudo systemctl start fx-live-engine
sudo systemctl status fx-live-engine --no-pager | head -10
```

Expected: `Active: active (running)`. The engine logs to systemd
journal; health-check after ~10 seconds:

```bash
curl -s http://localhost:8200/api/system | jq .
```

Expected: `{"engine": "running", "oms_halted": false}`.

If `oms_halted` is true on a fresh start, a kill switch fired
during startup self-check. Inspect:

```bash
journalctl -u fx-live-engine --since "5 minutes ago" | grep -i "kill switch\|ERROR"
```

---

## 5. Start the watchdog (last)

```bash
sudo systemctl start fx-watchdog
sudo systemctl status fx-watchdog --no-pager | head -8
```

The watchdog tails the engine's metrics endpoint and restarts the
engine if it goes silent for >60 seconds. Verify it sees the engine:

```bash
journalctl -u fx-watchdog --since "1 minute ago" | grep -i "engine OK"
```

Expected: at least one "engine OK" log line within 30 seconds.

---

## 6. Verify all four services

```bash
systemctl status fx-* --no-pager | grep -E "(●|Active:)"
```

All four should show `Active: active (running)`. Any unit in
`failed` state needs investigation before trading begins.

---

## 7. Tail logs for ~60 seconds

```bash
journalctl -fu fx-live-engine
```

Watch for:
- `INFO Starting curLit live engine (broker=…)`
- `INFO Strategy <id>: signal_interval=<n>s`
- No `ERROR` or `CRITICAL` lines

After ~60s, **Ctrl+C** to detach. Tail any failed unit specifically:

```bash
journalctl -fu fx-vault-agent
journalctl -fu fx-ingestion
journalctl -fu fx-watchdog
```

---

## Shutdown (reverse order)

```bash
sudo systemctl stop fx-watchdog
sudo systemctl stop fx-live-engine
sudo systemctl stop fx-ingestion
sudo systemctl stop fx-vault-agent
```

Verify with `systemctl status fx-*` — all should be `inactive (dead)`.

For an emergency halt (positions liquidated, no clean shutdown):

```bash
curl -X POST -H "X-API-Secret: $WEB_API_SECRET" \
  http://localhost:8200/api/system/halt
sudo systemctl stop fx-live-engine
```

The halt API drains pending OMS work first, so positions don't get
abandoned mid-rebalance.

---

## Troubleshooting

| Symptom                                      | First action                                              |
| -------------------------------------------- | --------------------------------------------------------- |
| vault-agent fails on start                   | Re-enter passphrase; check `/var/log/fx-vault-agent/`     |
| vault-agent active but socket missing        | Check `umask` in unit; should be 0077                     |
| engine starts then crashes within 30s        | `journalctl -u fx-live-engine -n 200`                     |
| ingestion DAG fails                          | Airflow UI at https://fx-server:8080 (VPN required)       |
| watchdog restart loop                        | Engine is exiting cleanly each time — see live-engine log |
| metrics endpoint 503                         | Engine is alive but Prometheus client thread died — restart |

For deeper alert response, see `docs/OPERATOR.md`. For BOOT.md (local
dev), see `docs/BOOT.md`.
