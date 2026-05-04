# Server provisioning + hardening guide (CL-env)

End-to-end ritual for bringing a fresh Ubuntu server up to production
hosting standard for curLit.

## Threat model (one paragraph)

The server holds OANDA API credentials with live trading authority and
~30 days of trade-journal data. Compromise = adversary trades in the
account or exfiltrates the journal. Highest-leverage controls: SSH
keypair-only, full-disk encryption (LUKS), least-privilege service
accounts, vault-isolated secrets. fail2ban on SSH catches the dumb
brute-force; UFW catches the lateral-discovery attempts.

## Prerequisites

- Ubuntu 22.04 LTS or newer (fresh install).
- Static public IPv4 + DNS A record (call it `fx-server`).
- LUKS-encrypted root volume — can ONLY be enabled at install time.
  If you skipped this, rebuild before going to live trading.
- One non-root operator user with SSH key already in
  `~/.ssh/authorized_keys`.

## Steps

### 1. Run the hardening script

```bash
# From your laptop:
scp deploy/scripts/provision_server.sh fx-server:/tmp/
ssh fx-server "sudo bash /tmp/provision_server.sh"
```

This applies: unattended-upgrades, SSH key-only auth, UFW firewall
(SSH-only externally; everything else via tunnel), fail2ban, chrony.

Verify:
```bash
ssh fx-server 'sudo ufw status verbose && sudo systemctl is-active fail2ban'
```

### 2. Reboot

```bash
ssh fx-server "sudo reboot"
```

### 3. Install the application stack

```bash
scp deploy/scripts/install_server.sh fx-server:/tmp/
ssh fx-server "sudo bash /tmp/install_server.sh"
```

Brings up Python 3.11, Postgres + TimescaleDB, Docker, and creates the
service users (`fx-vault`, `fx-engine`, `fx-operator`).

### 4. Pull the repo

```bash
ssh fx-server "sudo -u fx-operator git clone https://github.com/jackctj117/curLit.git /opt/curlit"
ssh fx-server "cd /opt/curlit && sudo -u fx-operator python3.11 -m venv .venv"
ssh fx-server "cd /opt/curlit && sudo -u fx-operator .venv/bin/pip install -e ."
```

### 5. Initialize the vault

Interactive — you'll be prompted for OANDA / FRED / Postgres credentials
and a master passphrase.

```bash
ssh -t fx-server "cd /opt/curlit && sudo bash deploy/scripts/init_vault.sh"
```

### 6. Install systemd units

```bash
ssh fx-server "sudo install -m 0644 /opt/curlit/deploy/systemd/*.service /etc/systemd/system/"
ssh fx-server "sudo systemctl daemon-reload"
ssh fx-server "sudo systemctl enable fx-vault-agent fx-ingestion fx-live-engine fx-watchdog"
```

### 7. Run migrations

```bash
ssh fx-server "cd /opt/curlit && sudo -u fx-operator .venv/bin/python -m migrations.run"
```

### 8. Start in paper mode

```bash
ssh fx-server "echo 'CURLIT_BROKER_MODE=paper' | sudo tee /etc/curlit/curlit.env > /dev/null"
ssh -t fx-server "sudo systemctl start fx-vault-agent"   # passphrase prompt
ssh fx-server "sudo systemctl start fx-ingestion fx-live-engine fx-watchdog"
```

Follow `docs/STARTUP.md` for the per-service verification.

## Promotion to live trading

Only after:
- 14+ days of clean reconciliation reports in paper mode
- Daily backup runs verified (one full restore drill)
- Operator ack via `docs/OPERATOR.md` checklist
- The `curlit.env` line `CURLIT_BROKER_MODE` flipped to `oanda-live`
- `--confirm-live` flag added to the engine ExecStart

## Quarterly rotation

```bash
ssh -t fx-server "cd /opt/curlit && sudo -u fx-operator \
    .venv/bin/python -m scripts.rotate_secrets --all"
```

See `docs/SECURITY.md` for the threat model and `scripts/rotate_secrets.py`
docstring for the rotation flow.

## Disaster recovery

See `deploy/scripts/restore_drill.sh` and the matching docs in
`docs/SECURITY.md`. TL;DR: restore the latest `.tar.gpg` to a
disposable Postgres instance, verify row counts, then promote
to production via `pg_dump | psql`.
