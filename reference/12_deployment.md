# 12 — Deployment

Systemd service units, Docker compose for observability, backup scripts, deployment workflow.

## Systemd: Vault Agent

**Location:** `/etc/systemd/system/fx-vault-agent.service`
**Purpose:** Long-running daemon holding vault unlocked. Started first; required by all other services.

```ini
[Unit]
Description=FX Vault Agent
After=network-online.target

[Service]
Type=simple
User=fx
Group=fx
WorkingDirectory=/opt/fx-system
Environment="PYTHONPATH=/opt/fx-system"
ExecStart=/opt/fx-system/venv/bin/python -m src.security.vault_agent
Restart=on-failure
RestartSec=15
StandardInput=tty-force
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Systemd: Live Engine

**Location:** `/etc/systemd/system/fx-live-engine.service`
**Purpose:** Main trading process. Runs all strategies, signal generation, order management.

```ini
[Unit]
Description=FX Live Trading Engine
After=network-online.target postgresql.service fx-vault-agent.service
Wants=network-online.target
Requires=postgresql.service fx-vault-agent.service

[Service]
Type=simple
User=fx
Group=fx
WorkingDirectory=/opt/fx-system
Environment="PYTHONPATH=/opt/fx-system"
Environment="FX_ENV=production"

ExecStart=/opt/fx-system/venv/bin/python -m src.runtime.run_engine
ExecStop=/bin/kill -SIGTERM $MAINPID

Restart=on-failure
RestartSec=15s
StartLimitInterval=300
StartLimitBurst=5
TimeoutStopSec=60

LimitNOFILE=65536
MemoryMax=8G

StandardOutput=journal
StandardError=journal
SyslogIdentifier=fx-live-engine

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/fx-system/logs /opt/fx-system/data /opt/fx-system/models

[Install]
WantedBy=multi-user.target
```

## Systemd: Ingestion

**Location:** `/etc/systemd/system/fx-ingestion.service`
**Purpose:** Periodic data ingestion (FRED, Stooq, CB statements, COT).

```ini
[Unit]
Description=FX Data Ingestion
After=network-online.target postgresql.service fx-vault-agent.service

[Service]
Type=simple
User=fx
Group=fx
WorkingDirectory=/opt/fx-system
Environment="PYTHONPATH=/opt/fx-system"

ExecStart=/opt/fx-system/venv/bin/python -m src.ingestion.run_scheduler

Restart=on-failure
RestartSec=30s

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Systemd: Watchdog

**Location:** `/etc/systemd/system/fx-watchdog.service`
**Purpose:** Independent process monitoring the live engine. Restarts on failure, sends alerts.

```ini
[Unit]
Description=FX System Watchdog
After=network-online.target

[Service]
Type=simple
User=fx
Group=fx
WorkingDirectory=/opt/fx-system
ExecStart=/opt/fx-system/venv/bin/python -m src.ops.watchdog
Restart=always
RestartSec=30s

[Install]
WantedBy=multi-user.target
```

## Docker Compose for Observability

**Location:** `docker/docker-compose.yml`
**Purpose:** Observability stack: Prometheus, Grafana, Loki, Promtail, Alertmanager, exporters.

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "127.0.0.1:9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=90d'
      - '--storage.tsdb.path=/prometheus'
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:10.4.2
    ports:
      - "127.0.0.1:3000:3000"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_AUTH_ANONYMOUS_ENABLED=false
    restart: unless-stopped
  
  loki:
    image: grafana/loki:3.0.0
    ports:
      - "127.0.0.1:3100:3100"
    volumes:
      - ./loki/loki-config.yml:/etc/loki/local-config.yaml:ro
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    restart: unless-stopped
  
  promtail:
    image: grafana/promtail:3.0.0
    volumes:
      - /var/log:/var/log:ro
      - /var/lib/systemd:/var/lib/systemd:ro
      - /etc/machine-id:/etc/machine-id:ro
      - ./promtail/promtail-config.yml:/etc/promtail/config.yml:ro
    command: -config.file=/etc/promtail/config.yml
    restart: unless-stopped
  
  alertmanager:
    image: prom/alertmanager:v0.27.0
    ports:
      - "127.0.0.1:9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    environment:
      - PUSHOVER_USER_KEY=${PUSHOVER_USER_KEY}
      - PUSHOVER_API_TOKEN=${PUSHOVER_API_TOKEN}
    restart: unless-stopped
  
  node_exporter:
    image: prom/node-exporter:v1.7.0
    ports:
      - "127.0.0.1:9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/rootfs'
    restart: unless-stopped
  
  postgres_exporter:
    image: prometheuscommunity/postgres-exporter:v0.15.0
    ports:
      - "127.0.0.1:9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://postgres_exporter:${PG_EXPORTER_PW}@host.docker.internal:5432/fx?sslmode=disable
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
  loki_data:
  alertmanager_data:
```

## Initial Server Hardening

**Location:** `scripts/initial_hardening.sh`
**Purpose:** First-time setup of new fx-server. Runs as root once.

```bash
#!/bin/bash
set -euo pipefail

echo "=== FX Server Initial Hardening ==="

apt update && apt upgrade -y

# Create non-root user (idempotent check)
if ! id fx >/dev/null 2>&1; then
    adduser fx
    usermod -aG sudo fx
fi

# SSH hardening
sed -i 's/^PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
systemctl restart ssh

# Firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw --force enable

# fail2ban
apt install -y fail2ban
systemctl enable fail2ban
systemctl start fail2ban

# Unattended security updates
apt install -y unattended-upgrades
echo 'APT::Periodic::Update-Package-Lists "1";' > /etc/apt/apt.conf.d/20auto-upgrades
echo 'APT::Periodic::Unattended-Upgrade "1";' >> /etc/apt/apt.conf.d/20auto-upgrades

# Essentials
apt install -y git curl wget build-essential \
    python3.12 python3.12-dev python3.12-venv \
    htop iotop ncdu tmux jq rsync

echo "Initial hardening complete"
```

## Backup Script

**Location:** `scripts/backup.sh`
**Purpose:** Daily encrypted backup of database, vault, state, configs.

```bash
#!/bin/bash
set -euo pipefail

BACKUP_DIR=/mnt/backup/fx
DATE=$(date +%Y%m%d)
STAGING=/tmp/fx-backup-$DATE

mkdir -p "$STAGING" "$BACKUP_DIR"

# DB dump
pg_dump -U fx fx | gzip > "$STAGING/fx-db.sql.gz"

# Encrypt with age (or wolfCrypt-based encrypt_file.py)
python /opt/fx-system/scripts/encrypt_file.py "$STAGING/fx-db.sql.gz"

# Vault and salts
cp /opt/fx-system/vault.enc "$STAGING/"
cp /opt/fx-system/vault.salt "$STAGING/"
cp /opt/fx-system/recovery.enc "$STAGING/"

# Strategy state
tar czf "$STAGING/state.tar.gz" -C /opt/fx-system state/

# Configs
tar czf "$STAGING/configs.tar.gz" -C /opt/fx-system configs/

# Bundle and hash
cd /tmp
tar czf "$BACKUP_DIR/fx-backup-$DATE.tar.gz" "fx-backup-$DATE"
sha256sum "$BACKUP_DIR/fx-backup-$DATE.tar.gz" > \
    "$BACKUP_DIR/fx-backup-$DATE.sha256"

# Offsite sync
if [ "${CLOUD_SYNC:-true}" = "true" ]; then
    rclone sync "$BACKUP_DIR" remote:fx-backup/ --max-age 31d
fi

# Cleanup
rm -rf "$STAGING"

# Retention
find "$BACKUP_DIR" -name "fx-backup-*" -mtime +30 -delete

echo "Backup complete: $(date)"
```

## Backup Restore Test

**Location:** `scripts/test_restore.sh`
**Purpose:** Quarterly verification that backups can actually be restored.

```bash
#!/bin/bash
set -euo pipefail

LATEST=$(ls -1 /mnt/backup/fx/fx-backup-*.tar.gz | tail -1)
mkdir /tmp/restore-test
cd /tmp/restore-test
tar xzf "$LATEST"

# Decrypt DB dump
python /opt/fx-system/scripts/decrypt_file.py fx-backup-*/fx-db.sql.gz.enc

# Create test DB and restore
sudo -u postgres createdb fx_restore_test
gunzip -c fx-db.sql.gz | psql -U postgres fx_restore_test

# Sanity check
ROW_COUNT=$(psql -U postgres fx_restore_test -t -c "SELECT COUNT(*) FROM fx_prices;")
echo "Restored rows in fx_prices: $ROW_COUNT"

# Teardown
sudo -u postgres dropdb fx_restore_test
rm -rf /tmp/restore-test

echo "Restore test passed"
```

## Deployment Script

**Location:** `scripts/deploy.sh`
**Purpose:** Git-based deployment with signed commit verification and test gating.

```bash
#!/bin/bash
set -euo pipefail

cd /opt/fx-system

echo "Fetching latest..."
git fetch origin main

LATEST_SHA=$(git rev-parse origin/main)
git verify-commit "$LATEST_SHA" || { echo "Unsigned commit!"; exit 1; }

echo "Running tests on new code..."
WORKTREE=/tmp/fx-deploy-test
git worktree add "$WORKTREE" "$LATEST_SHA"
cd "$WORKTREE"

/opt/fx-system/venv/bin/pip install -r requirements.txt --dry-run
/opt/fx-system/venv/bin/pytest tests/ -x -q

cd /opt/fx-system
echo "Deploying..."
git merge --ff-only origin/main

venv/bin/pip install -r requirements.txt
venv/bin/python scripts/migrate_db.py

sudo systemctl reload-or-restart fx-live-engine
sudo systemctl reload-or-restart fx-ingestion

git worktree remove "$WORKTREE"

echo "Deploy complete: $(git log -1 --oneline)"
```

## Morning Check Script

**Location:** `scripts/morning_check.sh`
**Purpose:** Daily automated health check, emailed to operator.

```bash
#!/bin/bash
set -e

echo "=== FX System Morning Check ==="
echo "Date: $(date)"
echo ""

echo "--- Service Status ---"
systemctl --no-pager status fx-vault-agent fx-live-engine fx-ingestion fx-watchdog | grep -E "(Active|Main PID)"

echo ""
echo "--- Disk Space ---"
df -h / /opt/fx-system

echo ""
echo "--- Docker Containers ---"
docker compose -f /opt/fx-system/docker/docker-compose.yml ps

echo ""
echo "--- Recent Errors ---"
sudo journalctl --since "24 hours ago" -p err -u fx-live-engine -u fx-ingestion --no-pager | tail -20

echo ""
echo "--- Last Ingestion Run ---"
psql -U fx -d fx -c "SELECT source, MAX(ingested_at) FROM ingestion_log GROUP BY source;"

echo ""
echo "--- Portfolio Status ---"
curl -s http://localhost:8000/metrics | grep -E "fx_(account_equity|portfolio_drawdown|positions_open|signal)" | head -20
```

## Cron Schedule

**Location:** `/etc/cron.d/fx-system`
**Purpose:** Scheduled jobs for the fx user.

```cron
# Daily backup at 01:00 UTC
0 1 * * * fx /opt/fx-system/scripts/backup.sh > /var/log/fx-backup.log 2>&1

# Morning health check at 07:00 local
0 7 * * * fx /opt/fx-system/scripts/morning_check.sh | mail -s "FX Daily Check" your@email.com

# Quarterly restore test (1st of Jan/Apr/Jul/Oct)
0 3 1 1,4,7,10 * fx /opt/fx-system/scripts/test_restore.sh
```

## Postgres Initialization

**Location:** `scripts/init_postgres.sh`
**Purpose:** Configure PostgreSQL with TimescaleDB and create users.

```bash
#!/bin/bash
set -euo pipefail

# Tune TimescaleDB
sudo timescaledb-tune --quiet --yes

# Get passwords from vault
FX_PW=$(python -c "from src.security.vault_client import VaultClient; print(VaultClient().get('POSTGRES_FX_PASSWORD'))")
GRAFANA_PW=$(python -c "from src.security.vault_client import VaultClient; print(VaultClient().get('POSTGRES_GRAFANA_PASSWORD'))")
EXPORTER_PW=$(python -c "from src.security.vault_client import VaultClient; print(VaultClient().get('POSTGRES_EXPORTER_PASSWORD'))")

# Create database and users
sudo -u postgres psql <<EOF
CREATE DATABASE fx;
CREATE USER fx WITH PASSWORD '${FX_PW}';
GRANT ALL PRIVILEGES ON DATABASE fx TO fx;

CREATE USER grafana WITH PASSWORD '${GRAFANA_PW}';
GRANT CONNECT ON DATABASE fx TO grafana;

CREATE USER postgres_exporter WITH PASSWORD '${EXPORTER_PW}';
GRANT CONNECT ON DATABASE fx TO postgres_exporter;
GRANT pg_monitor TO postgres_exporter;

\c fx
CREATE EXTENSION IF NOT EXISTS timescaledb;
GRANT ALL ON SCHEMA public TO fx;
GRANT USAGE ON SCHEMA public TO grafana, postgres_exporter;
EOF

# Lock down access to localhost only
sudo sed -i 's/host all all 0.0.0.0\/0 md5/# disabled/' /etc/postgresql/16/main/pg_hba.conf
sudo systemctl restart postgresql

echo "PostgreSQL initialized"
```

## Service Startup Sequence

**Location:** `scripts/start_all.sh`
**Purpose:** Manual startup ritual after server reboot. Vault agent needs interactive passphrase.

```bash
#!/bin/bash
# This script is run manually via SSH after a reboot.
# It cannot be fully automated because the vault needs the passphrase.

echo "=== Starting FX System ==="

# 1. Start vault agent — prompts for passphrase via systemd-ask-password
echo "Starting vault agent (will prompt for passphrase)..."
sudo systemctl start fx-vault-agent

# Verify
sleep 3
sudo systemctl status fx-vault-agent --no-pager

# 2. Start observability stack (no secrets needed)
echo "Starting observability stack..."
cd /opt/fx-system/docker
docker compose up -d
cd -

# 3. Start data services
echo "Starting ingestion..."
sudo systemctl start fx-ingestion

# 4. Start trading engine
echo "Starting live engine..."
sudo systemctl start fx-live-engine

# 5. Start watchdog
echo "Starting watchdog..."
sudo systemctl start fx-watchdog

# Final status
echo ""
echo "=== All Services ==="
sudo systemctl status fx-* --no-pager

echo ""
echo "Tail logs with: sudo journalctl -fu fx-live-engine"
```
