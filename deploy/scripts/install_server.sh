#!/usr/bin/env bash
# CL-cly: server software installation — Python, Postgres, TimescaleDB, Docker
# Idempotent — re-running on an already-provisioned server is safe; each
# step skips when the package is already present.
#
# Run on a fresh Ubuntu 22.04+ server, after the hardening guide in
# deploy/scripts/provision_server.sh has been applied.
#
# Usage (from a controller machine):
#   scp deploy/scripts/install_server.sh fx-server:/tmp/
#   ssh fx-server "sudo bash /tmp/install_server.sh"

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root (sudo)" >&2
    exit 1
fi

log() { printf '[install] %s\n' "$*"; }

log "Updating apt cache"
apt-get update -y

# -- Python 3.11 + venv --------------------------------------------------
if ! command -v python3.11 >/dev/null; then
    log "Installing Python 3.11"
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
    apt-get install -y python3.11 python3.11-venv python3.11-dev \
        python3-pip build-essential
fi

# -- Postgres + TimescaleDB ---------------------------------------------
if ! dpkg -l | grep -q timescaledb-2-postgresql-15; then
    log "Adding TimescaleDB apt repo + installing"
    apt-get install -y gnupg postgresql-common apt-transport-https \
        lsb-release wget curl
    /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
    echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/timescaledb.list
    wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | \
        apt-key add -
    apt-get update -y
    apt-get install -y postgresql-15 timescaledb-2-postgresql-15 \
        postgresql-contrib-15
    timescaledb-tune --quiet --yes
    systemctl restart postgresql
fi

# -- Docker --------------------------------------------------------------
if ! command -v docker >/dev/null; then
    log "Installing Docker (official repo)"
    apt-get install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
fi

# -- Service users -------------------------------------------------------
for u in fx-vault fx-engine fx-operator; do
    if ! id "$u" >/dev/null 2>&1; then
        log "Creating user $u"
        useradd --system --create-home --shell /bin/bash "$u"
    fi
done
# fx-operator needs docker group for compose ops; engine + vault DON'T.
usermod -aG docker fx-operator

# -- Working directory ---------------------------------------------------
install -d -m 0750 -o fx-operator -g fx-operator /opt/curlit
install -d -m 0750 -o fx-engine   -g fx-engine   /var/log/curlit
install -d -m 0700 -o fx-vault    -g fx-vault    /etc/curlit

# -- NTP (preempts CL-gnhu drift alarms with a real time sync source) ---
if ! systemctl is-active --quiet chrony && ! systemctl is-active --quiet systemd-timesyncd; then
    log "Installing chrony for NTP sync"
    apt-get install -y chrony
    systemctl enable --now chrony
fi

log "Done — verify with 'systemctl status postgresql docker chrony'"
