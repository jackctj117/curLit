#!/usr/bin/env bash
# CL-env: server hardening — applies the security baseline before
# install_server.sh adds the application stack.
#
# Idempotent. Run as root on a fresh Ubuntu 22.04+ server.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root (sudo)" >&2
    exit 1
fi

log() { printf '[provision] %s\n' "$*"; }

# -- Patches -------------------------------------------------------------
log "Apt update + unattended-upgrades"
apt-get update -y
apt-get upgrade -y
apt-get install -y unattended-upgrades apt-listchanges
dpkg-reconfigure -plow unattended-upgrades

# -- SSH hardening -------------------------------------------------------
log "SSH hardening (disable root login, disable password auth)"
SSH_CONFIG=/etc/ssh/sshd_config
cp -n "$SSH_CONFIG" "${SSH_CONFIG}.orig"
sed -i \
    -e 's/^#*PermitRootLogin.*/PermitRootLogin no/' \
    -e 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' \
    -e 's/^#*ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' \
    -e 's/^#*PermitEmptyPasswords.*/PermitEmptyPasswords no/' \
    -e 's/^#*X11Forwarding.*/X11Forwarding no/' \
    "$SSH_CONFIG"
# Whitelist authentication methods explicitly so a future config drift
# can't silently re-enable password auth.
grep -q '^AuthenticationMethods' "$SSH_CONFIG" || \
    echo 'AuthenticationMethods publickey' >> "$SSH_CONFIG"
systemctl reload sshd

# -- Firewall ------------------------------------------------------------
log "Configuring UFW (allow SSH only externally)"
apt-get install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
# All other ports (Grafana, Prometheus, engine API) bind to 127.0.0.1
# in docker-compose.yml — operator reaches them via SSH tunnel.
ufw --force enable

# -- fail2ban ------------------------------------------------------------
log "Installing fail2ban (default sshd jail)"
apt-get install -y fail2ban
systemctl enable --now fail2ban

# -- LUKS reminder -------------------------------------------------------
# We can't enable LUKS automatically on an existing system — it requires
# a fresh install or rebuild. We can detect whether root is encrypted and
# warn loudly if not.
log "Disk encryption status"
if blkid -t TYPE=crypto_LUKS >/dev/null 2>&1; then
    log "  LUKS volume(s) detected — good"
else
    cat <<EOF
[provision] WARNING: no LUKS-encrypted volumes detected.
  curLit secrets (vault, OANDA keys, Postgres data) sit on this disk.
  For production, rebuild with full-disk encryption before promoting
  to live trading. See docs/SECURITY.md.
EOF
fi

# -- Time sync -----------------------------------------------------------
log "Time sync (chrony)"
apt-get install -y chrony
systemctl enable --now chrony

log "Done. Reboot recommended after this script's first run."
