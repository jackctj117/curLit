#!/usr/bin/env bash
# Apply hardened SSH config — validates before installing
# Run as root: sudo ./scripts/apply_ssh_config.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSHD_CONFIG="$SCRIPT_DIR/../configs/sshd_config"
TARGET="/etc/ssh/sshd_config"

if [ ! -f "$SSHD_CONFIG" ]; then
    echo "ERROR: $SSHD_CONFIG not found"
    exit 1
fi

echo "Validating new sshd_config..."
sshd -t -f "$SSHD_CONFIG" 2>&1 && echo "  ✓ Syntax OK" || {
    echo "  ✗ Validation failed — aborting"
    exit 1
}

echo "Backing up current config..."
cp "$TARGET" "${TARGET}.bak.$(date +%Y%m%d_%H%M%S)"

echo "Installing new config..."
cp "$SSHD_CONFIG" "$TARGET"
chmod 600 "$TARGET"

echo "Restarting sshd..."
systemctl restart sshd && echo "  ✓ sshd restarted successfully" || {
    echo "  ✗ Restart failed — restoring backup"
    cp "${TARGET}.bak."* "$TARGET"
    systemctl restart sshd
    exit 1
}

echo "SSH hardening applied successfully."
