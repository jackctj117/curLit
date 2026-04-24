#!/usr/bin/env bash
# Install systemd unit files for curLit
# Run as root: sudo ./configs/systemd/install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

echo "Installing curLit systemd unit files..."

for unit in fx-live-engine.service fx-risk-monitor.service fx-watchdog.service; do
    echo "  Installing $unit..."
    cp "$SCRIPT_DIR/$unit" "$SYSTEMD_DIR/$unit"
    chmod 644 "$SYSTEMD_DIR/$unit"
    systemctl daemon-reload
    echo "    ✓ $unit installed"
done

echo ""
echo "All units installed. To enable on boot:"
echo "  systemctl enable fx-live-engine fx-risk-monitor fx-watchdog"
echo ""
echo "To start now:"
echo "  systemctl start fx-live-engine fx-risk-monitor fx-watchdog"
