#!/usr/bin/env bash
# Restore from encrypted backup
set -euo pipefail

echo "=== curLit Restore ==="
echo ""

TARBALL="${1:-}"
if [ -z "$TARBALL" ]; then
    read -rp "Backup tarball path: " TARBALL
fi

if [ ! -f "$TARBALL" ]; then
    echo "ERROR: $TARBALL not found"
    exit 1
fi

DRY_RUN="${2:-}"
STAGING="/tmp/fx-restore-$$"

echo "Extracting backup..."
mkdir -p "$STAGING"
tar xzf "$TARBALL" -C "$STAGING"

echo "Contents:"
find "$STAGING" -type f

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "Dry run complete — no changes made."
    rm -rf "$STAGING"
    exit 0
fi

read -rp "Restore to /opt/fx-system? [y/N] " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    rm -rf "$STAGING"
    exit 0
fi

# Restore vault artifacts
cp "$STAGING"/*/vault.enc /opt/fx-system/ 2>/dev/null || true
cp "$STAGING"/*/vault.salt /opt/fx-system/ 2>/dev/null || true
cp "$STAGING"/*/recovery.enc /opt/fx-system/ 2>/dev/null || true

# Restore DB
SQL_FILE="$(find "$STAGING" -name "*.sql" | head -1)"
if [ -n "$SQL_FILE" ]; then
    echo "Restoring database..."
    psql -U fx fx < "$SQL_FILE" 2>/dev/null && echo "  ✓ DB restored"
fi

rm -rf "$STAGING"
echo "Restore complete. Verify services: systemctl status fx-live-engine"
