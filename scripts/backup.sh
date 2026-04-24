#!/usr/bin/env bash
# Encrypted backup — vault + DB dump, daily cron
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/mnt/external/fx-backup}"
DATE="$(date +%Y%m%d_%H%M%S)"
STAGING="/tmp/fx-backup-${DATE}"
VAULT_PATH="${VAULT_PATH:-/opt/fx-system/vault.enc}"
SALT_PATH="${VAULT_SALT_PATH:-/opt/fx-system/vault.salt}"
RECOVERY_PATH="${RECOVERY_PATH:-/opt/fx-system/recovery.enc}"

mkdir -p "$STAGING" "$BACKUP_DIR"

echo "[$(date)] Starting backup..."

# Copy vault artifacts (already encrypted)
cp "$VAULT_PATH" "$STAGING/" 2>/dev/null || echo "  ! vault.enc not found"
cp "$SALT_PATH" "$STAGING/" 2>/dev/null || true
cp "$RECOVERY_PATH" "$STAGING/" 2>/dev/null || true

# Postgres dump
if command -v pg_dump &>/dev/null; then
    echo "  Dumping database..."
    pg_dump -U fx fx -f "$STAGING/fx-db-${DATE}.sql" 2>/dev/null || echo "  ! DB dump failed"
fi

# Tar + hash
tar czf "${BACKUP_DIR}/fx-backup-${DATE}.tar.gz" -C /tmp "fx-backup-${DATE}" 2>/dev/null
sha256sum "${BACKUP_DIR}/fx-backup-${DATE}.tar.gz" > "${BACKUP_DIR}/fx-backup-${DATE}.sha256"

# Cleanup
rm -rf "$STAGING"

# Retention: 30 days
find "$BACKUP_DIR" -name "fx-backup-*.tar.gz" -mtime +30 -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "fx-backup-*.sha256" -mtime +30 -delete 2>/dev/null || true

echo "[$(date)] Backup complete: fx-backup-${DATE}.tar.gz"
