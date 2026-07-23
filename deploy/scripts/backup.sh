#!/usr/bin/env bash
# CL-2e2: daily encrypted backup. Runs from cron at 03:00 UTC.
# Keeps 7 daily + 4 weekly + 12 monthly snapshots.
#
# Encryption: gpg symmetric with the passphrase from /etc/curlit/backup.key
# (mode 0400, owned by fx-operator). Quarterly restore drill is a
# separate script (restore_drill.sh below) that exercises the snapshot.

set -euo pipefail

BACKUP_DIR=/var/backups/curlit
PG_DB=${POSTGRES_DB:-fx}
PG_USER=${POSTGRES_USER:-fx}
KEYFILE=/etc/curlit/backup.key
DATE=$(date -u +%Y%m%d_%H%M%S)
NAME="curlit-${DATE}"

if [[ ! -r "$KEYFILE" ]]; then
    echo "Backup key not readable at $KEYFILE" >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR/daily" "$BACKUP_DIR/weekly" "$BACKUP_DIR/monthly"

# ----------------------------------------------------------------------
# Postgres dump (custom format — fast restore, parallel-friendly).
# Streamed straight into gpg (CL-fg46/E11): the plaintext dump — the whole
# database — never touches disk, so there is no window where it sits
# unencrypted next to the key material. pipefail makes a pg_dump failure
# fail the whole step loudly instead of sealing a truncated dump.
# ----------------------------------------------------------------------
ENC_DUMP="$BACKUP_DIR/daily/${NAME}.pgdump.gpg"
echo "[backup] pg_dump $PG_DB → $ENC_DUMP (streamed, no plaintext on disk)"
sudo -u postgres pg_dump -Fc -d "$PG_DB" -U "$PG_USER" \
    | gpg --batch --yes --symmetric --cipher-algo AES256 \
        --passphrase-file "$KEYFILE" \
        --output "$ENC_DUMP"

# ----------------------------------------------------------------------
# Tarball of operator state — vault, configs, recent reports — streamed
# straight into gpg for the same reason (no plaintext .tar intermediate).
# The vault file is already encrypted, but configs can carry secrets, so
# the archive itself is never written in the clear. A missing optional dir
# is tolerated (|| true on the subshell) without dropping pipefail on gpg.
# Trade journal lives in Postgres so the dump above captures it.
# ----------------------------------------------------------------------
ENC="${BACKUP_DIR}/daily/${NAME}.tar.gpg"
{ tar -cf - \
    -C /etc/curlit . \
    -C /opt/curlit configs \
    -C /opt/curlit reports/reconciliation 2>/dev/null || true ; } \
    | gpg --batch --yes --symmetric --cipher-algo AES256 \
        --passphrase-file "$KEYFILE" \
        --output "$ENC"

# ----------------------------------------------------------------------
# Promote: weekly = Sunday's daily, monthly = 1st-of-month's daily.
# ----------------------------------------------------------------------
DOW=$(date -u +%u)
DOM=$(date -u +%d)
if [[ $DOW == 7 ]]; then
    cp "$ENC" "$BACKUP_DIR/weekly/" && cp "$ENC_DUMP" "$BACKUP_DIR/weekly/"
fi
if [[ $DOM == 01 ]]; then
    cp "$ENC" "$BACKUP_DIR/monthly/" && cp "$ENC_DUMP" "$BACKUP_DIR/monthly/"
fi

# ----------------------------------------------------------------------
# Retention: 7 daily, 4 weekly, 12 monthly. -mtime is in days.
# ----------------------------------------------------------------------
find "$BACKUP_DIR/daily"   -name '*.gpg' -mtime +7   -delete
find "$BACKUP_DIR/weekly"  -name '*.gpg' -mtime +28  -delete
find "$BACKUP_DIR/monthly" -name '*.gpg' -mtime +365 -delete

echo "[backup] Done. Latest: $ENC ($(du -h "$ENC" | cut -f1))"
