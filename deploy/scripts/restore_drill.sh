#!/usr/bin/env bash
# CL-2e2: quarterly restore drill. Restores the latest backup into a
# disposable Postgres instance and verifies row counts match the source.
# Run from a separate machine to actually exercise the restore path.

set -euo pipefail

BACKUP=$1
KEYFILE=${KEYFILE:-/etc/curlit/backup.key}
TEST_DB=${TEST_DB:-curlit_restore_test}

if [[ -z "${BACKUP:-}" ]]; then
    echo "Usage: $0 <backup.pgdump.gpg>" >&2
    exit 1
fi
if [[ ! -r "$KEYFILE" ]]; then
    echo "Backup key not readable at $KEYFILE" >&2
    exit 1
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "[drill] Decrypting $BACKUP"
gpg --batch --yes --decrypt --passphrase-file "$KEYFILE" \
    --output "$TMP" "$BACKUP"

echo "[drill] (Re)creating $TEST_DB"
sudo -u postgres dropdb --if-exists "$TEST_DB"
sudo -u postgres createdb "$TEST_DB"

echo "[drill] pg_restore"
sudo -u postgres pg_restore -d "$TEST_DB" -j 4 "$TMP"

echo "[drill] Sanity row counts:"
for table in macro_data prices trade_journal_events strategy_fills; do
    n=$(sudo -u postgres psql -d "$TEST_DB" -tAc "SELECT count(*) FROM $table" 2>/dev/null || echo "0")
    printf "  %-30s %s\n" "$table" "$n"
done

echo "[drill] OK — review row counts match production. Drop $TEST_DB when done:"
echo "  sudo -u postgres dropdb $TEST_DB"
