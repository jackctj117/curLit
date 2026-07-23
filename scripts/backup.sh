#!/usr/bin/env bash
# Secure backup entrypoint (CL-fg46 / E11 remediation).
#
# The former body of this script was a security footgun: it staged the
# Postgres dump and the vault artifacts as PLAINTEXT in a world-readable
# /tmp directory and produced an UN-encrypted .tar.gz — anyone on the box
# could read the whole vault + database out of /tmp or the final archive.
#
# The canonical daily backup is deploy/scripts/backup.sh (CL-2e2): it
# gpg-encrypts every artifact (AES256, key from /etc/curlit/backup.key),
# never leaves an unencrypted archive, and prunes on a retention schedule.
# This wrapper delegates there, so any cron/doc still pointing at
# scripts/backup.sh gets the SECURE path instead of the old plaintext one.
# Environment overrides (BACKUP_DIR, POSTGRES_DB, ...) pass straight through.
#
# It fails LOUD in an environment the hardened script isn't provisioned for
# (e.g. a missing /etc/curlit/backup.key) rather than silently falling back
# to an insecure backup — a failed backup is recoverable, a leaked one is not.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARDENED="${HERE}/../deploy/scripts/backup.sh"

if [[ ! -f "$HARDENED" ]]; then
    echo "Hardened backup script not found at $HARDENED" >&2
    exit 1
fi

exec bash "$HARDENED" "$@"
