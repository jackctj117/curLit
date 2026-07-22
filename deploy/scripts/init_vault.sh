#!/usr/bin/env bash
# CL-r48 / CL-8lv6: initialize vault on a fresh server. Collects operator
# credentials, seals them with scripts/initialize_vault.py's re-seal mode,
# then registers the vault-agent systemd unit. Run ONCE per server.
#
# Secret hygiene (CL-8lv6 P0 — the old version cat'd the full credentials
# JSON into a mktemp file):
#   * plaintext NEVER touches disk — the JSON is built in a child Python's
#     memory from exported env vars and handed over as a /dev/fd/N pipe;
#   * no secret value is ever interpolated onto a command line (argv is
#     world-readable via ps) — children read them from the environment;
#   * the passphrase goes over stdin via the printf BUILTIN (no argv, no
#     temp file — bash herestrings/heredocs may hit disk, so none are used
#     for secrets).
#
# Idempotent against an already-initialized vault: refuses to overwrite
# unless --force is passed.
#
# Local/dev validation (no root, no systemd):
#   VAULT_DIR=/tmp/x CURLIT_SKIP_SYSTEMD=1 OANDA_API_KEY=... VAULT_PASS=... \
#       bash deploy/scripts/init_vault.sh
# Any credential var already set in the environment skips its prompt.

set -euo pipefail

REPO_ROOT=${CURLIT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${CURLIT_PYTHON:-${REPO_ROOT}/.venv/bin/python}
VAULT_DIR=${VAULT_DIR:-/etc/curlit/vault}
VAULT_FILE=${VAULT_FILE:-${VAULT_DIR}/credentials.enc}
SALT_FILE=${SALT_FILE:-${VAULT_DIR}/vault.salt}
VAULT_OWNER=${CURLIT_VAULT_OWNER:-fx-vault}
SKIP_SYSTEMD=${CURLIT_SKIP_SYSTEMD:-0}
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -e "$VAULT_FILE" && $FORCE -eq 0 ]]; then
    echo "Vault already exists at $VAULT_FILE — pass --force to recreate" >&2
    exit 1
fi

if [[ $EUID -ne 0 && $SKIP_SYSTEMD -ne 1 ]]; then
    echo "Must run as root (or set CURLIT_SKIP_SYSTEMD=1 for a local dry run)" >&2
    exit 1
fi

if [[ $EUID -eq 0 ]]; then
    install -d -o "$VAULT_OWNER" -g "$VAULT_OWNER" -m 0700 "$VAULT_DIR"
else
    install -d -m 0700 "$VAULT_DIR"
fi

# Prompt for $1 unless it is already set in the environment — even set-empty
# counts as "provided" so the non-interactive/dev path can skip optional
# credentials with VAR=''. Echo is disabled for secrets so keystrokes don't
# reach scrollback. The value is EXPORTED for the JSON builder child below —
# it is never placed on a command line.
prompt_var() {
    local name=$1 mode=${2:-plain} label=${3:-$1}
    local val
    if [[ -z ${!name+x} ]]; then
        if [[ $mode == silent ]]; then
            IFS= read -r -s -p "${label}: " val; echo
        else
            IFS= read -r -p "${label}: " val
        fi
        printf -v "$name" '%s' "$val"
    fi
    export "${name?}"
}

prompt_var OANDA_API_KEY silent
prompt_var OANDA_ACCOUNT_ID plain
prompt_var FRED_API_KEY silent
prompt_var POSTGRES_PASSWORD silent
prompt_var ANTHROPIC_API_KEY silent "ANTHROPIC_API_KEY (blank to skip)"
prompt_var TELEGRAM_BOT_TOKEN silent "TELEGRAM_BOT_TOKEN (blank to skip)"
prompt_var TELEGRAM_CHAT_ID plain "TELEGRAM_CHAT_ID (blank to skip)"

# Master passphrase: deliberately NOT exported — it travels via stdin only.
if [[ -z ${VAULT_PASS:-} ]]; then
    IFS= read -r -s -p "Vault master passphrase (used to unlock at boot): " VAULT_PASS; echo
    IFS= read -r -s -p "Confirm passphrase: " VAULT_PASS_CONFIRM; echo
    if [[ "$VAULT_PASS" != "$VAULT_PASS_CONFIRM" ]]; then
        echo "Passphrase mismatch" >&2
        exit 1
    fi
fi

# --force means RECREATE: move the old files aside first so initialize_vault
# sees a NEW vault and enforces the strong-passphrase policy (an existing file
# only warns — that leniency is for the rotation re-seal path, not fresh
# deploys). Timestamped .bak (same convention as rotate_secrets) rather than
# rm: if the recreate then fails, the old vault is still recoverable.
if [[ $FORCE -eq 1 ]]; then
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    for f in "$VAULT_FILE" "$SALT_FILE"; do
        if [[ -e "$f" ]]; then
            mv "$f" "${f}.bak-${STAMP}"
            echo "Kept old $(basename "$f") as ${f}.bak-${STAMP}" >&2
        fi
    done
fi

# Build the credentials JSON in a child Python's memory from the exported env
# vars (proper JSON escaping; a heredoc-interpolated template would corrupt on
# quotes/backslashes AND put secrets in a file). The heredoc below contains no
# secrets — it's just the program text. exec pins the process-substitution
# pipe to fd 9 (fixed number: bash 3.2 has no {var}< syntax) so the sealer
# child inherits it as /dev/fd/9.
exec 9< <("$PYTHON" - <<'PY'
import json, os
keys = (
    "OANDA_API_KEY", "OANDA_ACCOUNT_ID", "FRED_API_KEY",
    "POSTGRES_PASSWORD", "ANTHROPIC_API_KEY",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
)
print(json.dumps({k: os.environ.get(k, "") for k in keys}))
PY
)

# Seal: plaintext over the inherited pipe fd, passphrase over stdin (printf is
# a bash builtin — the passphrase never appears in a process list). The Python
# side writes credentials.enc atomically with mode 0600 and creates/reuses the
# salt file. Runs as the CURRENT user (root on deploy) and chowns after — an
# inherited fd does not survive sudo, which closes non-std fds by default.
printf '%s\n' "$VAULT_PASS" | PYTHONPATH="$REPO_ROOT" "$PYTHON" -m scripts.initialize_vault \
    --plaintext /dev/fd/9 \
    --encrypted "$VAULT_FILE" \
    --salt "$SALT_FILE" \
    --passphrase-stdin
exec 9<&-

chmod 0400 "$VAULT_FILE" "$SALT_FILE"
if [[ $EUID -eq 0 ]]; then
    chown "$VAULT_OWNER:$VAULT_OWNER" "$VAULT_FILE" "$SALT_FILE"
fi

# Install systemd unit (skipped on local/dev runs)
if [[ $EUID -eq 0 && $SKIP_SYSTEMD -ne 1 ]]; then
    install -m 0644 "$REPO_ROOT/deploy/systemd/fx-vault-agent.service" \
        /etc/systemd/system/fx-vault-agent.service
    systemctl daemon-reload
    systemctl enable fx-vault-agent
fi

echo "Vault initialized at $VAULT_FILE"
echo "Start with:  sudo systemctl start fx-vault-agent"
echo "             (you'll be prompted for the master passphrase)"
