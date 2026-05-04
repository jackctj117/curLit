#!/usr/bin/env bash
# CL-r48: initialize vault on a fresh server. Generates an encrypted
# credentials file from operator input, then registers the vault-agent
# systemd unit. Run ONCE per server.
#
# Idempotent against an already-initialized vault: refuses to overwrite
# unless --force is passed.

set -euo pipefail

VAULT_DIR=/etc/curlit/vault
VAULT_FILE=${VAULT_DIR}/credentials.enc
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -e "$VAULT_FILE" ]] && [[ $FORCE -eq 0 ]]; then
    echo "Vault already exists at $VAULT_FILE — pass --force to recreate" >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root" >&2
    exit 1
fi

install -d -o fx-vault -g fx-vault -m 0700 "$VAULT_DIR"

# Prompt for credentials. Echo is disabled per-prompt so the keystrokes
# don't leak to scrollback.
read -r -p "OANDA_API_KEY: " -s OANDA_API_KEY; echo
read -r -p "OANDA_ACCOUNT_ID: " OANDA_ACCOUNT_ID
read -r -p "FRED_API_KEY: " -s FRED_API_KEY; echo
read -r -p "POSTGRES_PASSWORD: " -s POSTGRES_PASSWORD; echo
read -r -p "ANTHROPIC_API_KEY (blank to skip): " -s ANTHROPIC_API_KEY; echo
read -r -p "PUSHOVER_USER_KEY (blank to skip): " -s PUSHOVER_USER_KEY; echo
read -r -p "PUSHOVER_API_TOKEN (blank to skip): " -s PUSHOVER_API_TOKEN; echo

read -r -p "Vault master passphrase (used to unlock at boot): " -s VAULT_PASS; echo
read -r -p "Confirm passphrase: " -s VAULT_PASS_CONFIRM; echo
if [[ "$VAULT_PASS" != "$VAULT_PASS_CONFIRM" ]]; then
    echo "Passphrase mismatch" >&2
    exit 1
fi

# Build a JSON credentials blob, then encrypt with the same scheme the
# vault agent uses (AES-GCM via cryptography.hazmat).
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<EOF
{
  "OANDA_API_KEY": "$OANDA_API_KEY",
  "OANDA_ACCOUNT_ID": "$OANDA_ACCOUNT_ID",
  "FRED_API_KEY": "$FRED_API_KEY",
  "POSTGRES_PASSWORD": "$POSTGRES_PASSWORD",
  "ANTHROPIC_API_KEY": "$ANTHROPIC_API_KEY",
  "PUSHOVER_USER_KEY": "$PUSHOVER_USER_KEY",
  "PUSHOVER_API_TOKEN": "$PUSHOVER_API_TOKEN"
}
EOF

# Use the project's encryption helper so the format matches what
# vault_agent.decrypt_vault expects at runtime.
sudo -u fx-vault /opt/curlit/.venv/bin/python -m scripts.initialize_vault \
    --plaintext "$TMP" --encrypted "$VAULT_FILE" --passphrase-stdin <<< "$VAULT_PASS"

chmod 0400 "$VAULT_FILE"
chown fx-vault:fx-vault "$VAULT_FILE"

# Install systemd unit
install -m 0644 deploy/systemd/fx-vault-agent.service \
    /etc/systemd/system/fx-vault-agent.service
systemctl daemon-reload
systemctl enable fx-vault-agent

echo "Vault initialized at $VAULT_FILE"
echo "Start with:  sudo systemctl start fx-vault-agent"
echo "             (you'll be prompted for the master passphrase)"
