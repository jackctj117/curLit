# Security Architecture — curLit

## Implementation status

The vault is an available component, **not a verified credential boundary for
the whole trading system**. The FX/Alpaca and research paths still consume
environment-provided credentials; the Polymarket secret loader is the current
vault-client integration. Encrypting a vault file does not remove secrets from
those services' environments or from child processes.

The systemd templates are not a working deployment recipe as written: the vault
agent and engine use different service UIDs despite the agent's same-UID peer
check; the agent declares `Type=notify` without readiness notification; and
interactive passphrase entry requires a terminal. CL-t398 tracks this deployment
work. Do not weaken peer authentication to make the templates appear to work.

The layers and operational procedures below describe the intended vault design,
not verified adoption, backup execution, or recovery drills on a running host.
Use `docs/CURRENT_OPERATIONS.md` for recorded operational context; code review
alone does not establish today's deployed state.

## Operational exposure controls (CL-esh6)

These are code-enforced today, independent of the vault:

| Control | Env var | Default | Behavior |
|---|---|---|---|
| Telegram gate approvals | `TELEGRAM_APPROVER_IDS` | unset = **deny** | `approve`/`reject`/`skip` require the sender's Telegram user id (`from.id`) to be listed (comma-separated positive ids). Unset/empty disables them, logged once at bot startup; malformed values fail startup (exit 2). Refusals are logged with the sender id, never the bot token, and change nothing. Messages posted as a chat/channel (`sender_chat`) or by bots never authorize. Read-only `help`/`pending`/`ideas`/`idea` remain open within `TELEGRAM_CHAT_ID`. |
| Prometheus `/metrics` | `METRICS_BIND_ADDR` | `127.0.0.1` | The unauthenticated metrics endpoint (positions, PnL, strategy state) listens on loopback. Exposing it is deliberate: set the variable (e.g. `0.0.0.0` inside a container whose published port is loopback-bound, or a specific private interface behind a firewall); a non-loopback bind is logged at WARNING. |
| `claude` CLI subprocess | — | allowlist | The research LLM driver passes only an explicit allowlist of variables to `claude -p` (process basics, config-dir/subscription-login variables, the output cap, proxy/CA settings — see `_CLI_ENV_ALLOWLIST` in `src/research/llm/claude_code.py`). Broker, DB, messaging and web-API secrets and `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` are withheld. |

Operational steps and deploy consequences: `docs/CURRENT_OPERATIONS.md` §6.

## Layers

```
Layer 1: Paper backup (air-gapped)
  └─ Master passphrase (6 words) + 24-word BIP39 recovery seed

Layer 2: Derived keys (on machine, not sensitive without passphrase)
  └─ Vault master key (PBKDF2-HMAC-SHA256, 600k iterations)
  └─ Salt stored at vault.salt

Layer 3: Encrypted vault (AES-256-GCM)
  └─ vault.enc — all credentials encrypted
  └─ recovery.enc — vault key encrypted with recovery seed

Layer 4: Working services (read from vault agent via Unix socket)
  └─ /run/fx-vault-agent.sock (mode 600, same-UID only)
```

## Vault Initialization (one-time)

```bash
python scripts/initialize_vault.py
# Prints paper recovery document → PRINT NOW
# Store in fireproof safe. Two copies, two locations.
```

## Daily Operations

```bash
# 1. SSH into fx-server (key auth only)
# 2. Start vault agent
systemctl --user start fx-vault-agent
# 3. Type 6-word passphrase
# 4. Start services
systemctl --user start fx-live-engine
```

## Credential Management

```bash
python scripts/vault_add.py add     # Add new credential
python scripts/vault_add.py remove  # Remove credential
python scripts/vault_add.py list    # List names only (safe to screenshare)
```

## Recovery Scenarios

### 1. Forgot passphrase, machine fine
1. Retrieve paper backup from safe
2. Type passphrase from paper

### 2. Paper destroyed, passphrase remembered
1. Log in normally
2. Run `scripts/initialize_vault.py` with `--regenerate` flag
3. Print new paper backup

### 3. Machine dies, disk unrecoverable
1. Provision new machine
2. Restore from backup tarball
3. Unlock vault with remembered passphrase or recovery seed

### 4. Everything destroyed
1. Two copies in two locations prevents this

### 5. Vault file corrupted
1. Restore vault.enc from latest daily backup
2. Re-add any credentials changed since backup

## Backup Schedule

- **Daily**: vault.enc, vault.salt, recovery.enc, DB dump → encrypted tarball
- **Weekly**: backup integrity check
- **Monthly**: credential change verification
- **Quarterly**: full disaster recovery drill

## Credential Rotation (Yearly)

1. Generate new API keys from each service
2. Update vault with new values
3. Test all services with new credentials
4. Revoke old API keys
5. Update paper backup with new passphrase
