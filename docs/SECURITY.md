# Security Architecture — curLit

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
