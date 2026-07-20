# Polymarket trading runbook

Operator reference for the three-phase Polymarket integration. The
companion design doc is committed alongside this runbook (look for the
"Polymarket Broker Integration — Plan & Reference" markdown the work
was scoped from). Beads tracking each phase: **CL-poly-1**, **CL-poly-2**,
**CL-poly-3**.

## Phase status (as of this runbook's last edit)

| Phase | Bead       | Code | Validation gate |
| :---- | :--------- | :--- | :-------------- |
| 1. Read-only deepening    | CL-poly-1 | shipped (PolymarketDataSource, POLY: symbol resolver, DataProvider already pulls POLY rows) | tests/unit/test_polymarket_data_source.py |
| 2. Paper trading          | CL-poly-2 | shipped (PolymarketPaperBroker, Kelly sizer, cost model, `--broker polymarket-paper`) | tests/unit/test_polymarket_paper_broker.py + ~75 unit tests |
| 3. Live trading           | CL-poly-3 | scaffolds + tests; **mainnet HARD-GATED** | operator must satisfy CL-poly-3 acceptance before flipping `POLYMARKET_MAINNET_UNLOCK=1` |

## Mental model — why this is not OANDA

| Axis                  | OANDA (FX)                       | Polymarket                                                  |
| --------------------- | -------------------------------- | ----------------------------------------------------------- |
| Position unit         | Notional currency                | Share count, each ∈ [0, 1] of USDC at resolution            |
| P&L surface           | Continuous, mark-to-market       | Path-irrelevant; resolves to {0, 1} per share at expiry     |
| Max loss              | Stop-loss bounded                | Equals position cost. **No stop-loss exists.**              |
| Sizer                 | Vol-target (`src/risk/sizing.py`) | Kelly fraction (`src/risk/polymarket_sizer.py`)            |
| Fee                   | Spread + commission              | Maker rebate / taker fee + gas on settlement                |
| Counterparty risk     | OANDA balance sheet              | Polygon CTF Exchange contract + UMA optimistic oracle       |
| Reconciliation source | OANDA REST fills                 | On-chain `OrderFilled` events on Polygon                    |
| Key custody           | API token (vault)                | EIP-712 signing key + on-chain wallet (vault, same model)   |

Read this once. The defaults baked into `PolymarketCostModel`,
`size_polymarket_order`, and `polymarket_preflight` only make sense
once you've internalized the table.

## Operator commands

### Phase 1 / 2 (read + paper, no chain)

```bash
# Discover FX/macro markets, refresh configs/polymarket_markets.yaml
.venv/bin/python -m scripts.discover_polymarket_markets

# Seed historical price-history into prices table (POLY: symbols)
.venv/bin/python -m scripts.seed_polymarket_history

# Run a strategy against the paper broker — no wallet, no chain
.venv/bin/python -m src.runtime.run_engine --broker polymarket-paper
```

### Phase 3 (live — gated)

The `polymarket-mainnet` broker mode raises a `RuntimeError` until
`POLYMARKET_MAINNET_UNLOCK=1` is set in the environment. Setting that
variable is the single documented gate. **Operator must complete
the CL-poly-3 acceptance checklist** (`bd show CL-poly-3`) before
unlocking. No automation flips this for you.

```bash
# 1. Provision the vault on the production server
sudo bash deploy/scripts/init_vault.sh   # CL-r48 — preexisting
# Add the polymarket secrets:
#   secret/trading/polymarket/{env}/{signer_pk, api_key, api_secret,
#                                    api_passphrase, funder_address, rpc_url}

# 2. Install the EVM extras
pip install '.[polymarket]'

# 3. Run Amoy testnet end-to-end (full sign + submit + reconcile)
.venv/bin/python -m src.runtime.run_engine --broker polymarket-amoy

# 4. After Amoy passes + safety review + funded wallet + ramp plan:
export POLYMARKET_MAINNET_UNLOCK=1
.venv/bin/python -m src.runtime.run_engine \
    --broker polymarket-mainnet --confirm-live
```

## Vault layout

```
secret/trading/polymarket/mainnet/signer_pk          # 0x-prefixed hex private key (signer)
secret/trading/polymarket/mainnet/api_key            # uuid from /auth/api-key
secret/trading/polymarket/mainnet/api_secret         # base64 hmac
secret/trading/polymarket/mainnet/api_passphrase     # passphrase string
secret/trading/polymarket/mainnet/funder_address     # 0x... — wallet holding the USDC
secret/trading/polymarket/mainnet/rpc_url            # Alchemy/Infura

secret/trading/polymarket/amoy/                      # mirror for testnet
```

Two key roles:
- **Funder key** (cold/manual): deposits USDC.e, approves `CTFExchange`
  allowance. Used rarely; stays cold. Not stored in the production vault.
- **Signer/API key** (hot, vault-resident): signs orders, submits to
  CLOB. This is what the broker uses at runtime.

## Sizing — Kelly only

`src/risk/polymarket_sizer.py:size_polymarket_order` returns either a
`PolySizeDecision` or `None`:

- `bankroll`: free USDC.
- `market_price`: current limit price ∈ [0.01, 0.99].
- `model_prob`: our q estimate.
- `side`: "YES" or "NO".
- `kelly_fraction` (default 0.25): fractional Kelly scale.
- `max_market_fraction` (default 0.05): hard cap per market.
- `min_edge` (default 0.03): below this, return None.

Returns `None` when:
- edge below `min_edge`,
- wrong-side edge (e.g. `side="YES"` with `q < p`),
- bankroll × fraction rounds to zero shares.

Hard rules baked in:
- Cap aggregate exposure across **correlated markets** (e.g.
  candidate-X-primary AND candidate-X-general). Maintain a market-
  cluster registry in `configs/polymarket_market_clusters.yaml`
  (file ships empty; populate before live runs).
- Treat resolution time as opportunity cost — capital locked in a
  6-month market has carry the Kelly formula ignores. Multiply
  `kelly_fraction` by `1 / (1 + days_to_resolution / 90)` for any
  market resolving more than 30 days out.
- Don't size beyond resting book depth at `price ± slippage_tol`. The
  paper broker enforces this via `OrderBookSnapshot.depth_at_or_better`.

## Settlement / oracle risk

The categorically different layer. UMA Optimistic Oracle resolves each
market; failure modes:

- **Ambiguous question wording.** UMA resolvers vote on the literal
  question. "Will X happen by Date" with fuzzy definition → adversarial
  resolution.
- **Disputed resolution.** Proposed outcome challenged → DVM vote →
  days of delay, market frozen.
- **Pre-resolution freeze.** Trading halts at "market closed" but
  tokens don't redeem until oracle posts. Capital illiquid hours-to-
  days.
- **Wrong-resolution risk.** UMA returns 0.5 (un-resolvable) → both
  YES and NO redeem at $0.50.

Mitigations:
- Question-text gating (TODO file `configs/polymarket_question_denylist.yaml`):
  reject markets with denylist words ("substantially", "majority of
  observers", "as widely reported"). Surface to operator for manual
  approval.
- Prefer machine-checkable resolution sources (CoinGecko price, BLS
  print, election certification). Skip "consensus of media reports".
- Default exit policy: close positions ≥ 24h before market close.
  Per-strategy override.
- `polymarket_oracle_monitor.py` watches `UmaCtfAdapter` for
  `QuestionResolved` events on positions we hold. Wired to alerting
  in CL-poly-3 acceptance.

## Reconciliation

Same shape as CL-unlt (OANDA reconciler), different transport. Sources
in increasing canonicality:

1. Internal journal (`trade_journal_events`).
2. CLOB REST `/data/trades?user=<funder>` — eventual, sometimes lossy.
3. **On-chain `OrderFilled` events** on `CTFExchange` — canonical.

`polymarket_reconciler.py:reconcile()` returns a `ReconcileSummary`
with onchain_count / journal_count / matched / onchain_only /
journal_only counts. The on-chain pass is two passes (maker filter +
taker filter) merged via dedup on `(tx_hash, order_hash)`.

Polygon block time ~2s. Reconcile loop runs every 30s in production
without stressing a paid RPC (Alchemy/Infura). Public RPCs throttle
log queries hard — production must use a paid provider.

## Failure modes that have no FX analogue

- **Oracle dispute.** A market we hold goes into UMA dispute → frozen
  for days. Capital locked. Action: alert via Telegram; don't size new
  positions in correlated markets.
- **Wrong-resolution.** UMA returns 0.5 → both sides redeem at $0.50.
  Action: post-mortem. The position cost is gone modulo whatever you
  paid above 0.50; this is a real expected loss in the cost model that
  operator should track per-resolution.
- **MATIC depleted.** Gas float runs out → fills can't settle → orders
  reject silently. Action: preflight checks MATIC ≥ `_MIN_MATIC_GAS_FLOAT`.
  Watchdog should refill below threshold.
- **Allowance revoked / expired.** Hostile takeover scenario or
  operator panic-revoke. Order placement still works but settlement
  reverts. Action: kill-switch detects via `get_balance_allowance`
  pre-order; cancels open orders if allowance == 0.

## Open questions to resolve before mainnet

Per the design doc §9 (still open as of CL-poly-3 filing):

1. **Funder model:** EOA (signature_type=0) vs magic-link (=2). Plan
   recommends EOA — fewer moving parts, clearer custody.
2. **Approval cap policy:** `MaxUint256` is the SDK default; operator
   must grant working-cap + buffer instead.
3. **RPC provider:** Alchemy vs Infura vs self-host. Mainnet should
   be paid for log-query reliability.
4. **Question-text denylist:** ownership + review cadence (suggested
   quarterly).
5. **Correlated-market clustering:** `event`-based (v1) vs manual (v2).

## Related runbooks + references

- [`docs/runbooks/KillSwitchTriggered.md`](KillSwitchTriggered.md) — kill-switch response (existing CL-a2p)
- [`docs/runbooks/OrderRejected.md`](OrderRejected.md) — generic order-reject playbook
- [`docs/runbooks/DrawdownExceeds15Pct.md`](DrawdownExceeds15Pct.md) — DD response
- [`reference/07_execution.md`](../../reference/07_execution.md) — broker ABC reference
- [`reference/09_security.md`](../../reference/09_security.md) — vault + key handling
