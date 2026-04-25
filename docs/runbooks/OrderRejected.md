# OrderRejected

## Severity: WARNING (varies by class)

## What it means
The broker (OANDA) rejected an order. The OMS classified the rejection and applied the configured policy. Most rejections are auto-handled; persistent ones surface as alerts.

## Rejection classes + auto-resolution

| Class | Trigger examples | Auto-resolution | Operator action if persistent |
|---|---|---|---|
| `liquidity` | FOK fail, "no liquidity at price" | Retry up to 3× halving size each attempt | Inspect spread / regime; consider widening max_slippage_bps |
| `margin` | "insufficient margin" | Abort + alert (size won't help) | Add capital or reduce gross exposure; check pre-trade gate calibration |
| `halt` | "instrument halted", "market closed" | Abort + halt the originating strategy | Confirm with broker; disable strategy until instrument resumes |
| `transient` | timeout, connection reset, 5xx | Retry up to 4× with exponential backoff (1s, 2s, 4s) | Check broker connectivity; consider failover |
| `malformed` | 400 / 422 / "invalid payload" | Abort + alert | This is a bug — escalate immediately, capture order payload |
| `unknown` | anything unclassified | Abort (conservative) | Read latest log; consider adding a classification pattern |

Defaults are in `RejectionPolicy.default()` in `src/execution/rejection.py`. Per-class behavior is configurable via `OrderManager(rejection_handler=...)` wiring.

## Immediate actions
1. Identify the rejection class from the alert / log: search for `Order reject` in the live engine logs.
2. Confirm the auto-resolution behaved as expected (Prometheus: `fx_orders_rejected_total{pair, reason}`).
3. If `halt` fired: confirm the strategy halt callback paused signal generation for the affected strategy.

## Common causes
- **Liquidity spikes**: low-liquidity windows (US close → Asia open) compounded with high regime volatility.
- **Margin exhaustion**: drawdown reduced equity below the pre-trade gate threshold; D1 PortfolioCoordinator scaling should normally prevent this — investigate why it didn't.
- **Halts**: scheduled (weekend, holiday) or unscheduled (broker maintenance, exchange events).
- **Transient**: typical broker API hiccups; auto-retry usually resolves.
- **Malformed**: programming error in our code or upstream change in OANDA's API contract.

## Resolution
1. Auto-resolution should already have happened. Verify via Prometheus that the counter incremented for the expected class.
2. For `halt`, decide whether to manually re-enable the strategy after the instrument resumes trading.
3. For repeated `liquidity` on the same pair: review `max_slippage_bps` and pair selection.
4. For repeated `margin`: PortfolioCoordinator is not constraining gross enough; review constraints.

## Escalation
- Persistent `malformed` → investigate IMMEDIATELY (likely API contract change or our bug).
- Persistent `unknown` → IMMEDIATELY add a classification pattern; an unexplained reject is a silent failure mode.
- `halt` without a known broker maintenance window → contact broker support to confirm.

## Related
- Pre-trade gate (CL-srsy) — rejects before broker sees order; orders that reach this runbook passed pre-trade.
- `docs/runbooks/KillSwitchTriggered.md` — kill switches act on aggregate behavior; rejection handler acts per-order.
