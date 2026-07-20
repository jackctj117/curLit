# KillSwitchTriggered

## Severity: CRITICAL

## What it means
A kill switch has fired — an automated risk rule was violated. Positions may have been flattened or trading halted.

## Immediate actions
1. Open incident dashboard: http://localhost:3000/d/fx-incident
2. Identify which switch fired from alert details
3. Check current positions match kill switch action
4. Review logs: `{service="live_engine"} |~ "kill|trigger|halt"` in Loki

## Common causes
- **drawdown_limit**: Portfolio drew down past threshold
- **reconciliation_failure**: Internal vs broker positions mismatched
- **vix_spike**: Risk-off regime triggered
- **equity_trailing_stop**: Equity fell 10% below the persisted all-time peak
- **open_position_correlation**: Open positions are effectively one trade (mean direction-adjusted corr > 0.85)

Switches are evaluated by the engine health tick every 60s — full list and thresholds in `docs/OPERATOR.md` and `configs/risk_profile.yaml`.

## Resolution
1. Confirm root cause
2. If data/position issue: fix reconciliation, verify clean state
3. If legitimate market event: review whether strategy exposure should remain reduced
4. Reset via web UI: POST /api/system/resume
5. **equity_trailing_stop only**: the cooldown (default 7 days) lives in `data/equity_trailing_stop_state.json` and survives restarts — the switch stays latched until the cooldown expires and a fresh equity mark arrives. Do not hand-edit the state file to resume without PM sign-off; a corrupt file will stop the engine from booting (by design).

## Escalation
If unsure, LEAVE THINGS HALTED. Do not resume trading to "fix" a problem.
