# DrawdownExceeds15Pct

## Severity: CRITICAL

## What it means
Portfolio drawdown from peak equity has exceeded 15%.

## Immediate actions
1. Open incident dashboard
2. Review which strategies contributed to losses
3. Check correlation regime — are all strategies losing together?
4. Verify kill switches already triggered (drawdown_limit at 20%)

## Common causes
- Multi-strategy correlation regime shift (all strategies losing together)
- Major unexpected market event
- Strategy model drift (R² degraded, signal generating bad trades)

## Resolution
1. If correlation crisis: wait for regime to normalize before resuming
2. If model drift: refit models on recent data, re-run backtests
3. If market event: assess whether thesis still holds post-event
4. Resume only after drawdown recovers above 10% or 30 days pass

## Escalation
Review with fresh eyes after 24 hours before making major changes.
