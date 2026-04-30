```markdown
# Hypothesis: A Signal Credibility Index (SCI) applied to FX price moves can identify self-fulfilling coordination regimes with higher Sharpe than baseline momentum

## Source extract
- **path**: `data/research/extracts/a8e0811487f74c5aebf744506d5f919637ab4f22295bb1ea1e78458f28d303fa.md`
- **paper title**: Price as Focal Point: Prediction Markets, Conditional Reflexivity, and the Politics of Common Knowledge
- **paper hash**: `a8e0811487f74c5aebf744506d5f919637ab4f22295bb1ea1e78458f28d303fa`

## Change to baseline
None — this is a new strategy class. The closest existing strategy is `src/strategies/momentum.py`, which uses a simple rolling return as signal. This hypothesis replaces that with a multi-component Signal Credibility Index (SCI) combining variance ratio, two-sidedness diagnostic, and a trader-concentration proxy (approximated via price impact per unit return). The SCI is used as a regime filter: only enter momentum trades when SCI exceeds a threshold, indicating the move has "behavioral traction."

## Prediction
- `predicted_sharpe_range`: [0.3, 0.7]            # annualized OOS; estimate derived from comparable regime-filtered momentum papers; not directly from this extract
- `predicted_hit_rate_range`: [0.52, 0.60]         # 0.0–1.0
- `expected_n_trades_per_year`: 40
- `regime_dependence`: "trend-following"
- `time_to_signal`: "daily"

## Abandon condition
- OOS Sharpe < 0 over any rolling 6-month window, OR realized hit rate < 48% over the most recent 50 trades.

## Data requirements
- `prices.{symbol}` — daily close price for the primary FX symbol (e.g., EURUSD)
- No additional datasets required. The SCI components are computed from the single close series:
  - Variance ratio VR(6) — computed from daily returns
  - Two-sidedness diagnostic — computed from the distribution of daily returns (skewness / kurtosis)
  - Trader-concentration adjustment — approximated by the ratio of absolute return to the number of days in the window (a crude proxy for price impact per unit return)

## References
- `data/research/extracts/a8e0811487f74c5aebf744506d5f919637ab4f22295bb1ea1e78458f28d303fa.md`

## Final position
**FINAL_POSITION**: PROPOSED

The extract provides a falsifiable thesis: a Signal Credibility Index (SCI) can distinguish price moves that are self-fulfilling coordination devices from noise. The SCI components (variance ratio, two-sidedness, trader-concentration) can be approximated from a single daily close series, making this testable within the current backtest harness constraints. The abandon condition is measurable. The predicted Sharpe range is conservative and grounded in comparable regime-filtered momentum literature. The thesis is adjacent to FX (price moves as coordination devices among traders and institutions) and does not require data not yet ingested.
```