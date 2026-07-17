```markdown
# Hypothesis: Robust shrinkage covariance estimation improves GMVP out-of-sample Sharpe in G10 currencies under heavy-tailed returns

## Source extract
- **path**: `data/research/extracts/68bf362a513f8b891483134f63473911fb8ab71af819c283ff53ea0503532b75.md`
- **paper title**: "The Decision Geometry of Covariance Estimation for the Global Minimum-Variance Portfolio under Heavy Tails"
- **paper hash**: `68bf362a513f8b891483134f63473911fb8ab71af819c283ff53ea0503532b75`

## Change to baseline
None – this introduces a new strategy class. The codebase currently contains no Global Minimum‑Variance Portfolios (GMVP). The hypothesis will implement a monthly‑rebalanced GMVP (long‑only, weights sum to 1) over a G10 currency basket, with covariance estimated using a shrinkage estimator that shrinks the sample covariance toward a constant‑correlation target. The baseline for comparison is the same GMVP using the standard sample covariance matrix. (Both strategies are new; no pre‑existing file is altered.)

## Prediction
- `predicted_sharpe_range`: [0.30, 0.60]               # annualized OOS for the robust GMVP
- `predicted_hit_rate_range`: [0.52, 0.65]             # fraction of months with positive portfolio return
- `expected_n_trades_per_year`: 12                     # monthly rebalance
- `regime_dependence`: vol-spike-only                  # advantage of robust estimation expected to materialise most clearly in high‑volatility / heavy‑tail clusters
- `time_to_signal`: monthly                            # weights are re‑estimated on daily returns but held constant for one month

## Abandon condition
If the annualised OOS Sharpe ratio of the robust‑covariance GMVP is **lower than** that of the sample‑covariance GMVP over the first complete rolling 12‑month window (252 trading days) of walk‑forward data, abandon the hypothesis.

## Data requirements
- Daily close prices for the following FX pairs (all vs USD), available via `prices.{symbol}`:
  - `EURUSD`, `GBPUSD`, `AUDUSD`, `NZDUSD`, `USDCAD`, `USDJPY`, `USDCHF`
- The implementer must apply quoting‑convention logic to construct consistent USD‑numeraire daily simple/log returns for each currency (e.g., for pairs where USD is the base, invert the price to obtain the USD value of one unit of foreign currency).
- No data beyond daily close is required; all symbols are assumed to be already ingested into the project’s `prices` table. (If any are absent, flag as `NOT YET INGESTED` before implementation.)

## References
- `data/research/extracts/68bf362a513f8b891483134f63473911fb8ab71af819c283ff53ea0503532b75.md`

## Final position
**FINAL_POSITION**: PROPOSED

The Fonseca (2026) paper provides a theoretical basis for robust covariance estimation in minimum‑variance portfolios under fat‑tailed return distributions. Although the paper itself does not prescribe a specific estimator, the derived decision geometry justifies testing a practical, well‑known robust estimator (shrinkage to constant correlation) against the classic sample covariance in a G10 currency basket – a domain known for heavy tails. The hypothesis is specific, falsifiable via a measurable abandon condition, uses only data already available (daily close prices), and fits the supported joint multi‑asset backtest mode (CL‑40n2 v2). The idea is therefore a valid candidate for implementation.
```