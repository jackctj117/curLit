```markdown
# Hypothesis: Expanding the candidate feature space and then applying basis pursuit to identify a sparse set of priced FX factors will yield out-of-sample performance that dominates a ridgeless (dense) benchmark.

## Source extract
- **path**: `data/research/extracts/ec4509ecfdad01bd088d81f83cd09b41943273682ffa4981cbb789b0b15ba57a.md`
- **paper title**: The Virtue of Sparsity in Complexity
- **paper hash**: `ec4509ecfdad01bd088d81f83cd09b41943273682ffa4981cbb789b0b15ba57a`

## Change to baseline
None — this is a new strategy class. The existing codebase contains single-factor momentum and mean-reversion strategies (e.g., `src/strategies/momentum.py`, `src/strategies/mean_reversion.py`), but no strategy that explicitly constructs a high-dimensional feature space and then applies a sparsity-inducing selection (basis pursuit / L1 regularization) to identify a sparse set of priced factors. This hypothesis proposes a new `SparseFactorFX` strategy.

## Prediction
- `predicted_sharpe_range`: [0.3, 0.8]            # annualized OOS; estimate derived from comparable sparse-factor papers in FX, not directly from this extract
- `predicted_hit_rate_range`: [0.52, 0.60]          # 0.0–1.0
- `expected_n_trades_per_year`: 52                  # weekly rebalancing
- `regime_dependence`: "regime-agnostic"
- `time_to_signal`: "weekly"

## Abandon condition
- OOS Sharpe < 0 over any rolling 6-month window.

## Data requirements
- `prices.{symbol}` for the primary FX pair (e.g., EURUSD, GBPUSD) — already ingested.
- The strategy requires constructing a high-dimensional feature space from the single close series. This can be done via lagged returns, rolling moments (volatility, skew, kurtosis), rolling correlations with a small set of macro proxies (e.g., US 10Y yield from FRED `DGS10`, VIX from FRED `VIXCLS`), and calendar-based dummies. All of these are derivable from the single `close` column plus existing FRED series in `src/data/fred.py`. No new data ingestion is required.

## References
- `data/research/extracts/ec4509ecfdad01bd088d81f83cd09b41943273682ffa4981cbb789b0b15ba57a.md`
- Didisheim et al. (2025) — cited in the extract but not ingested.

## Final position
**FINAL_POSITION**: PROPOSED

The extract provides a clear, falsifiable thesis: expanding the candidate feature space and then applying basis pursuit to identify a sparse set of factors should outperform a dense (ridgeless) benchmark. The principle is transferable to FX, and the required data (a single close series plus a few FRED macro series) is already available in the codebase. The abandon condition is measurable. The prediction ranges are conservative and grounded in comparable sparse-factor FX literature. This hypothesis is ready for implementation.
```