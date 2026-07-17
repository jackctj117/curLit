```markdown
# Hypothesis: Trend-following signals in FX should only be traded on pairs with high volatility-normalized tick size, as small-tick pairs degrade trend profits; testing a filtered SMA crossover strategy

## Source extract
- **path**: `data/research/extracts/a8296ffbb36e1f8ca670a59a67b6a086b12d2d95ecb0ac6b32a54db11f719c12.md`
- **paper title**: "Is Trend Still Your Friend?: A Microstructural Account of the Demise of Short-Term Trend-Following"
- **paper hash**: `a8296ffbb36e1f8ca670a59a67b6a086b12d2d95ecb0ac6b32a54db11f719c12`

## Change to baseline
The baseline for this hypothesis is an unconditional daily trend-following strategy (SMA(20)/SMA(50) crossover) applied uniformly to a basket of major FX pairs. This strategy would be implemented in a new file (e.g., `src/strategies/trend_filtered_tick_size.py`). It modifies the baseline by introducing a pre‑trade filter: trend signals are accepted only for pairs where the ratio `tick_size / (volatility * price)` exceeds a fixed threshold (the median of historical values). Pairs that fail the filter are excluded from the portfolio on that day, thereby selectively trading only “large‑tick” contracts where the paper finds trends remain intact. (The codebase currently does not contain a tick‑size‑filtered trend strategy; this is a new strategy class.)

## Prediction
- `predicted_sharpe_range`: [0.3, 0.7] annualised OOS, after typical transaction costs
- `predicted_hit_rate_range`: [0.45, 0.55]
- `expected_n_trades_per_year`: 200 (aggregate across all pairs in the basket)
- `regime_dependence`: "trend-following"
- `time_to_signal`: "daily"

(Estimate derived from comparable published CTA variations; the paper does not quote a precise Sharpe but indicates that post‑2008 large‑tick trends are still economically meaningful. A filter that discards ~50% of signals should improve the Sharpe by roughly 20‑30% over the unconditional baseline, placing the filtered strategy around 0.3‑0.7 depending on the universe and walk‑forward period.)

## Abandon condition
- OOS Sharpe < 0 over any rolling 6‑month window

## Data requirements
- Close prices for the following FX pairs (all already ingested via `src/data/fred.py` and the `prices` table, using symbols `EURUSD`, `USDJPY`, `GBPUSD`, `AUDUSD`, `NZDUSD`, `USDCAD`, `USDCHF`, `EURJPY`, `EURCHF`, `GBPJPY`). A broader G10 basket may be added if desired.
- Fixed tick sizes (pip values) for each pair: these are constants (e.g., `EURUSD` 0.0001, `USDJPY` 0.01) and will be hardcoded in the strategy. No additional data ingestion is required.

## References
- the source extract: `data/research/extracts/a8296ffbb36e1f8ca670a59a67b6a086b12d2d95ecb0ac6b32a54db11f719c12.md`

## Final position
**FINAL_POSITION**: PROPOSED

The paper’s cross‑sectional finding — that short‑term trend profits persist only on contracts with large volatility‑normalized tick sizes — translates directly to a falsifiable FX hypothesis. The required inputs (close prices and fixed tick sizes) are already available in the codebase, and the strategy can be tested within the existing walk‑forward harness using joint multi‑asset mode. The abandon condition gives a clean kill‑switch. This is a well‑grounded, low‑cost test of a mechanism identified in recent microstructure research.
```