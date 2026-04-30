```markdown
# Hypothesis: Minimum Regime Performance (MRP) can serve as a pre-trade filter to avoid deploying FX momentum strategies during regimes where their historical risk-adjusted return is lowest, thereby improving out-of-sample durability.

## Source extract
- **path**: `data/research/extracts/3f3a74a72587dab7703d4e5feb82924b9448dca2c326f9c1e69d028c6e906fc6.md`
- **paper title**: Measuring Strategy-Decay Risk: Minimum Regime Performance and the Durability of Systematic Investing
- **paper hash**: `3f3a74a72587dab7703d4e5feb82924b9448dca2c326f9c1e69d028c6e906fc6`

## Change to baseline
This modifies the existing `TrendFollowingStrategy` in `src/strategies/trend_following.py`. The baseline strategy is always-on: it generates a momentum signal and takes a position regardless of the prevailing regime. The proposed change adds a regime filter that disables the strategy when the current regime is identified as one where the strategy's historical MRP is lowest (e.g., a high-volatility, choppy market regime). The filter is estimated from in-sample data and applied out-of-sample.

## Prediction
- `predicted_sharpe_range`: [0.4, 0.8]            # annualized OOS; baseline trend-following in FX typically 0.3-0.6, filter should improve by reducing drawdowns
- `predicted_hit_rate_range`: [0.52, 0.60]         # 0.0–1.0; baseline hit rate ~50-55%, filter should increase by avoiding losing regimes
- `expected_n_trades_per_year`: 40                 # reduced from ~52 (weekly) due to regime filter disabling trades ~25% of the time
- `regime_dependence`: "regime-agnostic"           # the filter is designed to make the strategy regime-agnostic by avoiding its worst regime
- `time_to_signal`: "weekly"                       # regime classification updated weekly

The Sharpe range is an estimate derived from comparable papers on regime-filtered momentum; the extract itself does not quote a Sharpe. The hit rate range is estimated from the premise that avoiding the worst regime should improve the proportion of winning trades.

## Abandon condition
- "OOS Sharpe < 0 over any rolling 6-month window" — if the regime filter fails to prevent a sustained period of negative returns, the strategy is not durable and should be unwound.

## Data requirements
- `prices.{symbol}.close` — daily close prices for the primary FX symbol (e.g., EURUSD)
- A regime classification series derived from in-sample data. This is NOT YET INGESTED — it requires a data-seeding ticket to define and compute regime states (e.g., based on rolling volatility, trend strength, or a clustering algorithm) from the price series. The specific regime definition is part of the hypothesis and must be implemented by the Implementer agent.

## References
- `data/research/extracts/3f3a74a72587dab7703d4e5feb82924b9448dca2c326f9c1e69d028c6e906fc6.md` — source extract
- No other extracts or canonical works are directly cited in the abstract.

## Final position
**FINAL_POSITION**: PROPOSED

The extract introduces a clear, falsifiable concept (MRP) that can be operationalized as a regime filter for an existing single-asset time-series strategy (FX momentum). The hypothesis is testable within the current backtest harness constraints: it requires only a single close price series and a derived regime classification. The abandon condition is measurable. The primary risk is that the regime classification itself is not pre-defined in the extract and must be specified by the Implementer, but this is a standard implementation detail rather than a fatal ambiguity. The data requirement for a regime series is flagged as not yet ingested, but it can be derived from the existing price data, so no external data seeding is strictly required.
```