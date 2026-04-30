# Hypothesis: Representation homogeneity among AI-driven FX traders amplifies synchronized deleveraging during stress events, creating exploitable volatility clustering

## Source extract
- **path**: `data/research/extracts/e93d4a224d95b45a28dd5f7e0124262c267e61bee8f1da2cc01849551e293984.md`
- **paper title**: Representation Homogeneity and Systemic Instability in AI-Dominated Financial Markets: A Structural Approach
- **paper hash**: `e93d4a224d95b45a28dd5f7e0124262c267e61bee8f1da2cc01849551e293984`

## Change to baseline
None — this is a new strategy class. The existing codebase has no strategy that explicitly models representation homogeneity or synchronized deleveraging dynamics. The closest existing strategy is `src/strategies/volatility_targeting.py`, which adjusts position size based on realized volatility but does not attempt to detect or exploit regime shifts driven by AI trader synchronization.

## Prediction
- `predicted_sharpe_range`: [0.3, 0.7]            # annualized OOS
- `predicted_hit_rate_range`: [0.52, 0.60]          # 0.0–1.0
- `expected_n_trades_per_year`: 12
- `regime_dependence`: "vol-spike-only"
- `time_to_signal`: "daily"

**Rationale for ranges**: The paper does not quote a Sharpe ratio. Estimate derived from comparable volatility-clustering and regime-switching strategies in FX (e.g., Menkhoff et al. 2012 on FX volatility risk premia, which reports Sharpe ratios in the 0.3–0.8 range OOS). The strategy is inherently episodic — it only generates signals during periods of synchronized deleveraging, which occur infrequently. Hit rate is modest because the signal-to-noise ratio in identifying the onset of synchronized deleveraging is low.

## Abandon condition
- OOS Sharpe < 0 over any rolling 12-month window, OR realized hit rate < 45% over the most recent 30 trades.

## Data requirements
- `prices.{symbol}` — daily close prices for the primary FX pair (e.g., EURUSD, USDJPY). The strategy requires at least 5 years of daily data to calibrate the "normal times" volatility regime and detect deviations.
- Realized volatility series computed from daily returns (can be derived from `prices.{symbol}`).
- **No additional data sources required** — the strategy operates on a single time series of close prices and derived volatility. The representation homogeneity mechanism is proxied by the volatility clustering signature itself, not by direct measurement of AI agent representations (which is not publicly observable).

## References
- `data/research/extracts/e93d4a224d95b45a28dd5f7e0124262c267e61bee8f1da2cc01849551e293984.md`
- Menkhoff, L., Sarno, L., Schmeling, M., & Schrimpf, A. (2012). Carry trades and global foreign exchange volatility. *Journal of Finance*, 67(2), 681-718. (Canonical reference for FX volatility risk premia; not in extract backlog.)

## Final position
**FINAL_POSITION**: PROPOSED

**Rationale**: The extract describes a falsifiable mechanism — representation homogeneity among AI traders leads to synchronized deleveraging during stress, which manifests as volatility clustering in FX price series. This can be tested as a single-asset time-series strategy: detect periods of compressed volatility (where hidden leverage accumulates) followed by sudden volatility spikes (synchronized deleveraging), and position accordingly (e.g., short during the spike, or long volatility via a regime-switching position-sizing rule). The strategy requires only daily close prices and derived volatility, which the backtest harness supports. The abandon condition is measurable. The predicted Sharpe range is conservative and grounded in comparable FX volatility strategies. The primary risk is that the volatility clustering signature is too weak to trade profitably after transaction costs, but this is precisely what the backtest will determine.