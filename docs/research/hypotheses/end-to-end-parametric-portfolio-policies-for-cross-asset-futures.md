# Hypothesis: Transformer-based end-to-end parametric portfolio policy outperforms equal-weight and time-series momentum on a G10 FX basket after transaction costs

## Source extract
- **path**: `data/research/extracts/70b807054e41c36c9dd795a14bd79f1943a7fa3d8707c00ab4d3a53d6c714dfc.md`
- **paper title**: End-to-End Parametric Portfolio Policies for Cross-Asset Futures Timing: When Do AI Models Beat Simple Rules?
- **paper hash**: `70b807054e41c36c9dd795a14bd79f1943a7fa3d8707c00ab4d3a53d6c714dfc`

## Change to baseline
None — this is a new strategy class. The codebase currently contains single-instrument time-series momentum and carry strategies; this hypothesis introduces a joint multi‑asset allocation strategy that generates portfolio weights for a basket of G10 FX pairs directly from a trained transformer model.

## Prediction
- `predicted_sharpe_range`: [0.3, 0.8] (OOS, net of transaction costs, annualized) — estimate derived from comparable parametric‑portfolio studies (e.g., Ehlers & Bekkers 2022) and the paper’s own abstract, which finds transformer policies maintain positive returns with moderate turnover after costs.
- `predicted_hit_rate_range`: [0.52, 0.60] — approximate, as the paper does not quote hit rates; this is a plausible range for a low‑turnover, Sharpe‑optimized multi‑asset policy.
- `expected_n_trades_per_year`: 80 (quarterly rebalancing with occasional smaller adjustments when the transformer’s weights shift meaningfully — consistent with the paper’s description of "trades far less" than the LSTM after costs).
- `regime_dependence`: `regime-agnostic` (the transformer should learn to adapt to any regime present in the training window; no explicit ex‑ante regime classification is imposed).
- `time_to_signal`: `daily` (the model is trained on daily returns and outputs positions for the next day’s close; rebalancing occurs at end‑of‑day, but actual trade frequency is governed by the quarterly walk‑forward step and the transformer’s weight changes).

## Abandon condition
- OOS Sharpe ratio (net of transaction costs, annualized) < 0 over any rolling 6‑month (126‑trading‑day) window.

## Data requirements
- Daily close prices for the following G10 FX pairs (all accessible via the existing `prices` table):
  - `EURUSD`, `USDJPY`, `GBPUSD`, `AUDUSD`, `NZDUSD`, `USDCAD`, `USDCHF`, `USDNOK`, `USDSEK` (or a comparable set of 8–10 liquid crosses).
- No external datasets required; the model uses only daily returns computed from these close series.

## References
- the source extract: `data/research/extracts/70b807054e41c36c9dd795a14bd79f1943a7fa3d8707c00ab4d3a53d6c714dfc.md`
- canonical reference: Pollok & Robik (2026) "End-to-End Parametric Portfolio Policies for Cross-Asset Futures Timing", arXiv:2607.00475

## Final position
**FINAL_POSITION**: PROPOSED

The extract provides a clear, testable thesis — train a transformer policy end‑to‑end on daily FX returns using a differentiable Sharpe loss, then compare to simple rules — and the required data (daily FX close prices) is already ingested. The hypothesis is falsifiable via the Sharpe‑based abandon condition and falls within the harness’s support for joint multi‑asset strategies (CL‑40n2 v2). The implementer can train the transformer inside each walk‑forward in‑sample window and output a position matrix for the OOS quarter, fitting the harness constraints.