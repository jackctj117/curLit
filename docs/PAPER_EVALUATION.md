# Paper evaluation rubric (CL-1i7)

A 6-dimension scoring framework for triaging research papers into the
implementation queue. Total range: **-5 to +25**. Score ≥ 18 → propose
paper-mode strategy; 12–17 → discuss; < 12 → archive.

The rubric exists because the paper ingester (CL-5b6) will surface ~20
new arXiv/NBER/SSRN papers per week, and most of them won't be worth
implementing. Triage by gut produces inconsistent decisions across
weeks. Triage by score is reproducible and audit-able.

---

## Six dimensions

### 1. Economic Rationale (0 to 5)

Does the paper explain *why* the edge exists in terms of market structure,
behavioral bias, or institutional flow?

| Score | Description                                                              |
| ----- | ------------------------------------------------------------------------ |
| 5     | Identifies a specific friction or constraint (regulatory, liquidity, behavioral) tied to a known market structure |
| 4     | Plausible mechanism with prior literature support                        |
| 3     | Hand-wavy but consistent with macro intuition                             |
| 2     | Pure data-mining ("we found a pattern") with weak rationale              |
| 1     | Statistical anomaly with no theory                                        |
| 0     | No economic rationale offered                                             |

### 2. Data Quality (0 to 5)

Are the data sources realistic? Survivorship-biased? Look-ahead-biased?

| Score | Description                                                              |
| ----- | ------------------------------------------------------------------------ |
| 5     | Tradeable data (real prices, real volumes, real trade times); no leak    |
| 4     | Cleaned-but-real data with explicit handling of corporate actions / delistings |
| 3     | Daily data only; some look-ahead concerns flagged but mitigated          |
| 2     | Suspected look-ahead (e.g. "PIT" claims with no audit)                   |
| 1     | Survivorship-biased (ETF holdings as of today, applied to history)       |
| 0     | Synthetic / simulation only                                               |

### 3. Implementation Feasibility (0 to 5)

Can we build this with curLit's data + execution stack?

| Score | Description                                                              |
| ----- | ------------------------------------------------------------------------ |
| 5     | Already-ingested data (FRED + yfinance); standard execution              |
| 4     | One new data source (e.g. CFTC) on the roadmap                           |
| 3     | Needs paid feed (Bloomberg, Refinitiv) we don't have                     |
| 2     | Needs alternative data (satellite, web scraping) requiring pipeline work |
| 1     | Requires options/futures plumbing we don't currently support             |
| 0     | Requires HFT infra (microsecond execution, colocation)                   |

### 4. Edge Persistence (0 to 5)

Will the edge survive widespread knowledge?

| Score | Description                                                              |
| ----- | ------------------------------------------------------------------------ |
| 5     | Structural (e.g. central-bank intervention asymmetry); decade-stable     |
| 4     | Behavioral with limits-to-arbitrage support; multi-year persistence      |
| 3     | Crowded but still profitable; edge has been published 3+ years ago and tests still work |
| 2     | Pre-publication backtest only; could disappear post-publication          |
| 1     | Already-arbitraged signal (Sharpe 2.0 in paper, ~0.3 in recent live data)|
| 0     | Single-period anomaly with no replication                                 |

### 5. Diversification Value (0 to 5)

How does it correlate with our existing strategies?

| Score | Description                                                              |
| ----- | ------------------------------------------------------------------------ |
| 5     | Orthogonal to current book (\|ρ\| < 0.1)                                  |
| 4     | Mostly diversifying (\|ρ\| < 0.3)                                         |
| 3     | Some overlap (\|ρ\| < 0.5)                                                |
| 2     | Strong overlap with rate_diff or carry_vol                                |
| 1     | Essentially the same trade as something we already run                    |
| 0     | Unknown — paper doesn't expose enough to estimate                          |

### 6. Red Flag Check (0 to -5 penalty)

Subtract points for each of the following present:

| Penalty | Flag                                                                     |
| ------- | ------------------------------------------------------------------------ |
| -1      | Authors are at a fund that already trades this — they wouldn't publish unless it had decayed |
| -1      | Sample period ends before a known regime break (e.g. ends 2007 ignoring GFC) |
| -1      | Stationary-bootstrap CI not reported; only point Sharpe                  |
| -1      | More than 5 hyperparameters with no walk-forward validation              |
| -1      | Only the headline strategy works; sub-period robustness fails            |

Bottom-out at -5. Do not give bonuses (positive scores) here — this
section is purely for downward adjustment.

---

## Worked examples

### Menkhoff, Sarno, Schmeling, Schrimpf (2012) — *Carry Trades and Global Foreign Exchange Volatility*

| Dim | Score | Note |
| --- | ----- | ---- |
| 1 — Economic rationale | 5 | Volatility risk premium, well-grounded in ICAPM |
| 2 — Data quality | 5 | Bloomberg G10 spot+forward, real trade-able rates |
| 3 — Implementation | 4 | Need 3-month interbank rates + realized vol — both on the roadmap |
| 4 — Edge persistence | 4 | 30-year backtest; structural premium |
| 5 — Diversification | 4 | Correlated with rate_diff but distinct (vol-conditioned) |
| 6 — Red flags | -2 | Sample ends 2010 (no post-GFC test); 4 hyperparams (window, vol filter, weight, threshold) |
| **Total** | **20** | **PROMOTE** |

### Hypothetical: "Twitter sentiment predicts EUR/USD intraday"

| Dim | Score | Note |
| --- | ----- | ---- |
| 1 — Economic rationale | 1 | Pure pattern; weak mechanism |
| 2 — Data quality | 2 | Twitter API; survivorship of accounts undocumented |
| 3 — Implementation | 1 | Real-time sentiment pipeline we don't have |
| 4 — Edge persistence | 0 | Decay shown in extended sample |
| 5 — Diversification | 3 | Sentiment is at least different |
| 6 — Red flags | -5 | Authors at a sentiment-product company; no walk-forward; heavy hyperparam tuning; full sample picks the right model post-hoc; tested on a single year |
| **Total** | **2** | **ARCHIVE** |

---

## Pre-implementation sanity checks

After scoring ≥ 18, run through these before committing engineering time:

1. **Reproduce the headline statistic** on the paper's stated sample. If
   our reproduction misses by >0.1 Sharpe, the paper is hiding something —
   investigate before going further.
2. **Walk-forward test** with our standard cadence (CL-g7e
   `WalkForwardConfig`: 756d IS, 63d OOS). If OOS Sharpe < 0.3 or drops
   >50% vs IS, the paper's significance is overfit.
3. **Cost stress**: run with our cost model at 2× spread + 1bp impact.
   Strategies with Sharpe > 1.5 in paper but < 0.3 net of costs are not
   tradeable.
4. **Portfolio fit**: simulate 50/50 blend with current book; Sharpe of
   blend should beat current book by > 0.1.
5. **Decay test**: if the paper's sample ends >5 years ago, run on the
   period since publication. Live-period Sharpe should be at least 50%
   of the published Sharpe. If not, it's been arbitraged.

---

## Post-implementation tracking

Once a strategy is in paper mode, track these monthly until promotion:

- **Live-vs-paper Sharpe ratio**: live > 0.5 × paper for 3 consecutive months
- **Drawdown vs paper-claimed max DD**: live should not exceed 1.5×
- **Cost surprise**: TCA breakdown should match the cost stress model
  used in sanity check #3 within 0.5 bps
- **Capacity check**: at the strategy's planned size, what's the
  expected market impact? Compare to paper's stated capacity claims

A strategy that fails any of these for two consecutive months returns to
the research queue rather than getting promoted.
