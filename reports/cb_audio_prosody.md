# FOMC presser audio prosody vs EUR/USD reaction (CL-68e prototype)

Generated 2026-07-14T23:06:05.589104+00:00 — whisper `base.en`, first 15.0 min of each presser (statement reading; Q&A out of scope).

## Per-presser features

| meeting | words | wpm (speech) | pitch mean Hz | pitch std Hz | pauses/min | pause frac | text net | stmt net_shift (DB) | EURUSD reaction | window |
|---|---|---|---|---|---|---|---|---|---|---|
| 20260617 | 2155 | 200 | 144.2 | 37.3 | 17.3 | 0.21 | +0.00 | n/a | -0.358% | 5m_post_statement |
| 20260429 | 2506 | 226 | 113.1 | 28.3 | 15.5 | 0.15 | -0.31 | n/a | -0.128% | 60m_post_statement |
| 20260318 | 2487 | 204 | 120.3 | 37.3 | 6.5 | 0.04 | -0.75 | 0.0 | -0.322% | 60m_post_statement |

## Feature ↔ reaction correlations

| feature | pearson r | spearman r | n |
|---|---|---|---|
| pitch_mean_hz | -0.78 | -1.00 | 3 |
| pitch_std_hz | -0.99 | -0.50 | 3 |
| pitch_range_hz | -0.92 | -0.50 | 3 |
| pauses_per_min | +0.21 | -0.50 | 3 |
| pause_mean_s | -0.14 | -0.50 | 3 |
| pause_fraction | -0.02 | -0.50 | 3 |
| wpm_speech | +1.00 | +1.00 | 3 |
| wpm_total | +0.65 | +1.00 | 3 |
| rms_cv | -0.36 | -0.50 | 3 |
| text_net | -0.05 | -0.50 | 3 |

## Read

On this n=3 sample the strongest prosody correlate of the EUR/USD post-statement move is `wpm_speech` (r=+1.00) vs text-sentiment baseline |r|=0.05. Tone features look additive to text sentiment here — but n=2-3 is anecdotal: one degree of freedom, no p-values, and the reaction windows mix granularities (5m vs 60m). Treat this strictly as a pipeline proof and a hypothesis to test on 20+ pressers, not as signal.
