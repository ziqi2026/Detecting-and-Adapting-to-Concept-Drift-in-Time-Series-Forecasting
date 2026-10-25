# T4 — adaptation cost, all 17 configurations (post_only)

Regime convention: `post_only` (drift = event + 7 days after it). Mean ± sample SD over 5 seeds; MAE gain vs arm A = MAE_drift(arm) − MAE_drift(A), paired on region and seed (negative = better); gain per retrain = gain / retrains (computed per seed, then averaged; — when there are no retrains). Forecaster: `seasonal_naive`. Deterministic forecaster: all five seeds give identical results, so sd = 0.0 reflects no seed variation, not independent replicates. Seasonal-naive has no fitted state: `fit()` is a no-op and `predict()` reads the observation 48 half-hours (one day) earlier. A retrain therefore changes no forecast. `retrains` and `cumulative train samples` are the schedule's bookkeeping (the refit data selection is unchanged) and `wall clock` is the cost of an empty `fit()`.

| forecaster | arm_label | region | retrains | cumulative train samples | wall clock (s) | MAE gain vs arm A | gain per retrain | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive | A | NSW1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| seasonal_naive | B | NSW1 | 46 | 3,203,519 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C_adwin | NSW1 | 16 | 1,088,207 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C_kswin | NSW1 | 25 | 1,662,579 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C_page_hinkley | NSW1 | 5 | 390,967 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_30d_adwin | NSW1 | 16 | 58,095 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_30d_kswin | NSW1 | 25 | 71,064 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_30d_page_hinkley | NSW1 | 5 | 42,244 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_60d_adwin | NSW1 | 16 | 81,135 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_60d_kswin | NSW1 | 25 | 105,933 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_60d_page_hinkley | NSW1 | 5 | 49,444 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_120d_adwin | NSW1 | 16 | 127,215 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_120d_kswin | NSW1 | 25 | 174,877 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_120d_page_hinkley | NSW1 | 5 | 63,844 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_180d_adwin | NSW1 | 16 | 172,366 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_180d_kswin | NSW1 | 25 | 243,508 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_180d_page_hinkley | NSW1 | 5 | 78,244 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | A | SA1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| seasonal_naive | B | SA1 | 46 | 3,203,519 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C_adwin | SA1 | 13 | 910,258 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C_kswin | SA1 | 17 | 1,182,004 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C_page_hinkley | SA1 | 5 | 375,183 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_30d_adwin | SA1 | 13 | 53,772 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_30d_kswin | SA1 | 17 | 59,536 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_30d_page_hinkley | SA1 | 5 | 42,244 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_60d_adwin | SA1 | 13 | 72,492 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_60d_kswin | SA1 | 17 | 84,016 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_60d_page_hinkley | SA1 | 5 | 48,059 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_120d_adwin | SA1 | 13 | 107,691 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_120d_kswin | SA1 | 17 | 132,650 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_120d_page_hinkley | SA1 | 5 | 60,964 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_180d_adwin | SA1 | 13 | 144,492 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_180d_kswin | SA1 | 17 | 176,618 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D_180d_page_hinkley | SA1 | 5 | 75,364 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
