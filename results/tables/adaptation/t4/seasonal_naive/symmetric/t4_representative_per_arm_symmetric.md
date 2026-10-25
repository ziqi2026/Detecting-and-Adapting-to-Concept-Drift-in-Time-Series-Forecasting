# T4 — adaptation cost, representative configuration per arm (symmetric)

Regime convention: `symmetric` (drift = 7 days before + event + 7 days after). Mean ± sample SD over 5 seeds; MAE gain vs arm A = MAE_drift(arm) − MAE_drift(A), paired on region and seed (negative = better); gain per retrain = gain / retrains (computed per seed, then averaged; — when there are no retrains). Forecaster: `seasonal_naive`. Deterministic forecaster: all five seeds give identical results, so sd = 0.0 reflects no seed variation, not independent replicates. Seasonal-naive has no fitted state: `fit()` is a no-op and `predict()` reads the observation 48 half-hours (one day) earlier. A retrain therefore changes no forecast. `retrains` and `cumulative train samples` are the schedule's bookkeeping (the refit data selection is unchanged) and `wall clock` is the cost of an empty `fit()`.

| forecaster | arm_label | region | retrains | cumulative train samples | wall clock (s) | MAE gain vs arm A | gain per retrain | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive | A · never retrain | NSW1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| seasonal_naive | B · periodic 30d | NSW1 | 46 | 3,203,519 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C · drift + full history (KSWIN) | NSW1 | 25 | 1,662,579 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D · drift + 180d window (Page-Hinkley) | NSW1 | 5 | 78,244 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | A · never retrain | SA1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| seasonal_naive | B · periodic 30d | SA1 | 46 | 3,203,519 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | C · drift + full history (KSWIN) | SA1 | 17 | 1,182,004 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
| seasonal_naive | D · drift + 180d window (Page-Hinkley) | SA1 | 5 | 75,364 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 5 |
