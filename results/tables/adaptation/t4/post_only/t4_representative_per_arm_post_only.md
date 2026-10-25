# T4 — adaptation cost, representative configuration per arm (post_only)

Regime convention: `post_only` (drift = event + 7 days after it). Mean ± sample SD over 5 seeds; MAE gain vs arm A = MAE_drift(arm) − MAE_drift(A), paired on region and seed (negative = better); gain per retrain = gain / retrains (computed per seed, then averaged; — when there are no retrains).

| arm_label | region | retrains | cumulative train samples | wall clock (s) | MAE gain vs arm A | gain per retrain | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A · never retrain | NSW1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| B · periodic 30d | NSW1 | 46 | 3,203,519 | 23.1 ± 3.8 | -31.6 ± 2.1 | -0.69 ± 0.05 | 5 |
| C · drift + full history (KSWIN) | NSW1 | 25 | 1,662,579 | 11.4 ± 0.4 | -29.9 ± 1.4 | -1.19 ± 0.06 | 5 |
| D · drift + 180d window (Page-Hinkley) | NSW1 | 5 | 78,244 | 1.6 ± 0.0 | 64.1 ± 1.6 | 12.83 ± 0.32 | 5 |
| A · never retrain | SA1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| B · periodic 30d | SA1 | 46 | 3,203,519 | 22.8 ± 3.1 | -9.8 ± 1.0 | -0.21 ± 0.02 | 5 |
| C · drift + full history (KSWIN) | SA1 | 17 | 1,182,004 | 8.5 ± 1.0 | -9.4 ± 0.7 | -0.55 ± 0.04 | 5 |
| D · drift + 180d window (Page-Hinkley) | SA1 | 5 | 75,364 | 1.7 ± 0.1 | 31.0 ± 0.9 | 6.20 ± 0.18 | 5 |
