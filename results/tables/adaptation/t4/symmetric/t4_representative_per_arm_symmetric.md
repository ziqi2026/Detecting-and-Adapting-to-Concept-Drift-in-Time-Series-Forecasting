# T4 — adaptation cost, representative configuration per arm (symmetric)

Regime convention: `symmetric` (drift = 7 days before + event + 7 days after). Mean ± sample SD over 5 seeds; MAE gain vs arm A = MAE_drift(arm) − MAE_drift(A), paired on region and seed (negative = better); gain per retrain = gain / retrains (computed per seed, then averaged; — when there are no retrains).

| arm_label | region | retrains | cumulative train samples | wall clock (s) | MAE gain vs arm A | gain per retrain | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A · never retrain | NSW1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| B · periodic 30d | NSW1 | 46 | 3,203,519 | 23.1 ± 3.8 | -34.6 ± 2.3 | -0.75 ± 0.05 | 5 |
| C · drift + full history (KSWIN) | NSW1 | 25 | 1,662,579 | 11.4 ± 0.4 | -31.6 ± 1.3 | -1.26 ± 0.05 | 5 |
| D · drift + 180d window (Page-Hinkley) | NSW1 | 5 | 78,244 | 1.6 ± 0.0 | 72.6 ± 1.8 | 14.52 ± 0.36 | 5 |
| A · never retrain | SA1 | 0 | 35,039 | 0.0 ± 0.0 | 0.0 ± 0.0 | — | 5 |
| B · periodic 30d | SA1 | 46 | 3,203,519 | 22.8 ± 3.1 | -10.5 ± 0.5 | -0.23 ± 0.01 | 5 |
| C · drift + full history (KSWIN) | SA1 | 17 | 1,182,004 | 8.5 ± 1.0 | -9.7 ± 0.5 | -0.57 ± 0.03 | 5 |
| D · drift + 180d window (Page-Hinkley) | SA1 | 5 | 75,364 | 1.7 ± 0.1 | 29.7 ± 0.6 | 5.94 ± 0.12 | 5 |
