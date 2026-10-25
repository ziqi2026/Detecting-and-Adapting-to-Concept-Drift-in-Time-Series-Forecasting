# Cross-forecaster adaptation summary (symmetric)

Regime convention: `symmetric` (Drift window = 7 days before + documented event + 7 days after (`[event_start - 7 days, event_end + 1 day + 7 days)`); pre-drift = everything before the first drift window; post-drift = everything after it outside the union of drift windows.). Headline configuration per arm as in T3/T4 (C = KSWIN, D = 180d window with Page-Hinkley). Mean ± sample SD over 5 seeds; MAE in MW; MAE gain vs arm A is the drift-regime MAE difference to arm A (negative = better). `seasonal_naive` is deterministic: its five seeds are identical, so sd = 0.0 carries no information about variability. `seasonal_naive`: arms B-D have exactly the same forecasts as arm A (gain 0.0).

| forecaster | region | arm | MAE pre-drift | MAE drift | MAE post-drift | retrains | MAE gain vs arm A | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosted | NSW1 | A · never retrain | 336.9 ± 5.0 | 380.1 ± 1.4 | 365.8 ± 1.4 | 0 | 0.0 ± 0.0 | 5 |
| gradient_boosted | NSW1 | B · periodic 30d | 336.9 ± 5.0 | 345.6 ± 1.0 | 340.5 ± 0.4 | 46 | -34.6 ± 2.3 | 5 |
| gradient_boosted | NSW1 | C · drift + full history (KSWIN) | 336.9 ± 5.0 | 348.5 ± 0.6 | 341.3 ± 0.7 | 25 | -31.6 ± 1.3 | 5 |
| gradient_boosted | NSW1 | D · drift + 180d window (Page-Hinkley) | 336.9 ± 5.0 | 452.7 ± 0.7 | 441.4 ± 0.5 | 5 | 72.6 ± 1.8 | 5 |
| seasonal_naive | NSW1 | A · never retrain | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 0 | 0.0 ± 0.0 | 5 |
| seasonal_naive | NSW1 | B · periodic 30d | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 46 | 0.0 ± 0.0 | 5 |
| seasonal_naive | NSW1 | C · drift + full history (KSWIN) | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 25 | 0.0 ± 0.0 | 5 |
| seasonal_naive | NSW1 | D · drift + 180d window (Page-Hinkley) | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 5 | 0.0 ± 0.0 | 5 |
| gradient_boosted | SA1 | A · never retrain | 127.1 ± 2.4 | 153.2 ± 0.7 | 143.2 ± 0.4 | 0 | 0.0 ± 0.0 | 5 |
| gradient_boosted | SA1 | B · periodic 30d | 127.1 ± 2.4 | 142.7 ± 0.4 | 133.1 ± 0.4 | 46 | -10.5 ± 0.5 | 5 |
| gradient_boosted | SA1 | C · drift + full history (KSWIN) | 127.1 ± 2.4 | 143.5 ± 0.4 | 133.1 ± 0.3 | 17 | -9.7 ± 0.5 | 5 |
| gradient_boosted | SA1 | D · drift + 180d window (Page-Hinkley) | 127.1 ± 2.4 | 182.9 ± 0.2 | 180.1 ± 0.0 | 5 | 29.7 ± 0.6 | 5 |
| seasonal_naive | SA1 | A · never retrain | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 0 | 0.0 ± 0.0 | 5 |
| seasonal_naive | SA1 | B · periodic 30d | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 46 | 0.0 ± 0.0 | 5 |
| seasonal_naive | SA1 | C · drift + full history (KSWIN) | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 17 | 0.0 ± 0.0 | 5 |
| seasonal_naive | SA1 | D · drift + 180d window (Page-Hinkley) | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 5 | 0.0 ± 0.0 | 5 |
