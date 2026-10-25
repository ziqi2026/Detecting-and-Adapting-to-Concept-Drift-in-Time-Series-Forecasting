# Cross-forecaster adaptation summary (post_only)

Regime convention: `post_only` (Drift window = documented event + 7 days after it (`[event_start, event_end + 1 day + 7 days)`); pre-drift = everything before the first drift window; post-drift = everything after it outside the union of drift windows.). Headline configuration per arm as in T3/T4 (C = KSWIN, D = 180d window with Page-Hinkley). Mean ± sample SD over 5 seeds; MAE in MW; MAE gain vs arm A is the drift-regime MAE difference to arm A (negative = better). `seasonal_naive` is deterministic: its five seeds are identical, so sd = 0.0 carries no information about variability. `seasonal_naive`: arms B-D have exactly the same forecasts as arm A (gain 0.0).

| forecaster | region | arm | MAE pre-drift | MAE drift | MAE post-drift | retrains | MAE gain vs arm A | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosted | NSW1 | A · never retrain | 284.8 ± 4.4 | 366.4 ± 1.4 | 371.8 ± 1.4 | 0 | 0.0 ± 0.0 | 5 |
| gradient_boosted | NSW1 | B · periodic 30d | 284.8 ± 4.4 | 334.9 ± 1.0 | 344.7 ± 0.3 | 46 | -31.6 ± 2.1 | 5 |
| gradient_boosted | NSW1 | C · drift + full history (KSWIN) | 284.8 ± 4.4 | 336.6 ± 1.0 | 346.0 ± 0.8 | 25 | -29.9 ± 1.4 | 5 |
| gradient_boosted | NSW1 | D · drift + 180d window (Page-Hinkley) | 284.8 ± 4.4 | 430.6 ± 0.6 | 449.9 ± 0.3 | 5 | 64.1 ± 1.6 | 5 |
| seasonal_naive | NSW1 | A · never retrain | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 0 | 0.0 ± 0.0 | 5 |
| seasonal_naive | NSW1 | B · periodic 30d | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 46 | 0.0 ± 0.0 | 5 |
| seasonal_naive | NSW1 | C · drift + full history (KSWIN) | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 25 | 0.0 ± 0.0 | 5 |
| seasonal_naive | NSW1 | D · drift + 180d window (Page-Hinkley) | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 5 | 0.0 ± 0.0 | 5 |
| gradient_boosted | SA1 | A · never retrain | 145.4 ± 5.6 | 155.5 ± 0.9 | 143.6 ± 0.4 | 0 | 0.0 ± 0.0 | 5 |
| gradient_boosted | SA1 | B · periodic 30d | 145.4 ± 5.6 | 145.6 ± 0.3 | 133.2 ± 0.4 | 46 | -9.8 ± 1.0 | 5 |
| gradient_boosted | SA1 | C · drift + full history (KSWIN) | 145.4 ± 5.6 | 146.0 ± 0.4 | 133.4 ± 0.3 | 17 | -9.4 ± 0.7 | 5 |
| gradient_boosted | SA1 | D · drift + 180d window (Page-Hinkley) | 145.4 ± 5.6 | 186.5 ± 0.0 | 179.4 ± 0.0 | 5 | 31.0 ± 0.9 | 5 |
| seasonal_naive | SA1 | A · never retrain | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 0 | 0.0 ± 0.0 | 5 |
| seasonal_naive | SA1 | B · periodic 30d | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 46 | 0.0 ± 0.0 | 5 |
| seasonal_naive | SA1 | C · drift + full history (KSWIN) | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 17 | 0.0 ± 0.0 | 5 |
| seasonal_naive | SA1 | D · drift + 180d window (Page-Hinkley) | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 5 | 0.0 ± 0.0 | 5 |
