# T3 — full-stream forecast accuracy by regime (buffer mode: post_only)

Drift window = documented event + 7 days after it (`[event_start, event_end + 1 day + 7 days)`); pre-drift = everything before the first drift window; post-drift = everything after it outside the union of drift windows.

Forecaster: `seasonal_naive`.

Deterministic forecaster: all five seeds give identical results, so sd = 0.0 reflects no seed variation, not independent replicates.

| forecaster | arm | region | MAE pre-drift | MAE drift | MAE post-drift | pinball loss | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive | A · never retrain | NSW1 | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | B · periodic 30d | NSW1 | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | C · drift + full history (KSWIN) | NSW1 | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | D · drift + 180d window (Page-Hinkley) | NSW1 | 402.1 ± 0.0 | 430.8 ± 0.0 | 446.0 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | A · never retrain | SA1 | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 29.9 ± 0.0 | 5 |
| seasonal_naive | B · periodic 30d | SA1 | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 29.9 ± 0.0 | 5 |
| seasonal_naive | C · drift + full history (KSWIN) | SA1 | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 29.9 ± 0.0 | 5 |
| seasonal_naive | D · drift + 180d window (Page-Hinkley) | SA1 | 149.0 ± 0.0 | 161.8 ± 0.0 | 154.0 ± 0.0 | 29.9 ± 0.0 | 5 |
