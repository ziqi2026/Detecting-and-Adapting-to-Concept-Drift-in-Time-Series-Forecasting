# T3 — full-stream forecast accuracy by regime (buffer mode: symmetric)

Drift window = 7 days before + documented event + 7 days after (`[event_start - 7 days, event_end + 1 day + 7 days)`); pre-drift = everything before the first drift window; post-drift = everything after it outside the union of drift windows.

Forecaster: `seasonal_naive`.

Deterministic forecaster: all five seeds give identical results, so sd = 0.0 reflects no seed variation, not independent replicates.

| forecaster | arm | region | MAE pre-drift | MAE drift | MAE post-drift | pinball loss | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seasonal_naive | A · never retrain | NSW1 | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | B · periodic 30d | NSW1 | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | C · drift + full history (KSWIN) | NSW1 | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | D · drift + 180d window (Page-Hinkley) | NSW1 | 519.3 ± 0.0 | 441.1 ± 0.0 | 442.1 ± 0.0 | 89.0 ± 0.0 | 5 |
| seasonal_naive | A · never retrain | SA1 | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 29.9 ± 0.0 | 5 |
| seasonal_naive | B · periodic 30d | SA1 | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 29.9 ± 0.0 | 5 |
| seasonal_naive | C · drift + full history (KSWIN) | SA1 | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 29.9 ± 0.0 | 5 |
| seasonal_naive | D · drift + 180d window (Page-Hinkley) | SA1 | 121.5 ± 0.0 | 160.8 ± 0.0 | 153.6 ± 0.0 | 29.9 ± 0.0 | 5 |
