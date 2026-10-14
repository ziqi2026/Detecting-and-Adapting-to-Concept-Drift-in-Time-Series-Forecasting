# T3 — full forecast accuracy by regime

| arm | region | MAE pre-drift | MAE drift | MAE post-drift | pinball loss | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| A · never retrain | NSW1 | 336.9 ± 5.0 | 380.1 ± 1.4 | 365.8 ± 1.4 | 76.6 ± 1.0 | 5 |
| B · periodic 30d | NSW1 | 336.9 ± 5.0 | 345.6 ± 1.0 | 340.5 ± 0.4 | 75.1 ± 1.0 | 5 |
| C · drift + full history (KSWIN) | NSW1 | 336.9 ± 5.0 | 348.5 ± 0.6 | 341.3 ± 0.7 | 75.2 ± 1.1 | 5 |
| D · drift + 180d window (Page-Hinkley) | NSW1 | 336.9 ± 5.0 | 452.7 ± 0.7 | 441.4 ± 0.5 | 83.8 ± 0.7 | 5 |
| A · never retrain | SA1 | 127.1 ± 2.4 | 153.2 ± 0.7 | 143.2 ± 0.4 | 26.9 ± 0.3 | 5 |
| B · periodic 30d | SA1 | 127.1 ± 2.4 | 142.7 ± 0.4 | 133.1 ± 0.4 | 26.2 ± 0.3 | 5 |
| C · drift + full history (KSWIN) | SA1 | 127.1 ± 2.4 | 143.5 ± 0.4 | 133.1 ± 0.3 | 26.1 ± 0.3 | 5 |
| D · drift + 180d window (Page-Hinkley) | SA1 | 127.1 ± 2.4 | 182.9 ± 0.2 | 180.1 ± 0.0 | 31.1 ± 0.0 | 5 |
