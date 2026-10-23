# T3 — local forecast accuracy by regime

| arm | region | MAE pre-drift | MAE drift | MAE post-drift | pinball loss | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| A · never retrain | NSW1 | 417.8 ± 2.7 | 358.1 ± 1.8 | 377.8 ± 3.1 | 77.6 ± 1.0 | 5 |
| B · periodic 30d | NSW1 | 372.7 ± 1.8 | 329.8 ± 2.0 | 343.7 ± 2.9 | 75.6 ± 1.0 | 5 |
| C · drift + full history (KSWIN) | NSW1 | 379.2 ± 1.4 | 330.9 ± 1.4 | 346.2 ± 0.9 | 75.6 ± 1.1 | 5 |
| D · drift + 180d window (Page-Hinkley) | NSW1 | 507.2 ± 0.9 | 426.7 ± 1.1 | 439.1 ± 0.6 | 85.3 ± 0.6 | 5 |
| A · never retrain | SA1 | 147.6 ± 0.7 | 165.5 ± 1.4 | 138.7 ± 0.6 | 27.6 ± 0.2 | 5 |
| B · periodic 30d | SA1 | 134.8 ± 0.8 | 154.5 ± 0.2 | 131.5 ± 0.6 | 27.0 ± 0.2 | 5 |
| C · drift + full history (KSWIN) | SA1 | 136.4 ± 0.6 | 155.4 ± 0.6 | 131.6 ± 0.6 | 26.9 ± 0.3 | 5 |
| D · drift + 180d window (Page-Hinkley) | SA1 | 172.1 ± 0.5 | 194.7 ± 0.1 | 175.2 ± 0.0 | 32.5 ± 0.0 | 5 |
