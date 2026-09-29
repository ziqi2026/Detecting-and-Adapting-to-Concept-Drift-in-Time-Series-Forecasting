# T3 — Forecast accuracy, split by regime

Three regime columns are reported rather than one overall test-set MAE. A single overall MAE would conceal the forecast-accuracy effects of the COVID-19 demand-pattern change.

| arm | region | MAE pre-drift | MAE drift | MAE post-drift | pinball loss | seeds |
|---|---|---:|---:|---:|---:|---:|
| A · never retrain | SA1 | 149.4 | 112.3 | 148.9 | 26.8 | 1 |
| B · periodic 30d | SA1 | 149.4 | 110.8 | 137.5 | 26.1 | 1 |
| C · drift + full history (KSWIN) | SA1 | 149.4 | 112.3 | 137.4 | 26.0 | 1 |
| D · drift + 180d window (Page-Hinkley) | SA1 | 149.4 | 112.3 | 183.1 | 31.1 | 1 |
| A · never retrain | NSW1 | 280.9 | 236.8 | 380.1 | 76.6 | 1 |
| B · periodic 30d | NSW1 | 280.9 | 231.4 | 348.3 | 75.0 | 1 |
| C · drift + full history (KSWIN) | NSW1 | 280.9 | 231.3 | 347.5 | 75.0 | 1 |
| D · drift + 180d window (Page-Hinkley) | NSW1 | 280.9 | 236.8 | 501.0 | 87.9 | 1 |

## Reporting notes

- **Pre-drift:** before 16 March 2020.
- **Drift:** 16 March 2020 through 17 May 2020, inclusive.
- **Post-drift:** from 18 May 2020 onward.
- MAE is in MW; lower is better.
- One adaptation run is available (`seed = 0`), so each result is reported as one number rather than mean ± standard deviation.
- Pinball loss is the average of 5th- and 95th-percentile pinball losses. For all adaptation arms, intervals use the PGS242-141 region-specific fixed conformal radius around each arm's point forecast. This is a transferred fixed-radius interval score, not independently recalibrated uncertainty for B/C/D.
- C-KSWIN is the strict full-test MAE winner among Arm C variants. D-180 Page-Hinkley is the best tested Arm D configuration, but it still underperforms A, B, and C overall.

## Interpretation

For both regions, periodic retraining (B) and drift-triggered full-history retraining (C-KSWIN) improve post-drift accuracy relative to never retraining (A). The selected recent-window strategy (D) is substantially worse after drift, indicating that restricting retraining to a recent 180-day window was not effective for this Gradient Boosting setup.
