# Forecaster selection — SA1 and NSW1

**Validation:** October–December 2019, using only the 2018–2019 training partition. No calibration or test leakage.

Selection by lowest mean validation MAE among runtime-eligible models. Rolling MAE (336 half-hours), per-seed MAE and separate runtimes are also reported.

| Region | Model | 7-day rolling MAE (MW) | MAE (MW) | Fit (s) | Predict (s) | Total (s) | Seeds | Pick |
|---|---|---|---|---|---|---|---|---|
| SA1 | Seasonal Naive | 173.58 | 179.57 | 0.00 | 0.00 | 0.00 | 1/1 |  |
| SA1 | SARIMAX | 38.89 | 38.92 | 26.25 | 0.02 | 26.28 | 1/1 |  |
| SA1 | Gradient Boosting | 167.22 ± 1.81 | 174.47 ± 1.92 | 0.15 ± 0.01 | 0.01 ± 0.00 | 0.16 ± 0.01 | 5/5 |  |
| SA1 | PatchTST | 38.30 ± 3.13 | 38.52 ± 3.10 | 34.67 ± 6.00 | 0.14 ± 0.04 | 34.81 ± 6.00 | 5/5 | ✓ |
| NSW1 | Seasonal Naive | 457.59 | 449.70 | 0.00 | 0.00 | 0.00 | 1/1 |  |
| NSW1 | SARIMAX | 90.98 | 92.52 | 16.19 | 0.02 | 16.21 | 1/1 | ✓ |
| NSW1 | Gradient Boosting | 326.98 ± 3.63 | 324.22 ± 3.77 | 0.20 ± 0.10 | 0.01 ± 0.00 | 0.21 ± 0.10 | 5/5 |  |
| NSW1 | PatchTST | 116.54 ± 11.91 | 117.20 ± 11.42 | 18.10 ± 1.86 | 0.13 ± 0.01 | 18.23 ± 1.87 | 5/5 |  |

**Interpretation:** F1 still evaluates all frozen baselines on the test stream; this selection uses no F1 degradation results. One forecaster/family should be aligned between teams before T7 trigger comparisons.
